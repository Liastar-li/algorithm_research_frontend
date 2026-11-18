"""Stage 1 自检 —— 验证模型形状与论文一致（不需要数据、不需要训练）。

用法：
    conda activate vpp
    python algorithms/voiceprint_protection/scripts/test_stage1.py

检查什么：
  1. 各张量形状（对照 AutoVC 官方实测值）
  2. content code 的真实维度与个数（★ 论文写错了，这里以官方代码为准）
  3. 三个损失的行为（数值有限、可反传、方向正确）
  4. 熵最大化的效果：L_dis 确实能把分类器"骗瞎"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from stage1 import losses as stage1_losses  # noqa: E402
from stage1.model import Stage1Model, count_parameters  # noqa: E402

FAILURES: list[str] = []


def check(title: str, condition: bool, detail: str = "") -> None:
    status = "通过" if condition else "失败"
    print(f"  [{status}] {title}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(title)


def test_shapes() -> None:
    print("\n【1】张量形状（对照 AutoVC 官方实测）")
    torch.manual_seed(config.SEED)
    model = Stage1Model()
    print(f"  模型参数量：{count_parameters(model):,}")

    batch, frames = 2, 256
    mel = torch.randn(batch, frames, config.FRAME_DIM)
    embedding = torch.randn(batch, config.SPEAKER_DIM)
    embedding = torch.nn.functional.normalize(embedding, dim=-1)

    mel_out, mel_postnet, codes = model(mel, embedding, embedding)

    check("mel_out 形状 = (B, T, 80)", tuple(mel_out.shape) == (batch, frames, config.FRAME_DIM),
          f"实际 {tuple(mel_out.shape)}")
    check("mel_postnet 形状 = (B, T, 80)", tuple(mel_postnet.shape) == (batch, frames, config.FRAME_DIM),
          f"实际 {tuple(mel_postnet.shape)}")
    check("★ x 与 x_adv 帧数完全相等（契约硬要求）",
          mel_out.shape[1] == mel.shape[1], f"{mel_out.shape[1]} vs {mel.shape[1]}")

    freq = config.AUTOVC_MODEL["freq"]
    check(f"content code 个数 = T // freq = {frames // freq}", len(codes) == frames // freq,
          f"实际 {len(codes)}")
    check(f"content code 维度 = 2 * dim_neck = {config.CONTENT_ACTUAL_DIM}",
          tuple(codes[0].shape) == (batch, config.CONTENT_ACTUAL_DIM),
          f"实际 {tuple(codes[0].shape)}")

    # ★ 论文写 c ∈ R^{T'×32}、T' = T//2，与官方代码不符
    print(f"  [说明] 论文写 c ∈ R^{{T'×32}}、T' = T//2；")
    print(f"          官方代码实测是 (T//{freq}, {config.CONTENT_ACTUAL_DIM})，以代码为准。")


def test_losses() -> None:
    print("\n【2】三个损失")
    torch.manual_seed(config.SEED)
    batch, frames = 2, 256
    mel = torch.randn(batch, frames, config.FRAME_DIM)
    model = Stage1Model()
    embedding = torch.nn.functional.normalize(torch.randn(batch, config.SPEAKER_DIM), dim=-1)

    mel_out, mel_postnet, codes = model(mel, embedding, embedding)
    codes_out = model.content_codes(mel_out, embedding)
    codes_postnet = model.content_codes(mel_postnet, embedding)

    classifier = stage1_losses.SpeakerClassifier(
        num_speakers=config.STAGE1["num_speakers"], embedding_dim=config.SPEAKER_DIM)

    total, parts = stage1_losses.stage1_loss(
        mel, mel_out, mel_postnet, codes, codes_out, codes_postnet, classifier)

    for name in ("loss_rec", "loss_cd", "loss_dis", "total"):
        check(f"{name} 是有限数", np.isfinite(parts[name]), f"= {parts[name]:.4f}")

    total.backward()
    # LSTM 的参数名是 weight_ih_l0 / weight_hh_l0（双向有 _l0/_l1 两组）
    grad = model.encoder.lstm.weight_ih_l0.grad
    check("梯度能回传到编码器", grad is not None and bool(torch.isfinite(grad).all()))
    check("梯度非零（确实在学）", grad is not None and float(grad.abs().sum()) > 0)
    decoder_grad = model.decoder.lstm1.weight_ih_l0.grad
    check("梯度能回传到解码器", decoder_grad is not None and bool(torch.isfinite(decoder_grad).all()))

    # L_rec 的方向：越接近越 loss 越小
    perfect = stage1_losses.reconstruction_loss(mel, mel, mel)
    check("L_rec(完全相同) = 0", float(perfect) == 0.0, f"= {float(perfect)}")
    worse = stage1_losses.reconstruction_loss(mel, mel + 1.0, mel + 1.0)
    check("L_rec(偏差 1.0) 明显更大", float(worse) > 1.0, f"= {float(worse):.4f}")

    # L_cd 在 code 数量不一致时要报错（而不是静默算错）
    try:
        stage1_losses.content_consistency_loss(codes, codes[:-1], codes_postnet)
        check("code 数量不一致时报错", False)
    except ValueError:
        check("code 数量不一致时报错", True)


def test_entropy_maximization() -> None:
    print("\n【3】熵最大化（L_dis）的行为")
    torch.manual_seed(config.SEED)
    num_speakers = config.STAGE1["num_speakers"]

    classifier = stage1_losses.SpeakerClassifier(
        num_speakers=num_speakers, embedding_dim=config.SPEAKER_DIM)

    # 构造"不含说话人信息"的内容码：所有窗口都一样
    codes = [torch.randn(4, config.CONTENT_ACTUAL_DIM) * 0.01 for _ in range(6)]
    windows = stage1_losses.content_windows(codes)
    check("沿时间维拼接出多个窗口", len(windows) == len(codes) - config.STAGE1["content_window"] + 1,
          f"实际 {len(windows)}")
    check(f"★ 每个窗口维度 = 256（与论文 W_s ∈ R^{{K×256}} 一致）",
          windows[0].shape[-1] == config.SPEAKER_DIM,
          f"实际 {tuple(windows[0].shape)}")

    loss = stage1_losses.disentanglement_loss(codes, classifier)
    check("L_dis 是有限数", bool(torch.isfinite(loss)))

    # 极端情况：所有类概率相等 -> 熵最大 -> L_dis = log(K)
    uniform_logits = torch.zeros(1, num_speakers)
    uniform_probability = torch.softmax(uniform_logits, dim=-1)
    uniform_value = float(-(uniform_probability * uniform_probability.log()).sum())
    check(f"均匀分布时 L_dis ≈ log(K) = {np.log(num_speakers):.4f}",
          abs(uniform_value - float(loss.detach())) < 1e-3,
          f"实测 {float(loss.detach()):.4f}")
    print("  [说明] 最小化 L_dis = 最大化熵 = 让说话人分类器在内容码上完全猜不准。")

    # 极端情况：分类器非常确定 -> 熵小 -> L_dis 小（更容易被优化）
    confident_logits = torch.zeros(1, num_speakers)
    confident_logits[0, 0] = 50.0
    confident_probability = torch.softmax(confident_logits, dim=-1)
    confident_value = float(-(confident_probability * confident_probability.log()).sum())
    check("分类器很确定时 L_dis 明显更小", confident_value < float(loss.detach()),
          f"{confident_value:.6f} < {float(loss.detach()):.6f}")


def main() -> int:
    print("=" * 66)
    print("  Stage 1 自检（形状 / 损失 / 熵最大化）")
    print("=" * 66)

    test_shapes()
    test_losses()
    test_entropy_maximization()

    print()
    print("=" * 66)
    if FAILURES:
        print(f"  自检失败：{len(FAILURES)} 项未通过")
        for item in FAILURES:
            print(f"    - {item}")
        print("=" * 66)
        return 1
    print("  自检通过：Stage 1 的形状与损失都符合预期。")
    print("  下一步：准备 data/train.txt 后跑")
    print("    python algorithms/voiceprint_protection/stage1/train.py --stage classifier --iterations 2000")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
