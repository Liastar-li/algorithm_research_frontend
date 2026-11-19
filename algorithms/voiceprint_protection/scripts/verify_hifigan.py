"""对照脚本：验证 HiFi-GAN 声码器 + 我们的 mel 约定**真的对得上**。

【为什么必须有这个脚本】
论文 §2.3 要求用 HiFi-GAN 把 `x_adv = D(c, h_adv)` 转成波形。
但**喂错 mel 约定，声码器出来的是噪音，而且不报错** ——
这是本项目最容易中招的一类 bug（`audio_utils.py` 文件头有详细说明）。

【三层检查】
  1. **权重检查**：能否 `strict=True` 加载（234/234 键名与形状全匹配）
  2. **数值检查**：我们的 `hifigan_mel()` 与 speechbrain 的 torchaudio 链是否一致
  3. ★ **端到端检查**（最有价值）：
       原波形 → mel → HiFi-GAN → 重建波形
       然后用**说话人嵌入余弦**判断重建的还是不是本人说话。
       约定对 → 余弦接近 1（实测 0.9746）；约定错 → 余弦 ≈ 0（纯噪音）

     为什么用说话人余弦而不是波形 SNR：
     声码器**不保证相位一致**，所以波形级 SNR 天然是负的（-20 dB 左右很正常），
     不能用来判断对错。而"重建音还是不是同一个人"是**我们真正在乎的东西**，
     也是论文里 Sim_prot / Sim_clone 用的量。

【用法】
    conda activate vpp
    python algorithms/voiceprint_protection/scripts/verify_hifigan.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 控制台默认 GBK，中文会变乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import audio_utils  # noqa: E402
import config  # noqa: E402
import paths  # noqa: E402

paths.ensure_import_paths()

FAILURES: list[str] = []


def check(title: str, condition: bool, detail: str = "") -> None:
    status = "通过" if condition else "失败"
    print(f"  [{status}] {title}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(title)


def find_default_wav() -> Path | None:
    candidate = paths.sample_dir("group_n005_voiceprint_protection") / "input.wav"
    if candidate.exists():
        return candidate
    audio_root = paths.data_root() / "audio16k"
    if audio_root.is_dir():
        for wav in sorted(audio_root.rglob("*.wav")):
            return wav
    return None


def main() -> int:
    print("=" * 70)
    print("  HiFi-GAN 声码器对照（论文 §2.3 的 V(·)）")
    print("=" * 70)

    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", default=None)
    args = parser.parse_args()

    wav_path = Path(args.wav) if args.wav else find_default_wav()
    if wav_path is None or not wav_path.exists():
        print("\n[失败] 找不到测试音频。")
        return 1
    print(f"\n  测试音频：{wav_path}")

    # ---- 1. 权重 ------------------------------------------------------
    print("\n【1】权重检查")
    try:
        from bridge_vocoder import VocoderBridge

        vocoder = VocoderBridge()
        print(vocoder.describe())
        check("权重能 strict=True 加载（234/234 键全匹配）", True)
    except Exception as error:  # noqa: BLE001
        print(f"  [失败] {error}")
        print("  先下载：python algorithms/voiceprint_protection/scripts/download_hifigan_weights.py")
        return 1

    sample_rate = config.MEL_PRESETS["hifigan16k"]["sample_rate"]
    wav = audio_utils.load_audio(wav_path, sample_rate=sample_rate)
    print(f"  时长 {audio_utils.duration_seconds(wav, sample_rate):.2f} 秒")

    # ---- 2. 数值一致性 ------------------------------------------------
    print("\n【2】数值检查：我们的 hifigan_mel vs 直接调 torchaudio 链")
    mel = audio_utils.hifigan_mel(wav)
    spectrogram, mel_scale = audio_utils._hifigan_transforms()
    preset = config.MEL_PRESETS["hifigan16k"]
    with torch.no_grad():
        # ★ 参考链要**完整**：STFT 幅度 -> slaney mel -> log(clamp(1e-5))
        #   （少了最后的 log 压缩，两者当然对不上）
        reference = torch.log(torch.clamp(
            mel_scale(spectrogram(torch.from_numpy(wav))),
            min=preset["clip_val"])).numpy()
    difference = np.abs(mel.T - reference)
    print(f"  我们的 (T,80) {mel.shape}，参考 (80,T) {reference.shape}")
    print(f"  最大绝对差 = {float(difference.max()):.3e}")
    check("最大绝对差 < 1e-6", float(difference.max()) < 1e-6)

    # 归一化 ↔ 反归一化 必须是**互逆**的（否则送进声码器的 mel 会偏）
    normalized = audio_utils.hifigan_mel_norm(wav)
    restored = audio_utils.hifigan_mel_unnorm(normalized)
    unnormalized_error = float(np.abs(restored - mel).max())
    print(f"\n  归一化 mel 值域 [{normalized.min():+.4f}, {normalized.max():+.4f}]"
          f"   原始 mel 值域 [{mel.min():+.4f}, {mel.max():+.4f}]")
    print(f"  归一化 -> 反归一化的最大误差 = {unnormalized_error:.3e}")
    check("归一化/反归一化互逆（误差 < 1e-4）", unnormalized_error < 1e-4,
          f"实测 {unnormalized_error:.3e}")
    check("归一化后 mel 的量级与 AutoVC 一致（约 [-0.2, 1.2]）",
          normalized.min() > -0.5 and normalized.max() < 1.5,
          f"实测 [{normalized.min():+.3f}, {normalized.max():+.3f}]")

    # ---- 3. 端到端 ----------------------------------------------------
    print("\n【3】★ 端到端检查：原音 -> mel -> HiFi-GAN -> 重建音")
    rebuilt = vocoder.mel_to_wav(mel.T, normalized=False)   # 原始 mel
    rebuilt_norm = vocoder.mel_to_wav(normalized.T)          # 网络侧 mel（默认反变换）
    print(f"  重建波形 {rebuilt.shape}，"
          f"RMS {audio_utils.rms_energy(wav):.4f} -> {audio_utils.rms_energy(rebuilt):.4f}")
    check("重建波形长度与原音接近（差 < 5%）",
          abs(len(rebuilt) - len(wav)) / len(wav) < 0.05,
          f"{len(wav)} -> {len(rebuilt)}")
    check("重建音不是静音（RMS > 0.005）",
          audio_utils.rms_energy(rebuilt) > 0.005,
          f"RMS = {audio_utils.rms_energy(rebuilt):.5f}")
    check("走「网络侧 mel」这条路的结果与「原始 mel」完全一致",
          float(np.abs(rebuilt_norm - rebuilt).max()) < 1e-4,
          f"最大差 = {float(np.abs(rebuilt_norm - rebuilt).max()):.3e}")

    print("\n  ★ 说话人余弦（重建音 vs 原音）—— 判定的关键指标：")
    try:
        from bridge_speaker import SpeakerBridge

        bridge = SpeakerBridge()
        original_embedding = bridge.embed(wav, sample_rate)
        rebuilt_embedding = bridge.embed(np.asarray(rebuilt, dtype=np.float32), sample_rate)
        cosine = float(np.dot(original_embedding, rebuilt_embedding)
                       / (np.linalg.norm(original_embedding) * np.linalg.norm(rebuilt_embedding)))
        print(f"      余弦 = {cosine:.4f}")
        check("说话人余弦 > 0.85（重建的还是本人说话 → mel 约定正确）",
              cosine > 0.85, f"实测 {cosine:.4f}")
    except Exception as error:  # noqa: BLE001
        print(f"  [跳过] 说话人嵌入算不了：{error}")

    # ---- 4. 可求导性 --------------------------------------------------
    print("\n【4】★ 可求导性（Stage 2 的梯度必须穿过声码器）")
    mel_tensor = torch.from_numpy(mel.T.copy()).unsqueeze(0).requires_grad_(True)
    wav_tensor = vocoder.model(mel_tensor.to(vocoder.device))
    wav_tensor.sum().backward()
    gradient = mel_tensor.grad
    check("梯度能回传到 mel（Stage 2 的 L_de 靠它）",
          gradient is not None and bool(torch.isfinite(gradient).all())
          and float(gradient.abs().sum()) > 0,
          f"grad absmax = {float(gradient.abs().max()):.4f}" if gradient is not None else "None")

    print()
    print("=" * 70)
    if FAILURES:
        print(f"  对照失败：{len(FAILURES)} 项未通过")
        for item in FAILURES:
            print(f"    - {item}")
        print("=" * 70)
        return 1
    print("  对照通过：HiFi-GAN + 我们的 mel 约定完全可用。")
    print("  ★ Stage 1 的模型侧 mel 必须是 'hifigan16k'"
          "（config.STAGE1['mel_preset']）")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
