"""推理主流程 —— 把论文的推理路径完整跑通（L1 阶段）。

【论文的推理链 vs 我们的实现】
    论文                                本文件
    ──────────────────────────────────  ──────────────────────────────────────────
    x  = mel(波形)                       audio_utils.autovc_mel()
    h  = E_spk(波形)                     bridge_speaker.embed()
    δ  = G(h)        ← Stage 2 的生成器    此处先用随机高斯噪声占位（L1 不训练）
    h_adv = h + α·δ                       protect_embedding()
    c  = E_c(x, h)                        AutoVC 内部 encoder
    x_adv = D(c, h_adv)                   bridge_autovc.protect()
    波形 = V(x_adv)                       bridge_vocoder.synthesize()

【形象比喻】
整条链像"给一段录音做体检 + 打疫苗 + 重新体检"：
  1. 体检      —— 测出身份卡 h（256 个数字）
  2. 打疫苗    —— 在身份卡上做微小改动（h + α·δ），人耳听不出，机器却认错了
  3. 重新体检  —— 用改过的身份卡重新"念"一遍，念出来的是受保护的声音

【L1 与最终版的差别】
L1 阶段的 δ 是**随机高斯噪声**（不训练任何东西，先把链路跑通）。
Stage 2 训练完成后，这里换成训练好的生成器 G(h)，其余完全不变。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import audio_utils
import config
import paths


# ---------------------------------------------------------------------------
# 扰动
# ---------------------------------------------------------------------------
def random_perturbation(dimension: int = config.SPEAKER_DIM,
                         seed: int = config.SEED) -> np.ndarray:
    """随机高斯扰动 δ（L1 阶段的占位实现）。

    比喻：现在往身份卡上撒的是"随机沙粒"；
    Stage 2 训练完成后，换成"精心挑选的、能精准破坏机器识别的沙粒"。
    """
    generator = np.random.RandomState(seed)
    delta = generator.randn(dimension).astype(np.float32)
    # 归一化到单位长度，使 α 的含义与论文一致（α 控制扰动的相对强度）
    norm = float(np.linalg.norm(delta))
    if norm > 0:
        delta = delta / norm
    return delta


def perturb_embedding(embedding: np.ndarray, delta: np.ndarray,
                      alpha: float = config.STAGE2["alpha"]) -> np.ndarray:
    """h_adv = h + α·δ，然后重新 L2 归一化。

    归一化是必要的：AutoVC 的解码器是用**已归一化**的嵌入训练的，
    加了扰动后模长会变大，不归一化会让解码器跑到没见过的输入区域。
    """
    advanced = np.asarray(embedding, dtype=np.float32) + alpha * np.asarray(delta, dtype=np.float32)
    norm = float(np.linalg.norm(advanced))
    if norm > 0:
        advanced = advanced / norm
    return advanced.astype(np.float32)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
class Protector:
    """声纹保护器：把"体检 + 打疫苗 + 重新体检"封装成一个对象。"""

    def __init__(self, use_speaker_bridge: bool = True, device: str | None = None):
        from bridge_autovc import AutoVCBridge
        from bridge_speaker import SpeakerEncoderBridge
        from bridge_vocoder import VocoderBridge

        self.device = device
        self.autovc = AutoVCBridge(device=device)
        self.speaker = SpeakerEncoderBridge(device=device) if use_speaker_bridge else None
        self.vocoder = VocoderBridge(device=device)
        self.spec = audio_utils.MelSpec()

    # -- 单步，便于逐步打印与调试 ----------------------------------------
    def to_mel(self, wav: np.ndarray) -> np.ndarray:
        return self.spec.mel(wav)

    def embedding(self, mel: np.ndarray) -> np.ndarray:
        if self.speaker is None:
            raise RuntimeError("未加载说话人编码器，无法提取嵌入。")
        return self.speaker.embed(mel)

    def protect_mel(self, mel: np.ndarray, embedding: np.ndarray,
                    delta: np.ndarray, alpha: float) -> np.ndarray:
        embedding_adv = perturb_embedding(embedding, delta, alpha)
        return self.autovc.protect(mel, embedding, embedding_adv)

    def to_waveform(self, mel: np.ndarray) -> np.ndarray:
        return self.vocoder.synthesize(mel)

    # -- 端到端 ----------------------------------------------------------
    def protect(self, wav: np.ndarray, alpha: float = config.STAGE2["alpha"],
                delta: np.ndarray | None = None, seed: int = config.SEED,
                verbose: bool = True) -> dict:
        """一整条链：波形 -> 保护后波形。

        返回一个 dict，含中间的每一步，便于逐步核对形状。
        """
        if delta is None:
            delta = random_perturbation(seed=seed)

        steps = {}

        mel = self.to_mel(wav)
        steps["mel"] = mel.shape
        if verbose:
            print(f"  [1] mel 频谱      {tuple(mel.shape)}  取值 [{mel.min():.3f}, {mel.max():.3f}]")

        embedding = self.embedding(mel)
        steps["embedding"] = embedding.shape
        if verbose:
            norm = float(np.linalg.norm(embedding))
            print(f"  [2] 说话人嵌入    {tuple(embedding.shape)}  L2 范数 {norm:.4f}")

        embedding_adv = perturb_embedding(embedding, delta, alpha)
        steps["embedding_adv"] = embedding_adv.shape
        if verbose:
            similarity = float(np.dot(embedding, embedding_adv))
            print(f"  [3] 扰动后嵌入    {tuple(embedding_adv.shape)}  "
                  f"与原嵌入的余弦相似度 {similarity:.4f}（α={alpha}）")

        protected_mel = self.protect_mel(mel, embedding, delta, alpha)
        steps["protected_mel"] = protected_mel.shape
        if verbose:
            difference = float(np.mean(np.abs(protected_mel - mel)))
            print(f"  [4] 保护后 mel    {tuple(protected_mel.shape)}  "
                  f"与原 mel 平均绝对差 {difference:.4f}")

        waveform = self.to_waveform(protected_mel)
        steps["waveform"] = waveform.shape
        if verbose:
            print(f"  [5] 保护后波形    {tuple(waveform.shape)}  "
                  f"RMS {audio_utils.rms_energy(waveform):.5f}")

        return {
            "mel": mel,
            "embedding": embedding,
            "embedding_adv": embedding_adv,
            "protected_mel": protected_mel,
            "waveform": waveform,
            "shapes": {key: list(value) for key, value in steps.items()},
        }


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="声纹保护推理（L1 阶段）")
    parser.add_argument("audio", help="输入音频路径（wav）")
    parser.add_argument("--alpha", type=float, default=config.STAGE2["alpha"],
                        help=f"扰动强度 α，默认 {config.STAGE2['alpha']}")
    parser.add_argument("--seed", type=int, default=config.SEED, help="随机种子")
    parser.add_argument("--output", default=None, help="输出 wav 路径")
    parser.add_argument("--report", default=None, help="输出 JSON 报告路径")
    args = parser.parse_args()

    print("=" * 64)
    print("  声纹保护推理 · L1（随机扰动占位，未训练生成器 G）")
    print("=" * 64)
    print(paths.describe())
    print()

    wav = audio_utils.load_audio(args.audio, config.MEL_PRESETS[config.DEFAULT_PRESET]["sample_rate"])
    print(f"输入: {args.audio}  时长 {audio_utils.duration_seconds(wav, 16000):.2f} 秒")
    print()

    protector = Protector()
    print("【推理链】")
    result = protector.protect(wav, alpha=args.alpha, seed=args.seed)

    output = Path(args.output) if args.output else Path(args.audio).with_name(
        Path(args.audio).stem + "_protected.wav"
    )
    audio_utils.save_audio(output, result["waveform"], 16000)
    print(f"\n已保存保护后音频: {output}")

    if args.report:
        report = {
            "alpha": args.alpha,
            "seed": args.seed,
            "input": str(args.audio),
            "shapes": result["shapes"],
            "note": "L1 阶段使用随机高斯扰动占位；Stage 2 训练后替换为生成器 G(h)。",
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        print(f"已保存报告: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
