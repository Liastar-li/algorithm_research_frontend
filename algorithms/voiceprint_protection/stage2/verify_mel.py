"""Stage 2 自检：torch 版 GE2E 与上游 numpy 版**逐值一致**吗？

【为什么这件事必须验】
`stage2/models.py` 里的 `TorchSpeakerEncoder` 是我们自己用 torch 重写的
（因为上游的 numpy 封装断了梯度）。但如果它的 mel 算得和上游不一样：

  · 推理时（`bridge_speaker`，numpy 路径）与训练时（这里，torch 路径）
    得到的 h **不是同一个东西**；
  · 更糟的是**不报错** —— 嵌入只是"不对"，训练照跑，最后指标全是错的。

上游的 mel：
    librosa.feature.melspectrogram(y=wav, sr=16000, n_fft=400,
                                   hop_length=160, n_mels=40)
    → power=2.0, center=True, pad_mode='constant', window='hann'(周期),
      fmin=0, fmax=8000 → 取 .T 得到 (T, 40)

本脚本拿同一段波形，两个路径各算一遍，比对最大绝对误差。

用法：
    conda activate vpp
    python algorithms/voiceprint_protection/stage2/verify_mel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR.parent))

try:
    import numpy as np  # noqa: E402
    import torch  # noqa: E402

    import config  # noqa: E402
except ImportError as error:  # 依赖缺失（多半是忘了 conda activate vpp）
    from paths import print_env_hint  # noqa: E402

    print_env_hint(error)
    raise SystemExit(1) from error
from stage2.models import TorchSpeakerEncoder  # noqa: E402

PASSED = 0
FAILED = 0


def check(ok: bool, name: str, extra: str = "") -> bool:
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  [通过] {name}")
    else:
        FAILED += 1
        print(f"  [失败] {name}" + (f"  —— {extra}" if extra else ""))
    return ok


def main() -> int:
    print("=" * 66)
    print("  Stage 2 · torch GE2E 与上游 numpy 的数值对照")
    print("=" * 66)

    encoder = TorchSpeakerEncoder(device="cpu")

    # -- 1. 准备测试波形 ---------------------------------------------------
    torch.manual_seed(0)
    sample_rate = encoder.sample_rate
    duration = 2.0
    t = np.arange(int(sample_rate * duration)) / sample_rate
    # 带包络的类语音信号（比纯正弦更能暴露谱泄漏类的差异）
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3.1 * t)) ** 2
    wav = (envelope * (0.4 * np.sin(2 * np.pi * 220 * t)
                        + 0.2 * np.sin(2 * np.pi * 1100 * t)
                        + 0.1 * np.sin(2 * np.pi * 3300 * t))).astype(np.float32)

    # -- 2. 上游（numpy / librosa）-----------------------------------------
    print("\n【1】上游 librosa 路径")
    import librosa

    upstream_mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sample_rate,
        n_fft=encoder.n_fft,
        hop_length=encoder.hop_length,
        n_mels=encoder.mel_n_channels,
    ).astype(np.float32).T          # (T, 40)
    print(f"  上游 mel 形状 : {upstream_mel.shape}")

    # -- 3. 本项目的 torch 路径 --------------------------------------------
    print("\n【2】本项目 torch 路径")
    with torch.no_grad():
        torch_mel = encoder.mel(torch.from_numpy(wav)[None, :])[0].numpy()
    print(f"  torch mel 形状 : {torch_mel.shape}")

    # -- 4. 比对 ------------------------------------------------------------
    print("\n【3】数值对照")
    length = min(upstream_mel.shape[0], torch_mel.shape[0])
    difference = np.abs(upstream_mel[:length] - torch_mel[:length])
    max_difference = float(difference.max())
    relative = max_difference / max(float(np.abs(upstream_mel).max()), 1e-12)
    print(f"  最大绝对误差 : {max_difference:.3e}")
    print(f"  相对误差     : {relative:.3e}")
    print(f"  mel 幅值范围 : {upstream_mel.min():.4f} ~ {upstream_mel.max():.4f}")

    check(upstream_mel.shape == torch_mel.shape,
          f"mel 形状一致 —— {upstream_mel.shape} vs {torch_mel.shape}")
    check(max_difference < 1e-3,
          f"最大绝对误差 < 1e-3（实测 {max_difference:.3e}）",
          "误差过大 —— 训练与推理会算出不同的嵌入")
    check(relative < 1e-4,
          f"相对误差 < 1e-4（实测 {relative:.3e}）")

    # -- 5. 端到端：嵌入也要一致 -------------------------------------------
    print("\n【4】端到端：说话人嵌入（上游 numpy 路径 vs 本项目 torch 路径）")
    from bridge_speaker import SpeakerBridge

    bridge = SpeakerBridge(device="cpu")
    numpy_embedding = bridge.embed(wav, sample_rate)

    with torch.no_grad():
        torch_embedding = encoder(torch.from_numpy(wav)[None, :],
                                  torch.tensor([len(wav)]))[0].numpy()

    embedding_difference = float(np.abs(numpy_embedding - torch_embedding).max())
    cosine = float(np.dot(numpy_embedding, torch_embedding)
                   / (np.linalg.norm(numpy_embedding) * np.linalg.norm(torch_embedding)))
    print(f"  余弦相似度   : {cosine:.8f}")
    print(f"  最大绝对误差 : {embedding_difference:.3e}")
    check(cosine > 0.999,
          f"两个路径的嵌入余弦 > 0.999（实测 {cosine:.8f}）",
          "嵌入对不上 —— 训练与推理的说话人嵌入不是一回事")

    print("\n" + "=" * 66)
    print(f"  通过 {PASSED} 项，失败 {FAILED} 项")
    print("=" * 66)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
