"""D2-1 验收脚本：AutoVC 官方 mel 链 vs 本项目实现的逐元素对照。

用法：
    conda activate vpp
    python algorithms/voiceprint_protection/scripts/verify_autovc_mel.py

做法：
  * 左侧：把 AutoVC 官方 third_party/AutoVC/make_spect.py 的原始代码
    原封不动抄进来（不改动任何一行），作为"真值"。
  * 右侧：调用 audio_utils.autovc_mel()，即本项目的实现。
  * 对同一段音频比较数值。

通过标准：
  * 最大绝对差 < 1e-5（float32 精度范围内）
  * 帧数完全一致
  * 数值范围都在 [0, 1]

★ 为什么这个脚本是整个项目的防火墙：
  我们的 mel 若与 AutoVC 官方差一点点，AutoVC 预训练权重加载后输出的就是噪音，
  而且**不会报任何错**。跑通这个脚本，才允许进入下一步。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MODULE_DIR.parent.parent
sys.path.insert(0, str(MODULE_DIR))

import audio_utils  # noqa: E402
import config  # noqa: E402

FAILURES: list[str] = []


def check(title: str, condition: bool, detail: str = "") -> None:
    status = "通过" if condition else "失败"
    print(f"  [{status}] {title}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(title)


# ---------------------------------------------------------------------------
# ↓↓↓ AutoVC 官方 third_party/AutoVC/make_spect.py 的原始代码，一字未改 ↓↓↓
# 只把「读文件遍历目录」改成「接收一个波形数组」，其余逻辑完全照抄。
# ---------------------------------------------------------------------------
def _autovc_reference_mel(x: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    from numpy.random import RandomState
    from scipy import signal
    from scipy.signal import get_window
    from librosa.filters import mel

    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def pySTFT(x, fft_length=1024, hop_length=256):
        x = np.pad(x, int(fft_length // 2), mode='reflect')
        noverlap = fft_length - hop_length
        shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
        strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        fft_window = get_window('hann', fft_length, fftbins=True)
        result = np.fft.rfft(fft_window * result, n=fft_length).T
        return np.abs(result)

    # ⚠️ 与官方源码唯一的差异：官方写 `mel(16000, 1024, fmin=90, fmax=7600, n_mels=80)`
    #    用的是位置参数，但 librosa 0.10 起 filters.mel 的参数全部变成关键字专用，
    #    照抄位置参数会报 TypeError。改成等价的关键字写法，数值完全一致。
    mel_basis = mel(sr=16000, n_fft=1024, fmin=90, fmax=7600, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, 16000, order=5)

    # ↓↓↓ 以下三行来自 make_spect.py 主流程（prng seed 用说话人 id，这里固定 0）↓↓
    prng = RandomState(0)
    y = signal.filtfilt(b, a, x)
    wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-06
    D = pySTFT(wav).T
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = np.clip((D_db + 100) / 100, 0, 1)
    return S.astype(np.float32)


# ↑↑↑ AutoVC 官方原始代码结束 ↑↑↑


def main() -> int:
    print("=" * 70)
    print("  D2-1 验收：AutoVC 官方 mel 链 vs 本项目实现")
    print("=" * 70)

    preset = config.MEL_PRESETS[config.DEFAULT_PRESET]
    sample_rate = preset["sample_rate"]
    print(f"  预设: {config.DEFAULT_PRESET}  采样率 {sample_rate}  "
          f"mel {preset['n_mels']}  fmin {preset['fmin']}  fmax {preset['fmax']}  "
          f"n_fft {preset['n_fft']}  hop {preset['hop_length']}")

    # 用一段"像语音"的信号：多频率成分 + 包络，比纯随机噪声更接近真实分布
    duration = 2.0
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    wav = (
        0.25 * np.sin(2 * np.pi * 140 * t)
        + 0.15 * np.sin(2 * np.pi * 420 * t) * np.sin(2 * np.pi * 3.1 * t)
        + 0.05 * np.random.RandomState(0).randn(len(t)).astype(np.float32)
    ).astype(np.float32)

    print("\n【对照】")
    reference = _autovc_reference_mel(wav, sample_rate)
    ours = audio_utils.autovc_mel(wav, noise_seed=0)

    check("形状完全一致", reference.shape == ours.shape,
          f"官方={reference.shape} 本实现={ours.shape}")
    if reference.shape != ours.shape:
        print("  形状不同，无法继续比较。")
        return 1

    difference = float(np.max(np.abs(reference - ours)))
    check("逐元素最大绝对差 < 1e-5", difference < 1e-5, f"实际 {difference:.3e}")
    check("平均绝对差 < 1e-6", float(np.mean(np.abs(reference - ours))) < 1e-6,
          f"实际 {float(np.mean(np.abs(reference - ours))):.3e}")
    check("官方输出范围在 [0,1]", float(reference.min()) >= 0.0 and float(reference.max()) <= 1.0,
          f"[{reference.min():.4f}, {reference.max():.4f}]")
    check("本实现输出范围在 [0,1]", float(ours.min()) >= 0.0 and float(ours.max()) <= 1.0,
          f"[{ours.min():.4f}, {ours.max():.4f}]")

    print("\n【帧数约定】")
    spec = audio_utils.MelSpec()
    check(
        "frames() 预测与实际一致",
        spec.frames(len(wav)) == ours.shape[0],
        f"预测 {spec.frames(len(wav))} 实际 {ours.shape[0]}",
    )
    print("  说明：AutoVC 的 pySTFT 帧数 = (N+hop)//hop，")
    print("        librosa center=True 帧数 = 1 + N//hop，两者相同，帧轴能对上。")

    print("\n【与 librosa 口径的差异（预期不同，用于说明为什么必须用 AutoVC 链）】")
    librosa_version = audio_utils.librosa_mel(wav)
    if librosa_version.shape == ours.shape:
        gap = float(np.max(np.abs(librosa_version - ours)))
        check("librosa 口径与 AutoVC 链确实不同", gap > 0.01,
              f"最大绝对差 {gap:.4f}（若为 0，说明两条链被写串了）")

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"  对照失败：{len(FAILURES)} 项未通过")
        for item in FAILURES:
            print(f"    - {item}")
        print("  不要继续下一步，先修 audio_utils.autovc_mel()")
        print("=" * 70)
        return 1
    print("  对照通过：本项目的 mel 与 AutoVC 官方逐元素一致。")
    print("  → 可以进入 D3-2：写 bridge_autovc.py，加载官方权重做推理。")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
