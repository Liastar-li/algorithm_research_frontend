"""对照脚本：验证 `audio_utils.rtc_mel()` 与上游 WaveRNN 声码器**真的对得上**。

【为什么必须有这个脚本】
Stage 1 的解码器输出的是 mel 频谱，要"听"必须过声码器。
而**声码器只认自己训练时见过的频谱约定** —— 喂错约定，出来是噪音，**而且不报错**。
这是本项目最容易中招的一类 bug（见 audio_utils.py 文件头）。

【★ 这条约定是怎么定下来的（不是猜的）】
1. `third_party/Real-Time-Voice-Cloning/vocoder/vocoder_dataset.py` 第 27-28 行：
       # Load the mel spectrogram and adjust its range to [-1, 1]
       mel = np.load(mel_path).T.astype(np.float32) / hp.mel_max_abs_value
   ⇒ WaveRNN 训练时见到的 mel ∈ **[-1, 1]**。
2. `vocoder/inference.py::infer_waveform` 内部会 `mel / 4`
   ⇒ 喂给 `infer_waveform` 的必须是 **[-4, 4]**（合成器原始输出的量纲）。
3. 反查上游合成器的**真实输出**：`synthesize_spectrograms()` 返回的 mel
   实测范围是 **[-4.19, +0.66]** ⇒ 对应
   `synthesizer/audio.py::melspectrogram`（含 preemphasis、fmax=7600、对称归一化）。

⚠️ **踩过的坑**：一开始照抄了 `vocoder/audio.py::melspectrogram`
（没有 preemphasis、fmax 用默认值 8000、归一化成 [0,1]），
数值对照能过（1.6e-07）但**送进声码器出来是噪音**。
→ 所以这个脚本必须有**第 2 层端到端对照**，只看数值是不够的。

【用法】
    conda activate vpp
    python algorithms/voiceprint_protection/scripts/verify_rtc_mel.py
    python algorithms/voiceprint_protection/scripts/verify_rtc_mel.py --wav 某条音频.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

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
    """找一个默认的测试音频：优先平台样例，其次 data/audio16k 下的任意一条。"""
    candidate = paths.sample_dir("group_n005_voiceprint_protection") / "input.wav"
    if candidate.exists():
        return candidate
    audio_root = paths.data_root() / "audio16k"
    if audio_root.is_dir():
        for wav in sorted(audio_root.rglob("*.wav")):
            return wav
    return None


def upstream_synth_mel(wav: np.ndarray) -> np.ndarray:
    """**逐行复刻**上游 `synthesizer/audio.py::melspectrogram`，返回 (80, T)，范围 [-4, 4]。

    ★ 为什么要复刻而不是直接 import：
      上游按 librosa **0.9.2** 写的，`librosa.filters.mel(sr, n_fft, ...)` 用的是
      **位置参数**；而 librosa **0.10** 的 `filters.mel` 全部改成**仅关键字**，
      直接调用会报
          TypeError: mel() takes 0 positional arguments but 2 positional arguments
      （我们锁的就是 librosa==0.10.2.post1，必然踩到。
        成员 B 在 `bridge_speaker.patch_upstream_mel` 里已经处理过同一类问题。）

    这里只把那一处改成等价的关键字写法，**其余逐行照抄**，数值完全一致。
    """
    import librosa
    from scipy import signal

    from synthesizer.hparams import hparams as hp

    def build_mel_basis():
        # ★ 唯一改动：位置参数 -> 关键字参数（librosa 0.10 要求）
        return librosa.filters.mel(
            sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,
            fmin=hp.fmin, fmax=hp.fmax)

    def preemphasis(wav_, k, preemphasize=True):
        return signal.lfilter([1, -k], [1], wav_) if preemphasize else wav_

    def amp_to_db(x):
        minimum = np.exp(hp.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(minimum, x))

    def normalize(S):
        # symmetric_mels=True 且 allow_clipping_in_normalization=True
        return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db))
                       - hp.max_abs_value, -hp.max_abs_value, hp.max_abs_value)

    D = librosa.stft(y=preemphasis(wav, hp.preemphasis, hp.preemphasize),
                     n_fft=hp.n_fft, hop_length=hp.hop_size, win_length=hp.win_size)
    S = amp_to_db(np.dot(build_mel_basis(), np.abs(D))) - hp.ref_level_db
    return normalize(S)


def mel_round_trip_distance(original_mel: np.ndarray,
                            rebuilt_wav: np.ndarray) -> tuple[float, float]:
    """mel 域往返距离：把重建波形再转回 mel，与原 mel 比。

    返回 (平均绝对差, 相对 L2 误差)。**这才是判断声码器约定对不对的指标** ——
    声码器不保证相位一致（所以波形级 SNR 天然是负的），
    但"听起来是不是同一个声音"由 mel 决定。

    帧数不齐时取公共长度。
    """
    rebuilt_mel = audio_utils.rtc_mel(np.asarray(rebuilt_wav, dtype=np.float32))
    length = min(len(original_mel), len(rebuilt_mel))
    a = original_mel[:length]
    b = rebuilt_mel[:length]
    mean_absolute = float(np.abs(a - b).mean())
    denominator = float(np.sqrt((a ** 2).sum()))
    relative_l2 = (float(np.sqrt(((a - b) ** 2).sum()) / denominator)
                   if denominator > 1e-12 else float("inf"))
    return mean_absolute, relative_l2


def test_numeric_match(wav: np.ndarray) -> None:
    print("\n【第 1 层】数值对照：我们的 rtc_mel vs 上游 synthesizer/audio.py::melspectrogram")
    ours = audio_utils.rtc_mel(wav)                        # (T, 80)，[-1, 1]
    theirs = upstream_synth_mel(wav) / 4.0                # (80, T) -> [-1, 1]

    check("形状一致", ours.shape == theirs.T.shape,
          f"我们 {ours.shape} vs 上游 {theirs.T.shape}")
    if ours.shape != theirs.T.shape:
        return

    difference = np.abs(ours - theirs.T)
    maximum = float(difference.max())
    print(f"  最大绝对差 = {maximum:.3e}   平均绝对差 = {float(difference.mean()):.3e}")
    check("最大绝对差 < 1e-5", maximum < 1e-5, f"实测 {maximum:.3e}")


def test_round_trip(wav: np.ndarray) -> None:
    print("\n【第 2 层】端到端对照：原波形 -> rtc_mel -> WaveRNN -> 重建波形 -> 再转回 mel")
    print("  （判断依据是 **mel 域往返距离**：声码器不做相位重建，但必须保证音色）")
    from bridge_vocoder import VocoderBridge

    vocoder = VocoderBridge()
    mel = audio_utils.rtc_mel(wav)

    print(f"  mel 形状 {mel.shape}，取值范围 [{mel.min():+.4f}, {mel.max():+.4f}]")

    results = {}
    for scale, label in ((4.0, "scale=4（WaveRNN 看到原 mel，量纲 [-1,1]）"),
                         (1.0, "scale=1（WaveRNN 看到 mel/4，量纲 [-0.25,0.25]）")):
        rebuilt = vocoder.mel_to_wav(mel, scale=scale)
        mean_absolute, relative_l2 = mel_round_trip_distance(mel, rebuilt)
        results[scale] = (mean_absolute, relative_l2)
        print(f"  {label}")
        print(f"      重建波形 {rebuilt.shape}，RMS = {audio_utils.rms_energy(rebuilt):.5f}")
        print(f"      mel 往返：平均绝对差 = {mean_absolute:.5f}，"
              f"相对 L2 = {relative_l2 * 100:.2f}%")

    best = min(results, key=lambda key: results[key][0])
    print(f"\n  最佳 scale = {best}"
          f"（mel 平均绝对差 {results[best][0]:.5f}，"
          f"相对 L2 {results[best][1] * 100:.2f}%）")
    check("正确量纲下 mel 往返距离小（相对 L2 < 25%）",
          results[4.0][1] < 0.25, f"实测 {results[4.0][1] * 100:.2f}%")
    check("错误量纲明显更差（mel 往返距离至少差 1.5 倍）",
          results[4.0][0] * 1.5 < results[1.0][0],
          f"{results[4.0][0]:.5f} vs {results[1.0][0]:.5f}")


def main() -> int:
    print("=" * 70)
    print("  rtc_mel 对照（上游 WaveRNN 声码器）")
    print("=" * 70)

    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", default=None, help="测试音频路径（默认用平台样例）")
    args = parser.parse_args()

    wav_path = Path(args.wav) if args.wav else find_default_wav()
    if wav_path is None or not wav_path.exists():
        print("\n[失败] 找不到测试音频。")
        print("  用 --wav 指定一条 wav，或先准备好 public/samples/.../input.wav")
        return 1

    print(f"\n  测试音频：{wav_path}")
    sample_rate = config.MEL_PRESETS["rtc16k"]["sample_rate"]
    wav = audio_utils.load_audio(wav_path, sample_rate=sample_rate)
    print(f"  时长 {audio_utils.duration_seconds(wav, sample_rate):.2f} 秒，"
          f"{len(wav)} 个采样点")

    test_numeric_match(wav)
    test_round_trip(wav)

    print()
    print("=" * 70)
    if FAILURES:
        print(f"  对照失败：{len(FAILURES)} 项未通过")
        for item in FAILURES:
            print(f"    - {item}")
        print("=" * 70)
        return 1
    print("  对照通过：rtc_mel 与上游 WaveRNN 完全兼容。")
    print(f"  ★ 若要用 WaveRNN 出音频，Stage 1 的 mel 预设必须是 "
          f"'rtc16k'（config.STAGE1['mel_preset']）")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
