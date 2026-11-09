"""mel 参数契约的自检脚本 —— 把"一致性"变成会自动失败的断言。

用法：
    conda activate vpp
    python algorithms/voiceprint_protection/scripts/test_audio_utils.py

通过标准：所有检查项输出 [通过]，最后显示"契约锁定成功"。

为什么需要它：
  mel 参数一旦在多处不一致，x 与 x_adv 的帧轴会对不上，
  L_mel 会退化成恒定大常数，模型学不到东西 —— 而且**不报错**。
  这个脚本把"一致性"变成一条自动断言。

注意：
  * 本脚本只校验**我们自己这一侧**的契约（预设、形状、帧数、超参）。
  * "与 AutoVC / HiFi-GAN 官方权重的数值对照"属于 L1 阶段的任务，
    因为那需要真的把它们的仓库和权重拉下来才能做。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import audio_utils  # noqa: E402
import config  # noqa: E402

FAILURES: list[str] = []


def check(title: str, condition: bool, detail: str = "") -> None:
    status = "通过" if condition else "失败"
    print(f"  [{status}] {title}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(title)


def test_preset_consistency() -> None:
    print("\n【两个预设的一致性】")
    paper = config.MEL_PRESETS["paper16k"]
    autovc = config.MEL_PRESETS["autovc22k"]
    differing = [key for key in paper if paper[key] != autovc[key]]
    check(
        "两个预设只在 sample_rate 上不同",
        differing == ["sample_rate"],
        f"实际差异字段: {differing}",
    )
    check("论文预设为 16 kHz", paper["sample_rate"] == 16000)
    check("AutoVC 预设为 22.05 kHz", autovc["sample_rate"] == 22050)
    check("mel 为 80 维", paper["n_mels"] == 80)
    check("hop 256 / n_fft 1024 / win 1024",
          paper["hop_length"] == 256 and paper["n_fft"] == 1024 and paper["win_length"] == 1024)
    check("fmin 40 / fmax 7600", paper["fmin"] == 40 and paper["fmax"] == 7600)


def test_paper_hyperparameters() -> None:
    print("\n【接口契约（第 8 章）与论文超参】")
    check("FRAME_DIM = 80", config.FRAME_DIM == 80)
    check("CONTENT_DIM = 32", config.CONTENT_DIM == 32)
    check("SPEAKER_DIM = 256", config.SPEAKER_DIM == 256)
    check("Stage1 λ_cd = 1.0（论文值）", config.STAGE1["lambda_cd"] == 1.0)
    check("Stage1 λ_dis = 0.01（论文值）", config.STAGE1["lambda_dis"] == 0.01)
    check("Stage1 迭代 500k（论文值）", config.STAGE1["iterations"] == 500_000)
    check("Stage2 α = 0.10（论文值）", config.STAGE2["alpha"] == 0.10)
    check("Stage2 λ_per = 15.0（论文值）", config.STAGE2["lambda_per"] == 15.0)
    check("Stage2 λ_de = 5.0（论文值）", config.STAGE2["lambda_de"] == 5.0)
    check("Stage2 λ_mask = 0.1（论文值）", config.STAGE2["lambda_mask"] == 0.1)
    check("Stage2 迭代 200k（论文值）", config.STAGE2["iterations"] == 200_000)


def test_mel_shapes(name: str) -> None:
    print(f"\n【预设 {name} 的 mel 计算】")
    spec = audio_utils.MelSpec(name)
    seconds = 2.0
    wav = np.random.RandomState(0).randn(int(spec.sample_rate * seconds)).astype(np.float32) * 0.05

    mel = spec.mel(wav)
    check(
        "mel 形状为 (T, 80)",
        mel.ndim == 2 and mel.shape[1] == config.FRAME_DIM,
        f"实际 {mel.shape}",
    )
    check("mel 数值有限", bool(np.isfinite(mel).all()))
    check("mel 非全零", float(np.abs(mel).sum()) > 0)
    check(
        "帧数与 librosa 实际输出一致（frames() 公式正确）",
        mel.shape[0] == spec.frames(len(wav)),
        f"mel={mel.shape[0]} 预测={spec.frames(len(wav))}",
    )

    # 直接用 librosa 的原生接口复核一次，确认预设参数被正确传下去
    import librosa

    preset = config.MEL_PRESETS[name]
    raw = librosa.feature.melspectrogram(
        y=wav,
        sr=preset["sample_rate"],
        n_fft=preset["n_fft"],
        hop_length=preset["hop_length"],
        win_length=preset["win_length"],
        n_mels=preset["n_mels"],
        fmin=preset["fmin"],
        fmax=preset["fmax"],
        power=preset["power"],
    )
    if preset["log_base"]:
        raw = np.log10(np.clip(raw, preset["log_floor"], None))
    difference = float(np.max(np.abs(raw.T - mel)))
    check(
        "与 librosa 原生调用数值完全一致",
        difference < 1e-6,
        f"最大绝对差 {difference:.3e}",
    )

    # 变长批次不应报错
    shorter = wav[: spec.sample_rate]
    batch = spec.mel_batch([wav, shorter])
    check("mel_batch 支持变长输入", len(batch) == 2 and batch[0].shape[0] > batch[1].shape[0])

    # 太短的音频应给出明确错误，而不是静默出错
    try:
        spec.mel(wav[:100])
        too_short_ok = False
    except ValueError:
        too_short_ok = True
    check("过短音频抛出明确错误", too_short_ok)


def test_vocoder_mel() -> None:
    print("\n【HiFi-GAN 声码器侧的 mel】")
    import librosa

    spec = audio_utils.MelSpec("autovc22k")
    wav = np.random.RandomState(2).randn(spec.sample_rate).astype(np.float32) * 0.05
    mel = audio_utils.vocoder_mel(wav)
    check("vocoder_mel 形状为 (T, 80)", mel.ndim == 2 and mel.shape[1] == 80, f"实际 {mel.shape}")
    check("vocoder_mel 数值有限", bool(np.isfinite(mel).all()))

    # 独立复算一遍，确认实现严格符合 HiFi-GAN 的 (x + 4.5) / 5.0 约定
    mel_basis = librosa.filters.mel(sr=8000, n_fft=1024, n_mels=80)
    spectrum = librosa.stft(
        y=wav, n_fft=1024, hop_length=256, win_length=1024, window="hann", center=True
    )
    raw = mel_basis @ np.log10(np.clip(np.abs(spectrum), 1e-5, None))
    expected = ((raw + 4.5) / 5.0).T
    difference = float(np.max(np.abs(expected - mel)))
    check(
        "严格符合 HiFi-GAN 的 (x + 4.5) / 5.0 约定",
        difference < 1e-6,
        f"最大绝对差 {difference:.3e}",
    )

    # 归一化后应落在大致 [-2, 2] 的区间（真实语音通常在 [-1.2, 1.0] 附近）
    check(
        "归一化后数值范围合理",
        float(mel.min()) > -2.0 and float(mel.max()) < 2.0,
        f"实际范围 [{mel.min():.3f}, {mel.max():.3f}]",
    )

    # 模型侧与声码器侧的约定必须不同，否则说明参数被写串了
    model_mel = spec.mel(wav)
    check(
        "模型侧与声码器侧的 mel 明显不同（约定没有被混用）",
        float(np.max(np.abs(model_mel - mel[: model_mel.shape[0]]))) > 0.1,
    )


def test_waveform_tools() -> None:
    print("\n【波形工具】")
    wav = np.random.RandomState(1).randn(16000).astype(np.float32) * 0.05
    difference = audio_utils.waveform_difference(wav, wav * 0.5)
    check("waveform_difference 正确", float(np.max(np.abs(difference - wav * 0.5))) < 1e-6)
    spectrum = audio_utils.power_spectrum(wav)
    check("power_spectrum 形状 (513, frames)", spectrum.shape[0] == 513, f"实际 {spectrum.shape}")
    check("rms_energy > 0", audio_utils.rms_energy(wav) > 0)
    check("spectral_distance 可计算", audio_utils.spectral_distance(wav, wav * 0.9) >= 0)

    import torch

    unit = audio_utils.normalize_l2(audio_utils.to_tensor(wav))
    check("L2 归一化后模长为 1", abs(float(unit.norm()) - 1.0) < 1e-5)

    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    c = torch.tensor([0.0, 1.0, 0.0])
    check("cosine_similarity 同向为 1", abs(float(audio_utils.cosine_similarity(a, b)) - 1.0) < 1e-6)
    check("cosine_similarity 正交为 0", abs(float(audio_utils.cosine_similarity(a, c))) < 1e-6)


def main() -> int:
    print("=" * 64)
    print("  mel 参数契约自检")
    print("=" * 64)

    test_preset_consistency()
    test_paper_hyperparameters()
    for name in config.MEL_PRESETS:
        test_mel_shapes(name)
    test_vocoder_mel()
    test_waveform_tools()

    print()
    print("=" * 64)
    if FAILURES:
        print(f"  契约未锁定：{len(FAILURES)} 项失败")
        for item in FAILURES:
            print(f"    - {item}")
        print("=" * 64)
        return 1
    print("  契约锁定成功：mel 参数与形状约定已一致，可以开始写 Stage 1。")
    print("  提醒：与 AutoVC / HiFi-GAN 官方权重的数值对照属于 L1 阶段，仍需完成。")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
