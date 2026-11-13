"""mel 参数契约的自检脚本 —— 把"一致性"变成会自动失败的断言。

用法：
    conda activate vpp
    python algorithms/voiceprint_protection/scripts/test_audio_utils.py

★ 另有一个更重要的脚本：scripts/verify_autovc_mel.py
  它用 AutoVC 官方原始代码与本项目实现做逐元素对照，**那个才是关键**。

通过标准：所有检查项输出 [通过]，最后显示"契约锁定成功"。
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


def test_preset_values() -> None:
    print("\n【模型侧 mel 预设（实测对齐 AutoVC 官方）】")
    preset = config.MEL_PRESETS[config.DEFAULT_PRESET]
    check("默认预设为 model16k", config.DEFAULT_PRESET == "model16k")
    check("采样率 16000（AutoVC 官方权重就是 16k）", preset["sample_rate"] == 16000,
          f"实际 {preset['sample_rate']}")
    check("mel 80 维", preset["n_mels"] == 80)
    check("fmin 90（取自 make_spect.py，不是 hparams.py 的 125）", preset["fmin"] == 90,
          f"实际 {preset['fmin']}")
    check("fmax 7600", preset["fmax"] == 7600)
    check("n_fft 1024 / hop 256 / win 1024",
          preset["n_fft"] == 1024 and preset["hop_length"] == 256 and preset["win_length"] == 1024)
    check("预处理链标记为 autovc", preset["chain"] == "autovc")


def test_paper_hyperparameters() -> None:
    print("\n【接口契约与论文超参】")
    check("FRAME_DIM = 80", config.FRAME_DIM == 80)
    check("CONTENT_DIM(dim_neck) = 32", config.CONTENT_DIM == 32)
    check("CONTENT_ACTUAL_DIM = 64（双向拼接后）", config.CONTENT_ACTUAL_DIM == 64)
    check("CONTENT_DOWNSAMPLE(freq) = 32", config.CONTENT_DOWNSAMPLE == 32)
    check("SPEAKER_DIM = 256", config.SPEAKER_DIM == 256)
    check("AutoVC dim_emb = 256", config.AUTOVC_MODEL["dim_emb"] == 256)
    check("AutoVC dim_pre = 512", config.AUTOVC_MODEL["dim_pre"] == 512)
    check("Stage1 λ_cd = 1.0（论文值）", config.STAGE1["lambda_cd"] == 1.0)
    check("Stage1 λ_dis = 0.01（论文值）", config.STAGE1["lambda_dis"] == 0.01)
    check("Stage1 迭代 500k（论文值）", config.STAGE1["iterations"] == 500_000)
    check("Stage2 α = 0.10（论文值）", config.STAGE2["alpha"] == 0.10)
    check("Stage2 λ_per = 15.0（论文值）", config.STAGE2["lambda_per"] == 15.0)
    check("Stage2 λ_de = 5.0（论文值）", config.STAGE2["lambda_de"] == 5.0)
    check("Stage2 λ_mask = 0.1（论文值）", config.STAGE2["lambda_mask"] == 0.1)
    check("Stage2 迭代 200k（论文值）", config.STAGE2["iterations"] == 200_000)


def test_speaker_encoders() -> None:
    print("\n【说话人编码器（两套权重不可混用）】")
    autovc = config.SPEAKER_ENCODERS["autovc_bl"]
    ge2e = config.SPEAKER_ENCODERS["ge2e_rtc"]
    check("AutoVC BL 输入 80 维 mel", autovc["dim_input"] == 80)
    check("GE2E(RTC) 输入 40 维 mel", ge2e["dim_input"] == 40)
    check("两者输出都是 256 维", autovc["dim_emb"] == 256 and ge2e["dim_emb"] == 256)
    check(
        "两者输入维度不同（所以权重不能混用）",
        autovc["dim_input"] != ge2e["dim_input"],
    )


def test_mel_chain() -> None:
    print("\n【模型侧 mel 计算】")
    spec = audio_utils.MelSpec()
    sample_rate = spec.sample_rate
    seconds = 2.0
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    wav = (
        0.25 * np.sin(2 * np.pi * 140 * t)
        + 0.15 * np.sin(2 * np.pi * 420 * t) * np.sin(2 * np.pi * 3.1 * t)
    ).astype(np.float32)

    mel = spec.mel(wav)
    check("mel 形状为 (T, 80)", mel.ndim == 2 and mel.shape[1] == config.FRAME_DIM,
          f"实际 {mel.shape}")
    check("mel 数值有限", bool(np.isfinite(mel).all()))
    check("mel 取值在 [0,1]（AutoVC 归一化约定）",
          float(mel.min()) >= 0.0 and float(mel.max()) <= 1.0,
          f"[{mel.min():.4f}, {mel.max():.4f}]")
    check("mel 非全零", float(np.abs(mel).sum()) > 0)
    check(
        "帧数与 frames() 预测一致",
        mel.shape[0] == spec.frames(len(wav)),
        f"mel={mel.shape[0]} 预测={spec.frames(len(wav))}",
    )

    # 固定 seed 必须可复现
    again = spec.mel(wav)
    check("固定 seed 时结果可复现", float(np.max(np.abs(mel - again))) < 1e-6)

    # 关闭噪声后也应稳定
    quiet = audio_utils.autovc_mel(wav, noise_seed=None)
    check("关闭噪声也能正常计算", quiet.shape == mel.shape and bool(np.isfinite(quiet).all()))

    # 变长批次
    batch = spec.mel_batch([wav, wav[: sample_rate]])
    check("mel_batch 支持变长输入", len(batch) == 2 and batch[0].shape[0] > batch[1].shape[0])


def test_vocoder_mel() -> None:
    print("\n【HiFi-GAN 声码器侧的 mel】")
    import librosa

    settings = config.VOCODER_MEL
    spec = audio_utils.MelSpec()
    wav = np.random.RandomState(2).randn(22050).astype(np.float32) * 0.05
    mel = audio_utils.vocoder_mel(wav)
    check("vocoder_mel 形状为 (T, 80)", mel.ndim == 2 and mel.shape[1] == 80, f"实际 {mel.shape}")
    check("vocoder_mel 数值有限", bool(np.isfinite(mel).all()))

    mel_basis = librosa.filters.mel(sr=settings["mel_basis_sr"], n_fft=settings["n_fft"], n_mels=80)
    spectrum = librosa.stft(
        y=wav, n_fft=settings["n_fft"], hop_length=settings["hop_length"],
        win_length=settings["win_length"], window="hann", center=True,
    )
    raw = mel_basis @ np.log10(np.clip(np.abs(spectrum), settings["log_floor"], None))
    expected = ((raw + settings["offset"]) / settings["scale"]).T
    difference = float(np.max(np.abs(expected - mel)))
    check("严格符合 HiFi-GAN 的 (x + 4.5) / 5.0 约定", difference < 1e-6,
          f"最大绝对差 {difference:.3e}")
    check("归一化后数值范围合理",
          float(mel.min()) > -2.0 and float(mel.max()) < 2.0,
          f"实际范围 [{mel.min():.3f}, {mel.max():.3f}]")

    # 模型侧与声码器侧必须不同，防止将来写串
    model_mel = spec.mel(wav)
    length = min(model_mel.shape[0], mel.shape[0])
    check("模型侧与声码器侧的 mel 明显不同（约定没被混用）",
          float(np.max(np.abs(model_mel[:length] - mel[:length]))) > 0.1)


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
    print("=" * 70)
    print("  mel 参数契约自检")
    print("=" * 70)

    test_preset_values()
    test_paper_hyperparameters()
    test_speaker_encoders()
    test_mel_chain()
    test_vocoder_mel()
    test_waveform_tools()

    print()
    print("=" * 70)
    if FAILURES:
        print(f"  契约未锁定：{len(FAILURES)} 项失败")
        for item in FAILURES:
            print(f"    - {item}")
        print("=" * 70)
        return 1
    print("  契约锁定成功。")
    print("  ★ 还要跑 scripts/verify_autovc_mel.py —— 那个才是与官方权重的关键对照。")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
