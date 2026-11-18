"""Stage 2 自检：心理声学掩蔽这一块对不对？

【为什么要有这个脚本】
`psychoacoustics.py` 里的每个数字（MPEG-1 的临界频带表、绝对听阈公式、
掩蔽指数、扩散函数）都是**手抄标准**的 —— 抄错一个数不会报错，
只会让 L_mask 悄悄失去意义（训练照跑，指标全错）。

所以这里把它变成一组**会失败的断言**：
  · 阈值必须是有限数、量级合理
  · 阈值必须随信号变响而**单调升高**
  · 零扰动的 L_mask 必须**严格等于 0**
  · 超阈值的扰动 L_mask 必须 > 0
  · torch 版与 numpy 版的数值必须**一致**
  · torch 版必须**可导**

用法：
    conda activate vpp
    python algorithms/voiceprint_protection/stage2/verify_psychoacoustics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR.parent))

import torch  # noqa: E402

from stage2.psychoacoustics import (  # noqa: E402
    describe,
    masking_loss,
    masking_loss_torch,
    masking_threshold,
    perturbation_psd,
)

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


def tone(frequency: float, seconds: float = 1.0, sample_rate: int = 16000,
         amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float64)


def main() -> int:
    print("=" * 68)
    print("  Stage 2 · 心理声学掩蔽自检")
    print("=" * 68)
    print(describe())
    print()

    sample_rate = 16000
    n_fft = 1024

    # -- 1. 形状与数值范围 --------------------------------------------------
    print("【1】掩蔽阈值：形状与量级")
    threshold = masking_threshold(tone(1000.0), sample_rate=sample_rate, n_fft=n_fft)
    check(threshold.shape == (n_fft // 2 + 1,),
          f"阈值形状 (K,) = (513,) —— 实际 {threshold.shape}")
    check(np.all(np.isfinite(threshold)), "阈值全是有限数（没有 nan/inf）")
    # 阈值是**均方 dB**（满幅正弦 = 94 dB SPL，对应均方 dB = -3.01），
    # 所以量级在 −120 ~ 0 dB 之间
    check(-130.0 < float(threshold.min()) and float(threshold.max()) < 10.0,
          f"阈值量级合理（{threshold.min():.1f} ~ {threshold.max():.1f} dB，均方口径）")

    # -- 2. 阈值随信号变响而升高 ---------------------------------------------
    print("\n【2】单调性：信号变响，阈值必须升高")
    quiet = masking_threshold(tone(1000.0, amplitude=0.05),
                              sample_rate=sample_rate, n_fft=n_fft)
    loud = masking_threshold(tone(1000.0, amplitude=0.5),
                             sample_rate=sample_rate, n_fft=n_fft)
    # 1 kHz 对应的 FFT 频点（n_fft=1024, sr=16000 -> bin 64）
    bin_1k = int(round(1000.0 * n_fft / sample_rate))
    check(float(loud[bin_1k]) > float(quiet[bin_1k]),
          f"1 kHz 处：安静 {quiet[bin_1k]:.1f} dB -> 响亮 {loud[bin_1k]:.1f} dB")
    check(float(loud.mean()) > float(quiet.mean()),
          f"全带平均也升高（{quiet.mean():.1f} -> {loud.mean():.1f} dB）")

    # -- 3. PSD -------------------------------------------------------------
    print("\n【3】扰动 PSD")
    zeros_psd = perturbation_psd(np.zeros(sample_rate), sample_rate=sample_rate, n_fft=n_fft)
    check(float(zeros_psd.max()) < -100.0,
          f"零扰动的 PSD 极小（最大 {zeros_psd.max():.1f} dB）")

    rng = np.random.default_rng(0)
    noise = 0.1 * rng.standard_normal(sample_rate)
    noise_psd = perturbation_psd(noise, sample_rate=sample_rate, n_fft=n_fft)
    usable = noise_psd[10:-10]
    spread = float(usable.max() - usable.min())
    check(spread < 12.0,
          f"白噪声的 PSD 谱形平坦（展布 {spread:.1f} dB）")

    # -- 4. L_mask -----------------------------------------------------------
    print("\n【4】L_mask（论文式 8）")
    zero_loss = masking_loss(np.zeros(sample_rate), threshold,
                              sample_rate=sample_rate, n_fft=n_fft)
    check(zero_loss == 0.0, f"零扰动的 L_mask 严格等于 0（实际 {zero_loss}）")

    # 阈值比掩蔽器电平低约 20 dB（调性掩蔽器），所以扰动要**跟原音频一个量级**
    # 才会超出阈值 —— 这正是心理声学掩蔽的含义：小于阈值的扰动人耳听不见。
    loud_loss = masking_loss(2.0 * rng.standard_normal(sample_rate), threshold,
                              sample_rate=sample_rate, n_fft=n_fft)
    check(loud_loss > 0.0, f"超阈值扰动的 L_mask > 0（实际 {loud_loss:.2f} dB）")

    quiet_loss = masking_loss(0.01 * rng.standard_normal(sample_rate), threshold,
                              sample_rate=sample_rate, n_fft=n_fft)
    check(quiet_loss <= loud_loss,
          f"扰动越小 L_mask 越小（{quiet_loss:.3f} <= {loud_loss:.2f}）")

    # ★ 注意：在**原音频很安静的频带**上，哪怕很小的扰动也会超过绝对听阈，
    #   所以 L_mask 不会严格等于 0 —— 这是对的（那部分确实听得见）。
    #   要让 L_mask 严格为 0，扰动必须低于**所有**频带的绝对听阈。
    inaudible_loss = masking_loss(1e-4 * rng.standard_normal(sample_rate), threshold,
                                  sample_rate=sample_rate, n_fft=n_fft)
    check(inaudible_loss == 0.0,
          f"低于绝对听阈的扰动 L_mask = 0（实际 {inaudible_loss:.3f}）")
    check(quiet_loss < 1.0,
          f"很小的扰动 L_mask 很小（{quiet_loss:.3f} dB）—— 这就是掩蔽的含义")

    # -- 5. torch 版 ----------------------------------------------------------
    print("\n【5】torch 版：数值一致 + 可导")
    perturbation = 0.05 * rng.standard_normal(sample_rate)
    numpy_value = masking_loss(perturbation, threshold, sample_rate=sample_rate, n_fft=n_fft)

    tensor = torch.tensor(perturbation, dtype=torch.float32, requires_grad=True)
    torch_value = masking_loss_torch(tensor, threshold, sample_rate=sample_rate, n_fft=n_fft)
    check(bool(torch_value.requires_grad), "torch 版带梯度")
    check(abs(float(torch_value) - numpy_value) <= max(1e-3, abs(numpy_value) * 1e-3),
          f"两个实现数值一致（numpy {numpy_value:.6f} vs torch {float(torch_value):.6f}）")

    torch_value.backward()
    gradient = tensor.grad
    check(gradient is not None, "梯度存在")
    check(gradient is not None and bool(torch.all(torch.isfinite(gradient))),
          "梯度全是有限数")
    check(gradient is not None and float(gradient.abs().max()) > 0.0,
          "梯度不为 0")

    # -- 6. 批量形状 ----------------------------------------------------------
    print("\n【6】批量输入 (B, N)")
    batch = torch.tensor(0.05 * rng.standard_normal((3, sample_rate)),
                         dtype=torch.float32, requires_grad=True)
    batch_loss = masking_loss_torch(batch, threshold, sample_rate=sample_rate, n_fft=n_fft)
    check(batch_loss.dim() == 0, f"批量返回标量（实际 {tuple(batch_loss.shape)}）")
    batch_loss.backward()
    check(batch.grad is not None and bool(torch.all(torch.isfinite(batch.grad))),
          "批量梯度有限")

    # -- 7. 真实语音 ----------------------------------------------------------
    print("\n【7】真实语音跑通")
    import paths

    manifest = paths.data_root() / "test.txt"
    if manifest.exists():
        sample = None
        for line in manifest.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                candidate = paths.data_root() / parts[0]
                if candidate.exists():
                    sample = candidate
                    break
        if sample is not None:
            import soundfile

            wav, sr = soundfile.read(str(sample), dtype="float64")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            real_threshold = masking_threshold(wav, sample_rate=sr, n_fft=n_fft)
            real_psd = perturbation_psd(0.01 * rng.standard_normal(len(wav)),
                                        sample_rate=sr, n_fft=n_fft)
            real_loss = masking_loss(0.01 * rng.standard_normal(len(wav)), real_threshold,
                                     sample_rate=sr, n_fft=n_fft)
            check(np.all(np.isfinite(real_threshold)) and real_loss >= 0.0,
                  f"真实语音（{sample.name}）跑通：阈值均值 {real_threshold.mean():.1f} dB，"
                  f"L_mask {real_loss:.3f}")
        else:
            print("  [跳过] 找不到测试音频")
    else:
        print("  [跳过] 没有 test.txt")

    print("\n" + "=" * 68)
    print(f"  通过 {PASSED} 项，失败 {FAILED} 项")
    print("=" * 68)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
