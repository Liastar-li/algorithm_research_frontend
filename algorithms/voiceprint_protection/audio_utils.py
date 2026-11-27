"""音频与 mel 频谱处理 —— 全项目唯一的实现入口。

【为什么这个文件必须存在】
论文要求 16 kHz / 80 维 mel，但三个第三方组件的原生配置互不相同。
如果 mel 参数在多处手写，出现不一致时：
  * x_adv 与 x 的帧轴对不上
  * L_mel 变成一个恒定的大常数，模型什么都学不到
  * **而且不报错** —— 这是最难查的一类 bug。
所以：任何地方都不许再手写 mel 参数，一律调用本模块。

【★ 2026-11 实测对齐结论（来自 AutoVC 官方仓库，见 third_party/AutoVC）】
AutoVC 官方 `make_spect.py` 的 mel 预处理链**不是 librosa.feature.melspectrogram**，
而是一条自定义链。本模块的 `autovc_mel()` 逐行复刻它：

  1. 30 Hz Butterworth 高通（filtfilt 双向，五阶）
  2. 幅度缩放 ×0.96，再叠加 1e-6 量级随机噪声（AutoVC 训练时按说话人固定 seed）
  3. STFT：n_fft=1024, hop=256, **reflect 填充**（librosa 是 constant！），输出**幅度**（不是功率）
  4. mel 滤波器组：librosa.filters.mel(16000, 1024, fmin=90, fmax=7600, n_mels=80)
  5. D_db = 20*log10(max(1e-5, D_mel)) - 16
  6. S = clip((D_db + 100) / 100, 0, 1)   ← 归一化到 [0,1]

  ⚠️ 注意 AutoVC 自己内部不一致：`hparams.py` 写 fmin=125，`make_spect.py` 实际用 fmin=90。
     真正生效的是 make_spect.py（训练数据由它生成），所以 **fmin=90** 才是对的。

  ⚠️ 帧数：AutoVC 的 pySTFT 给出 (N+256)//256 帧，与 librosa center=True 的
     1 + N//256 **相同**，所以帧轴能对上，差别只在数值。

【三个 mel 约定，别混用】
  ┌──────────────┬────────────────────────┬──────────────────────────────┐
  │ 约定          │ 模型侧（本项目/AutoVC）  │ 声码器侧（HiFi-GAN）          │
  ├──────────────┼────────────────────────┼──────────────────────────────┤
  │ 采样率        │ 16000                  │ 22050（官方权重）             │
  │ fmin/fmax    │ 90 / 7600              │ 0 / 8000（mel 基用 sr=8000） │
  │ STFT 幅度    │ 幅度（非功率）          │ 幅度                        │
  │ 填充          │ reflect                │ reflect（librosa stft）      │
  │ 取对数        │ 20*log10 - 16          │ log10|STFT|，下限 1e-5       │
  │ 归一化        │ clip((x+100)/100,0,1)  │ (x + 4.5) / 5.0             │
  └──────────────┴────────────────────────┴──────────────────────────────┘

  混用会让声码器输出变成噪音，且**不报错**。

【⚠️ 接入预训练权重时的必做动作】
`scripts/verify_autovc_mel.py` 会用 AutoVC 官方 `make_spect.py` 的原始代码
与本模块的 `autovc_mel()` 做逐元素对照。**先跑它，再动模型。**
"""


from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

import config


# ---------------------------------------------------------------------------
# 读取 / 写入 / 重采样
# ---------------------------------------------------------------------------
def load_audio(path: str | Path, sample_rate: int | None = None) -> np.ndarray:
    """读取音频，返回 float32 单声道波形，取值范围约 [-1, 1]。

    只依赖 soundfile（不需要 ffmpeg），wav/flac/ogg 可读；
    mp3 读不了时会给出明确提示，让用户先转成 wav。
    """
    import soundfile as sf

    path = Path(path)
    try:
        wav, native_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            f"无法读取音频 {path}：{error}。"
            "若为 mp3，请先转成 wav：ffmpeg -i in.mp3 -ac 1 -ar 16000 out.wav"
        ) from error

    wav = wav.mean(axis=1)  # 混成单声道
    if sample_rate is not None and native_sr != sample_rate:
        wav = resample(wav, native_sr, sample_rate)
    return np.ascontiguousarray(wav, dtype=np.float32)


def save_audio(path: str | Path, wav: np.ndarray, sample_rate: int) -> Path:
    """写出 16-bit PCM wav。"""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), clipped, sample_rate, subtype="PCM_16")
    return path


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """重采样。优先用 soxr（质量高、快），缺失时退回 librosa。"""
    if orig_sr == target_sr:
        return wav.astype(np.float32, copy=False)
    try:
        import soxr

        return soxr.resample(wav.astype(np.float32), orig_sr, target_sr).astype(np.float32)
    except ImportError:
        import librosa

        return librosa.resample(
            wav.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr
        ).astype(np.float32)


def to_mono16k(wav: np.ndarray, orig_sr: int) -> np.ndarray:
    """任意采样率 -> 16 kHz 单声道（GE2E 说话人编码器只吃 16 kHz）。"""
    return resample(wav, orig_sr, 16000)


def duration_seconds(wav: np.ndarray, sample_rate: int) -> float:
    return len(wav) / float(sample_rate)


# ---------------------------------------------------------------------------
# mel 频谱：GE2E 侧（第二套约定，只喂给说话人编码器）
# ---------------------------------------------------------------------------
# ⚠️ **这是第三套 mel 约定**，来自上游 Real-Time-Voice-Cloning 的
#    `encoder/params_data.py`：
#       mel_window_length = 25 ms  -> n_fft       = int(16000*25/1000) = 400
#       mel_window_step   = 10 ms  -> hop_length = int(16000*10/1000) = 160
#       mel_n_channels    = 40
#       fmin/fmax = 0 / 8000（librosa 默认）
#       power = 2（librosa 默认）
#       ★ **线性幅度，不取 log**（上游源码注释原话："Note: this not a log-mel spectrogram."）
#
# 【为什么 Stage 1 要同时用两套 mel】
#   AutoVC 架构的 Ec/D 吃 **80 维** AutoVC mel；
#   论文的说话人编码器是 GE2E，吃 **40 维** mel。
#   两者维度不同、不能互换，所以 Stage 1 训练时每条音频要算两次 mel。
#
# 【不可微也没关系】
#   mel 是从波形算出来的前处理，梯度不需要穿过它；
#   Stage 1 要"端到端微调"的是 GE2E 的**参数**，不是 mel 的计算过程。
#   所以这里用 numpy 实现即可（快、稳、无需 autograd）。

GE2E_MEL = {
    "sample_rate": 16000,
    "n_fft": 400,
    "hop_length": 160,
    "n_mels": 40,
    "fmin": 0,
    "fmax": 8000,
    "power": 2.0,
}


def ge2e_mel(wav: np.ndarray) -> np.ndarray:
    """GE2E 说话人编码器用的 mel，形状 (T, 40)，**线性幅度不取 log**。

    与上游 `encoder/audio.py::wav_to_mel_spectrogram` 数值等价
    （上游按 librosa 0.9.2 写的，用位置参数；这里改成 0.10 的仅关键字写法）。
    """
    import librosa

    wav = np.asarray(wav, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=GE2E_MEL["sample_rate"],
        n_fft=GE2E_MEL["n_fft"],
        hop_length=GE2E_MEL["hop_length"],
        n_mels=GE2E_MEL["n_mels"],
        fmin=GE2E_MEL["fmin"],
        fmax=GE2E_MEL["fmax"],
        power=GE2E_MEL["power"],
    )
    return np.ascontiguousarray(mel.T, dtype=np.float32)   # (T, 40)


# ---------------------------------------------------------------------------
# mel 频谱：模型侧（AutoVC 链，★ 实测对齐后的唯一实现）
# ---------------------------------------------------------------------------
def _butter_highpass(cutoff: float, fs: int, order: int = 5):
    """30 Hz 高通滤波器系数。复刻 AutoVC make_spect.py。"""
    from scipy import signal

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return signal.butter(order, normal_cutoff, btype="high", analog=False)


def _py_stft(x: np.ndarray, fft_length: int = 1024, hop_length: int = 256) -> np.ndarray:
    """AutoVC 的 pySTFT：reflect 填充 + 幅度谱。

    与 librosa.stft 的两点关键差异：
      * 填充用 **reflect**（librosa 的 center=True 用 constant）
      * 输出 **幅度** |X|（librosa.feature.melspectrogram 默认 power=2）

    ⚠️ 返回形状是 **(n_fft//2+1, frames) = (513, T)**，**不是 (T, 513)**。
    这是因为官方源码里 rfft 之后做了 `.T`（转置最后两维）。
    官方 make_spect.py 里的 `D = pySTFT(wav).T` 会再转一次得到 (T, 513)，
    然后 `np.dot(D, mel_basis)`。我们这里保持函数返回值与官方一致，
    在 autovc_mel() 里做同样的 `.T`。
    """
    from scipy.signal import get_window

    x = np.pad(x, int(fft_length // 2), mode="reflect")
    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    fft_window = get_window("hann", fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    return np.abs(result)


def autovc_mel_filterbank() -> np.ndarray:
    """AutoVC 的 mel 滤波器组，形状 (n_fft//2+1, n_mels) = (513, 80)。

    注意 AutoVC 内部不一致：`hparams.py` 写 fmin=125，但真正生成训练数据的
    `make_spect.py` 用的是 **fmin=90**，以它为准。

    ⚠️ **与官方源码的一处等价改写**：AutoVC 原文写的是
        `mel(16000, 1024, fmin=90, fmax=7600, n_mels=80)`（位置参数），
        但 **librosa 0.10 起 `filters.mel` 的参数全部变成关键字专用**，
        照抄位置参数会报
        `TypeError: mel() takes 0 positional arguments but 2 positional arguments ...`。
        这里改成等价的关键字写法，数值完全一致。
    """
    from librosa.filters import mel

    preset = config.MEL_PRESETS[config.DEFAULT_PRESET]
    return mel(
        sr=preset["sample_rate"],
        n_fft=preset["n_fft"],
        fmin=preset["fmin"],
        fmax=preset["fmax"],
        n_mels=preset["n_mels"],
    ).T.astype(np.float32)


def autovc_mel(wav: np.ndarray, noise_seed: int | None = 0) -> np.ndarray:
    """AutoVC 官方 mel 预处理链的逐行复刻，输出 (T, 80)，取值 [0, 1]。

    对应 third_party/AutoVC/make_spect.py 的第 58-66 行。

    参数
    ----
    wav : 原始波形（float32，采样率须为 16000）
    noise_seed : AutoVC 训练时按说话人用固定 seed 叠加 1e-6 噪声以增强鲁棒性。
                 这里默认固定 seed=0 以保证可复现；传 None 可关闭噪声。
    """
    from scipy import signal

    preset = config.MEL_PRESETS[config.DEFAULT_PRESET]
    sample_rate = preset["sample_rate"]
    n_fft = preset["n_fft"]
    hop = preset["hop_length"]

    x = np.asarray(wav, dtype=np.float32)

    # 1) 30 Hz 高通，去除漂移噪声
    b, a = _butter_highpass(30, sample_rate, order=5)
    y = signal.filtfilt(b, a, x)

    # 2) 幅度缩放 + 1e-6 随机噪声
    y = y * 0.96
    if noise_seed is not None:
        generator = np.random.RandomState(noise_seed)
        y = y + (generator.rand(y.shape[0]) - 0.5) * 1e-06

    # 3) STFT（幅度）。官方 pySTFT 返回 (513, frames)，这里同样转成 (frames, 513)
    spectrum = _py_stft(y, fft_length=n_fft, hop_length=hop).T

    # 4) mel：(frames, 513) @ (513, n_mels) -> (frames, n_mels)
    mel_energy = spectrum @ autovc_mel_filterbank()

    # 5) dB
    min_level = np.exp(-100 / 20 * np.log(10))   # = 1e-5
    decibels = 20 * np.log10(np.maximum(min_level, mel_energy)) - 16

    # 6) 归一化到 [0, 1]
    return np.ascontiguousarray(np.clip((decibels + 100) / 100, 0, 1), dtype=np.float32)


# ---------------------------------------------------------------------------
# mel 频谱：模型侧第二套约定（**WaveRNN 声码器那套**）
# ---------------------------------------------------------------------------
def rtc_mel_filterbank() -> np.ndarray:
    """上游的 mel 滤波器组，形状 (n_fft//2+1, 80) = (401, 80)。

    对应 `third_party/Real-Time-Voice-Cloning/synthesizer/audio.py::_build_mel_basis`：
        librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)

    ⚠️ 上游的 `vocoder/audio.py::build_mel_basis` 用的是 **fmax 默认值 8000**，
       而 `synthesizer/audio.py` 用的是 **fmax=7600**（hparams 里显式写的）。
       两者**不一样**！我们跟的是 synthesizer 那份（实测验证过，见
       `scripts/verify_rtc_mel.py` 与 docs/notes/autovc_alignment.md 的同类记录）。
    """
    from librosa.filters import mel

    preset = config.MEL_PRESETS["rtc16k"]
    return mel(
        sr=preset["sample_rate"],
        n_fft=preset["n_fft"],
        fmin=preset["fmin"],
        fmax=preset["fmax"],
        n_mels=preset["n_mels"],
    ).T.astype(np.float32)


def rtc_mel_raw(wav: np.ndarray) -> np.ndarray:
    """上游 `synthesizer/audio.py::melspectrogram` 的等价复刻，形状 (T, 80)，取值 **[-4, 4]**。

    步骤（逐行对照上游源码）：
      1. preemphasis：y = lfilter([1, -0.97], [1], wav)
      2. STFT：librosa.stft(n_fft=800, hop=200, win=800, center=True) -> **幅度**
      3. mel：librosa.filters.mel(16000, 800, n_mels=80, fmin=55, fmax=7600)
      4. dB：S = 20*log10(max(1e-5, mel)) - ref_level_db(20)
      5. 对称归一化（hparams.symmetric_mels=True）：
         m = clip(8*(S+100)/100 - 4, -4, 4)
    """
    import librosa
    from scipy import signal

    preset = config.MEL_PRESETS["rtc16k"]
    wav = np.asarray(wav, dtype=np.float32)

    # 1) preemphasis
    y = signal.lfilter([1, -preset["preemphasis"]], [1], wav)

    # 2) STFT（幅度）
    spectrum = librosa.stft(
        y=y,
        n_fft=preset["n_fft"],
        hop_length=preset["hop_length"],
        win_length=preset["win_length"],
        window="hann",
        center=True,
    )

    # 3) mel
    mel_energy = np.abs(spectrum).T @ rtc_mel_filterbank()

    # 4) dB（减 ref_level_db）
    decibels = (20 * np.log10(np.maximum(1e-5, mel_energy))
                - preset["ref_level_db"])

    # 5) 对称归一化 -> [-4, 4]
    maximum = preset["max_abs_value"]
    normalized = ((2 * maximum) * ((decibels - preset["min_level_db"])
                                   / (-preset["min_level_db"])) - maximum)
    return np.ascontiguousarray(
        np.clip(normalized, -maximum, maximum), dtype=np.float32)


def rtc_mel(wav: np.ndarray) -> np.ndarray:
    """**WaveRNN 声码器的输入 mel**，形状 (T, 80)，取值 **[-1, 1]**。

    = `rtc_mel_raw(wav) / max_abs_value`。

    【为什么是 [-1, 1】—— 依据是上游的**训练代码**，不是猜的】
    `third_party/Real-Time-Voice-Cloning/vocoder/vocoder_dataset.py` 第 27-28 行：

        # Load the mel spectrogram and adjust its range to [-1, 1]
        mel = np.load(mel_path).T.astype(np.float32) / hp.mel_max_abs_value

    而 `hp.mel_max_abs_value = 4.0`，所以 WaveRNN 训练时见到的 mel ∈ [-1, 1]。
    反过来，`vocoder/inference.py::infer_waveform` 内部会 `mel / 4`，
    所以**喂给它之前要先乘 4**（见 `bridge_vocoder.VocoderBridge.mel_to_wav`）。

    【实测怎么验的】
    先取上游合成器**真实输出**的 mel（范围实测 [-4.19, +0.66]）当基准，
    再看哪种约定能算回来 —— 见 `scripts/verify_rtc_mel.py`。
    """
    preset = config.MEL_PRESETS["rtc16k"]
    return np.ascontiguousarray(
        rtc_mel_raw(wav) / preset["max_abs_value"], dtype=np.float32)


# ---------------------------------------------------------------------------
# mel 频谱：模型侧第三套约定（**HiFi-GAN 声码器**那套）★ 当前默认
# ---------------------------------------------------------------------------
def _hifigan_transforms():
    """构造 torchaudio 的 Spectrogram + MelScale（HiFi-GAN 那套）。

    ★ 为什么用 torchaudio 而不是自己写 numpy：
      我们这版权重（HF `speechbrain/tts-hifigan-libritts-16kHz`）是
      SpeechBrain 用**他们自己的** torchaudio 变换链训练的。
      用**同一套 torchaudio 调用**算 mel，才能保证**逐位一致** ——
      自己重写 numpy 版很容易在"窗口是不是周期的""reflect 填不填"这类细节上偏一点，
      而 mel 偏一点，声码器出来的就是噪音，**而且不报错**。

    参数来自该模型的 README（"IMPORTANT: Use these specific parameters to
    match the Vocoder's training settings"）：

        Spectrogram(hop_length=256, win_length=1024, n_fft=1024,
                    power=1, normalized=False)
        MelScale(sample_rate=16000, n_mels=80, f_min=0.0, f_max=8000.0,
                 norm="slaney", mel_scale="slaney")
    """
    import torchaudio

    preset = config.MEL_PRESETS["hifigan16k"]
    spectrogram = torchaudio.transforms.Spectrogram(
        hop_length=preset["hop_length"],
        win_length=preset["win_length"],
        n_fft=preset["n_fft"],
        power=preset["power"],
        normalized=preset["normalized"],
    )
    mel_scale = torchaudio.transforms.MelScale(
        sample_rate=preset["sample_rate"],
        n_stft=preset["n_fft"] // 2 + 1,
        n_mels=preset["n_mels"],
        f_min=preset["fmin"],
        f_max=preset["fmax"],
        norm=preset["norm"],
        mel_scale=preset["mel_scale"],
    )
    return spectrogram, mel_scale


def hifigan_mel_torch(wav: torch.Tensor) -> torch.Tensor:
    """HiFi-GAN 约定的 mel，torch 版，形状 **(80, T)**。

    ⚠️ 方向注意：HiFi-GAN 生成器吃的是 **(80, T)**（通道在前），
    而本项目其它 mel 函数返回 **(T, 80)**。
    用 `bridge_vocoder.VocoderBridge.mel_to_wav()` 就不会搞混。
    """
    spectrogram, mel_scale = _hifigan_transforms()
    spectrum = spectrogram(wav)
    mel = mel_scale(spectrum)
    preset = config.MEL_PRESETS["hifigan16k"]
    return torch.log(torch.clamp(mel, min=preset["clip_val"]))


def hifigan_mel(wav: np.ndarray) -> np.ndarray:
    """HiFi-GAN 约定的**原始** mel，形状 **(T, 80)**（与本项目其它 mel 函数同向）。

    ★ 这是**声码器要的量纲** —— 直接喂 `bridge_vocoder` 的 HiFi-GAN。
      网络侧（Ec/D 与损失）用的是**归一化**后的版本，见 `hifigan_mel_norm()`。

    与 `hifigan_mel_torch()` 数值一致（同一套 torchaudio 变换）。

    【实测验证】`scripts/verify_hifigan.py`：
      重建音与原音的说话人嵌入余弦 = **0.9746**，RMS 0.0539 -> 0.0585。
    """
    tensor = torch.from_numpy(np.ascontiguousarray(np.asarray(wav, dtype=np.float32)))
    mel = hifigan_mel_torch(tensor)
    return np.ascontiguousarray(mel.numpy().T, dtype=np.float32)   # (80, T) -> (T, 80)


def hifigan_mel_norm(wav: np.ndarray) -> np.ndarray:
    """**网络侧**的 HiFi-GAN mel：原始 mel 做一次固定仿射变换，形状 **(T, 80)**。

        a = scale * s + offset      （s = 原始 mel，a ≈ [-0.11, 0.92]）

    ★ **为什么要这一步**：论文的 `λ_cd = 1.0`、`λ_dis = 0.01` 是在
      **AutoVC 的 mel 量纲**（值域 ~[0,1]，宽约 1.0）下网格搜索出来的。
      原始的 HiFi-GAN mel 是 `ln(m)`，值域约 **[-11.5, 0]**，
      直接用会让 `L_rec` 比论文大 ~**100 倍**，
      `λ_cd` 的相对权重被稀释 ~40 倍 ⇒ **论文的超参就失效了**。

    ★ 仿射**可逆**，所以送进声码器前用 `hifigan_mel_unnorm()` 反变换回去即可，
      **零损失**（不需要 clip，clip 会让变换不可逆）。
    """
    preset = config.MEL_PRESETS["hifigan16k"]
    raw = hifigan_mel(wav)
    return np.ascontiguousarray(
        raw * preset["norm_scale"] + preset["norm_offset"], dtype=np.float32)


def hifigan_mel_unnorm(mel: np.ndarray) -> np.ndarray:
    """`hifigan_mel_norm()` 的**逆变换**：网络侧 mel -> 声码器要的原始 mel。"""
    preset = config.MEL_PRESETS["hifigan16k"]
    mel = np.asarray(mel, dtype=np.float32)
    return np.ascontiguousarray(
        (mel - preset["norm_offset"]) / preset["norm_scale"], dtype=np.float32)


def model_mel_padding_value(preset_name: str | None = None) -> float:
    """返回模型侧 mel 的真实静音值，任意长度推理补齐时使用。"""
    name = preset_name or config.STAGE1.get("mel_preset", config.DEFAULT_PRESET)
    preset = config.MEL_PRESETS[name]
    if name == "hifigan16k":
        import math

        return math.log(preset["clip_val"]) * preset["norm_scale"] + preset["norm_offset"]
    return 0.0


def model_mel(wav: np.ndarray, preset_name: str | None = None) -> np.ndarray:
    """**模型侧 mel 的统一入口**（Ec / D 的输入 + 损失的计算基准）。

    preset_name 省略时用 `config.STAGE1["mel_preset"]`（契约字段，两人共同区）。

    ★ 别再直接调 autovc_mel / rtc_mel / hifigan_mel —— 统一从这里走，
      否则"训练用一套 mel、评测用另一套"这种坑会以"指标莫名不对"的形式出现。

    ⚠️ 对 `hifigan16k`：这里返回的是**归一化**后的 mel（网络侧量纲）。
      送进声码器前要用 `hifigan_mel_unnorm()` 反变换 ——
      `bridge_vocoder.VocoderBridge.mel_to_wav()` 已经帮你做了。
    """
    name = preset_name or config.STAGE1.get("mel_preset", config.DEFAULT_PRESET)
    if name == "model16k":
        return autovc_mel(wav)
    if name == "rtc16k":
        return rtc_mel(wav)
    if name == "hifigan16k":
        return hifigan_mel_norm(wav)
    raise ValueError(
        f"未知的模型侧 mel 预设：{name!r}。"
        f"可选：{sorted(config.MEL_PRESETS)}"
    )


class MelSpec:
    """模型侧 mel 的可复用计算器。

    用法：
        spec = MelSpec()                  # 默认 config.DEFAULT_PRESET（AutoVC 链）
        mel = spec.mel(wav)              # (samples,) -> (T, 80)

    ★ **别再直接调 `autovc_mel`**（2026-11-20 修）——
      这里原来写死了 `return autovc_mel(wav)`，于是 `MelSpec("hifigan16k")`
      也会返回 AutoVC 口径的 mel：**名字写着 A、算出来是 B**，
      而且**不报错**。现在统一走 `model_mel(wav, self.name)`。

    ★ Stage 1 训练请用 `config.STAGE1["mel_preset"]`（契约 §7.1）。
    """

    def __init__(self, name: str | None = None):
        self.name = name or config.DEFAULT_PRESET
        self.preset = config.MEL_PRESETS[self.name]
        self.sample_rate = self.preset["sample_rate"]
        self.n_fft = self.preset["n_fft"]
        self.hop_length = self.preset["hop_length"]
        self.win_length = self.preset["win_length"]
        self.n_mels = self.preset["n_mels"]
        self.fmin = self.preset["fmin"]
        self.fmax = self.preset["fmax"]

    def mel(self, wav: np.ndarray) -> np.ndarray:
        """numpy 波形 -> 模型侧 mel，形状 (T, n_mels)。"""
        return model_mel(wav, self.name)

    def mel_batch(self, waveforms: list[np.ndarray]) -> list[np.ndarray]:
        """批量计算。变长语音不 padding，直接返回 list（Stage 1 训练用）。"""
        return [self.mel(wav) for wav in waveforms]

    def frames(self, num_samples: int) -> int:
        """给定采样点数，输出帧数 T。

        AutoVC 的 pySTFT 给 (N+256)//256 帧，与 librosa center=True 的
        1 + N//256 相同 —— 所以帧轴能对上，差别只在数值。
        """
        return (num_samples + self.hop_length) // self.hop_length


# ---------------------------------------------------------------------------
# mel 频谱：参照实现（librosa 口径，仅用于对照，不用于喂模型）
# ---------------------------------------------------------------------------
def librosa_mel(wav: np.ndarray, name: str | None = None) -> np.ndarray:
    """用 librosa.feature.melspectrogram 算 mel，形状 (T, n_mels)。

    ⚠️ **不要用它喂模型** —— 它的数值约定与 AutoVC 官方权重不兼容。
       保留它是为了：(1) 与 AutoVC 官方链做数值对照；(2) 论文式参考实现。
    """
    import librosa

    preset = config.MEL_PRESETS[name or config.DEFAULT_PRESET]
    mel = librosa.feature.melspectrogram(
        y=np.asarray(wav, dtype=np.float32),
        sr=preset["sample_rate"],
        n_fft=preset["n_fft"],
        hop_length=preset["hop_length"],
        win_length=preset["win_length"],
        n_mels=preset["n_mels"],
        fmin=preset["fmin"],
        fmax=preset["fmax"],
        power=2.0,
    )
    return np.ascontiguousarray(mel.T, dtype=np.float32)


# ---------------------------------------------------------------------------
# mel 频谱：声码器侧（HiFi-GAN 的约定，与模型侧不同！）
# ---------------------------------------------------------------------------
def vocoder_mel(wav: np.ndarray, n_mels: int | None = None) -> np.ndarray:
    """HiFi-GAN 声码器使用的 mel，形状 (T, n_mels)。

    ⚠️ **HiFi-GAN 的 mel 约定和模型侧不一样**，不要混用：
      | 约定      | 模型侧（AutoVC / 本项目）    | 声码器侧（HiFi-GAN）          |
      |-----------|-----------------------------|-------------------------------|
      | 采样率    | 16000                       | 22050（官方权重）             |
      | fmin/fmax | 90 / 7600                   | 0 / 8000（mel 基用 sr=8000） |
      | 幅度      | 幅度（非功率）                | 幅度                        |
      | 填充      | reflect                     | reflect（librosa stft）      |
      | 取对数    | 20*log10(max(1e-5,·)) - 16  | log10(|STFT|)，下限 1e-5     |
      | 归一化    | clip((x+100)/100, 0, 1)      | (x + 4.5) / 5.0             |

    实现按 HiFi-GAN 官方 mel_processing.py 的写法。
    **真正接入其权重时，必须与其仓库代码做一次数值对照。**
    """
    import librosa

    settings = config.VOCODER_MEL
    wav = np.asarray(wav, dtype=np.float32)
    mel_basis = librosa.filters.mel(
        sr=settings["mel_basis_sr"],
        n_fft=settings["n_fft"],
        n_mels=n_mels or settings["n_mels"],
    ).astype(np.float32)
    spectrum = librosa.stft(
        y=wav,
        n_fft=settings["n_fft"],
        hop_length=settings["hop_length"],
        win_length=settings["win_length"],
        window="hann",
        center=True,
    )
    log_spectrum = np.log10(np.clip(np.abs(spectrum), settings["log_floor"], None))
    mel = mel_basis @ log_spectrum
    mel = (mel + settings["offset"]) / settings["scale"]
    return np.ascontiguousarray(mel.T, dtype=np.float32)


# ---------------------------------------------------------------------------
# 波形工具（Stage 2 的心理声学掩蔽与 GAN 训练会用到）
# ---------------------------------------------------------------------------
def waveform_difference(wav_a: np.ndarray, wav_b: np.ndarray) -> np.ndarray:
    """两段波形的差，论文里 δ = V(x_adv) − V(x)。长度不足时补零到一致。"""
    length = max(len(wav_a), len(wav_b))
    a = np.pad(np.asarray(wav_a, dtype=np.float32), (0, length - len(wav_a)))
    b = np.pad(np.asarray(wav_b, dtype=np.float32), (0, length - len(wav_b)))
    return a - b


def power_spectrum(wav: np.ndarray, n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
    """功率谱，形状 (n_fft//2+1, frames)，供 L_mask 使用。"""
    import librosa

    spectrum = librosa.stft(
        y=np.asarray(wav, dtype=np.float32),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        center=True,
    )
    return np.abs(spectrum) ** 2.0


def rms_energy(wav: np.ndarray) -> float:
    """均方根能量，用于快速判断"输出是不是变成静音/噪声"。"""
    wav = np.asarray(wav, dtype=np.float32)
    return float(np.sqrt(np.mean(wav ** 2))) if len(wav) else 0.0


def spectral_distance(wav_a: np.ndarray, wav_b: np.ndarray) -> float:
    """两段波形的对数谱距离，用于快速判断音质变化幅度。"""
    a = power_spectrum(wav_a)
    b = power_spectrum(wav_b)
    length = min(a.shape[1], b.shape[1])
    a, b = a[:, :length], b[:, :length]
    return float(np.mean(np.abs(np.log10(a + 1e-10) - np.log10(b + 1e-10))))


def to_tensor(wav: np.ndarray) -> torch.Tensor:
    """numpy 波形 -> torch 张量 (samples,)。"""
    return torch.from_numpy(np.asarray(wav, dtype=np.float32))


def to_tensor_batch(mels: list[np.ndarray]) -> list[torch.Tensor]:
    """一批 mel -> 一批 torch 张量，形状各自为 (T, n_mels)。"""
    return [torch.from_numpy(mel) for mel in mels]


def normalize_l2(vector: torch.Tensor) -> torch.Tensor:
    """L2 归一化。说话人嵌入 h 的约定（见接口契约第 5 条）。"""
    return vector / (vector.norm(p=2, dim=-1, keepdim=True) + 1e-8)


def cosine_similarity(vector_a: torch.Tensor, vector_b: torch.Tensor) -> torch.Tensor:
    """余弦相似度。论文用它定义 L_de 与 Sim_prot / Sim_clone。"""
    return torch.nn.functional.cosine_similarity(
        vector_a.unsqueeze(0), vector_b.unsqueeze(0), dim=-1
    )


def frames_to_seconds(num_frames: int, hop_length: int, sample_rate: int) -> float:
    return num_frames * hop_length / sample_rate


def seconds_to_frames(seconds: float, hop_length: int, sample_rate: int) -> int:
    return int(math.floor(seconds * sample_rate / hop_length))
