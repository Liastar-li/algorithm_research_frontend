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


class MelSpec:
    """模型侧 mel 的可复用计算器（AutoVC 链）。

    用法：
        spec = MelSpec()
        mel = spec.mel(wav)              # (samples,) -> (T, 80)，取值 [0,1]
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
        return autovc_mel(wav)

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
