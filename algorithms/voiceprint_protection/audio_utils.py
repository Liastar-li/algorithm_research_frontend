"""音频与 mel 频谱处理 —— 全项目唯一的实现入口。

【为什么这个文件必须存在】
论文要求 16 kHz / 80 维 mel，但三个第三方组件的原生配置互不相同
（AutoVC 22.05k、GE2E 16k/40维、HiFi-GAN 22.05k）。
如果 mel 参数在多处手写，出现不一致时：
  * x_adv 与 x 的帧轴对不上
  * L_mel 变成一个恒定的大常数，模型什么都学不到
  * **而且不报错** —— 这是最难查的一类 bug。
所以：任何地方都不许再手写 mel 参数，一律调用本模块。

【设计原则：不做"看起来等价"的重新实现】
第一版这里自己重写了 STFT，但 librosa 内部有窗口归一化、padding 方式、
mel 刻度等一堆细节，一旦猜错就会和 AutoVC 官方权重对不上，且不报错。
现在改成**直接调用 librosa 自己算**：
  * mel()         —— 与 AutoVC 官方代码路径完全一致（都用 librosa.feature.melspectrogram）
  * vocoder_mel() —— HiFi-GAN 声码器的 mel 约定（与上面不同，见该函数注释）
这样"mel 契约"就不依赖我们对第三方库的猜测。

【⚠️ 接入预训练权重时的必做动作】
AutoVC / HiFi-GAN 的 mel 约定各不相同（见下方两个函数）。
**第一次真正加载它们的权重时，必须用它们仓库里的原始 mel 代码做一次数值对照**
（同一段音频，两边算出来的 mel 应当逐元素接近），并把对照结论写进本文件注释。
本项目当前的两个 preset 是"按论文配置"和"按 AutoVC 公开配置"填的，
属于"文献与通行配置"，**以实际权重仓库的代码为准**。
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
# mel 频谱：模型侧（与 AutoVC 一致）
# ---------------------------------------------------------------------------
def mel_filterbank(name: str | None = None) -> np.ndarray:
    """mel 滤波器组，形状 (n_mels, n_fft//2+1)。

    用 librosa.filters.mel 的默认约定（Slaney 刻度 + Slaney 归一化），
    与 AutoVC 官方实现保持一致。
    """
    import librosa

    preset = config.MEL_PRESETS[name or config.DEFAULT_PRESET]
    return librosa.filters.mel(
        sr=preset["sample_rate"],
        n_fft=preset["n_fft"],
        n_mels=preset["n_mels"],
        fmin=preset["fmin"],
        fmax=preset["fmax"],
    ).astype(np.float32)


class MelSpec:
    """一个 mel 预设的可复用计算器。

    用法：
        spec = MelSpec("paper16k")
        mel = spec.mel(wav)              # (samples,) -> (T, 80)
        batch = spec.mel_batch([w1, w2]) # 变长时返回 list
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
        self.power = self.preset["power"]
        self.log_base = self.preset["log_base"]
        self.log_floor = self.preset["log_floor"]

    # -- 主路径：直接用 librosa，保证与 AutoVC 官方一致 ------------------
    def mel(self, wav: np.ndarray) -> np.ndarray:
        """numpy 波形 -> mel 频谱，形状 (T, n_mels)。"""
        import librosa

        wav = np.asarray(wav, dtype=np.float32)
        if len(wav) < self.win_length:
            raise ValueError(
                f"音频太短（{len(wav)} 采样点），至少需要 {self.win_length} 点。"
            )
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=self.power,
        )
        if self.log_base:
            mel = np.log10(np.clip(mel, self.log_floor, None))
        return np.ascontiguousarray(mel.T, dtype=np.float32)  # (T, n_mels)

    def mel_batch(self, waveforms: list[np.ndarray]) -> list[np.ndarray]:
        """批量计算。变长语音不 padding，直接返回 list（Stage 1 训练用）。"""
        return [self.mel(wav) for wav in waveforms]

    def frames(self, num_samples: int) -> int:
        """给定采样点数，输出帧数 T（用于与 librosa 的实际输出做对齐检查）。"""
        import librosa

        padded = num_samples + self.n_fft  # center=True 时前后各补 n_fft//2
        return 1 + (padded - self.win_length) // self.hop_length


# ---------------------------------------------------------------------------
# mel 频谱：声码器侧（HiFi-GAN 的约定，与模型侧不同！）
# ---------------------------------------------------------------------------
def vocoder_mel(wav: np.ndarray, n_mels: int = 80) -> np.ndarray:
    """HiFi-GAN 声码器使用的 mel，形状 (T, n_mels)。

    ⚠️ **HiFi-GAN 的 mel 约定和 AutoVC 不一样**，不要混用：
      | 约定      | AutoVC / 模型侧           | HiFi-GAN / 声码器侧        |
      |-----------|---------------------------|---------------------------|
      | fmin/fmax | 40 / 7600                 | 0 / 8000（mel 基用 sr=8000）|
      | 幅度      | 幅度平方 (power=2)         | 幅度 (power=1)             |
      | 取对数    | log10(mel)，下限 1e-10    | log10(|STFT|)，下限 1e-5   |
      | 归一化    | 无                        | (x + 4.5) / 5.0           |
      | 输入采样率| 16k 或 22.05k（按预设）   | 22.05k（官方权重）        |

    本函数按 HiFi-GAN 官方 mel_processing.py 的写法实现。
    **真正接入其权重时，必须与其仓库代码做一次数值对照**（见文件头说明）。
    """
    import librosa

    wav = np.asarray(wav, dtype=np.float32)
    mel_basis = librosa.filters.mel(sr=8000, n_fft=1024, n_mels=n_mels).astype(np.float32)
    spectrum = librosa.stft(
        y=wav,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window="hann",
        center=True,
    )
    log_spectrum = np.log10(np.clip(np.abs(spectrum), 1e-5, None))
    mel = mel_basis @ log_spectrum
    mel = (mel + 4.5) / 5.0
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
