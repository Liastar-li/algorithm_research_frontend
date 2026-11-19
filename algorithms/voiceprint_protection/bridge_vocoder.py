"""桥接：声码器 V(·) —— mel 频谱 -> 波形。

【它是干什么的】
Stage 1 的解码器 D 输出的是 **mel 频谱**，不是声音。
要"听"出重建得好不好，必须再过一个**声码器**把 mel 变成波形。

    mel (80, T)  ──V(·)──►  wav (T*256,)  @16 kHz

论文 §2.3 原文：
    "The protected mel-spectrogram is reconstructed as x_adv = D(c, h_adv),
     and converted to waveform using **HiFi-GAN vocoder V(·)**."

【★ 为什么是 HiFi-GAN，而不是上游 WaveRNN】
| | HiFi-GAN | 上游 WaveRNN |
|---|---|---|
| 论文指定 | ✅ §2.3 明确写的就是它 | ❌ |
| **可求导** | ✅ 纯卷积网络 | ❌ **自回归采样 9 bit 离散码，梯度断** |
| 权重可得 | ✅ HF `speechbrain/tts-hifigan-libritts-16kHz` | ✅ HF `CorentinJ/SV2TTS` |
| 16 kHz | ✅ | ✅ |

**可求导这一条是决定性的**：Stage 2 的梯度必须穿过声码器才能回到生成器 G
```
G → h_adv → D(c, h_adv) → V(·) → E_spk(·) → L_de
                              ↑
                        这里必须可导
```
拿 WaveRNN 训练，G 收不到梯度，**loss 一步都不动，而且不报错**。

【★ 权重从哪来】
NVIDIA 官方的 `NVIDIA/torch.hifi_gan` 仓库**已整体下架**（2026-11-18 实测：
GitHub API / codeload / raw / release 附件全部 404）。
我们用 HuggingFace 上的等价替代 `speechbrain/tts-hifigan-libritts-16kHz`：
**同一套 HiFi-GAN v1 架构、同一套 mel 约定、同样 13,936,130 参数**，
只是训练数据不同（LibriTTS 16 kHz）。
加载时用 `strict=True` —— 键名或形状对不上立刻报错（实测 234/234 全匹配）。

下载：`scripts/download_hifigan_weights.py`

【★ mel 约定必须对上，否则出来是噪音且不报错】
HiFi-GAN 只认这套（见 `audio_utils.hifigan_mel()`）：
    STFT(n_fft=1024, hop=256, win=1024, center=True) 的**幅度**
    -> slaney mel（fmin 0 / fmax 8000，80 维）
    -> **自然对数** log(clamp(mel, 1e-5))
实测：喂对的约定，重建音与原音的说话人嵌入余弦 = **0.9746**；
喂 NVIDIA 那条 log10 链 → 2512% 的相对误差（纯噪音）。

【形象比喻】
mel 频谱像乐谱，声码器像演奏乐谱的乐团。
乐团只认自己那本谱子的记谱法 —— 拿另一本谱子给它，
它照样"演奏"，但出来的是噪音，而且**不会报错**。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

import config
import paths


class VocoderBridge:
    """HiFi-GAN 声码器（论文 §2.3 的 V(·)）：mel -> 波形。

    用法：
        vocoder = VocoderBridge()
        wav = vocoder.mel_to_wav(mel)          # mel: (80, T) 或 (T, 80)
    """

    def __init__(self, weights_path: str | Path | None = None,
                 device: str | torch.device | None = None):
        import hifigan

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.weights_path = Path(weights_path) if weights_path else \
            self._default_weights_path()
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"找不到 HiFi-GAN 权重：{self.weights_path}\n"
                "请先下载：\n"
                "    python algorithms/voiceprint_protection/scripts/download_hifigan_weights.py\n"
                "（NVIDIA 官方仓库已下架，我们用 HF 上等价的 "
                "speechbrain/tts-hifigan-libritts-16kHz）"
            )

        # ★ strict=True：架构对不上立刻报错（实测 234/234 键全匹配）
        self.model = hifigan.load_hifigan_weights(
            self.weights_path, device=self.device, strict=True)
        self.sample_rate = 16000

    @staticmethod
    def _default_weights_path() -> Path:
        candidates = [
            paths.models_dir() / "default" / "hifigan_16k.ckpt",
            paths.models_dir() / "hifigan_16k.ckpt",
            paths.WEIGHTS_DIR / "hifigan_16k.ckpt",
            paths.REPO_ROOT / "weights" / "hifigan_16k.ckpt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    # -- 对外 ------------------------------------------------------------
    def mel_to_wav(self, mel: np.ndarray | torch.Tensor,
                   normalized: bool = True,
                   batch: bool = False) -> np.ndarray | torch.Tensor:
        """mel -> 波形。

        参数
        ----
        mel       : 2 维 **(T, 80)**（项目约定：时间在前）或 (80, T)，会自动转置。
                    3 维（批量）时用 **(B, T, 80)**（时间在前）。
        normalized : True（默认）= 输入是**网络侧**的 mel（归一化过的），
                     内部会反变换成声码器要的原始量纲；
                     False = 输入已经是原始 mel，直接喂。
        batch     : True 时按批量处理，输入必须是 (B, T, 80)，返回 (B, N) tensor。

        返回 : 波形。numpy 输入 -> numpy 输出；批量模式 -> torch tensor。
        """
        if isinstance(mel, torch.Tensor):
            tensor = mel
        else:
            tensor = torch.from_numpy(
                np.ascontiguousarray(np.asarray(mel, dtype=np.float32)))

        n_mels = config.MEL_PRESETS["hifigan16k"]["n_mels"]

        if tensor.dim() == 2:
            # 单条：(T, 80) 或 (80, T) -> 一律转成 (80, T)
            if tensor.shape[1] == n_mels:
                tensor = tensor.transpose(0, 1).contiguous()      # (T,80) -> (80,T)
            elif tensor.shape[0] != n_mels:
                raise ValueError(
                    f"mel 形状应为 (T, 80) 或 (80, 80)，实际 {tuple(tensor.shape)}")
        elif tensor.dim() == 3:
            # 批量：(B, T, 80) -> (B, 80, T)（HiFi-GAN 吃通道在前）
            if tensor.shape[2] == n_mels:
                tensor = tensor.transpose(1, 2).contiguous()
            elif tensor.shape[1] != n_mels:
                raise ValueError(
                    f"批量 mel 形状应为 (B, T, 80)，实际 {tuple(tensor.shape)}")
        else:
            raise ValueError(f"mel 维度应为 2 或 3，实际 {tensor.dim()}")

        if normalized:
            preset = config.MEL_PRESETS["hifigan16k"]
            tensor = (tensor - preset["norm_offset"]) / preset["norm_scale"]

        if batch or tensor.dim() == 3:
            return self.model(tensor.to(self.device)).squeeze(1)

        with torch.no_grad():
            wav = self.model(tensor.unsqueeze(0).to(self.device))
        return wav.squeeze().cpu().numpy()

    def wav_to_mel(self, wav: np.ndarray) -> np.ndarray:
        """波形 -> mel（HiFi-GAN **原始**约定），形状 (T, 80)。"""
        import audio_utils

        return audio_utils.hifigan_mel(np.asarray(wav, dtype=np.float32))

    def describe(self) -> str:
        return (
            "HiFi-GAN 声码器（论文 §2.3 的 V(·)）\n"
            f"  权重      : {self.weights_path}\n"
            f"  设备      : {self.device}\n"
            f"  采样率    : {self.sample_rate}\n"
            "  输入/输出 : mel (80, T)  ->  wav (T*256,)\n"
            "  可求导    : ✅（Stage 2 的梯度要穿过它）"
        )


# ---------------------------------------------------------------------------
# 兼容旧接口：上游 WaveRNN 声码器（**不可求导**，只用于对照/兜底）
# ---------------------------------------------------------------------------
class WaveRNNBridge:
    """上游 Real-Time-Voice-Cloning 的 WaveRNN 声码器（**不可求导**）。

    ⚠️ **不要用它训练 Stage 2** —— 它是自回归生成器（采样 9 bit 离散码），
    梯度断在那里，G 收不到梯度而且**不报错**。
    保留它是因为：它的 mel 约定（`rtc16k`）与 HiFi-GAN 不同，
    两者可以互相对照，验证"mel 约定错了会出噪音"。
    """

    def __init__(self, rtc_dir: str | Path | None = None,
                 device: str | None = None,
                 models_dir: str | Path | None = None):
        from bridge_speaker import find_rtc_dir

        self.rtc_dir = find_rtc_dir(str(rtc_dir) if rtc_dir else None)
        self.models_dir = Path(models_dir) if models_dir else paths.models_dir()
        self.device = device

        if str(self.rtc_dir) not in sys.path:
            sys.path.insert(0, str(self.rtc_dir))

        self.vocoder_path = self._find_file("vocoder.pt")

        from vocoder import inference as vocoder_inference

        self._vocoder = vocoder_inference
        self._vocoder.load_model(self.vocoder_path, verbose=False)
        self.sample_rate = 16000

    def _find_file(self, name: str) -> Path:
        candidates = [
            self.models_dir / "default" / name,
            self.models_dir / name,
            self.rtc_dir / name,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.stat().st_size > 1024:
                with candidate.open("rb") as handle:
                    if handle.read(1) == b"<":
                        raise FileNotFoundError(
                            f"{candidate} 是一个 HTML 错误页，不是权重文件！\n"
                            "请删掉它，然后重新下载：\n"
                            "    python algorithms/voiceprint_protection/scripts/download_rtc_weights.py")
                return candidate
        raise FileNotFoundError(
            f"找不到 {name}。请先下载上游权重：\n"
            "    python algorithms/voiceprint_protection/scripts/download_rtc_weights.py")

    def mel_to_wav(self, mel: np.ndarray, scale: float | None = None) -> np.ndarray:
        """`rtc16k` 约定的 mel -> 波形（numpy，不可求导）。"""
        mel = np.asarray(mel, dtype=np.float32)
        if mel.ndim != 2 or mel.shape[1] != 80:
            raise ValueError(f"mel 形状应为 (T, 80)，实际 {mel.shape}")
        scale = 4.0 if scale is None else scale
        spec = (mel * scale).T
        wav = self._vocoder.infer_waveform(
            spec, normalize=True, batched=True,
            progress_callback=lambda *args, **kwargs: None)
        return np.asarray(wav, dtype=np.float32)

    def describe(self) -> str:
        return (
            "上游 WaveRNN 声码器（不可求导，**不要用于 Stage 2 训练**）\n"
            f"  权重      : {self.vocoder_path}\n"
            "  mel 约定  : rtc16k（与 HiFi-GAN 不同）"
        )


def main() -> int:
    """命令行自检：算一段 mel，过声码器，看能不能出声。"""
    import audio_utils

    vocoder = VocoderBridge()
    print(vocoder.describe())
    mel = np.zeros((80, 50), dtype=np.float32)
    mel[:, :] = np.linspace(-11, 0, 80)[None, :]     # 一个假的、单调的 mel
    wav = vocoder.mel_to_wav(mel)
    print(f"  假 mel -> 波形 {wav.shape}，RMS = {float(np.sqrt((wav ** 2).mean())):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
