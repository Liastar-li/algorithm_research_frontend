"""Stage 2 的前向流水线：把「原始语音」和「保护后语音」两条路接起来。

【为什么单独一个文件】
训练循环里反复要用这几件事，每一件都有**容易搞错的地方**，
集中放一处并写清楚，比散在 `train.py` 里安全：

  1. `mel(waveforms)`      原始语音 → mel（AutoVC 约定）
  2. `vocode(mel)`         mel → 波形（★ 必须可导）
  3. `align(a, b)`         两条波形长度可能不同 → 裁到一样长（否则相减报错）
  4. `thresholds(waveform)`原始波形 → 心理声学掩蔽阈值 θ_x
  5. `content(mel, emb)`   内容码 c

【★ 声码器：Stage 2 目前缺一个"可求导"的】
论文用的是 HiFi-GAN [15]，但我们有两个候选，各有坑：

  · **HiFi-GAN**（torch.hub，论文同款）
      坑：它的 mel 约定是 `log10` 幅度 + `(x+4.5)/5`，
          而 AutoVC 侧是 `20log10` + `clip[0,1]` —— **两者不能直接互换**。
          本文件提供 `autovc_mel_to_hifigan()` 做一次**近似**换算
          （假定两套滤波器组看到的底层能量相同，**不严格成立**）。

  · **AutoVC 配套的 wavenet_vocoder**（与 Stage 1 的 mel 天然匹配）
      坑：上游 `vocoder.generate()` 最后一步转 numpy，梯度断在那里。
          要用得把最后几步搬成 torch —— **这一步还没做**（见下方 TODO）。

  · **Stub**（随机权重）—— 只用于离线冒烟测试，指标无意义。

★ 这一项是 Stage 2 的**公开阻塞点**，已写进 `docs/notes/stage2_blockers.md`。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import audio_utils  # noqa: E402
import config  # noqa: E402


# ---------------------------------------------------------------------------
# mel 换算
# ---------------------------------------------------------------------------
def autovc_mel_to_hifigan(mel: torch.Tensor) -> torch.Tensor:
    """AutoVC 口径的 mel → HiFi-GAN 口径的 mel（**近似**）。

    AutoVC：`a = clip((20·log10(m) − 16 + 100)/100, 0, 1)`
        → 反解：`log10(m) = (a·100 − 84)/20`
    HiFi-GAN：`h = (log10(m') + 4.5)/5`
        → 反解：`log10(m') = 5h − 4.5`

    ★ **近似在哪**：两者的 mel 滤波器组不同
      （AutoVC: sr=16000, fmin=90, fmax=7600, 带 30 Hz 高通；
        HiFi-GAN: mel 基 sr=8000, fmin=0, fmax=8000），
      所以 `m' ≠ m`。这里是"假定底层能量相同"的近似换算。
      正式训练建议用 wavenet_vocoder（约定天然一致），别用这个。
    """
    log10_m = (mel * 100.0 - 84.0) / 20.0
    hifigan_normalized = (log10_m + 4.5) / 5.0
    return hifigan_normalized


def batched_autovc_mel(waveforms: torch.Tensor) -> torch.Tensor:
    """一批波形 → AutoVC 口径的 mel (B, T, 80)。

    逐条调 `audio_utils.autovc_mel`（numpy）再拼起来。
    慢，但**保证与项目里其他地方算出来的 mel 完全一致**（唯一真源原则）。
    这里只用于**原始语音**（不需要梯度），所以慢一点没关系。
    """
    frames = []
    for index in range(waveforms.shape[0]):
        single = waveforms[index].detach().cpu().numpy()
        frames.append(audio_utils.autovc_mel(single))
    length = min(frame.shape[0] for frame in frames)
    stacked = np.stack([frame[:length] for frame in frames], axis=0)
    return torch.from_numpy(stacked.astype(np.float32))


# ---------------------------------------------------------------------------
# 流水线
# ---------------------------------------------------------------------------
class Stage2Pipeline:
    """把冻结组件和辅助算子装在一起，给训练循环用。"""

    def __init__(self, speaker_encoder, stage1, vocoder=None,
                 vocoder_convention: str = "autovc",
                 device: torch.device | str = "cpu",
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 sample_rate: int = 16000):
        self.speaker = speaker_encoder
        self.stage1 = stage1
        self.vocoder = vocoder
        self.vocoder_convention = vocoder_convention
        self.device = torch.device(device)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate

    # -- 1. mel ----------------------------------------------------------
    def mel(self, waveforms: torch.Tensor) -> torch.Tensor:
        """原始波形 (B, N) → AutoVC mel (B, T, 80)。不需要梯度。"""
        return batched_autovc_mel(waveforms).to(self.device)

    # -- 2. 声码器 --------------------------------------------------------
    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        """mel → 波形。★ 必须可导（梯度要穿过它回到 G）。

        `vocoder_convention` 决定喂给声码器之前要不要换算 mel 口径。
        """
        if self.vocoder is None:
            raise RuntimeError("没有装载声码器 —— 见文件头的说明。")
        if self.vocoder_convention == "hifigan":
            mel = autovc_mel_to_hifigan(mel)
            # HiFi-GAN 要 (B, n_mels, T)
            return self.vocoder(mel.transpose(1, 2))
        return self.vocoder(mel)

    # -- 3. 波形对齐 -------------------------------------------------------
    @staticmethod
    def align(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """把两条波形裁到同样长（相减之前必须对齐）。

        为什么要对齐：δ_wav = V(x_adv) − V(x) 是**逐采样点相减**，
        长度不一样就没法减。x 与 x_adv 的帧数相同（契约 §2 的形状不变量），
        所以正常情况下两个波形本来就等长；这里只是兜底。
        """
        length = min(first.shape[-1], second.shape[-1])
        return first[..., :length], second[..., :length]

    # -- 4. 心理声学阈值 ---------------------------------------------------
    def thresholds(self, waveform: torch.Tensor) -> torch.Tensor | None:
        """原始波形 → 掩蔽阈值 θ_x (B, K)。

        逐条算（numpy），再拼成 batch。
        """
        try:
            from stage2.psychoacoustics import masking_threshold
        except ImportError:
            return None

        rows = []
        for index in range(waveform.shape[0]):
            single = waveform[index].detach().cpu().numpy()
            rows.append(masking_threshold(single, sample_rate=self.sample_rate,
                                         n_fft=self.n_fft))
        stacked = np.stack(rows, axis=0)
        return torch.from_numpy(stacked.astype(np.float32)).to(self.device)

    # -- 5. 内容码 ---------------------------------------------------------
    def content(self, mel: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """内容码 c（不需要梯度，它是"参照物"）。"""
        with torch.no_grad():
            return self.stage1.content_code(mel, embedding)

    # -- 自述 --------------------------------------------------------------
    def describe(self) -> str:
        vocoder_name = type(self.vocoder).__name__ if self.vocoder is not None else "（无）"
        return (
            f"Stage 2 流水线\n"
            f"  E_spk      : {type(self.speaker).__name__}\n"
            f"  Stage 1    : {type(self.stage1).__name__}\n"
            f"  声码器     : {vocoder_name}（mel 口径：{self.vocoder_convention}）\n"
            f"  采样率     : {self.sample_rate} Hz\n"
            f"  STFT       : n_fft={self.n_fft} hop={self.hop_length}"
        )


# ---------------------------------------------------------------------------
# 声码器们
# ---------------------------------------------------------------------------
class HiFiGANVocoder(nn.Module):
    """HiFi-GAN（论文同款），torch 版，可导。

    需要联网（torch.hub 从 GitHub 下载）。
    """

    def __init__(self, device: str | None = None, model_name: str = "hifigan_melbins_16k"):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            self.model = torch.hub.load("NVIDIA/torch.hifi_gan", model_name)
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(
                f"加载 HiFi-GAN 失败（{error}）。\n"
                "它需要联网从 GitHub 下载（torch.hub）。\n"
                "离线环境请用 --vocoder stub，或先在有网的环境里把 torch.hub 缓存好。"
            ) from error
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """HiFi-GAN 口径的 mel (B, 80, T) → 波形 (B, N)。"""
        return self.model(mel).squeeze(1)


class WaveNetVocoder(nn.Module):
    """AutoVC 配套的 wavenet_vocoder（**TODO：还没实现**，见文件头）。

    ★ 为什么需要它
      它的 mel 约定与 AutoVC **天然一致**（dB 归一化到 [0,1]），
      不需要像 HiFi-GAN 那样做近似换算 → 论文的 L_mel / L_mask 才严格成立。

    ★ 怎么实现（照着上游 `wavenet_vocoder/vocoder.py` 搬）
      1. `from wavenet_vocoder import builder` → `builder.build_model()`
      2. 加载 `checkpoint_step001000000_ema.pth` 的 `state_dict`
      3. 上游 `extract_features(mel)`：把 (B, T, 80) 的 mel 变成模型要的
         上采样条件（这一步**是 torch 的**，可以照抄）
      4. 上游 `model(c, None)` 出波形（**是 torch 的**，可以照抄）
      5. ★ **只有最后"转 numpy"那一步要改成留在 torch 里** ——
         梯度就是断在那里的

    在这之前，用 `--vocoder stub`（冒烟测试）或 `--vocoder hifigan`（近似换算）。
    """

    def __init__(self, checkpoint=None, device: str | None = None):
        super().__init__()
        raise NotImplementedError(
            "WaveNetVocoder 还没实现 —— 见本类的 docstring 里的 5 步实现说明。\n"
            "当前可用：--vocoder stub（冒烟测试）/ --vocoder hifigan（近似换算）。"
        )
