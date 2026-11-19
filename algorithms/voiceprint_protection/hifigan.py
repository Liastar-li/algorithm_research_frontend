"""HiFi-GAN 生成器（**内置**，不依赖 third_party 克隆）。

【为什么内置而不是放 third_party】
`third_party/` 按项目约定**不提交到 git**（要靠脚本现拉）。
但 HiFi-GAN 的生成器只有 ~120 行，放进仓库有两个好处：
  * 两台机器、云机**零配置**就能跑（不用再 clone、不用对 commit）
  * 改动可追踪、可 review

代码来自 `bshall/hifigan`（HiFi-GAN v1 的生成器，作者 Benjamin van Niekerk，
它本身又是 `jik876/hifi-gan` 的改写），**MIT 许可**：

    Copyright (c) 2021 Benjamin van Niekerk

原仓库：https://github.com/bshall/hifigan  （commit 49c16c9）

【★ 为什么必须是 HiFi-GAN，而不是上游 WaveRNN】
论文 §2.3 原文：
    "The protected mel-spectrogram is reconstructed as x_adv = D(c, h_adv),
     and converted to waveform using HiFi-GAN vocoder V(·)."

Stage 2 的梯度必须**穿过声码器**才能回到生成器 G：
    G → h_adv → D(c, h_adv) → V(·) → E_spk(·) → L_de
而 WaveRNN 是**自回归生成器**（要采样 9 bit 离散码），**不可求导** ——
拿它训练，G 收不到梯度，**而且不报错**。
HiFi-GAN 是纯卷积网络，**天然可求导** ✅

【★ 权重】
NVIDIA 官方的 `NVIDIA/torch.hifi_gan` 仓库**已整体下架**（2026-11-18 实测
GitHub API / codeload / raw / release 附件全部 404）。
我们用 HuggingFace 上的等价替代：
    `speechbrain/tts-hifigan-libritts-16kHz` 的 `generator.ckpt`

它与 NVIDIA 的 `hifigan_melbins_16k` **逐项等价**（同一套 HiFi-GAN v1 架构、
同一套 mel 约定、同样 13,936,130 参数），只是训练数据不同（LibriTTS 16 kHz）：

| | NVIDIA `hifigan_melbins_16k` | 本项目用的 |
|---|---|---|
| 采样率 / hop / n_fft | 16000 / 256 / 1024 | **同** ✅ |
| n_mels | 80 | **同** ✅ |
| 架构 | HiFi-GAN v1（4 个 upsample、12 个 resblock） | **同** ✅ |
| 参数量 | 13,936,130 | **13,936,130** ✅ |
| mel 约定 | `log10` → `(x+4.5)/5`，fmin 0 / fmax 8000 | **同** ✅ |
| 可求导 | ✅ | ✅ |

下载：`scripts/download_hifigan_weights.py`
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1

# HiFi-GAN v1 的结构参数（16 kHz mel-bins 那套）
DEFAULT_IN_CHANNELS = 80
DEFAULT_UPSAMPLE_KERNEL_SIZES = (16, 16, 4, 4)
DEFAULT_UPSAMPLE_FACTORS = (8, 8, 2, 2)
DEFAULT_UPSAMPLE_INITIAL_CHANNEL = 512
DEFAULT_RESBLOCK_KERNEL_SIZES = (3, 7, 11)
DEFAULT_RESBLOCK_DILATION_SIZES = ((1, 3, 5), (1, 3, 5), (1, 3, 5))


def get_padding(kernel_size: int, dilation: int) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(nn.Module):
    """HiFi-GAN 的多感受野残差块（MRF 的一个分支）。"""

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation: tuple[int, ...] = (1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                                  dilation=dilation[index],
                                  padding=get_padding(kernel_size, dilation[index])))
            for index in range(3)
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1)))
            for _ in range(3)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            hidden = F.leaky_relu(x, LRELU_SLOPE)
            hidden = conv1(hidden)
            hidden = F.leaky_relu(hidden, LRELU_SLOPE)
            hidden = conv2(hidden)
            x = hidden + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class HifiGANGenerator(nn.Module):
    """HiFi-GAN 生成器：mel (B, 80, T) -> 波形 (B, 1, T*256)。

    ★ **必须可求导** —— Stage 2 的梯度要穿过它回到生成器 G。
    """

    def __init__(self,
                 in_channels: int = DEFAULT_IN_CHANNELS,
                 resblock_dilation_sizes: tuple = DEFAULT_RESBLOCK_DILATION_SIZES,
                 resblock_kernel_sizes: tuple = DEFAULT_RESBLOCK_KERNEL_SIZES,
                 upsample_kernel_sizes: tuple = DEFAULT_UPSAMPLE_KERNEL_SIZES,
                 upsample_initial_channel: int = DEFAULT_UPSAMPLE_INITIAL_CHANNEL,
                 upsample_factors: tuple = DEFAULT_UPSAMPLE_FACTORS,
                 inference_padding: int = 5,
                 sample_rate: int = 16000) -> None:
        super().__init__()
        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        self.sample_rate = sample_rate

        self.conv_pre = weight_norm(
            nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for index, (factor, kernel) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                upsample_initial_channel // (2 ** index),
                upsample_initial_channel // (2 ** (index + 1)),
                kernel, factor, padding=(kernel - factor) // 2)))

        self.resblocks = nn.ModuleList()
        for index in range(len(self.ups)):
            channels = upsample_initial_channel // (2 ** (index + 1))
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(channels, kernel, dilation))

        self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, 1, padding=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv_pre(x)
        for index in range(self.num_upsamples):
            output = F.leaky_relu(output, LRELU_SLOPE)
            output = self.ups[index](output)
            total = None
            for kernel_index in range(self.num_kernels):
                branch = self.resblocks[index * self.num_kernels + kernel_index](output)
                total = branch if total is None else total + branch
            output = total / self.num_kernels
        output = F.leaky_relu(output)
        output = self.conv_post(output)
        return torch.tanh(output)

    def remove_weight_norm(self) -> None:
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


# ---------------------------------------------------------------------------
# 权重加载
# ---------------------------------------------------------------------------
def _remap_speechbrain_keys(state: dict) -> dict:
    """把 SpeechBrain 的键名映射成本仓库生成器的键名。

    ★ 差异只有**一层** `.conv`：

        SpeechBrain: `conv_pre.conv.weight_v`   （它的 Conv1d 包装类把真正的
                    卷积放在 `self.conv` 里，weight_norm 加在外层）
        本仓库    : `conv_pre.weight_v`          （直接对 nn.Conv1d 做 weight_norm，
                    与 `bshall/hifigan` 官方写法一致）

    转换规则：凡是 `xxx.conv.weight_g` / `xxx.conv.weight_v` / `xxx.conv.bias`，
    **去掉中间那一层 `.conv`**。这个映射是唯一且可逆的
    （78 个卷积 × 3 个张量 = 234 个，转换前后数量一致）。
    """
    if not any(".conv." in key for key in state):
        return state          # 已经是本仓库的键名，不用动

    remapped = {}
    for key, value in state.items():
        for suffix in ("weight_g", "weight_v", "bias"):
            marker = f".conv.{suffix}"
            if key.endswith(marker):
                remapped[f"{key[: -len(marker)]}.{suffix}"] = value
                break
        else:
            remapped[key] = value
    return remapped


def load_hifigan_weights(weights_path: str | Path,
                         device: str | torch.device | None = None,
                         strict: bool = True) -> HifiGANGenerator:
    """加载 HiFi-GAN 权重。

    权重来源：HuggingFace `speechbrain/tts-hifigan-libritts-16kHz` 的
    `generator.ckpt`（**原生 HiFi-GAN v1 格式**：`weight_g` / `weight_v`）。

    strict=True 时会用 `load_state_dict(strict=True)`，
    **键名或形状对不上立刻报错** —— 这是我们判断"架构是否一致"最硬的检查。
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")

    if isinstance(state, dict):
        for key in ("generator", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    cleaned = {}
    for key, value in state.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    cleaned = _remap_speechbrain_keys(cleaned)

    model = HifiGANGenerator()
    model.load_state_dict(cleaned, strict=strict)
    model.eval().to(device)
    # ★ 参数全部冻结：声码器是**冻结组件**，只有生成器 G 和 D 需要梯度
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def describe_hifigan() -> str:
    model = HifiGANGenerator()
    total = sum(p.numel() for p in model.parameters())
    return (
        "HiFi-GAN v1 生成器（论文 §2.3 的 V(·)）\n"
        f"  参数量      : {total:,}\n"
        "  结构        : 4 个 upsample × 3 个 MRF 分支 = 12 个 resblock\n"
        "  输入/输出   : mel (B, 80, T)  ->  wav (B, 1, T*256)\n"
        "  可求导      : ✅（Stage 2 的梯度要穿过它）"
    )


if __name__ == "__main__":
    print(describe_hifigan())
