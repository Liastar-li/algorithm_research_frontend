"""Stage 2 的判别器 D_adv：看一段 mel，判断"这是不是真的说话"。

【论文里的位置】
论文式(10)(11)：
    L_D  = E[(D_adv(x) − 1)²] + E[D_adv(x_adv)²]        ← 判别器：把真的判成 1、假的判成 0
    L_GAN = E[(D_adv(x_adv) − 1)²]                       ← 生成器：想让 D_adv 把假的也判成 1

注意括号里的项是**平方**而不是 log —— 这是 **LSGAN**（最小二乘 GAN），
论文明确写了 "We employ LSGAN [17] to enforce perceptual realism"。
所以输出**不接 sigmoid**，直接输出一个实数分数。

【D_adv 在这里防的是什么】
只靠 L_de（让克隆音不像本人）和 L_per（听感别变）的话，
G 很容易走偏：它可能找到一些"能骗过克隆系统但把原说话人也毁掉了"的捷径。
D_adv 的作用是逼 x_adv **看起来像一段正常人声**，
也就是论文里说的 "enforce perceptual realism"。

【输入是什么】
D_adv(x) 与 D_adv(x_adv) 里的 x 是 **mel 频谱**（论文式 6/13 的 x），
不是波形 —— 因为 mel 才是模型内部的表示，判别器在那个空间里工作更稳。
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config


class MelDiscriminator(nn.Module):
    """mel 频谱判别器（LSGAN 版）。

    结构
    ----
        输入 (B, 1, T, 80)
          -> Conv2d(1, 32, 3x3, stride 2x2) + LeakyReLU(0.2)
          -> Conv2d(32, 64, 3x3, stride 2x2) + LeakyReLU(0.2)
          -> Conv2d(64, 128, 3x3, stride 2x2) + LeakyReLU(0.2)
          -> Conv2d(128, 256, 3x3, stride 2x2) + LeakyReLU(0.2)
          -> Conv2d(256, 1, 3x3)          -> (B, 1, T', 80')
          -> AdaptiveAvgPool2d(1) -> (B, 1, 1) -> 展平 -> (B, 1)

    为什么最后用全局平均池化
    ------------------------
    mel 的时间长度 T 会随语音长短变化。全局平均池化让输出**与 T 无关**，
    这样 batch 里长短不一的 mel 可以直接拼在一起算损失，
    不用逐条 pad 成一样长再截断（截断会丢掉信息）。

    为什么用 2D 卷积
    ---------------
    mel 的两个轴含义不同：**时间**和**频率**。频率轴上的邻近关系是有意义的
    （相邻 mel 滤波器覆盖相邻频带），2D 卷积能利用这一点；
    1D 卷积只看时间轴，会白白丢掉频率结构。

    输出
    ----
    (B, 1) —— 实数分数，**没有 sigmoid**（LSGAN 的要求）。
    """

    def __init__(self, n_mels: int = config.FRAME_DIM, channels: tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()

        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        self.head = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel (B, T, 80) -> (B, 1) 分数。

        也接受 (B, 1, T, 80)（已经带通道维的）。
        """
        if mel.dim() != 3:
            raise ValueError(f"期望 (B, T, 80) 的 mel，收到 {tuple(mel.shape)}")
        x = mel.unsqueeze(1)                 # (B, 1, T, 80)
        x = self.features(x)                 # (B, C, T', 80')
        x = self.head(x)                     # (B, 1, T'', 80'')
        x = self.pool(x).flatten(1)          # (B, 1)
        return x


class SpectralDiscriminator(nn.Module):
    """可选的第二个判别器：看**波形**。

    为什么可能需要它
    --------------
    mel 判别器看不见 mel 之外的失真（比如相位噪声、高频毛刺）。
    L_mask 是在**波形域**约束扰动的，所以配一个波形判别器更对得上。
    但它更吃算力、也更容易训崩。

    **默认不用**（论文只提了一个 D_adv，作用在 mel 上）。
    留在这里是因为消融实验可能会想试"加一个波形判别器"。

    输入 (B, 1, N) 波形 -> 一维卷积栈 -> (B, 1)
    """

    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()

        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=25, stride=8, padding=12),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.head = nn.Conv1d(in_channels, 1, kernel_size=25, stride=1, padding=12)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform (B, N) -> (B, 1) 分数。"""
        if waveform.dim() != 2:
            raise ValueError(f"期望 (B, N) 的波形，收到 {tuple(waveform.shape)}")
        x = waveform.unsqueeze(1)
        x = self.features(x)
        x = self.head(x)
        x = self.pool(x).flatten(1)
        return x
