"""Stage 1 模型：内容-说话人解耦-重建网络。

【对应论文 §2.2】
    c   = E_c(x, h)          内容码（信息瓶颈，丢掉说话人）
    x̂   = D(c, h)            重建
    x̂_psnt = Postnet(x̂) + x̂ 精修后的重建

【架构来自哪里】
严格照搬 AutoVC 官方 `third_party/AutoVC/model_vc.py` 的结构
（ConvNorm / LinearNorm / BatchNorm / 双向 LSTM / Postnet 残差）。
**为什么照搬而不是自己设计**：
论文明确说 Stage 1 用 AutoVC 架构，而 AutoVC 官方权重链已 404 失效，
所以我们**自己实现并训练** —— 架构照搬能让 `marlenezw` 那份
`ckpt_autovc.pth`（含 Ec + D + GE2E）能作为初始化。

【★ 一个必须知道的事实：content code 的真实形状】
AutoVC 的编码器把 T 帧按 `freq=32` 抽样，得到 **T//32 个** content code，
每个是 **64 维**（dim_neck=32 × 双向 LSTM 两个方向拼接）。

    x      : (B, T, 80)
    codes  : list，共 T//32 个，每个 (B, 64)
    ↑ 每个 code 在解码时被"摊开"回 32 帧，解码器因此仍在 T 帧上工作，
      输出 mel 与输入**等长**（这点很关键，契约要求 x 与 x_adv 帧数完全相等）

⚠️ **常见误解**：content code 的时间分辨率是 32 帧 = 0.5 秒（@16 kHz），
不是"逐帧"。论文写 `c ∈ R^{T'×32}`、`T' = T//2` 与官方代码不符，
以**官方代码为准**（成员 A 实测）。详见 docs/interface-contract.md。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ---------------------------------------------------------------------------
# 基础层（照抄 AutoVC 官方 model_vc.py）
# ---------------------------------------------------------------------------
class LinearNorm(nn.Module):
    """带 Xavier 初始化的线性层（官方 w_init_gain='linear'）。"""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True,
                 w_init_gain: str = "linear"):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight,
                                gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


class ConvNorm(nn.Module):
    """一维卷积 + 权重归一化的官方写法（padding 使时间轴长度不变）。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1,
                 stride: int = 1, padding: int | None = None, dilation: int = 1,
                 bias: bool = True, w_init_gain: str = "linear"):
        super().__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=dilation,
                                    bias=bias)
        nn.init.xavier_uniform_(self.conv_layer.weight,
                                gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.conv_layer(signal)


# ---------------------------------------------------------------------------
# 内容编码器 E_c
# ---------------------------------------------------------------------------
class ContentEncoder(nn.Module):
    """把 mel（+说话人嵌入作为条件）压成 content code。

    ★ 官方实现里编码器**也吃说话人嵌入**（concat 到输入通道）。
      这一点常被误解 —— 官方 `Encoder.forward(x, c_org)` 把 c_org 拼进特征。
      论文说 "c = Ec(x, h)"，与此一致。
    """

    def __init__(self, dim_neck: int = 32, dim_emb: int = 256,
                 n_mels: int = 80, freq: int = 32):
        super().__init__()
        self.dim_neck = dim_neck
        self.freq = freq

        convolutions = []
        for index in range(3):
            in_channels = n_mels + dim_emb if index == 0 else 512
            convolutions.append(nn.Sequential(
                ConvNorm(in_channels, 512, kernel_size=5, stride=1, padding=2,
                         dilation=1, w_init_gain="relu"),
                nn.BatchNorm1d(512),
            ))
        self.convolutions = nn.ModuleList(convolutions)

        # ★ bidirectional=True -> 输出 2*dim_neck 维（32*2 = 64）
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, mel: torch.Tensor, embedding: torch.Tensor):
        """
        mel      : (B, T, 80)
        embedding: (B, 256)
        返回      : list of (B, 2*dim_neck)，长度 T // freq
        """
        # ★ 顺序要紧：官方写的是 `x.squeeze(1).transpose(2, 1)`，
        #   那是为兼容 (B, 1, 80, T) 的老输入格式；
        #   我们直接吃 (B, T, 80)，所以一步 transpose(1, 2) 即可。
        #   得到 (B, 80, T) 后沿 dim=1（通道维）拼上说话人嵌入 -> (B, 336, T)
        x = mel.transpose(1, 2)                                  # (B, 80, T)
        condition = embedding.unsqueeze(-1).expand(-1, -1, x.size(-1))   # (B, 256, T)
        x = torch.cat((x, condition), dim=1)                      # (B, 336, T)

        for conv in self.convolutions:
            x = F.relu(conv(x))                                   # (B, 512, T)
        # ★ .contiguous()：transpose 出来的是**非连续**张量，
        #   而 cuDNN 的 LSTM 只在输入连续时才走快路径，否则 PyTorch 会退回
        #   纯 Python 实现的慢路径（慢好几倍）。这是训练提速的大头。
        x = x.transpose(1, 2).contiguous()                         # (B, T, 512)

        outputs, _ = self.lstm(x)                                 # (B, T, 2*dim_neck)
        out_forward = outputs[:, :, : self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]

        codes = []
        for index in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat(
                (out_forward[:, index + self.freq - 1, :], out_backward[:, index, :]),
                dim=-1,
            ))
        return codes


# ---------------------------------------------------------------------------
# 解码器 D
# ---------------------------------------------------------------------------
class SpeakerDecoder(nn.Module):
    """content code + 目标说话人嵌入 -> mel。"""

    def __init__(self, dim_neck: int = 32, dim_emb: int = 256, dim_pre: int = 512,
                 n_mels: int = 80):
        super().__init__()
        self.lstm1 = nn.LSTM(dim_neck * 2 + dim_emb, dim_pre, 1, batch_first=True)

        convolutions = []
        for _ in range(3):
            convolutions.append(nn.Sequential(
                ConvNorm(dim_pre, dim_pre, kernel_size=5, stride=1, padding=2,
                         dilation=1, w_init_gain="relu"),
                nn.BatchNorm1d(dim_pre),
            ))
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        self.linear_projection = LinearNorm(1024, n_mels)

    def forward(self, codes: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """
        codes    : (B, T, 2*dim_neck)
        embedding: (B, 256)
        返回      : (B, T, n_mels)
        """
        x, _ = self.lstm1(codes)                        # (B, T, dim_pre)
        x = x.transpose(1, 2).contiguous()              # (B, dim_pre, T)
        for conv in self.convolutions:
            x = F.relu(conv(x))
        # ★ 同上：给 LSTM 的输入必须是连续张量，否则走慢路径
        x = x.transpose(1, 2).contiguous()              # (B, T, dim_pre)
        outputs, _ = self.lstm2(x)                     # (B, T, 1024)
        return self.linear_projection(outputs)          # (B, T, n_mels)


# ---------------------------------------------------------------------------
# Postnet
# ---------------------------------------------------------------------------
class Postnet(nn.Module):
    """5 层卷积的残差精修（官方：80 -> 512 -> 512 -> 512 -> 512 -> 80）。"""

    def __init__(self, n_mels: int = 80, channels: int = 512):
        super().__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(nn.Sequential(
            ConvNorm(n_mels, channels, kernel_size=5, stride=1, padding=2,
                     dilation=1, w_init_gain="tanh"),
            nn.BatchNorm1d(channels),
        ))
        for _ in range(1, 5 - 1):
            self.convolutions.append(nn.Sequential(
                ConvNorm(channels, channels, kernel_size=5, stride=1, padding=2,
                         dilation=1, w_init_gain="tanh"),
                nn.BatchNorm1d(channels),
            ))
        self.convolutions.append(nn.Sequential(
            ConvNorm(channels, n_mels, kernel_size=5, stride=1, padding=2,
                     dilation=1, w_init_gain="linear"),
            nn.BatchNorm1d(n_mels),
        ))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        for index in range(len(self.convolutions) - 1):
            mel = torch.tanh(self.convolutions[index](mel))
        return self.convolutions[-1](mel)


# ---------------------------------------------------------------------------
# 完整 Stage 1 模型
# ---------------------------------------------------------------------------
class Stage1Model(nn.Module):
    """内容编码器 + 解码器 + Postnet（AutoVC 架构）。

    用法：
        model = Stage1Model()
        mel_out, mel_postnet, codes = model(mel, emb_org, emb_trg)
    """

    def __init__(self, dim_neck: int | None = None, dim_emb: int | None = None,
                 dim_pre: int | None = None, freq: int | None = None,
                 n_mels: int = 80):
        super().__init__()
        settings = config.AUTOVC_MODEL
        self.dim_neck = dim_neck or settings["dim_neck"]
        self.dim_emb = dim_emb or settings["dim_emb"]
        self.dim_pre = dim_pre or settings["dim_pre"]
        self.freq = freq or settings["freq"]

        self.encoder = ContentEncoder(self.dim_neck, self.dim_emb, n_mels, self.freq)
        self.decoder = SpeakerDecoder(self.dim_neck, self.dim_emb, self.dim_pre, n_mels)
        self.postnet = Postnet(n_mels)

    def forward(self, mel: torch.Tensor, source_embedding: torch.Tensor,
                target_embedding: torch.Tensor):
        """
        mel             : (B, T, 80)
        source_embedding: (B, 256)  编码侧条件（原始说话人）
        target_embedding: (B, 256)  解码侧条件（可换成被扰动的嵌入）

        返回
        ----
        mel_out    : (B, T, 80)   x̂        （解码器输出）
        mel_postnet: (B, T, 80)   x̂_psnt   （含 Postnet 残差）
        codes      : list of (B, 64)，长度 T // freq
        """
        codes = self.encoder(mel, source_embedding)

        # 每个 code 摊回 freq 帧，再拼上目标说话人嵌入
        expanded = []
        for code in codes:
            expanded.append(code.unsqueeze(1).expand(
                -1, int(mel.size(1) / len(codes)), -1))
        code_expanded = torch.cat(expanded, dim=1)       # (B, T, 64)

        decoder_input = torch.cat(
            (code_expanded,
             target_embedding.unsqueeze(1).expand(-1, mel.size(1), -1)),
            dim=-1,
        )                                                  # (B, T, 64+256)

        mel_out = self.decoder(decoder_input, target_embedding)      # (B, T, 80)
        mel_postnet = self.postnet(mel_out.transpose(2, 1).contiguous())   # (B, 80, T)
        mel_postnet = mel_out + mel_postnet.transpose(2, 1)            # 残差
        return mel_out, mel_postnet, codes

    # -- 便捷：只重建（源=目标）-------------------------------------------
    def reconstruct(self, mel: torch.Tensor, embedding: torch.Tensor):
        """x̂_psnt = D(c, h)，用于"重建质量"自检。"""
        _, mel_postnet, _ = self.forward(mel, embedding, embedding)
        return mel_postnet

    def content_codes(self, mel: torch.Tensor, embedding: torch.Tensor):
        """只做编码：c = E_c(x, h)。"""
        return self.encoder(mel, embedding)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
