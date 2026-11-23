"""Stage 2 的全部损失函数（论文 §2.3，式 7–13）。

【一句话总览】
    L_per  = L_mel + λ_mask · L_mask      （式 7、8、9）  —— 保证**人耳**听不出来
    L_de   = sim(...)                     （式 12）        —— 保证**机器**认不出来
    L_GAN  = E[(D_adv(x_adv) − 1)²]       （式 11）        —— 保证保护后音频**像人话**
    L_D    = E[(D_adv(x)−1)²] + E[D_adv(x_adv)²]（式 10）—— 判别器自己学
    L_G    = L_GAN + λ_de·L_de + λ_per·L_per（式 13）    —— 生成器的总目标

【★ 最容易写反的一条：L_de 是要"最小化"的】
论文原文：*"Minimizing L_de ensures that speech cloned using the stolen embedding
does not resemble the original speaker."*
——`L_de` 越小，攻击者克隆出来的声音就越不像本人，防御越有效。

如果写成"最大化相似度"（比如写成 `1 - sim` 却忘了加负号，或者不小心用了
`-sim` 但前面对 h_stolen 做了 detach 导致梯度方向反了），
训练会**把音频彻底毁掉**。任务板 §「三个最容易踩的坑」第 1 条就是这个。

**自检方法**：训练 100 步后，L_de 应该**下降**；如果它上升，说明符号反了。
`train.py` 每步都会打印这几个损失，跑一眼就知道。

【超参来自哪里】
全部读 `config.STAGE2`（唯一真源），论文值：
    α = 0.10, λ_per = 15.0, λ_de = 5.0, λ_mask = 0.1, λ_GAN = 1.0
"""

from __future__ import annotations

import torch

import config
from stage2.common import cosine_similarity


# ---------------------------------------------------------------------------
# 式(7) L_mel —— 频谱保持
# ---------------------------------------------------------------------------
def mel_loss(original_mel: torch.Tensor, protected_mel: torch.Tensor) -> torch.Tensor:
    """L_mel = ‖x − x_adv‖² （论文式 7）

    含义：保护前后的频谱要**尽量一样**。
    频谱变了 = 音色变了 = 人耳听得出来 = 防御失败。

    参数
    ----
    original_mel : (B, T, 80) 原始语音的 mel
    protected_mel: (B, T, 80) 保护后语音的 mel

    ★ 帧数必须一致，否则广播会出错或悄悄退化成常数。
      `train.py` 里显式断言这一点（这是契约 §2 的"形状不变量"）。
    """
    if original_mel.shape != protected_mel.shape:
        raise ValueError(
            f"L_mel 要求两个 mel 形状一致，收到 {tuple(original_mel.shape)} vs "
            f"{tuple(protected_mel.shape)}。\n"
            "这几乎总是因为 x_adv 的帧数与 x 不一样 —— 契约里写死的形状不变量。"
        )
    return torch.mean((original_mel - protected_mel) ** 2)


# ---------------------------------------------------------------------------
# 式(8) L_mask —— 心理声学掩蔽
# ---------------------------------------------------------------------------
def mask_loss(delta_waveform: torch.Tensor,
              threshold: torch.Tensor | None,
              sample_rate: int = 16000) -> torch.Tensor:
    """L_mask = (1/K) Σ_k max(P_δ(k) − θ_x(k), 0) （论文式 8）

    含义：扰动 δ 在每个频点上的能量，超过人耳掩蔽阈值的部分要被罚。

    参数
    ----
    delta_waveform: (B, N) 波形扰动 δ = V(x_adv) − V(x)
    threshold      : (K,) 每个频点的掩蔽阈值 θ_x（dB），None 表示不做这一项
    sample_rate    : 采样率

    实现说明
    --------
    PSD 与阈值都在 **dB** 域比较（掩蔽阈值本来就是 dB 量纲）。
    `psychoacoustics.masking_loss_torch` 是同一条公式的可导实现；
    这里保留一个薄封装，方便调用方不必知道模块内部细节。
    """
    if threshold is None:
        return torch.zeros((), device=delta_waveform.device, dtype=delta_waveform.dtype)
    from stage2.psychoacoustics import masking_loss_torch

    return masking_loss_torch(delta_waveform, threshold, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# 式(9) L_per —— 感知质量
# ---------------------------------------------------------------------------
def perceptual_loss(original_mel: torch.Tensor, protected_mel: torch.Tensor,
                    delta_waveform: torch.Tensor,
                    threshold: torch.Tensor | None,
                    lambda_mask: float = config.STAGE2["lambda_mask"],
                    use_mel_loss: bool = True) -> torch.Tensor:
    """L_per = L_mel + λ_mask · L_mask （论文式 9）

    两个子项管两件不同的事：
      · L_mel ：整体频谱别乱动
      · L_mask：**人耳听不见的那些频点**上别有能量（哪怕 L_mel 看着还行）

    ★ `use_mel_loss=False` 是**消融**用的（论文 Table 2 的 "w/o L_mel"）：
      只留 L_mask。训练脚本用 `--no-mel-loss` 打开它。
    """
    total = mask_loss(delta_waveform, threshold) * lambda_mask
    if use_mel_loss:
        total = total + mel_loss(original_mel, protected_mel)
    return total


# ---------------------------------------------------------------------------
# 式(10)(11) LSGAN
# ---------------------------------------------------------------------------
def discriminator_gan_loss(real_scores: torch.Tensor,
                            fake_scores: torch.Tensor) -> torch.Tensor:
    """L_D = E[(D_adv(x) − 1)²] + E[D_adv(x_adv)²] （论文式 10）

    判别器的目标：真的给高分（→1），假的给低分（→0）。

    ★ `fake_scores` 那一项**不能**对生成器求梯度 ——
      更新判别器时必须把 x_adv 从图里 detach 掉，否则会同时去改生成器。
      `train.py` 里做这个 detach。
    """
    real_loss = torch.mean((real_scores - 1.0) ** 2)
    fake_loss = torch.mean(fake_scores ** 2)
    return real_loss + fake_loss


def generator_gan_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """L_GAN = E[(D_adv(x_adv) − 1)²] （论文式 11）

    生成器的目标：让判别器把保护后的音频也判成"真的"。
    """
    return torch.mean((fake_scores - 1.0) ** 2)


# ---------------------------------------------------------------------------
# 式(12) L_de —— 防御损失
# ---------------------------------------------------------------------------
def defense_loss(cloned_embedding: torch.Tensor,
                 original_embedding: torch.Tensor) -> torch.Tensor:
    """L_de = sim(E_spk(V(x_clone)), E_spk(V(x))) （论文式 12）—— **要最小化**

    参数
    ----
    cloned_embedding  : (B, 256) 攻击者克隆音的嵌入 E_spk(V(x_clone))
    original_embedding: (B, 256) 原始语音的嵌入 E_spk(V(x))

    ★ 方向：这是一个**余弦相似度**，取值 [-1, 1]。
      训练是"最小化"它 —— 让攻击者克隆出来的声音**远离**原说话人。
      论文报告训练后 Sim_clone = 0.13（对应 DSR 87.2%）。

    ★ `original_embedding` 必须是**常量**（detach 过的）。
      它是"参照系"，不参与梯度；否则梯度会试图去改原说话人的嵌入，
      整个目标就歪了。
    """
    return torch.mean(cosine_similarity(cloned_embedding, original_embedding.detach()))


# ---------------------------------------------------------------------------
# 式(13) L_G —— 生成器总目标
# ---------------------------------------------------------------------------
def generator_loss(gan_loss: torch.Tensor, defense_loss_value: torch.Tensor,
                   perceptual_loss_value: torch.Tensor,
                   lambda_de: float = config.STAGE2["lambda_de"],
                   lambda_per: float = config.STAGE2["lambda_per"],
                   lambda_gan: float = config.STAGE2["lambda_gan"]) -> torch.Tensor:
    """L_G = L_GAN + λ_de·L_de + λ_per·L_per （论文式 13）

    超参（论文 §3.1）：λ_de = 5.0, λ_per = 15.0。
    """
    return (lambda_gan * gan_loss
            + lambda_de * defense_loss_value
            + lambda_per * perceptual_loss_value)


def summarize(losses: dict[str, float]) -> str:
    """把一个损失字典排成一行，方便训练时打日志。"""
    order = ["total", "gan", "de", "per", "mel", "mask", "d"]
    parts = []
    for key in order:
        if key in losses:
            parts.append(f"{key}={losses[key]:.4f}")
    for key, value in losses.items():
        if key not in order:
            parts.append(f"{key}={value:.4f}")
    return "  ".join(parts)
