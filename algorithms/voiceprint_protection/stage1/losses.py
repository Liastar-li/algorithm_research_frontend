"""Stage 1 的三个损失（论文 §2.2）。

    L_S1 = L_rec + λ_cd·L_cd + λ_dis·L_dis

【形象比喻】
想象一台"拆解机"：
  · L_rec —— 拆完再装回去，要跟原件**一模一样**（否则说明信息丢了）
  · L_cd  —— 装回去的机器，说的话要跟原件**一样**（内容不能变）
  · L_dis —— ★ 最关键：装回去的机器，**听不出是谁在说话**
              （谁的音色都不能残留，否则"解耦"就失败了）

【★ L_dis 的实现：论文那处维度矛盾怎么解的】
论文原文：
    "c̃ ∈ R^256 is the concatenated content representation along the time dimension,
     p_s(k|c̃) = softmax(W_s c̃), and W_s ∈ R^{K×256} is a single-layer speaker
      classifier trained on the speaker embedding h"

矛盾：论文同一节写 content code 是 `c ∈ R^{T'×32}`，可说话人分类器是 `R^{K×256}`
—— 若字面"拼接"，维度会随 T' 变化，两者对不上。

**我们的解法（与论文的每一个数字都对得上）**：
  · AutoVC 官方代码实测：content code 实际是 **(T//32, 64)**（dim_neck=32 × 双向 LSTM）
  · 论文说 c̃ 是 "**concatenated** ... **along the time dimension**"
    → 我们就**沿时间维拼接**：
        c̃ = concat(codes[i], codes[i+1], codes[i+2], codes[i+3]) ∈ R^{4×64} = R^{256} ✅
  · 于是 **W_s ∈ R^{K×256}**，与论文**完全一致**；
    而且它既能吃说话人嵌入 h (R^256)（论文原话 "trained on the speaker embedding h"），
    也能吃内容聚合向量 c̃ (R^256) —— **两个空间维度相同，分类器才有意义**。
  · 拼接窗口 = 4 个 content code = 4 × 0.5 秒 = **2 秒**上下文，语义上也说得通。

窗口长度见 `config.STAGE1["content_window"]`（默认 4）。

【为什么 W_s 要先在 h 上训好】
如果 W_s 随机初始化，熵最大化会退化成"让 W_s 输出均匀"，
而不是"让内容码不含说话人"。所以顺序是：
    第 1 步：用训练集的说话人嵌入 h 训练 W_s（一个普通的 K 类分类器）
    第 2 步：冻结 W_s，训内容编码器，最小化 L_S1
本模块的 `SpeakerClassifier` 提供第 1 步，`L_dis` 提供第 2 步。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ---------------------------------------------------------------------------
# 说话人分类器 W_s（第 1 步用：先在说话人嵌入 h 上训好）
# ---------------------------------------------------------------------------
class SpeakerClassifier(nn.Module):
    """单层说话人分类器 W_s：R^{K×256}，输入 256 维向量，输出 K 类。

    论文说它 "trained on the speaker embedding h" ——
    所以先用它做普通的说话人识别训练，学好后再冻结。

    ★ 输入维度固定 256：它既要吃说话人嵌入 h (R^256)，
      也要吃沿时间维拼接后的内容表示 c̃ (R^{4×64} = R^256)，
      **两个空间维度必须相同**，否则同一个分类器没法两边都用。

    用法：
        # 第 1 步：训练分类器
        loss = F.cross_entropy(classifier(embedding), speaker_id)
        # 第 2 步：冻结
        classifier.freeze()
    """

    def __init__(self, num_speakers: int = 80, embedding_dim: int = 256):
        super().__init__()
        self.num_speakers = num_speakers
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim, num_speakers)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """(B, 256) -> (B, K) 的 logits。"""
        return self.linear(embedding)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def unfreeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)


# ---------------------------------------------------------------------------
# L_rec：重建损失
# ---------------------------------------------------------------------------
def reconstruction_loss(mel: torch.Tensor, mel_hat: torch.Tensor,
                        mel_hat_postnet: torch.Tensor) -> torch.Tensor:
    """L_rec = ‖x − x̂‖² + ‖x − x̂_psnt‖²    （论文式 3）

    mel / mel_hat / mel_hat_postnet 形状均为 (B, T, 80)。
    """
    return F.mse_loss(mel_hat, mel) + F.mse_loss(mel_hat_postnet, mel)


# ---------------------------------------------------------------------------
# L_cd：内容一致性损失
# ---------------------------------------------------------------------------
def content_consistency_loss(codes: list[torch.Tensor],
                             codes_hat: list[torch.Tensor],
                             codes_postnet: list[torch.Tensor]) -> torch.Tensor:
    """L_cd = ‖E_c(x,h) − E_c(x̂,h)‖₁ + ‖E_c(x,h) − E_c(x̂_psnt,h)‖₁   （论文式 4）

    三个 code 列表必须**长度相同**（因为 x 与 x̂ 帧数相同），
    每个元素形状相同 (B, 64)。

    ★ 注意：这里比较的是**编码器的输出**，所以重建出来的 mel 要再过一次编码器。
      论文的式(4)正是如此：E_c(x̂, h) 而不是直接比较 mel。
    """
    if len(codes) != len(codes_hat) or len(codes) != len(codes_postnet):
        raise ValueError(
            f"content code 数量不一致：原 {len(codes)}，重建 {len(codes_hat)}，"
            f"精修 {len(codes_postnet)}。\n"
            "这说明 x 与 x̂ 的帧数不等 —— 检查 pad 与下采样因子 freq。"
        )

    total = codes[0].new_zeros(())
    for code, code_hat, code_postnet in zip(codes, codes_hat, codes_postnet):
        total = total + (code - code_hat).abs().mean() + (code - code_postnet).abs().mean()
    return total


# ---------------------------------------------------------------------------
# L_dis：解耦损失（熵最大化）
# ---------------------------------------------------------------------------
def content_windows(codes: list[torch.Tensor], window: int | None = None
                    ) -> list[torch.Tensor]:
    """把 content code **沿时间维拼接**成固定长度的窗口。

    codes : list of (B, D)，长度 N
    返回  : list of (B, window*D)，长度 N - window + 1

    ★ 这就是论文里 "c̃ ∈ R^256 is the **concatenated** content representation
      **along the time dimension**" 的字面实现：
      window=4 且 D=64 时，c̃ 正好是 R^256，与论文的 W_s ∈ R^{K×256} 对上。
    """
    window = window or config.STAGE1["content_window"]
    if len(codes) < window:
        # 片段太短（不足一个窗口）时**重复最后一个 code 凑够窗口**。
        # ★ 不能直接 torch.cat(codes)：那样维度会变成 len(codes)*D ≠ window*D，
        #   分类器 W_s（输入固定 256 维）会直接报形状错误。
        padded = list(codes) + [codes[-1]] * (window - len(codes))
        return [torch.cat(padded, dim=-1)]

    return [
        torch.cat(codes[index:index + window], dim=-1)
        for index in range(len(codes) - window + 1)
    ]


def disentanglement_loss(codes: list[torch.Tensor],
                         classifier: SpeakerClassifier,
                         window: int | None = None) -> torch.Tensor:
    """L_dis = −H(p_s) = Σ_k p_s(k|c̃) log p_s(k|c̃)   （论文式 5）

    最小化它 = **最大化熵** = 让说话人分类器在内容码上**完全猜不准**。

    ★★ 2026-11-20（成员 A）修了一个**符号错误**（很隐蔽，靠看曲线才发现）：
      原来写的是 `-(p * log p).sum()`，那是 **+H(p)**（熵本身），
      最小化它 = **最小化熵** = 让分类器**更确定** —— 与论文**正好相反**。
      现象：训练出来的 `dis` 停在 0.0016（≈ H=0，分类器完全确定），
            说明内容码里**说话人信息一点没去掉**，解耦是反的。

      论文式(5)写的是 `L_dis = −H(p_s) = Σ_k p_s(k|c̃) log p_s(k|c̃)`，
      而 `Σ p log p` **本身就是 −H**（熵的负数）⇒ **不要再多加一个负号**。

      数值对照（K = 80）：
        p 均匀   → Σ p log p = **−4.3820**（= −log K，最负 = 最均匀）
        p one-hot→ Σ p log p = **≈ 0**
      最小化它 ⇒ 越负越均匀 ⇒ 分类器越猜不准 ✅
    """
    windows = content_windows(codes, window)
    if not windows:
        raise ValueError("content code 列表为空。")

    losses = []
    for aggregated in windows:
        logits = classifier(aggregated)               # (B, K)
        log_probabilities = F.log_softmax(logits, dim=-1)
        probability = log_probabilities.exp()
        # ★ 就是 Σ p log p（= −H），**不要再加负号**
        losses.append((probability * log_probabilities).sum(dim=-1).mean())
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# 总损失
# ---------------------------------------------------------------------------
def stage1_loss(mel: torch.Tensor,
                mel_hat: torch.Tensor,
                mel_hat_postnet: torch.Tensor,
                codes: list[torch.Tensor],
                codes_hat: list[torch.Tensor],
                codes_postnet: list[torch.Tensor],
                classifier: SpeakerClassifier,
                lambda_cd: float | None = None,
                lambda_dis: float | None = None,
                *,
                return_metrics: bool = True):
    """L_S1 = L_rec + λ_cd·L_cd + λ_dis·L_dis    （论文式 6）

    返回 (total, dict of 各分项)，便于记录训练曲线。
    """
    lambda_cd = config.STAGE1["lambda_cd"] if lambda_cd is None else lambda_cd
    lambda_dis = config.STAGE1["lambda_dis"] if lambda_dis is None else lambda_dis

    loss_rec = reconstruction_loss(mel, mel_hat, mel_hat_postnet)
    loss_cd = content_consistency_loss(codes, codes_hat, codes_postnet)
    loss_dis = disentanglement_loss(codes, classifier)

    total = loss_rec + lambda_cd * loss_cd + lambda_dis * loss_dis
    if not return_metrics:
        return total, {}
    return total, {
        "loss_rec": float(loss_rec.detach()),
        "loss_cd": float(loss_cd.detach()),
        "loss_dis": float(loss_dis.detach()),
        "total": float(total.detach()),
    }
