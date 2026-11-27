"""Stage 2 的公共件：扰动、余弦相似度、随机种子。

【为什么要单独一个文件】
`perturb_embedding` 这个操作在**三处**出现，含义必须完全一致：
  1. `infer.py` 的 L1 随机扰动（推理时用）
  2. `stage2/train.py` 的训练循环
  3. `stage2/losses.py` 里评估扰动强度时
数值一旦不一致，"α=0.1 到底扰动了多少" 就说不清了。
所以这里放**唯一一份实现**，其他地方一律 import 它。

⚠️ 提醒：`infer.py` 里也有一份同名函数（那是 A 写的，属于推理链）。
**两边的公式必须一样**（见下面 `perturb_embedding` 的注释），
改动时两边一起改 —— `tests/` 里有对照断言。
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """把三个随机源都固定住（论文 §3.1 要求可复现）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def perturb_embedding(embedding: torch.Tensor, delta: torch.Tensor,
                      alpha: float, renormalize: bool = False) -> torch.Tensor:
    """论文式 h_adv = h + α·δ，默认不重新归一化。"""
    advanced = embedding + alpha * delta
    if renormalize:
        norm = torch.linalg.vector_norm(advanced, dim=-1, keepdim=True)
        advanced = advanced / norm.clamp_min(1e-12)
    return advanced


def cosine_similarity(a: torch.Tensor, b: torch.Tensor,
                      dim: int = -1) -> torch.Tensor:
    """余弦相似度，支持任意 batch 形状。

    与 `audio_utils.cosine_similarity`（处理 torch tensor 的那个）语义一致，
    但这里不假定维度、也不做 L2 归一化的前置假设。
    """
    a = a.float()
    b = b.float()
    numerator = (a * b).sum(dim=dim)
    denominator = a.norm(dim=dim) * b.norm(dim=dim)
    return numerator / denominator.clamp_min(1e-12)


def pad_batch(waveforms: Sequence[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """把长度不一的波形拼成一个 batch。

    返回
    ----
    padded : (B, L) 右侧补零的波形
    lengths: (B,) 每条的真实长度（补零段不参与损失/统计）

    为什么不用 DataLoader 的默认 collate：默认会报错（长度不同），
    而这里"补零 + 记长度"是音频模型的标准做法，
    后面算 RMS、算 PSD 时都能按真实长度裁掉补零段。
    """
    if not waveforms:
        raise ValueError("空 batch")
    lengths = np.array([len(w) for w in waveforms], dtype=np.int64)
    longest = int(lengths.max())
    padded = np.zeros((len(waveforms), longest), dtype=np.float32)
    for index, wave in enumerate(waveforms):
        padded[index, : len(wave)] = wave
    return torch.from_numpy(padded), torch.from_numpy(lengths)


def trim_to_length(waveform: torch.Tensor, lengths: torch.Tensor) -> list[torch.Tensor]:
    """按真实长度把补零的 batch 裁回每条一条（算逐条指标时用）。"""
    trimmed = []
    for index in range(waveform.shape[0]):
        trimmed.append(waveform[index, : int(lengths[index])])
    return trimmed


def l2_normalize(vector: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """L2 归一化到单位长度（上游 GE2E 出口也是这么做的）。"""
    return vector / vector.norm(dim=dim, keepdim=True).clamp_min(1e-12)
