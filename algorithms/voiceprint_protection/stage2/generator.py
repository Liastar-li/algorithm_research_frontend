"""Stage 2 的生成器 G：说话人嵌入 h → 扰动 δ。

【论文里的位置】
论文 §2.3：*"the generator produces perturbation δ in the speaker embedding space"*，
    h_adv = h + α·δ
论文没有给出 G 的具体结构（只给了超参 α=0.10），所以**结构是我们的实现选择**，
下面会写清楚选了它以及为什么。

【比喻】
身份卡 h 是一张 256 个数字的卡片。G 是个"改卡员"：
它看一张卡，决定往哪个方向拧一下，让机器认错，但人耳听不出来。
拧的幅度由 α 统一控制（G 只决定**方向**）。

【为什么 G 的输出固定成单位长度】
    δ = G(h) / ‖G(h)‖
这样 α 就是**唯一**控制扰动强度的旋钮：
    ‖h_adv − h‖ ≈ α（h、h_adv 都在单位球面上，小 α 时近似成立）
论文式(11)(12) 里 α 是一个写死的 0.10，平台上也暴露给用户调（见方法 JSON），
如果 G 自己还能改长度，那"α=0.1"这个说法就没有意义了。
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config


class PerturbationGenerator(nn.Module):
    """G：把说话人嵌入映成单位长度的扰动方向。

    结构（我们的实现选择，论文未指定）
    ------------------------------------
        h (256)
          -> Linear(256, hidden) -> LayerNorm -> GELU
          -> [Linear(hidden, hidden) -> LayerNorm -> GELU] × (layers-1)
          -> Linear(hidden, 256)
          -> L2 归一化  =>  δ (256)

    为什么这么选
    ------------
    * **MLP 而不是卷积**：输入只有 256 维，没有空间结构，卷积没有意义。
    * **LayerNorm**：h 的模长恒为 1，但各维度的分布不均匀；
      没有归一化层时，深 MLP 很容易在前几层就把梯度带崩。
    * **最后 L2 归一化**：见文件头的解释（α 是唯一的强度旋钮）。
    * **残差直连（skip）**：把 h 直接加到输出上，
      让网络学的是"在 h 的基础上往哪拧"，比从零学一个绝对方向更容易训。
      权重初始化为 0，所以**训练刚开始时 δ 就是 h 的一个微小扰动**，
      不会一上来就把音频毁掉。

    参数
    ----
    dim        : 说话人嵌入维度（config.SPEAKER_DIM = 256）
    hidden     : 隐层宽度
    layers     : 隐层个数（≥1）
    use_skip   : 是否加 h 的残差直连
    """

    def __init__(self, dim: int = config.SPEAKER_DIM, hidden: int = 512,
                 layers: int = 2, use_skip: bool = True):
        super().__init__()
        if layers < 1:
            raise ValueError("layers 至少为 1")
        self.dim = dim
        self.use_skip = use_skip

        blocks: list[nn.Module] = []
        in_features = dim
        for _ in range(layers):
            blocks.extend([
                nn.Linear(in_features, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            ])
            in_features = hidden
        self.trunk = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, dim)

        # 残差直连的权重初始化为 0：训练刚开始时输出 ≈ 0（除以范数后方向由随机小扰动决定），
        # 这样第 0 步的 h_adv ≈ h，不会突然把音频毁掉。
        if use_skip:
            self.skip = nn.Linear(dim, dim, bias=False)
            nn.init.zeros_(self.skip.weight)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """h (..., 256) -> δ (..., 256)，**单位长度**。

        注意：输出没有再归一化到和 h 同一个"尺度"上 ——
        δ 只是方向，强度由 α 在外面乘。
        """
        raw = self.head(self.trunk(embedding))
        if self.use_skip:
            raw = raw + self.skip(embedding)
        norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
        # 范数为 0（理论上不会，因为 skip 权重为 0 但 head 有偏置）时给一个小的默认值，
        # 避免除零产生 nan —— nan 一次，整个训练就白跑了。
        return raw / norm.clamp_min(1e-6)

    def describe(self) -> str:
        parameters = sum(p.numel() for p in self.parameters())
        return (
            f"生成器 G：{self.dim} → {self.dim}（单位长度扰动）\n"
            f"  可训练参数 : {parameters:,}\n"
            f"  结构       : MLP + LayerNorm + GELU + 残差直连"
        )


def config_from_state(state: dict) -> dict:
    """从生成器的权重里**反推**它的结构参数（`PerturbationGenerator` 的构造参数）。

    为什么需要：checkpoint 里存了 `generator_config` 就直接用它；
    万一没有（旧 checkpoint），也得能从权重形状把结构还原出来 ——
    **结构差一点，`load_state_dict` 就报形状不符**；结构对但顺序错，
    更糟的是能加载上、结果却全错。

    形状（别搞反）：
        trunk.<i>.weight : (hidden, dim)   ← Linear
        trunk.<i>.weight : (hidden,)       ← LayerNorm（1 维，靠这个区分）
        head.weight      : (dim, hidden)
        skip.weight      : (dim, dim)

    唯一真源：`infer.py` 与 `evaluate.py` 都用这个函数，
    免得两处各写一份、慢慢跑偏。
    """
    settings = {"dim": config.SPEAKER_DIM, "hidden": 512, "layers": 2}
    if not isinstance(state, dict):
        return settings

    linear_indices = set()
    for key, value in state.items():
        if not (key.startswith("trunk.") and key.endswith(".weight")):
            continue
        if value.dim() != 2:
            continue                      # LayerNorm 的权重是 1 维，跳过
        index = int(key.split(".")[1])
        linear_indices.add(index)
        if index == 0:
            settings["hidden"] = int(value.shape[0])
            settings["dim"] = int(value.shape[1])
    if linear_indices:
        settings["layers"] = len(linear_indices)

    for key, value in state.items():
        if key.endswith("head.weight") and value.dim() == 2:
            settings["dim"] = int(value.shape[0])
            settings["hidden"] = int(value.shape[1])

    settings["use_skip"] = any(key.startswith("skip.") for key in state)
    return settings


class RandomPerturbationGenerator(nn.Module):
    """随机方向占位（L1 用）。

    **Stage 2 训练之前**，`infer.py` 用的是随机高斯噪声当扰动。
    这个类把同样的行为包成 `nn.Module`，好处是：
      * 训练脚本可以先跑通整条流程（用 `--perturb random`），再换成训练好的 G；
      * 评测脚本能拿它当**高斯噪声基线**（论文 Table 1 的 "Gaussian Noise" 那一行）。
    """

    def __init__(self, dim: int = config.SPEAKER_DIM, seed: int = config.SEED):
        super().__init__()
        self.dim = dim
        self.seed = seed
        self.register_buffer("_generator", torch.Generator().manual_seed(seed))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        shape = embedding.shape[:-1] + (self.dim,)
        delta = torch.randn(shape, generator=self._generator,
                            device=embedding.device, dtype=embedding.dtype)
        return delta / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
