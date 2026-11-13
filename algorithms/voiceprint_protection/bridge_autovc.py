"""桥接：AutoVC 官方模型 —— Stage 1 的"配音员"。

【它在做什么（比喻）】
配音员拿到两样东西：
  · 一份"乐谱"（原始语音的 mel 频谱，代表说了什么）
  · 一张"身份卡"（说话人嵌入，代表是谁在说）
它把两张卡拼在一起，重新"念"出一段话。
**关键点：它可以用两张不同的身份卡**——
  · 编码时用**原始身份卡**（保证"说了什么"不变）
  · 解码时用**被污染的身份卡**（这就是保护生效的地方）

【与论文的对应】
论文式(2)：x̂ = D(c, h_trg)
论文式(13)：x_adv = D(c, h_adv)，其中 h_adv = h + α·δ

AutoVC 的 `Generator(x, c_org, c_trg)` 恰好支持这个用法：
  c_org = h（原始）  ->  提取内容码 c
  c_trg = h_adv（污染）->  解码出 x_adv
所以我们**不需要改 AutoVC 的模型代码**，直接用它自带的接口就能实现论文的攻击路径。

【官方接口（third_party/AutoVC/model_vc.py）】
    G = Generator(dim_neck=32, dim_emb=256, dim_pre=512, freq=32)
    _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
      uttr_org : (B, T, 80)   原始 mel
      emb_org  : (B, 256)     原始说话人嵌入
      emb_trg  : (B, 256)     目标（=被污染的）说话人嵌入
      返回 x_identic_psnt : (B, 1, T, 80)   保护后 mel（已含 Postnet 残差）
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

import config
import paths


def _pad_to_multiple(array: np.ndarray, base: int) -> tuple[np.ndarray, int]:
    """把时间维补齐到 base 的整数倍。

    官方 `conversion.ipynb` 里的 `pad_seq(x, base=32)`。
    必须补齐的原因：AutoVC 的 Encoder 按 freq=32 抽样取 content code，
    时间轴不是 32 的整数倍会取不满、且解码时长度对不上。
    """
    length = array.shape[0]
    target = int(base * math.ceil(float(length) / base))
    padding = target - length
    if padding > 0:
        array = np.pad(array, ((0, padding), (0, 0)), "constant")
    return array, padding


class AutoVCBridge:
    """AutoVC 官方模型的薄封装。

    用法：
        bridge = AutoVCBridge()
        x_adv = bridge.protect(mel, embedding, embedding_adv)
    """

    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None):
        paths.ensure_import_paths()
        from model_vc import Generator

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        settings = config.AUTOVC_MODEL
        self.freq = settings["freq"]
        self.pad_base = settings["pad_base"]

        self.model = Generator(
            dim_neck=settings["dim_neck"],
            dim_emb=settings["dim_emb"],
            dim_pre=settings["dim_pre"],
            freq=settings["freq"],
        )
        self.model.eval().to(self.device)

        self.checkpoint_path = Path(checkpoint) if checkpoint else paths.weight_path("autovc")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"找不到 AutoVC 权重：{self.checkpoint_path}\n"
                f"请从 AutoVC 官方 README 的 IBM Box 链接下载 autovc.ckpt，放到：\n"
                f"  {paths.WEIGHTS_DIR}\n"
                f"下载地址见 third_party/README.md"
            )
        # torch 2.6 起 weights_only 默认 True，会导致加载含非张量对象的 ckpt 失败。
        # 我们锁定 torch 2.4.1，但这里显式写明，避免以后升级踩坑。
        checkpoint_data = torch.load(self.checkpoint_path, map_location="cpu")
        state = checkpoint_data["model"] if "model" in checkpoint_data else checkpoint_data
        self.model.load_state_dict(state)
        self.model.to(self.device)

    # -- 内部 ------------------------------------------------------------
    def _forward(self, mel: np.ndarray, source: np.ndarray, target: np.ndarray):
        padded, pad = _pad_to_multiple(mel, self.pad_base)
        utterance = torch.from_numpy(padded[None, :, :]).to(self.device)
        emb_source = torch.from_numpy(source[None, :]).to(self.device)
        emb_target = torch.from_numpy(target[None, :]).to(self.device)
        with torch.no_grad():
            _, mel_postnet, codes = self.model(utterance, emb_source, emb_target)
        output = mel_postnet[0, 0].cpu().numpy()
        if pad:
            output = output[:-pad, :]
        return output, codes

    # -- 对外 ------------------------------------------------------------
    def content_code(self, mel: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        """只做编码：mel + 原始嵌入 -> 内容码 c，形状 (T', 64)。

        论文里 c ∈ R^(T'×32) 指的是 LSTM 每个方向的维度（dim_neck=32），
        AutoVC 前向 32 + 后向 32 拼接后实际是 64 维。
        """
        padded, _ = _pad_to_multiple(mel, self.pad_base)
        utterance = torch.from_numpy(padded[None, :, :]).to(self.device)
        emb = torch.from_numpy(embedding[None, :]).to(self.device)
        with torch.no_grad():
            codes = self.model.encoder(utterance, emb)
        return codes

    def protect(self, mel: np.ndarray, embedding: np.ndarray,
                embedding_adv: np.ndarray) -> np.ndarray:
        """论文的核心一步：x_adv = D(c, h_adv)，返回保护后的 mel (T, 80)。

        mel          : (T, 80) 原始语音的 mel
        embedding    : (256,)  原始说话人嵌入 h
        embedding_adv: (256,)  被扰动的说话人嵌入 h + α·δ
        """
        output, _ = self._forward(mel, embedding, embedding_adv)
        return output

    def reconstruct(self, mel: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        """只做重建（不加扰动）：x̂_psnt = D(c, h)。

        用途：验证"AutoVC 权重 + 我们的 mel 是否配套"——
        如果这里出来的音频是噪音，说明 mel 对照没做对，
        不用再怀疑保护逻辑。
        """
        output, _ = self._forward(mel, embedding, embedding)
        return output

    def describe(self) -> str:
        return (
            f"AutoVC 桥接就绪\n"
            f"  权重      : {self.checkpoint_path}\n"
            f"  设备      : {self.device}\n"
            f"  模型参数  : dim_neck={config.AUTOVC_MODEL['dim_neck']} "
            f"dim_emb={config.AUTOVC_MODEL['dim_emb']} "
            f"dim_pre={config.AUTOVC_MODEL['dim_pre']} freq={self.freq}\n"
            f"  输出      : (T, 80) mel"
        )
