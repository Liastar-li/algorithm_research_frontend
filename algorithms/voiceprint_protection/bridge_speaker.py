"""桥接：说话人编码器 —— "体检仪"，把声音扫描成身份卡。

【它在做什么（比喻）】
你把一段录音递给体检仪，它扫描后给你一张 256 个数字组成的"身份卡"。
同一段内容换个人说，身份卡就变了；同一个人说不同的话，身份卡基本不变。

【两套权重，★ 不可混用】
| | AutoVC 自带 D_VECTOR | 论文脚注的 GE2E (RTC) |
|---|---|---|
| 权重文件 | `3000000-BL.ckpt` | `pretrained.pt` |
| **输入 mel 维度** | **80** | **40** |
| 结构 | 3 层单向 LSTM + Linear + L2 归一化 | 同构 |
| 输出 | 256 维，已 L2 归一化 | 256 维，已 L2 归一化 |

结构相同但第一层输入维度不同 -> **权重不能互相加载**。
L1 阶段用 AutoVC 自带的（与官方权重链完全配套）；
最终系统按论文改用 GE2E (RTC)。

【官方实现（third_party/AutoVC/model_bl.py）】
    class D_VECTOR(nn.Module):
        def __init__(self, num_layers=3, dim_input=80, dim_cell=768, dim_emb=256):
            self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_cell,
                                num_layers=num_layers, batch_batch_first=True)
            self.embedding = nn.Linear(dim_cell, dim_emb)
        def forward(self, x):                      # x: (B, T, dim_input)
            lstm_out, _ = self.lstm(x)
            embeds = self.embedding(lstm_out[:, -1, :])   # 取最后一个时间步
            return embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))   # L2 归一化

【官方输入约定（third_party/AutoVC/make_metadata.py）】
    melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()  # (1, 128, 80)
    emb = C(melsp)          # -> (1, 256)
即：**输入是 128 帧（2 秒 @16kHz/hop256）× 80 维的 mel**，取最后时间步的隐藏态。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import config
import paths

# 官方 make_metadata.py 里的 len_crop
LEN_CROP = 128


class SpeakerEncoderBridge:
    """说话人编码器的薄封装。"""

    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None,
                 encoder: str = "autovc_bl"):
        paths.ensure_import_paths()
        from model_bl import D_VECTOR

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.encoder_name = encoder
        settings = config.SPEAKER_ENCODERS[encoder]

        self.model = D_VECTOR(
            num_layers=settings["num_layers"],
            dim_input=settings["dim_input"],
            dim_cell=settings["dim_cell"],
            dim_emb=settings["dim_emb"],
        )
        self.model.eval().to(self.device)

        key = "speaker_bl" if encoder == "autovc_bl" else "ge2e_rtc"
        self.checkpoint_path = Path(checkpoint) if checkpoint else paths.weight_path(key)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"找不到说话人编码器权重：{self.checkpoint_path}\n"
                f"AutoVC 的 3000000-BL.ckpt 请从官方 README 的 IBM Box 链接下载，"
                f"放到：{paths.WEIGHTS_DIR}"
            )
        checkpoint_data = torch.load(self.checkpoint_path, map_location="cpu")
        if "model_b" in checkpoint_data:
            # 官方 make_metadata.py 的写法：键名前 7 个字符是 "module."（DataParallel 包装）
            state = {key[7:]: value for key, value in checkpoint_data["model_b"].items()}
        elif "model" in checkpoint_data:
            state = checkpoint_data["model"]
        else:
            state = checkpoint_data
        self.model.load_state_dict(state)
        self.model.to(self.device)

    # -- 对外 ------------------------------------------------------------
    def embed(self, mel: np.ndarray, crop: int = LEN_CROP) -> np.ndarray:
        """mel (T, 80) -> 说话人嵌入 (256,)，已 L2 归一化。

        官方做法是在时间轴上随机截取 `crop` 帧；
        为了可复现，这里固定取**居中**的 crop 帧（seed 固定的随机截取也可）。
        """
        mel = np.asarray(mel, dtype=np.float32)
        length = mel.shape[0]
        if length >= crop:
            start = (length - crop) // 2
            window = mel[start:start + crop]
        else:
            # 不足 crop 帧时补零（官方数据每条都有 128 帧以上）
            window = np.pad(mel, ((0, crop - length), (0, 0)), "constant")
        tensor = torch.from_numpy(window[None, :, :]).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)
        return embedding[0].cpu().numpy()

    def describe(self) -> str:
        return (
            f"说话人编码器桥接就绪\n"
            f"  权重    : {self.checkpoint_path}\n"
            f"  编码器  : {self.encoder_name}"
            f"（输入 {config.SPEAKER_ENCODERS[self.encoder_name]['dim_input']} 维 mel）\n"
            f"  设备    : {self.device}\n"
            f"  输出    : (256,) 已 L2 归一化"
        )
