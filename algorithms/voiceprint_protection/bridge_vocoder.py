"""桥接：声码器 —— "扩音器"，把乐谱(mel频谱)还原成人耳能听的波形。

【它在做什么（比喻）】
AutoVC 的解码器吐出的是一张"乐谱"（80 维的 mel 频谱，每行代表一帧）。
乐谱本身不是声音。声码器是乐队：它照着乐谱把每个音符演奏出来，
最后合成出波形（真正的音频数据）。

【为什么必须是 AutoVC 配套的 WaveNet，而不是随便一个 HiFi-GAN】
两者的 mel **约定不同**，混用会输出噪音且不报错：
  · AutoVC 侧 mel：dB 归一化到 [0,1]，采样率 16 kHz
  · HiFi-GAN 侧 mel：log10 幅度 + (x+4.5)/5，采样率 22.05 kHz
AutoVC 官方配套的 WaveNet 声码器就是用 [0,1] dB 格式训练的，**天然对接**。

【官方接口（third_party/AutoVC/synthesis.py + vocoder.ipynb）】
    from synthesis import build_model, wavegen
    model = build_model().to(device)
    checkpoint = torch.load("checkpoint_step001000000_ema.pth")
    model.load_state_dict(checkpoint["state_dict"])
    waveform = wavegen(model, c=mel)        # mel: (T, 80)
    sf.write(name+'.wav', waveform, samplerate=16000)

【★ 版本要求】
`wavenet_vocoder` 必须用 **v0.1.0**（含 `builder.py`）。
2019-08 之后的新版删掉了 builder.py 且模型结构有变，会导致权重加载失败。
第三方仓库已固定在 tag v0.1.0（commit 6c208ee）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import paths


class VocoderBridge:
    """WaveNet 声码器的薄封装。"""

    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None):
        paths.ensure_import_paths()
        from synthesis import build_model

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = build_model().to(self.device)

        self.checkpoint_path = Path(checkpoint) if checkpoint else paths.weight_path("vocoder")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"找不到声码器权重：{self.checkpoint_path}\n"
                f"请从 AutoVC 官方 README 的 IBM Box 链接下载 "
                f"checkpoint_step001000000_ema.pth，放到：{paths.WEIGHTS_DIR}"
            )
        checkpoint_data = torch.load(self.checkpoint_path, map_location="cpu")
        state = checkpoint_data["state_dict"] if "state_dict" in checkpoint_data else checkpoint_data
        self.model.load_state_dict(state)
        self.model.to(self.device)

    # -- 对外 ------------------------------------------------------------
    def synthesize(self, mel: np.ndarray) -> np.ndarray:
        """mel (T, 80) -> 波形 (T*256,)。

        注意：官方 `wavegen` 用的是**模块级** device 变量，
        我们在调用前把它切到本桥接器的 device。
        """
        from synthesis import wavegen
        import synthesis

        mel = np.ascontiguousarray(np.asarray(mel, dtype=np.float32))
        previous_device = synthesis.device
        synthesis.device = self.device
        try:
            with torch.no_grad():
                waveform = wavegen(self.model, c=mel)
        finally:
            synthesis.device = previous_device
        return np.asarray(waveform, dtype=np.float32)

    def describe(self) -> str:
        return (
            f"声码器桥接就绪\n"
            f"  权重    : {self.checkpoint_path}\n"
            f"  设备    : {self.device}\n"
            f"  输入    : (T, 80) AutoVC 格式 mel（dB 归一化到 [0,1]）\n"
            f"  输出    : 16 kHz 波形"
        )
