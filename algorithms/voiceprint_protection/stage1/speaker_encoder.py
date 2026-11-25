"""Stage 1 用的说话人编码器 —— **可微**的 GE2E。

【为什么要单独写一个，而不是直接用 bridge_speaker】
`bridge_speaker.py`（成员 B）做的是**推理**：加载权重、算嵌入、返回 numpy。
Stage 1 需要的是**训练**：
  · 论文 §3.1："Speaker embeddings are extracted using a GE2E model initialized from
    pretrained weights and **fine-tuned end-to-end during Stage 1 training**"
  · 也就是：GE2E 的**参数**要被 L_rec / L_cd 的梯度更新。

所以这里把上游的 `SpeakerEncoder` 拿来，让它保持 nn.Module 形态，
并把上游 `encoder.pt` 的权重装进去。

【上游前向（third_party/Real-Time-Voice-Cloning/encoder/model.py）】
    def forward(self, utterances, hidden_init=None):
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)
        return embeds

  · 输入 : (B, n_frames, 40)   ← 上游自己的 40 维线性 mel
  · 输出 : (B, 256)             ← 已 L2 归一化

【★ 两套 mel 的分工（Stage 1 每条音频要算两次）】
  · 80 维 AutoVC mel  →  Ec / D      （`audio_utils.autovc_mel()`）
  · 40 维 GE2E mel    →  GE2E         （`audio_utils.ge2e_mel()`）
  两者维度不同、不可互换。见 docs/interface-contract.md §1。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import paths


class GE2EMel(nn.Module):
    """Differentiable equivalent of audio_utils.ge2e_mel for waveform batches."""

    def __init__(self):
        super().__init__()
        from librosa.filters import mel
        from audio_utils import GE2E_MEL

        self.n_fft = GE2E_MEL["n_fft"]
        self.hop_length = GE2E_MEL["hop_length"]
        self.power = GE2E_MEL["power"]
        basis = mel(sr=GE2E_MEL["sample_rate"], n_fft=self.n_fft,
                    n_mels=GE2E_MEL["n_mels"], fmin=GE2E_MEL["fmin"], fmax=GE2E_MEL["fmax"])
        self.register_buffer("basis", torch.from_numpy(basis), persistent=False)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(waveform, self.n_fft, self.hop_length, window=self.window,
                              pad_mode="constant", return_complex=True)
        return (self.basis @ spectrum.abs().pow(self.power)).transpose(1, 2)


class TrainableSpeakerEncoder(nn.Module):
    """可微的 GE2E 说话人编码器（Stage 1 训练用）。

    用法：
        encoder = TrainableSpeakerEncoder(fine_tune=True)
        h = encoder(ge2e_mel_batch)      # (B, T, 40) -> (B, 256)
    """

    def __init__(self, rtc_dir: str | Path | None = None, device: str | None = None,
                 weights_path: str | Path | None = None, fine_tune: bool = True):
        super().__init__()
        from bridge_speaker import find_rtc_dir, find_weights

        self.rtc_dir = find_rtc_dir(str(rtc_dir) if rtc_dir else None)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.weights_path = find_weights(self.rtc_dir, str(weights_path) if weights_path else None)

        if str(self.rtc_dir) not in sys.path:
            sys.path.insert(0, str(self.rtc_dir))

        from encoder.model import SpeakerEncoder

        self.model = SpeakerEncoder(self.device, self.device)
        self._load_weights()
        self.model.to(self.device)

        # fine_tune=False 时冻结全部参数（用于消融：嵌入固定时的 Stage 1）
        self.fine_tune = fine_tune
        if not fine_tune:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    def _load_weights(self) -> None:
        """加载上游 `encoder.pt`。

        上游的 checkpoint 结构是 {'model_state': ..., 'step': ...}。
        """
        try:
            loaded = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        except TypeError:
            loaded = torch.load(self.weights_path, map_location="cpu")

        state = None
        for key in ("model_state", "model", "state_dict", "model_state_dict"):
            if isinstance(loaded, dict) and key in loaded and isinstance(loaded[key], dict):
                state = loaded[key]
                break
        if state is None:
            state = loaded

        # 去掉可能存在的 "module." 前缀（DataParallel 包装）
        cleaned = {}
        for key, value in state.items():
            cleaned[key[7:] if key.startswith("module.") else key] = value

        self.model.load_state_dict(cleaned, strict=False)

    def load_checkpoint_state(self, state: dict) -> None:
        """Load Stage 1 weights across devices with strict forward-key checks."""
        # Upstream registers these loss-only tensors differently on CPU and CUDA.
        loss_keys = {"similarity_weight", "similarity_bias"}
        target = self.model.state_dict()
        merged = {key: value for key, value in state.items()
                  if key in target or key not in loss_keys}
        for key in loss_keys & target.keys():
            merged.setdefault(key, target[key])
        self.model.load_state_dict(merged)

    def forward(self, ge2e_mel: torch.Tensor) -> torch.Tensor:
        """
        ge2e_mel : (B, T, 40)   上游约定的 40 维线性 mel
        返回      : (B, 256)     已 L2 归一化的说话人嵌入
        """
        mel = ge2e_mel.to(self.device)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        return self.model(mel)

    def embed_numpy(self, wav: np.ndarray) -> np.ndarray:
        """单条音频 -> (256,) numpy（调试用）。"""
        import audio_utils

        mel = audio_utils.ge2e_mel(wav)
        with torch.no_grad():
            embedding = self(torch.from_numpy(mel).unsqueeze(0))
        return embedding[0].cpu().numpy()

    def describe(self) -> str:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return (
            "可微 GE2E 说话人编码器\n"
            f"  权重      : {self.weights_path}\n"
            f"  设备      : {self.device}\n"
            f"  参数      : {trainable:,} / {total:,} 可训练"
            f"（fine_tune={self.fine_tune}）\n"
            "  输入/输出 : (B, T, 40) 40维线性 mel  ->  (B, 256) 已归一化嵌入"
        )
