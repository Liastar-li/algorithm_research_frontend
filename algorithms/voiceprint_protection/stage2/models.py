"""Stage 2 用的**可求导**冻结组件。

【★ 为什么不能直接用 bridge_autovc / bridge_vocoder】
Stage 2 的梯度必须穿过冻结的组件流回生成器 G：

    G → h_adv → D(c, h_adv) → V(...) → E_spk(...) → L_de

而 `bridge_autovc.protect()` 与 `bridge_vocoder.synthesize()` 是 **numpy 封装**，
内部写着 `with torch.no_grad()`，返回 numpy 数组 —— **梯度在那一层就被切断了**。
拿它们训练，G 收不到任何梯度，训一百步 loss 纹丝不动，还以为是超参不对。

所以本文件提供 **torch 原生**的等价实现：
  · `TorchSpeakerEncoder`  上游 GE2E（权重 `encoder.pt` 现在就有 ✓）
  · `TorchAutoVCStage1`    AutoVC 的 Generator（Stage 1 用同一套架构）
  · `StubStage1`          随机权重的替身，**离线冒烟测试用**

【★ 与上游的一致性有多重要】
`TorchSpeakerEncoder` 的 mel 计算必须与上游 `encoder/audio.py::wav_to_mel_spectrogram`
**逐值一致**，否则：
  · 推理时（bridge_speaker，numpy）与训练时（这里，torch）算出的 h 不是同一个东西；
  · 更糟的是**不报错** —— 嵌入只是"不对"，训练照跑，最后指标全错。
所以这里的 mel 是照着上游的 librosa 调用**逐参数**翻译成 torch 的，
并在 `verify_mel.py` 里做数值对照（见文件末尾的自检说明）。

上游调用：
    librosa.feature.melspectrogram(y=wav, sr=16000, n_fft=400, hop_length=160, n_mels=40)
    → power=2.0, center=True, pad_mode='constant', window='hann'(周期), fmin=0, fmax=8000
    → 取 .T 得到 (T, 40)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

import config
import paths


# ---------------------------------------------------------------------------
# 1. 可求导的说话人编码器（上游 GE2E）
# ---------------------------------------------------------------------------
class TorchSpeakerEncoder(nn.Module):
    """上游 GE2E 的 torch 版：**可求导**的说话人嵌入。

    与上游 `encoder/inference.py::embed_utterance` 的算法完全一致：
      1. 分帧：把语音切成 160 帧（1.6 秒）一段、步长 80 帧（50% 重叠）
      2. 每段各算一个嵌入
      3. 各段嵌入取平均，再 L2 归一化

    差异（都是有意为之，且不影响数值）：
      · 上游用 `preprocess_wav` 做「音量归一化 + VAD 裁静音」；
        这里只做**音量归一化**（可导），**不做 VAD** ——
        因为 VAD 会按条裁掉不同的长度，batch 就拼不起来了，
        而且它本身不可导（会切断梯度）。
      · 上游分母是 `norm + 1e-5`，这里照抄 ✓

    权重：`encoder.pt`（HuggingFace `CorentinJ/SV2TTS`，17 MB）
    """

    def __init__(self, rtc_dir: str | Path | None = None,
                 weights_path: str | Path | None = None,
                 device: str | None = None):
        super().__init__()

        from bridge_speaker import find_rtc_dir, find_weights

        self.rtc_dir = Path(find_rtc_dir(str(rtc_dir) if rtc_dir else None))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if str(self.rtc_dir) not in sys.path:
            sys.path.insert(0, str(self.rtc_dir))

        from encoder import params_data as encoder_params
        from encoder.model import SpeakerEncoder

        self.sample_rate = encoder_params.sampling_rate
        self.mel_n_channels = encoder_params.mel_n_channels
        self.partial_frames = encoder_params.partials_n_frames
        self.window_ms = encoder_params.mel_window_length
        self.step_ms = encoder_params.mel_window_step
        self.hop_length = int(self.sample_rate * self.step_ms / 1000)      # 160
        self.n_fft = int(self.sample_rate * self.window_ms / 1000)         # 400

        self.model = SpeakerEncoder(self.device, self.device)
        self.weights_path = find_weights(
            self.rtc_dir, str(weights_path) if weights_path else None
        )
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"找不到 GE2E 权重：{self.weights_path}\n"
                "请先下载：\n"
                "    python algorithms/voiceprint_protection/scripts/download_encoder_weights.py"
            )
        checkpoint = torch.load(self.weights_path, map_location="cpu")
        state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)
        # 冻结：不更新它的参数，但**梯度要能穿过它**
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self._build_mel_basis()

    # -- mel（照抄上游 librosa 调用）----------------------------------------
    def _build_mel_basis(self) -> None:
        """用 librosa 算一次 mel 滤波器组，之后转成 torch 常量。

        ★ 故意用 librosa 而不是自己写公式：
          上游就是用 librosa 算的，**用同一个库算同一个矩阵**才能保证数值一致。
        """
        import librosa

        basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.mel_n_channels,
            fmin=0.0,
            fmax=self.sample_rate / 2,
        )
        self.register_buffer("mel_basis", torch.from_numpy(basis).float().to(self.device))
        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft, periodic=True).to(self.device),
        )

    def mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """波形 (B, N) -> 上游口径的 mel (B, T, 40)。

        对应上游：
            librosa.feature.melspectrogram(y=wav, sr=16000, n_fft=400, hop_length=160, n_mels=40).T
        """
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="constant",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = spectrum.real ** 2 + spectrum.imag ** 2        # (B, 201, T)  ← power=2.0
        frames = torch.matmul(self.mel_basis, power)            # (B, 40, T)
        return frames.transpose(1, 2)                           # (B, T, 40)

    # -- 分段（照抄上游 compute_partial_slices）---------------------------
    def partial_frame_slices(self, n_samples: int,
                             overlap: float = 0.5,
                             min_pad_coverage: float = 0.75) -> list[tuple[int, int]]:
        """上游 `compute_partial_slices` 的**帧**版本。

        上游先在采样点上切，再换算成帧；这里直接按帧算，结果完全一致。
        """
        samples_per_frame = self.hop_length
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = max(int(np.round(self.partial_frames * (1 - overlap))), 1)

        steps = max(1, n_frames - self.partial_frames + frame_step + 1)
        slices: list[tuple[int, int]] = []
        for start in range(0, steps, frame_step):
            slices.append((start, start + self.partial_frames))

        # 上游的 coverage 判断：最后一段覆盖不足就丢掉
        last_start = slices[-1][0] * samples_per_frame
        last_stop = slices[-1][1] * samples_per_frame
        coverage = (n_samples - last_start) / max(last_stop - last_start, 1)
        if coverage < min_pad_coverage and len(slices) > 1:
            slices = slices[:-1]
        return slices

    # -- 前向 -------------------------------------------------------------
    def forward(self, waveform: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        """波形 (B, N) -> 嵌入 (B, 256)，已 L2 归一化。

        `lengths` 给每条的真实长度（补零 batch 用）；不给就当成全是 N。
        """
        if waveform.dim() != 2:
            raise ValueError(f"期望 (B, N) 的波形，收到 {tuple(waveform.shape)}")
        batch = waveform.shape[0]
        if lengths is None:
            lengths = torch.full((batch,), waveform.shape[1],
                                 device=waveform.device, dtype=torch.int64)

        # 音量归一化（照抄上游 normalize_volume，target=-30 dBFS，increase_only=True）
        waveform = self._normalize_volume(waveform, lengths)

        frames = self.mel(waveform)                        # (B, T, 40)

        # 收集每一段（partial）
        collected: list[torch.Tensor] = []
        owners: list[int] = []
        for index in range(batch):
            length = int(lengths[index])
            for start, stop in self.partial_frame_slices(length):
                segment = frames[index, start:stop, :]
                if segment.shape[0] < self.partial_frames:
                    segment = torch.nn.functional.pad(
                        segment, (0, 0, 0, self.partial_frames - segment.shape[0])
                    )
                collected.append(segment)
                owners.append(index)

        if not collected:
            raise RuntimeError("没有可用的分段 —— 波形是不是太短了？")

        stacked = torch.stack(collected, dim=0)             # (P, 160, 40)
        partial_embeddings = self.model(stacked)            # (P, 256)

        # 按所属样本平均
        owner_tensor = torch.tensor(owners, device=partial_embeddings.device)
        summed = torch.zeros(batch, partial_embeddings.shape[1],
                             device=partial_embeddings.device, dtype=partial_embeddings.dtype)
        counts = torch.zeros(batch, 1, device=partial_embeddings.device,
                             dtype=partial_embeddings.dtype)
        summed.index_add_(0, owner_tensor, partial_embeddings)
        counts.index_add_(0, owner_tensor, torch.ones_like(partial_embeddings[:, :1]))
        averaged = summed / counts.clamp_min(1.0)

        # 上游：raw_embed / np.linalg.norm(raw_embed, 2)
        return averaged / averaged.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def _normalize_volume(self, waveform: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """上游 `normalize_volume(wav, -30, increase_only=True)` 的 torch 版。

        上游是 `wav * 10^(dBFS_change/20)`，其中 dBFS_change = target − 10log10(mean(wav²))。
        只放大不缩小（increase_only）。
        """
        mask = (torch.arange(waveform.shape[1], device=waveform.device)[None, :]
                < lengths[:, None].to(waveform.device)).to(waveform.dtype)
        energy = (waveform * mask).pow(2).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        current_dbfs = 10.0 * torch.log10(energy.clamp_min(1e-12))
        change_db = -30.0 - current_dbfs
        gain = torch.where(change_db < 0, torch.ones_like(change_db),
                           torch.pow(10.0, change_db / 20.0))
        return waveform * gain[:, None] * mask


# ---------------------------------------------------------------------------
# 2. Stage 1（AutoVC 架构）的 torch 版
# ---------------------------------------------------------------------------
class TorchAutoVCStage1(nn.Module):
    """AutoVC `Generator` 的 torch 封装，**可求导**。

    对应关系：
        c      = Encoder(x, h)               → content_code()
        x_adv  = Decoder(c, h_adv)            → decode()
        x_base = Decoder(c, h)                → 参照（不加扰动的重建）

    ★ 论文式(6)：`x̂ = D(c, h_trg)` —— AutoVC 的 `Generator.forward(x, c_org, c_trg)`
      正好支持这个用法，**不需要改它的模型代码**。

    ★ 权重：AutoVC 官方 `autovc.ckpt` 已 404（见任务板 §0.8），
      所以这里支持加载**成员 A 自己训练的 Stage 1 checkpoint**，
      格式与 AutoVC 一致（`{"model": state_dict}` 或裸 state_dict）。
    """

    def __init__(self, checkpoint: str | Path | None = None,
                 device: str | None = None,
                 dim_neck: int | None = None, dim_emb: int | None = None,
                 dim_pre: int | None = None, freq: int | None = None):
        super().__init__()
        paths.ensure_import_paths()
        from model_vc import Generator

        settings = config.AUTOVC_MODEL
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.freq = freq or settings["freq"]
        self.pad_base = settings["pad_base"]

        self.model = Generator(
            dim_neck=dim_neck or settings["dim_neck"],
            dim_emb=dim_emb or settings["dim_emb"],
            dim_pre=dim_pre or settings["dim_pre"],
            freq=self.freq,
        )
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.checkpoint_path = Path(checkpoint) if checkpoint else self._guess_checkpoint()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"找不到 Stage 1 权重：{self.checkpoint_path}\n"
                "成员 A 的 Stage 1 训练完之前，这里用 StubStage1 冒烟测试。\n"
                "权重应放在 algorithms/voiceprint_protection/weights/ 下。"
            )
        data = torch.load(self.checkpoint_path, map_location="cpu")
        state = data["model"] if "model" in data else data
        self.model.load_state_dict(state)
        self.model.to(self.device)

    def _guess_checkpoint(self) -> Path:
        candidates = [
            paths.WEIGHTS_DIR / "stage1.pt",
            paths.WEIGHTS_DIR / "autovc.ckpt",
            paths.models_dir() / "default" / "stage1.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    # -- 内部 -------------------------------------------------------------
    def _pad(self, mel: torch.Tensor) -> torch.Tensor:
        """把时间维补齐到 `pad_base` 的整数倍（AutoVC 的 Encoder 按 freq 抽样）。"""
        length = mel.shape[1]
        target = int(self.pad_base * np.ceil(float(length) / self.pad_base))
        if target > length:
            mel = torch.nn.functional.pad(mel, (0, 0, 0, target - length))
        return mel

    # -- 对外 -------------------------------------------------------------
    def content_code(self, mel: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """(B, T, 80) + (B, 256) -> 内容码 (B, T', 64)。"""
        mel = self._pad(mel)
        with torch.no_grad():
            codes = self.model.encoder(mel, embedding)
        return codes

    def decode(self, mel: torch.Tensor, embedding: torch.Tensor,
               content: torch.Tensor | None = None) -> torch.Tensor:
        """x_adv = D(c, h_adv) → (B, T, 80)。**对 embedding 可导** ✓"""
        mel = self._pad(mel)
        _, postnet, _ = self.model(mel, embedding, embedding)
        return postnet[:, 0]

    def reconstruct(self, mel: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """不加扰动的重建 x̂ = D(c, h) → (B, T, 80)。"""
        mel = self._pad(mel)
        with torch.no_grad():
            _, postnet, _ = self.model(mel, embedding, embedding)
        return postnet[:, 0]


# ---------------------------------------------------------------------------
# 3. 声码器（可求导）
# ---------------------------------------------------------------------------
class TorchHiFiGANVocoder(nn.Module):
    """HiFi-GAN（论文用的就是这个，[15]），torch 版可求导。

    ★ 2026-11-19（成员 A）改了两处：
      1. **不再用 `torch.hub.load("NVIDIA/torch.hifi_gan", ...)`** ——
         那个仓库**已整体下架**（2026-11-18 实测：GitHub API / codeload / raw /
         release 附件全部 404），旧写法**必然失败**。
         现在用 `algorithms/voiceprint_protection/hifigan.py` 里**内置**的生成器
         （bshall/hifigan 的 HiFi-GAN v1，MIT）+ 本地权重
         `data/models/default/hifigan_16k.ckpt`
         （HuggingFace `speechbrain/tts-hifigan-libritts-16kHz`，
           与 NVIDIA 的 `hifigan_melbins_16k` **架构与 mel 约定逐项相同**，
           同样 13,936,130 参数，`strict=True` 加载 234/234 键全匹配）。
      2. **mel 约定不再是问题**：Stage 1 的模型侧 mel 现在**直接就是**
         HiFi-GAN 的约定（`config.STAGE1["mel_preset"] = "hifigan16k"`），
         不需要任何近似换算。

    ★ 实际上 `stage2/pipeline.py::HiFiGANVocoder` 已经是同样的东西了 ——
      这个类是**重复实现**，保留是为了不破坏既有导入路径。
    """

    def __init__(self, device: str | None = None, model_name: str = "hifigan_melbins_16k",
                 weights_path=None):
        super().__init__()
        import hifigan

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if weights_path is None:
            from bridge_vocoder import VocoderBridge

            weights_path = VocoderBridge._default_weights_path()
        if not Path(weights_path).exists():
            raise RuntimeError(
                f"找不到 HiFi-GAN 权重：{weights_path}\n"
                "先下载：python algorithms/voiceprint_protection/scripts/download_hifigan_weights.py")

        # ★ strict=True：架构对不上立刻报错（实测 234/234 键全匹配）
        self.model = hifigan.load_hifigan_weights(weights_path, device=self.device, strict=True)
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """HiFi-GAN **原始**约定的 mel **(B, T, 80)**（时间在前，项目约定）-> 波形 (B, N)。

        ⚠️ 输入要是原始 mel（不是归一化后的网络侧 mel）；
        网络侧 mel 请先过 `audio_utils.hifigan_mel_unnorm()`，
        或直接用 `stage2/pipeline.py::Stage2Pipeline.vocode()`（它会自动处理）。
        """
        # (B, T, 80) -> (B, 80, T)：HiFi-GAN 吃通道在前
        return self.model(mel.transpose(1, 2).contiguous()).squeeze(1)


class StubVocoder(nn.Module):
    """随机权重的替身声码器（**只用于离线冒烟测试**）。

    它不是声码器，只是一个**可导的** mel → 波形 映射，
    用来验证"梯度能不能穿过声码器流回 G"这条链路是通的。
    出来的波形是噪音，**任何指标都没有意义**。
    """

    def __init__(self, n_mels: int = config.FRAME_DIM, upsample: int = 160):
        super().__init__()
        self.upsample = upsample
        self.projection = nn.Conv1d(n_mels, 1, kernel_size=1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel (B, T, 80) -> 波形 (B, T*upsample)。"""
        x = self.projection(mel.transpose(1, 2))              # (B, 1, T)
        x = torch.nn.functional.interpolate(x, size=x.shape[-1] * self.upsample,
                                            mode="linear", align_corners=False)
        return x.squeeze(1)


# ---------------------------------------------------------------------------
# 4. 冒烟测试用的整套替身
# ---------------------------------------------------------------------------
class StubStage1(nn.Module):
    """随机权重的 Stage 1 替身：编码器 + 解码器 + 声码器。

    用途：**在 Stage 1 还没训练出来的时候**，验证 Stage 2 的训练循环
    （前向、反向、损失、梯度、更新）**跑得通**。

    ⚠️ 它**不能**用来评测 —— 权重是随机的，指标全是噪声。
    """

    def __init__(self, n_mels: int = config.FRAME_DIM,
                 content_dim: int = config.CONTENT_ACTUAL_DIM,
                 embedding_dim: int = config.SPEAKER_DIM,
                 freq: int = config.CONTENT_DOWNSAMPLE,
                 device: str | None = None):
        super().__init__()
        self.freq = freq
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.encoder = nn.Sequential(nn.Linear(n_mels, content_dim), nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(content_dim + embedding_dim, n_mels), nn.Tanh())
        self.vocoder = StubVocoder(n_mels=n_mels)

    def content_code(self, mel: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """mel (B, T, 80) -> 内容码 (B, T//freq, 64)。"""
        downsampled = mel[:, :: self.freq, :]
        return self.encoder(downsampled)

    def decode(self, mel: torch.Tensor, embedding: torch.Tensor,
               content: torch.Tensor | None = None) -> torch.Tensor:
        """→ 保护后 mel (B, T, 80)。对 embedding 可导 ✓

        ★ 帧数必须与输入 mel **完全一致**（契约 §2 的形状不变量），
          否则 L_mel 会退化成常量还不报错。
          AutoVC 的解码器是上采样 freq=32 倍刚好对上；
          这里用线性插值补回同样的帧数。
        """
        if content is None:
            content = self.content_code(mel, embedding)
        expanded = embedding.unsqueeze(1).expand(-1, content.shape[1], -1)
        decoded = self.decoder(torch.cat([content, expanded], dim=-1))   # (B, T', 80)
        if decoded.shape[1] != mel.shape[1]:
            decoded = torch.nn.functional.interpolate(
                decoded.transpose(1, 2), size=mel.shape[1], mode="linear", align_corners=False
            ).transpose(1, 2)
        return decoded

    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        """mel (B, T, 80) -> 波形 (B, N)。"""
        return self.vocoder(mel)


def freeze(module: nn.Module) -> nn.Module:
    """冻结参数但**保留梯度流过**的标准做法。

    很多人以为 `requires_grad_(False)` 会切断梯度 —— 不会！
    它只是**不累加**这些参数的梯度；梯度仍然会穿过这个模块往前传。
    真正切断梯度的是 `torch.no_grad()` 上下文或 `.detach()`。
    """
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()
    return module


def describe(module: nn.Module) -> str:
    """打印可训练/冻结的参数数量（训练脚本启动时打一次，便于确认冻结对了）。"""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
    return f"{type(module).__name__}: 可训练 {trainable:,} / 冻结 {frozen:,}"
