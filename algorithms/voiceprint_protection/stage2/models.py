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
# 1b. E_spk 的另一种来源：随 Stage 1 一起微调过的 GE2E
# ---------------------------------------------------------------------------
class Stage1SpeakerEncoder(nn.Module):
    """E_spk = **随 Stage 1 一起微调过**的 GE2E（论文 §3.1 的原意）。

    【★ 为什么需要这一个类 —— 这是 Stage 2 的"隐藏前提"】
    论文 §3.1 原文：
        *"Speaker embeddings are extracted using a GE2E model initialized from
          pretrained weights and **fine-tuned end-to-end during Stage 1 training**."*

    也就是说，**Stage 2 用的 E_spk 必须是 Stage 1 微调完的那一个**。
    如果 Stage 2 拿**上游没微调过的** GE2E 去算嵌入：
      · 训练时算的 `L_de` 与推理时算的 `Sim_clone` **不是同一个度量**；
      · 生成器 G 学到的是"怎么扰动**上游** GE2E 的嵌入空间"，
        而推理时喂给 G 的是**微调后**的嵌入 ⇒ **G 的输出是离分布的**，
        扰动基本等于随机噪声，**而且不报错**。

    ⇒ 所以 Stage 2 训练时必须显式选一个 E_spk，并且**把这个选择写进 checkpoint**，
      推理时照着做 —— 两边不可能对不上。

    【算法必须与推理时逐字一致】
    这里**直接复用** `stage1/speaker_encoder.py::TrainableSpeakerEncoder`
    （成员 A 写的），**不重新实现一遍** ——
    同一个类、同一个权重，才谈得上"一致"。
    注意它与上面 `TorchSpeakerEncoder` 的差别：
      · `TorchSpeakerEncoder`：上游 `embed_utterance` 的做法（分 160 帧的段、
        50% 重叠、各段嵌入取平均）+ 音量归一化；
      · `TrainableSpeakerEncoder`：整段 mel 一次过 LSTM（成员 A 训 Stage 1 用的就是这个）。
    两个都能用，但**训练与推理必须用同一个**。
    """

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        super().__init__()
        from stage1.speaker_encoder import TrainableSpeakerEncoder

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"找不到 Stage 1 checkpoint：{path}")

        state = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or "speaker_encoder" not in state:
            raise KeyError(
                f"{path.name} 里没有 'speaker_encoder' 这一段。\n"
                "⇒ 这个 checkpoint 不是本项目 stage1/train.py 训出来的"
                "（或当时没开 GE2E 微调），不能当作 Stage 2 的 E_spk。\n"
                "可以用 --speaker-encoder upstream 退回上游的 GE2E。"
            )

        self.encoder = TrainableSpeakerEncoder(device=str(self.device), fine_tune=False)
        self.encoder.model.load_state_dict(state["speaker_encoder"])
        self.encoder.model.eval().to(self.device)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.checkpoint_path = path

    def forward(self, waveform: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        """波形 (B, N) -> 嵌入 (B, 256)（已 L2 归一化）。"""
        import audio_utils

        if waveform.dim() != 2:
            raise ValueError(f"期望 (B, N) 的波形，收到 {tuple(waveform.shape)}")
        batch = waveform.shape[0]
        if lengths is None:
            lengths = torch.full((batch,), waveform.shape[1],
                                 device=waveform.device, dtype=torch.int64)

        rows = []
        for index in range(batch):
            wav = waveform[index, : int(lengths[index])].detach().cpu().numpy()
            mel = audio_utils.ge2e_mel(wav)
            rows.append(self.encoder(torch.from_numpy(mel).unsqueeze(0).to(self.device))[0])
        return torch.stack(rows, dim=0)


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
                 dim_pre: int | None = None, freq: int | None = None,
                 random_init: bool = False):
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

        # ★ random_init=True：权重是**随机初始化**的，指标毫无意义。
        #   它只用来验"Stage 1 <-> Stage 2 的接口对不对"：
        #     · content_code 的形状是不是 (B, T', 64)
        #     · h_adv 有没有真的进解码器（D(c,h) 与 D(c,h_adv) 要有差）
        #     · content 有没有被真的用上
        #     · 梯度能不能回到 h_adv
        #   这几条**不依赖训练好的权重** ⇒ Stage 1 还没训完也能验。
        #   （A 修的两个致命 bug 之所以能溜过去，就是因为当时只用 StubStage1 验，
        #     而 StubStage1 的 decode 逻辑跟官方 Generator 根本不是一回事。）
        if random_init:
            self.checkpoint_path = None
            return

        self.checkpoint_path = Path(checkpoint) if checkpoint else self._guess_checkpoint()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"找不到 Stage 1 权重：{self.checkpoint_path}\n"
                "成员 A 的 Stage 1 训练完之前，这里用 StubStage1 冒烟测试。\n"
                "权重应放在 algorithms/voiceprint_protection/weights/ 下。"
            )
        data = torch.load(self.checkpoint_path, map_location="cpu")
        state = data["model"] if "model" in data else data
        # ★ 兼容早期 checkpoint 的键名（`conv_layer` -> `conv`）。
        #   现在的键名与**官方 AutoVC 的 Generator 完全一致**，
        #   所以这里能直接用官方 `Generator` 加载成员 A 训出来的权重。
        from stage1.model import remap_to_official_keys

        self.model.load_state_dict(remap_to_official_keys(state))
        self.model.to(self.device)

    def _guess_checkpoint(self) -> Path:
        # ★ 2026-11-19（成员 A）：成员 A 的 Stage 1 训练脚本产出的文件名是
        #   `stage1_model.pt`（另有断点 `stage1_last_ckpt.pt`），
        #   与这里的 `stage1.pt` 不同 —— 把 A 的实际文件名加进候选。
        candidates = [
            paths.WEIGHTS_DIR / "stage1_model.pt",
            paths.WEIGHTS_DIR / "stage1_last_ckpt.pt",
            paths.WEIGHTS_DIR / "stage1.pt",
            paths.WEIGHTS_DIR / "autovc.ckpt",
            paths.models_dir() / "default" / "stage1.pt",
            paths.models_dir() / "default" / "stage1_model.pt",
        ]
        skipped: list[tuple[Path, int]] = []
        for candidate in candidates:
            if not candidate.exists():
                continue
            # ★★ 2026-11-20（成员 B 建议，A 实现）：**跳过"冒烟 checkpoint"**。
            #   冒烟测试（`--iterations 3 --smoke`）也会写 `stage1_model.pt`，
            #   它只训了几步、**指标全是垃圾**，但会被这里静默捡去用，
            #   **而且不报错** —— 最坏的一类。
            #   ⇒ 只接受训够步数的权重；不够的**明确跳过并说明**。
            step = self._read_step(candidate)
            minimum = config.STAGE1.get("min_steps_for_real_use", 1000)
            if step is not None and step < minimum:
                print(f"  [跳过] {candidate.name} 只训了 {step} 步"
                      f"（< {minimum}），看着像冒烟产物 —— 指标无意义。")
                skipped.append((candidate, step))
                continue
            return candidate

        if skipped:
            # ★ 全部候选都是冒烟产物 ⇒ **明确报错**，不要静默兜底
            #   （兜底会把刚跳过的垃圾权重又捡回来，等于白跳）。
            detail = "\n".join(f"    {path.name}：{step} 步"
                               for path, step in skipped)
            raise FileNotFoundError(
                "找到的 Stage 1 权重都只训了很少的步数（看着像冒烟产物），已拒绝使用：\n"
                f"{detail}\n"
                "正式训练请用 --stage1-checkpoint 显式指定正式权重。")

        return candidates[0]

    @staticmethod
    def _read_step(path: Path) -> int | None:
        """从 checkpoint 里读训练步数（读不出来就返回 None，不拦）。"""
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            return None
        if isinstance(data, dict):
            step = data.get("step")
            if isinstance(step, int):
                return step
        return None

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
        """(B, T, 80) + (B, 256) -> 内容码 **(B, T', 64)**。

        ★ 2026-11-19（成员 A）修：**返回张量，不是 list**。
          官方 AutoVC 的 `Encoder.forward` 返回 **list**（T' 个 (B,64)），
          但契约 §4.5.1 规定 `c` 是 `(B, T', 64)` 的**张量**，
          而且 `stage2/train.py` 要写 `content_codes[permutation]`
          （batch 内随机置换 = 论文的 "arbitrary content"）——
          对 list 做这种索引是错的。
        """
        mel = self._pad(mel)
        with torch.no_grad():
            codes = self.model.encoder(mel, embedding)      # list，len = T'
        if isinstance(codes, (list, tuple)):
            codes = torch.stack(codes, dim=1)              # (B, T', 64)
        return codes

    def decode(self, mel: torch.Tensor, embedding: torch.Tensor,
               content: torch.Tensor | None = None,
               source_embedding: torch.Tensor | None = None) -> torch.Tensor:
        """**x_adv = D(c, h_adv)** → (B, T, 80)。**对 embedding 可导** ✓

        参数
        ----
        mel              : 原始 mel (B, T, 80)（**内容从它来**）
        embedding        : **目标**说话人嵌入（h_adv / h_stolen）(B, 256) —— 解码用
        content          : 内容码 c (B, T', 64)。不给就从 mel 现场算
        source_embedding : 编码侧条件 c_org。默认用 `embedding`

        ★ 2026-11-19（成员 A）修的两个**致命 bug**：
          ① 原来写的是 `self.model(mel, embedding, embedding)` ——
             官方 `Generator.forward(x, c_org, c_trg)` 的**第三个参数才是目标嵌入**，
             两个都传 embedding 等于**h_adv 根本没进解码器**，
             整个防御机制（扰动嵌入 → 重建变样）就失效了，而且**不报错**。
          ② 原来的 `content` 参数**被完全忽略** ——
             论文的 "arbitrary content"（`c_arb`）没起作用，
             攻击者克隆的其实是原句内容。
        """
        # ★ 记下**原始**帧数：AutoVC 的解码器要求 T 是 freq 的整数倍，
        #   所以内部会 pad。但契约 §2 要求 **x_adv 与 x 帧数完全相等**，
        #   因此最后要把多出来的帧裁掉 —— 不裁的话调用方直接报
        #   "x 与 x_adv 的帧数必须相等"（实测 188 vs 192）。
        target_frames = mel.shape[1]
        mel = self._pad(mel)

        if content is None:
            content = self.content_code(mel, source_embedding
                                        if source_embedding is not None else embedding)
        codes = list(content.unbind(1))               # (B,T',64) -> T' 个 (B,64)

        # 官方 Generator.forward 的"摊开"逻辑：每个 code 摊回 freq 帧
        expanded = []
        for code in codes:
            expanded.append(code.unsqueeze(1).expand(
                -1, int(mel.size(1) / len(codes)), -1))
        code_expanded = torch.cat(expanded, dim=1)    # (B, T, 64)

        # ★ c_trg = 目标嵌入（这一位原来传错了）
        decoder_input = torch.cat(
            (code_expanded, embedding.unsqueeze(1).expand(-1, mel.size(1), -1)),
            dim=-1)                                     # (B, T, 64+256)

        mel_outputs = self.model.decoder(decoder_input)                 # (B, T, 80)
        mel_postnet = self.model.postnet(mel_outputs.transpose(2, 1))   # (B, 80, T)
        mel_postnet = mel_outputs + mel_postnet.transpose(2, 1)         # (B, T, 80)
        if mel_postnet.shape[1] != target_frames:
            # 裁回原始帧数 —— 保证 "输出与输入等长" 这个不变式**由构造保证**，
            # 而不是依赖调用方喂进来的 T 恰好是 freq 的整数倍。
            mel_postnet = mel_postnet[:, :target_frames]
        return mel_postnet

    def reconstruct(self, mel: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """不加扰动的重建 x̂ = D(c, h) → (B, T, 80)。"""
        mel = self._pad(mel)
        with torch.no_grad():
            return self.decode(mel, embedding, source_embedding=embedding)


# ---------------------------------------------------------------------------
# 3. 声码器（可求导）
# ---------------------------------------------------------------------------
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
