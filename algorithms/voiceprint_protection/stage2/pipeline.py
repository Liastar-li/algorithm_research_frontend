"""Stage 2 的前向流水线：把「原始语音」和「保护后语音」两条路接起来。

【为什么单独一个文件】
训练循环里反复要用这几件事，每一件都有**容易搞错的地方**，
集中放一处并写清楚，比散在 `train.py` 里安全：

  1. `mel(waveforms)`      原始语音 → mel（AutoVC 约定）
  2. `vocode(mel)`         mel → 波形（★ 必须可导）
  3. `align(a, b)`         两条波形长度可能不同 → 裁到一样长（否则相减报错）
  4. `thresholds(waveform)`原始波形 → 心理声学掩蔽阈值 θ_x
  5. `content(mel, emb)`   内容码 c

【★ 声码器：Stage 2 目前缺一个"可求导"的】
论文用的是 HiFi-GAN [15]，但我们有两个候选，各有坑：

  · **HiFi-GAN**（torch.hub，论文同款）
      坑：它的 mel 约定是 `log10` 幅度 + `(x+4.5)/5`，
          而 AutoVC 侧是 `20log10` + `clip[0,1]` —— **两者不能直接互换**。
          本文件提供 `autovc_mel_to_hifigan()` 做一次**近似**换算
          （假定两套滤波器组看到的底层能量相同，**不严格成立**）。

  · **AutoVC 配套的 wavenet_vocoder**（与 Stage 1 的 mel 天然匹配）
      坑：上游 `vocoder.generate()` 最后一步转 numpy，梯度断在那里。
          要用得把最后几步搬成 torch —— **这一步还没做**（见下方 TODO）。

  · **Stub**（随机权重）—— 只用于离线冒烟测试，指标无意义。

★ 这一项是 Stage 2 的**公开阻塞点**，已写进 `docs/notes/stage2_blockers.md`。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import audio_utils  # noqa: E402
import config  # noqa: E402


# ---------------------------------------------------------------------------
# mel 换算
# ---------------------------------------------------------------------------
# ★★ 2026-11-19（成员 A）：**`autovc_mel_to_hifigan()` 已删除**
#
# 原因（实测数据，见 docs/interface-contract.md §3.4 与 TASKS.md §0.10.4c）：
#   那个近似换算会把重建音的说话人余弦从 **0.9746 掉到 0.6983** ——
#   两套 mel 的**滤波器组不同**（AutoVC: fmin 90 / fmax 7600 / 带 30 Hz 高通；
#   HiFi-GAN: fmin 0 / fmax 8000 / slaney），底层能量分布不同，**仿射补不回来**。
#
# ⇒ 现在 Stage 1 的模型侧 mel **直接就是 HiFi-GAN 的约定**
#   （`config.STAGE1["mel_preset"] = "hifigan16k"`），
#   所以这里**不需要任何换算**。
# ⇒ 顺带解决了另一个隐患：HiFi-GAN 原始 mel 值域是 [-11.5, 0]，
#   直接用会让 L_rec 比论文大 ~100 倍、λ_cd 被稀释 40 倍；
#   `audio_utils.model_mel()` 返回的是**归一化后**的 mel（值域 [-0.11, 0.92]，
#   与 AutoVC 同量纲），论文的 λ_cd=1.0 / λ_dis=0.01 直接可用。
#
# 送进声码器前的反归一化由 `bridge_vocoder.VocoderBridge.mel_to_wav()` 负责。


def batched_model_mel(waveforms: torch.Tensor) -> torch.Tensor:
    """一批波形 → **模型侧** mel (B, T, 80)。

    逐条调 `audio_utils.model_mel()`（numpy）再拼起来。
    慢，但**保证与项目里其他地方算出来的 mel 完全一致**（唯一真源原则）。
    这里只用于**原始语音**（不需要梯度），所以慢一点没关系。
    """
    frames = []
    for index in range(waveforms.shape[0]):
        single = waveforms[index].detach().cpu().numpy()
        frames.append(audio_utils.model_mel(single))
    length = min(frame.shape[0] for frame in frames)
    stacked = np.stack([frame[:length] for frame in frames], axis=0)
    return torch.from_numpy(stacked.astype(np.float32))


# ---------------------------------------------------------------------------
# 流水线
# ---------------------------------------------------------------------------
class Stage2Pipeline:
    """把冻结组件和辅助算子装在一起，给训练循环用。"""

    def __init__(self, speaker_encoder, stage1, vocoder=None,
                 vocoder_convention: str = "hifigan",
                 device: torch.device | str = "cpu",
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 sample_rate: int = 16000):
        self.speaker = speaker_encoder
        self.stage1 = stage1
        self.vocoder = vocoder
        self.vocoder_convention = vocoder_convention
        self.device = torch.device(device)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        # ★ 模型侧 mel 预设（契约字段 config.STAGE1["mel_preset"]）。
        #   声码器要吃的是**原始量纲**的 mel，而网络侧喂模型的是**归一化后**的，
        #   中间那个仿射变换的系数就来自这个预设 —— 所以这里必须跟着预设走，
        #   绝不能写死 "hifigan16k"（写死的话，换了预设就会送出错的 mel，
        #   声码器出来是噪音，**而且不报错**）。
        self.mel_preset_name = config.STAGE1.get("mel_preset", config.DEFAULT_PRESET)
        self.mel_preset = config.MEL_PRESETS[self.mel_preset_name]

    # -- 1. mel ----------------------------------------------------------
    def mel(self, waveforms: torch.Tensor) -> torch.Tensor:
        """原始波形 (B, N) → **模型侧** mel (B, T, 80)。不需要梯度。"""
        return batched_model_mel(waveforms).to(self.device)

    # -- 2. 声码器 --------------------------------------------------------
    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        """mel → 波形。★ 必须可导（梯度要穿过它回到 G）。

        ★ 2026-11-19（成员 A）：**不再需要 mel 口径换算**。
        Stage 1 的模型侧 mel 已经是 HiFi-GAN 的约定
        （`config.STAGE1["mel_preset"] = "hifigan16k"`），
        这里只做一次**可逆的反归一化**（把它变回声码器要的原始量纲）。

        `vocoder_convention` 现在只有两种取值：
          * `"hifigan"`（默认）：输入是**网络侧** mel，这里反归一化后喂
          * `"raw"`：输入已经是**原始** mel，直接喂，不做任何处理
        """
        if self.vocoder is None:
            raise RuntimeError("没有装载声码器 —— 见文件头的说明。")
        if self.vocoder_convention == "raw":
            return self.vocoder(mel)
        # 网络侧 mel (B, T, 80) -> 原始 mel (B, T, 80)
        # ★ 用的是**当前模型侧预设**的归一化系数，不是写死的 "hifigan16k"。
        preset = self.mel_preset
        if "norm_scale" not in preset:
            raise RuntimeError(
                f"模型侧 mel 预设 {self.mel_preset_name!r} 没有归一化系数"
                "（norm_scale / norm_offset），无法反变换成声码器的原始量纲。\n"
                "要么把 config.STAGE1['mel_preset'] 改回 'hifigan16k'，"
                "要么把 vocoder_convention 设成 'raw'。"
            )
        raw = (mel - preset["norm_offset"]) / preset["norm_scale"]
        return self.vocoder(raw)

    # -- 3. 波形对齐 -------------------------------------------------------
    @staticmethod
    def align(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """把两条波形裁到同样长（相减之前必须对齐）。

        为什么要对齐：δ_wav = V(x_adv) − V(x) 是**逐采样点相减**，
        长度不一样就没法减。x 与 x_adv 的帧数相同（契约 §2 的形状不变量），
        所以正常情况下两个波形本来就等长；这里只是兜底。
        """
        length = min(first.shape[-1], second.shape[-1])
        return first[..., :length], second[..., :length]

    # -- 4. 心理声学阈值 ---------------------------------------------------
    def thresholds(self, waveform: torch.Tensor) -> torch.Tensor | None:
        """原始波形 → 掩蔽阈值 θ_x (B, K)。

        逐条算（numpy），再拼成 batch。
        """
        try:
            from stage2.psychoacoustics import masking_threshold
        except ImportError:
            return None

        rows = []
        for index in range(waveform.shape[0]):
            single = waveform[index].detach().cpu().numpy()
            rows.append(masking_threshold(single, sample_rate=self.sample_rate,
                                         n_fft=self.n_fft))
        stacked = np.stack(rows, axis=0)
        return torch.from_numpy(stacked.astype(np.float32)).to(self.device)

    # -- 5. 内容码 ---------------------------------------------------------
    def content(self, mel: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """内容码 c（不需要梯度，它是"参照物"）。"""
        with torch.no_grad():
            return self.stage1.content_code(mel, embedding)

    # -- 自述 --------------------------------------------------------------
    def describe(self) -> str:
        vocoder_name = type(self.vocoder).__name__ if self.vocoder is not None else "（无）"
        return (
            f"Stage 2 流水线\n"
            f"  E_spk      : {type(self.speaker).__name__}\n"
            f"  Stage 1    : {type(self.stage1).__name__}\n"
            f"  声码器     : {vocoder_name}（mel 口径：{self.vocoder_convention}）\n"
            f"  模型侧 mel : {self.mel_preset_name}\n"
            f"  采样率     : {self.sample_rate} Hz\n"
            f"  STFT       : n_fft={self.n_fft} hop={self.hop_length}"
        )


# ---------------------------------------------------------------------------
# 声码器们
# ---------------------------------------------------------------------------
class HiFiGANVocoder(nn.Module):
    """HiFi-GAN（论文同款），torch 版，可导。

    ★ 2026-11-19（成员 A）改了两处：
      1. **不再用 `torch.hub.load("NVIDIA/torch.hifi_gan", ...)`** ——
         那个仓库**已整体下架**（2026-11-18 实测：GitHub API / codeload / raw /
         release 附件全部 404），所以旧写法**必然失败**。
         现在用 `algorithms/voiceprint_protection/hifigan.py` 里**内置**的生成器
         （bshall/hifigan 的 HiFi-GAN v1，MIT）+ 本地权重：
             `<数据根>/models/default/hifigan_16k.ckpt`
         权重来源：HuggingFace `speechbrain/tts-hifigan-libritts-16kHz`
         （与 NVIDIA 的 `hifigan_melbins_16k` **架构与 mel 约定逐项相同**，
           同样 13,936,130 参数，`strict=True` 加载 234/234 键全匹配）。
      2. mel 口径：Stage 1 现在直接产出 HiFi-GAN 约定的 mel，
         **不需要 `autovc_mel_to_hifigan()` 那个近似换算了**。

    ★ 2026-11-20（成员 B）收口：内部改成包 `bridge_vocoder.VocoderBridge`
      —— 契约 §4.5.3 指定的**唯一**权重加载入口。两边加载的是**同一个**
      checkpoint，架构一致，不会出现"两份实现慢慢跑偏"。
      （原先 `stage2/models.py::TorchHiFiGANVocoder` 是同一份逻辑的第二份拷贝，
        全仓库没有任何地方 import 它 —— 已删除，只留这一个。）
    """

    def __init__(self, device: str | None = None, weights_path=None):
        super().__init__()
        from bridge_vocoder import VocoderBridge

        # ★ 权重加载收敛到 bridge_vocoder（契约 §4.5.3 指定的唯一入口），
        #   那边已经做过 strict=True 校验（234/234 键全匹配）与参数冻结。
        self.bridge = VocoderBridge(weights_path=weights_path, device=device)
        self.device = self.bridge.device
        # 生成器挂成子模块：这样 .to(device) / state_dict() 都对得上
        self.model = self.bridge.model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """HiFi-GAN 口径的 mel **(B, T, 80)**（时间在前，项目约定）→ 波形 (B, N)。

        ⚠️ 输入要是**原始** HiFi-GAN mel（不是归一化后的网络侧 mel）。
        若拿的是网络侧 mel，请先 `audio_utils.hifigan_mel_unnorm()`；
        或直接用 `stage2/pipeline.py::Stage2Pipeline.vocode()`（它会自动反变换）。
        """
        # (B, T, 80) -> (B, 80, T)：HiFi-GAN 吃通道在前
        return self.model(mel.transpose(1, 2).contiguous()).squeeze(1)


class WaveNetVocoder(nn.Module):
    """AutoVC 配套的 wavenet_vocoder（**TODO：还没实现**，见文件头）。

    ★ 为什么需要它
      它的 mel 约定与 AutoVC **天然一致**（dB 归一化到 [0,1]），
      不需要像 HiFi-GAN 那样做近似换算 → 论文的 L_mel / L_mask 才严格成立。

    ★ 怎么实现（照着上游 `wavenet_vocoder/vocoder.py` 搬）
      1. `from wavenet_vocoder import builder` → `builder.build_model()`
      2. 加载 `checkpoint_step001000000_ema.pth` 的 `state_dict`
      3. 上游 `extract_features(mel)`：把 (B, T, 80) 的 mel 变成模型要的
         上采样条件（这一步**是 torch 的**，可以照抄）
      4. 上游 `model(c, None)` 出波形（**是 torch 的**，可以照抄）
      5. ★ **只有最后"转 numpy"那一步要改成留在 torch 里** ——
         梯度就是断在那里的

    在这之前，用 `--vocoder stub`（冒烟测试）或 `--vocoder hifigan`（近似换算）。
    """

    def __init__(self, checkpoint=None, device: str | None = None):
        super().__init__()
        raise NotImplementedError(
            "WaveNetVocoder 还没实现 —— 见本类的 docstring 里的 5 步实现说明。\n"
            "当前可用：--vocoder stub（冒烟测试）/ --vocoder hifigan（近似换算）。"
        )
