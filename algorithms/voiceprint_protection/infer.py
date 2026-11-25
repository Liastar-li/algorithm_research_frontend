"""推理主流程 —— 把论文的推理路径完整跑通。

【论文的推理链 vs 我们的实现】
    论文                                      本文件
    ──────────────────────────────────────    ────────────────────────────────────────
    x  = mel(波形)                             audio_utils.autovc_mel()   （Stage 1 用）
    h  = E_spk(波形)                           bridge_speaker（上游 GE2E）★ 权重可用
    δ  = G(h)        ← Stage 2 的生成器         此处先用随机高斯噪声占位（L1 不训练）
    h_adv = h + α·δ                             perturb_embedding()
    c  = E_c(x, h)                              Stage 1 训练产出（L1 阶段暂不需要）
    x_adv = D(c, h_adv)                         ★ 见下方【★ 路线调整】
    波形 = V(x_adv)                             bridge_cloner（上游合成器+声码器）

【★ 路线调整（2026-11-17，成员 A）】
原计划用 AutoVC 官方解码器当"配音员"（x_adv = D(c, h_adv)），
但 **AutoVC 官方权重链实测 404 失效**（IBM Box），拿不到主模型与配套声码器。

于是改用**上游 Real-Time-Voice-Cloning 的合成器 + 声码器**做同一件事：
    x_adv  ≡  用**被污染的音色** h_adv 去念**同一句话**（文本相同 -> 内容相同）
    波形   ≡  合成器输出 mel -> 声码器输出波形

这样做的三个好处：
  1. 三个权重全部可下载（HF `CorentinJ/SV2TTS`），不依赖任何失效链接；
  2. 合成器与声码器**互相配套**，我们完全不用自己猜 mel 约定；
  3. ★ 合成器**就是论文 Figure 1 红框里的"攻击者"** ——
     用它跑两次（原始嵌入 vs 被扰动嵌入），两者的差别就是防御效果本身。

【形象比喻】
整条链像"给一段录音打疫苗，再让骗子去用这段录音行骗"：
  1. 体检      —— 测出身份卡 h（256 个数字）
  2. 打疫苗    —— 在卡上做微小改动（h + α·δ），人耳听不出，机器却认错了
  3. 骗子行骗  —— 用这张卡去念一段话，得到"克隆音"
                 · 用原始卡   -> 像本人（未防御的基线）
                 · 用打过疫苗的卡 -> 不像本人（防御生效）
  4. 验收      —— 人耳听"保护后语音"，确认听感几乎没变

【L1 与最终版的差别】
L1 阶段的 δ 是**随机高斯噪声**（不训练任何东西，先把链路跑通）。
Stage 2 训练完成后，这里换成训练好的生成器 G(h)，其余完全不变。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import audio_utils
import config
import paths

# 默认参考文本：上游 Tacotron 是英文模型，用一句常见英文句子。
# 也可以用 Whisper 转写输入音频，或在平台参数里让用户自己填。
DEFAULT_REFERENCE_TEXT = "The quick brown fox jumps over the lazy dog."


# ---------------------------------------------------------------------------
# 扰动
# ---------------------------------------------------------------------------
def random_perturbation(dimension: int = config.SPEAKER_DIM,
                         seed: int = config.SEED) -> np.ndarray:
    """随机高斯扰动 δ（L1 阶段的占位实现）。

    比喻：现在往身份卡上撒的是"随机沙粒"；
    Stage 2 训练完成后，换成"精心挑选的、能精准破坏机器识别的沙粒"。
    """
    generator = np.random.RandomState(seed)
    delta = generator.randn(dimension).astype(np.float32)
    norm = float(np.linalg.norm(delta))
    if norm > 0:
        delta = delta / norm
    return delta


def perturb_embedding(embedding: np.ndarray, delta: np.ndarray,
                      alpha: float = config.STAGE2["alpha"]) -> np.ndarray:
    """h_adv = h + α·δ，然后重新 L2 归一化。

    归一化是必要的：说话人编码器的输出是单位向量，
    加了扰动后模长会变大，不归一化会让下游模型跑到没见过的输入区域。
    """
    advanced = np.asarray(embedding, dtype=np.float32) + alpha * np.asarray(delta, dtype=np.float32)
    norm = float(np.linalg.norm(advanced))
    if norm > 0:
        advanced = advanced / norm
    return advanced.astype(np.float32)


def _infer_generator_config(state: dict) -> dict:
    """从生成器的权重里反推它的结构参数。

    ★ 唯一实现在 `stage2/generator.py::config_from_state` ——
      这里只是薄薄一层包装，免得两处各写一份、慢慢跑偏。
    """
    from stage2.generator import config_from_state

    return config_from_state(state)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
class Protector:
    """声纹保护器：把"体检 + 打疫苗 + 骗子行骗"封装成一个对象。

    三件套（都是上游 Real-Time-Voice-Cloning，权重全部可下载）：
      · speaker  —— 体检仪：波形 -> 256 维身份卡
      · cloner   —— 骗子：(文本, 身份卡) -> 音频
      · stage1   —— ★ 可选：**论文原版**的重建器（成员 A 训的 Stage 1 + HiFi-GAN）

    【三条路线，差别很大】
    | 路线 | 保护后语音怎么来的 | Sim_prot 上限 |
    |---|---|---|
    | "clone"（路线 B，默认） | `clone(text, h_adv)` —— 用克隆机**重新合成** | **≈0.82**（重新合成本身就改音色/韵律） |
    | "stage1"（★ 论文原版） | `D(c, h_adv)` —— **重建原句**，只换音色 | 论文 0.95 |
    | "stage2"（★ 论文原版 + 训练过的 G） | 同上，但 **δ = G(h)** 而不是随机高斯 | 论文 0.95 + 防御真的生效 |

    成员 B 在 `docs/experiments.md §3c` 实测过：路线 B 的 Sim_prot 有 **0.82 的
    结构性天花板**，与扰动强弱无关 —— 论文那组数字**必须走 stage1/stage2 路线**。

【★★★ stage1 与 stage2 的区别只有一处：δ 从哪来】
    stage1 : δ = 随机高斯噪声（L1 占位，**没有防御效果**）
    stage2 : δ = G(h) —— Stage 2 训练出来的生成器，**这才是论文的防御**
    其余（内容码、重建、声码器、攻击者对比）**完全一样**。
    """

    def __init__(self, use_speaker_bridge: bool = True, device: str | None = None,
                 with_cloner: bool = True, stage1_checkpoint: str | Path | None = None,
                 with_stage1: bool = True,
                 stage2_checkpoint: str | Path | None = None,
                 with_stage2: bool = True):
        from bridge_speaker import SpeakerBridge

        self.device = device
        self.speaker = SpeakerBridge(device=device) if use_speaker_bridge else None
        self.cloner = None
        if with_cloner:
            from bridge_cloner import ClonerBridge

            self.cloner = ClonerBridge(device=device)
        self._autovc = None   # AutoVC 路线按需加载（需要 AutoVC 权重）

        # ★ Stage 1 路线（论文原版）：成员 A 训的解耦-重建网络 + HiFi-GAN
        self.stage1 = None
        self.stage1_speaker = None
        self.vocoder = None
        if with_stage1:
            self.load_stage1(stage1_checkpoint)

        # ★ Stage 2 路线：训练好的生成器 G（δ = G(h)）
        self.stage2_generator = None
        self.stage2_checkpoint = None
        self.stage2_speaker_encoder = "upstream"   # E_spk 用哪个，由 checkpoint 自己说
        self.stage2_de_space = None                # L_de 训练时在哪个空间考核（仅用于报告）
        if with_stage2:
            self.load_stage2(stage2_checkpoint)

        self.spec = audio_utils.MelSpec()

    # -- Stage 1 路线 ----------------------------------------------------
    def load_stage1(self, checkpoint: str | Path | None = None) -> bool:
        """加载 Stage 1（解耦-重建网络）+ 随训微调的 GE2E + HiFi-GAN 声码器。

        返回是否成功（没有 checkpoint 时返回 False，调用方退回 "clone" 路线）。
        """
        import torch

        from stage1.model import Stage1Model
        from stage1.speaker_encoder import TrainableSpeakerEncoder

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 优先用训练完的最终权重；训练中就退回断点（它也有 "model" 键）
        candidates = [Path(checkpoint)] if checkpoint else [
            paths.WEIGHTS_DIR / "stage1_model.pt",
            paths.WEIGHTS_DIR / "stage1_last_ckpt.pt",
        ]
        path = next((p for p in candidates if p.exists()), candidates[0])
        if not path.exists():
            return False

        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            model = Stage1Model()
            model.load_state_dict(state["model"])
            model.eval().to(device)
            for parameter in model.parameters():
                parameter.requires_grad_(False)

            # ★ 用**随 Stage 1 一起微调过的** GE2E（论文的 E_spk），
            #   而不是上游那个原始的 —— 契约 §7.5
            encoder = TrainableSpeakerEncoder(device=device, fine_tune=False)
            if "speaker_encoder" in state:
                encoder.load_checkpoint_state(state["speaker_encoder"])
            encoder.model.eval().to(device)

            from bridge_vocoder import VocoderBridge

            self.stage1 = model
            self.stage1_speaker = encoder
            self.vocoder = VocoderBridge(device=device)
            self.stage1_checkpoint = path
            return True
        except Exception as error:  # noqa: BLE001
            print(f"  [警告] Stage 1 加载失败（{error}），退回 clone 路线。")
            self.stage1 = None
            return False

    @property
    def has_stage1(self) -> bool:
        return self.stage1 is not None

    # -- Stage 2 路线（训练好的生成器 G）---------------------------------
    def load_stage2(self, checkpoint: str | Path | None = None) -> bool:
        """加载 Stage 2 训练好的生成器 **G**（δ = G(h)）。

        返回是否成功（没有 checkpoint 时返回 False，调用方退回 stage1 路线）。

        ★★ **checkpoint 自带契约**（这是最重要的一点）：
        Stage 2 训练时用的 E_spk 与 G 的结构都写在 checkpoint 里
        （`speaker_encoder` / `generator_config`），
        这里**照着做** ⇒ 训练与推理不可能对不上。

        对不上的后果是"不报错、但防御完全失效"：
        G 学的是"怎么扰动**训练时那个** E_spk 的嵌入空间"，
        推理时喂进去的却是另一个嵌入空间 ⇒ δ 退化成随机方向。
        """
        import torch

        from stage2.generator import PerturbationGenerator

        candidates = [Path(checkpoint)] if checkpoint else [
            paths.WEIGHTS_DIR / "stage2" / "stage2_last.pt",
            paths.WEIGHTS_DIR / "stage2_last.pt",
        ]
        path = next((p for p in candidates if p.exists()), candidates[0])
        if not path.exists():
            return False

        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            generator_config = state.get("generator_config")
            if generator_config is None:
                # 早期 checkpoint 没存结构 ⇒ 从权重形状反推
                generator_config = _infer_generator_config(state["generator"])
            generator = PerturbationGenerator(**generator_config)
            generator.load_state_dict(state["generator"])
            generator.eval()
            for parameter in generator.parameters():
                parameter.requires_grad_(False)

            self.stage2_generator = generator
            self.stage2_checkpoint = path
            self.stage2_speaker_encoder = state.get("speaker_encoder", "upstream")
            self.stage2_de_space = state.get("de_space", self.stage2_speaker_encoder)
            print(f"  Stage 2 生成器 G：{path.name}"
                  f"（E_spk = {self.stage2_speaker_encoder}"
                  f"，L_de 考核于 {self.stage2_de_space} 空间）")
            return True
        except Exception as error:  # noqa: BLE001
            print(f"  [警告] Stage 2 加载失败（{error}），退回随机扰动。")
            self.stage2_generator = None
            return False

    @property
    def has_stage2(self) -> bool:
        return self.stage2_generator is not None

    def _embed_for_stage2(self, wav: np.ndarray) -> np.ndarray:
        """★ 用**训练时那个** E_spk 算嵌入 h。

        checkpoint 说用哪个就用哪个 ——
        见 `load_stage2` 里关于"对不上会静默失效"的说明。
        """
        if self.stage2_speaker_encoder == "stage1":
            if self.stage1_speaker is None:
                raise RuntimeError(
                    "Stage 2 的 checkpoint 说是用 Stage 1 微调过的 E_spk 训的，"
                    "但 Stage 1 权重没加载 —— 先 load_stage1()。"
                )
            return self.stage1_speaker.embed_numpy(wav)
        if self.speaker is None:
            raise RuntimeError("未加载说话人编码器，无法提取嵌入。")
        return self.speaker.embed(wav, 16000)

    def route_encoder_space(self, route: str = "clone") -> str:
        """这条路线**实际**用的是哪个 E_spk 空间（"stage1_finetuned" / "upstream"）。

        ★ 双口径评测要靠它算出「另一个空间」是谁 ——
        同一批音频在两个空间里的余弦尺度完全不同（微调后已塌缩，见
        `docs/notes/member_b_runbook.md` §1.4），报数时必须写清算的哪一套。
        """
        if route == "stage2":
            return ("stage1_finetuned" if self.stage2_speaker_encoder == "stage1"
                    else "upstream")
        if route == "stage1" and self.stage1_speaker is not None:
            return "stage1_finetuned"
        return "upstream"

    def embed_in_space(self, wav: np.ndarray, space: str) -> np.ndarray:
        """波形 -> (256,)，**按指定的 E_spk 空间**算（只做度量，不改生成链路）。"""
        if space == "stage1_finetuned":
            if self.stage1_speaker is None:
                raise RuntimeError(
                    "要算「Stage 1 微调后 E_spk」口径，但 Stage 1 权重没加载 —— "
                    "用 --stage1-checkpoint 指定。")
            return self.stage1_speaker.embed_numpy(wav)
        if space != "upstream":
            raise ValueError(f"未知的 E_spk 空间：{space}")
        return self.embedding(wav)

    def embed_for_route(self, wav: np.ndarray, route: str = "clone") -> np.ndarray:
        """按**路线**取说话人嵌入 h。

        为什么不能随便取：不同路线用的 E_spk 不一样
        （stage1 路线用随 Stage 1 微调过的那个，stage2 路线用 checkpoint 指定的那个），
        而生成器 G 是在**某个 E_spk 的嵌入空间上**学出来的 ——
        用错 E_spk 算出来的 δ 是**离分布**的，防御会静默失效。

        所以：**谁要算 δ，就用这条路线自己的嵌入。**
        """
        if route == "stage2":
            # 走 _embed_for_stage2 —— 它的报错会点名
            # 「Stage 2 checkpoint 说用微调 E_spk，但 Stage 1 没加载」这种情况。
            return self._embed_for_stage2(wav)
        return self.embed_in_space(wav, self.route_encoder_space(route))

    def protect_stage2(self, wav: np.ndarray, alpha: float = config.STAGE2["alpha"],
                       delta: np.ndarray | None = None,
                       seed: int = config.SEED, verbose: bool = True) -> dict:
        """★★ 论文原版 + **训练好的生成器**：δ = G(h)。

        与 `protect_stage1` 的**唯一区别**是 δ 的来源：
          stage1 : δ = 随机高斯噪声（L1 占位，**没有防御效果**）
          stage2 : δ = G(h) —— Stage 2 训出来的方向，**这才是论文的防御**

        ★ `delta` 给了就用给的（调用方自己算好的，例如评测时"同一批 δ 换路线对比"），
          不给才用 G(h) 现算（E_spk 取 checkpoint 指定的那个，保证与训练一致）。

        其余步骤与论文 Figure 1 完全一致：
          1. h      = E_spk(x)              身份卡
          2. δ      = G(h)                  ★ 生成器
          3. h_adv  = h + α·δ               打疫苗
          4. x_adv  = D(c, h_adv)           重建原句，只换音色
          5. protected = V(x_adv)           过声码器成波形
          6. h_stolen = E_spk(protected)    攻击者从保护后语音重新提卡
          7. cloned    = V(D(c, h_stolen))  攻击者用偷到的卡重建
        """
        import torch

        if self.stage1 is None:
            raise RuntimeError(
                "Stage 2 路线需要 Stage 1 的重建器（D(c, h_adv)）—— 先调 load_stage1()。"
            )
        if self.stage2_generator is None and delta is None:
            if alpha == 0.0:
                # α=0（"无防御"基线）⇒ δ 取什么都一样，用零向量占位。
                # 这样"无防御基线"不需要 Stage 2 的 checkpoint 也能跑。
                delta = np.zeros(config.SPEAKER_DIM, dtype=np.float32)
            else:
                raise RuntimeError(
                    "Stage 2 生成器未加载 —— 先调 load_stage2()，"
                    "或显式传入 delta（α=0 的无防御基线不需要生成器）。"
                )

        device = next(self.stage1.parameters()).device

        # 1. 体检：波形 -> 身份卡 h（★ 用训练时那个 E_spk）
        with torch.no_grad():
            embedding = self._embed_for_stage2(wav)

            # 2. 扰动方向
            if delta is None:
                # 用 G(h) 现算 —— E_spk 与训练时一致（checkpoint 里写着）
                embedding_tensor = torch.from_numpy(
                    np.asarray(embedding, dtype=np.float32)).unsqueeze(0).to(device)
                delta_tensor = self.stage2_generator(embedding_tensor)
                delta = delta_tensor[0].cpu().numpy()
                delta_source = "G(h)"
            else:
                # 调用方自己给的（例如评测时"同一批 δ 换路线对比"）
                delta = np.asarray(delta, dtype=np.float32).reshape(-1)
                delta_source = "调用方给定"
                if float(np.linalg.norm(delta)) == 0.0:
                    # α=0 的"无防御"基线：δ 取零占位（乘上 α 还是零）
                    delta_source = "零向量（α=0 基线）"

            # G 的输出是**单位长度**的方向，α 才是强度旋钮（见 stage2/generator.py）
            embedding_adv = perturb_embedding(embedding, delta, alpha)

            # 3. 内容码 + 重建
            from stage1.model import pad_mel_to_freq

            mel_np, original_frames = pad_mel_to_freq(audio_utils.model_mel(wav))
            mel = torch.from_numpy(mel_np).unsqueeze(0).to(device)      # (1, T', 80)
            h = torch.from_numpy(embedding).unsqueeze(0).to(device)
            h_adv = torch.from_numpy(embedding_adv).unsqueeze(0).to(device)

            _, protected_mel, codes = self.stage1(mel, h, h_adv)           # x_adv
            content = torch.stack(codes, dim=1)                            # (1, T', 64)

            # 4. 过声码器（先裁回原始帧数）
            protected = self.vocoder.mel_to_wav(
                protected_mel[0, :original_frames].cpu().numpy())

            # 5. 攻击者重新提卡（★ 必须用同一个 E_spk，否则度量不可比）
            stolen = self._embed_for_stage2(protected)

            # 6. 攻击者用偷到的卡重建（同一份内容码，只换音色）
            h_stolen = torch.from_numpy(stolen).unsqueeze(0).to(device)
            _, cloned_mel, _ = self.stage1(mel, h_stolen, h_stolen)
            cloned = self.vocoder.mel_to_wav(
                cloned_mel[0, :original_frames].cpu().numpy())

            # 对照：未防御的重建 D(c, h)
            _, baseline_mel, _ = self.stage1(mel, h, h)
            baseline = self.vocoder.mel_to_wav(
                baseline_mel[0, :original_frames].cpu().numpy())

        if verbose:
            print(f"  [1] 身份卡 h        {tuple(embedding.shape)}  "
                  f"L2 范数 {float(np.linalg.norm(embedding)):.4f}")
            print(f"  [2] 扰动 δ          {tuple(delta.shape)}  "
                  f"L2 范数 {float(np.linalg.norm(delta)):.4f}"
                  f"{'（生成器输出应为 1）' if delta_source == 'G(h)' else ''}"
                  f"  与原卡余弦 {float(np.dot(embedding, delta)):.4f}"
                  f"  来源：{delta_source}")
            print(f"  [3] 扰动后身份卡    与原卡余弦 {float(np.dot(embedding, embedding_adv)):.4f}"
                  f"（α={alpha}）")
            print(f"  [4] 保护后语音      {tuple(protected.shape)}  "
                  f"RMS {audio_utils.rms_energy(protected):.5f}")
            print(f"  [5] 攻击者重新提的卡 与原卡余弦 {float(np.dot(embedding, stolen)):.4f}"
                  f"（越小 -> 防御越有效；论文用 Sim_clone 衡量）")
            print(f"  [6] 未防御重建 D(c,h) {tuple(baseline.shape)}  "
                  f"RMS {audio_utils.rms_energy(baseline):.5f}")
            print(f"  [7] 防御后重建      {tuple(cloned.shape)}  "
                  f"RMS {audio_utils.rms_energy(cloned):.5f}")

        return {
            "embedding": embedding,
            "embedding_adv": embedding_adv,
            "delta": delta,
            "delta_source": delta_source,
            "stolen_embedding": stolen,
            "protected_waveform": protected,
            "clone_baseline_waveform": baseline,
            "clone_protected_waveform": cloned,
            "shapes": {
                "embedding": list(embedding.shape),
                "embedding_adv": list(embedding_adv.shape),
                "delta": list(delta.shape),
                "stolen_embedding": list(stolen.shape),
                "protected_waveform": list(protected.shape),
            },
            "route": "stage2",
        }

    def protect_stage1(self, wav: np.ndarray, alpha: float = config.STAGE2["alpha"],
                       delta: np.ndarray | None = None,
                       seed: int = config.SEED, verbose: bool = True) -> dict:
        """★ 论文原版：x_adv = D(c, h_adv) -> HiFi-GAN。

        步骤（与论文 Figure 1 的 Stage 2 一一对应）：
          1. c = E_c(x, h)              内容码（说的是什么）
          2. h_adv = h + α·δ            被污染的音色
          3. x_adv = D(c, h_adv)        重建原句，只换音色
          4. protected = V(x_adv)       过声码器成波形
          5. h_stolen = E_spk(protected) 攻击者从保护后语音重新提卡
          6. cloned = V(D(c, h_stolen)) 攻击者用偷到的卡重建
             （★ 与 x_adv 用**同一份内容码 c**，只有音色不同 ——
               这样两者的差异就**纯粹是防御效果**，比随机换内容更干净）
        """
        import torch

        if self.stage1 is None:
            raise RuntimeError("Stage 1 未加载 —— 先调 load_stage1()。")

        if delta is None:
            delta = random_perturbation(seed=seed)

        device = next(self.stage1.parameters()).device
        sample_rate = config.MEL_PRESETS[config.DEFAULT_PRESET]["sample_rate"]

        # 1. 体检：波形 -> 身份卡 h（用的是随 Stage 1 微调过的 E_spk）
        with torch.no_grad():
            embedding = self.stage1_speaker.embed_numpy(wav)
            steps = {"embedding": embedding.shape}

            # 2. 打疫苗
            embedding_adv = perturb_embedding(embedding, delta, alpha)
            steps["embedding_adv"] = embedding_adv.shape

            # 3. 内容码 + 重建
            # ★ 帧数必须补到 freq 的整数倍（编码器要取 index+freq-1），
            #   补完再把输出**裁回原始帧数** —— 保证"输入输出等长"（契约 §2）
            from stage1.model import pad_mel_to_freq

            mel_np, original_frames = pad_mel_to_freq(audio_utils.model_mel(wav))
            mel = torch.from_numpy(mel_np).unsqueeze(0).to(device)      # (1, T', 80)
            h = torch.from_numpy(embedding).unsqueeze(0).to(device)
            h_adv = torch.from_numpy(embedding_adv).unsqueeze(0).to(device)

            _, protected_mel, codes = self.stage1(mel, h, h_adv)           # x_adv
            content = torch.stack(codes, dim=1)                            # (1, T', 64)

            # 4. 过声码器（先裁回原始帧数）
            protected = self.vocoder.mel_to_wav(
                protected_mel[0, :original_frames].cpu().numpy())
            steps["protected_waveform"] = protected.shape

            # 5. 攻击者重新提卡
            stolen = self.stage1_speaker.embed_numpy(protected)
            steps["stolen_embedding"] = stolen.shape

            # 6. 攻击者用偷到的卡重建（同一份内容码，只换音色）
            h_stolen = torch.from_numpy(stolen).unsqueeze(0).to(device)
            _, cloned_mel, _ = self.stage1(mel, h_stolen, h_stolen)
            cloned = self.vocoder.mel_to_wav(
                cloned_mel[0, :original_frames].cpu().numpy())

            # 对照：未防御的重建 D(c, h)
            _, baseline_mel, _ = self.stage1(mel, h, h)
            baseline = self.vocoder.mel_to_wav(
                baseline_mel[0, :original_frames].cpu().numpy())

        if verbose:
            print(f"  [1] 身份卡 h        {tuple(embedding.shape)}  "
                  f"L2 范数 {float(np.linalg.norm(embedding)):.4f}")
            print(f"  [2] 扰动后身份卡    {tuple(embedding_adv.shape)}  "
                  f"与原卡余弦 {float(np.dot(embedding, embedding_adv)):.4f}（α={alpha}）")
            print(f"  [3] 内容码 c        {tuple(content.shape)}（时间分辨率 1/{config.CONTENT_DOWNSAMPLE}）")
            print(f"  [4] 保护后语音      {tuple(protected.shape)}  "
                  f"RMS {audio_utils.rms_energy(protected):.5f}  ★D(c,h_adv)→HiFi-GAN")
            print(f"  [5] 攻击者重新提的卡 {tuple(stolen.shape)}  "
                  f"与原卡余弦 {float(np.dot(embedding, stolen)):.4f}")
            print(f"      （越小 -> 防御越有效；论文用 Sim_clone 衡量）")
            print(f"  [6] 未防御重建 D(c,h) {tuple(baseline.shape)}  "
                  f"RMS {audio_utils.rms_energy(baseline):.5f}")
            print(f"  [7] 防御后重建      {tuple(cloned.shape)}  "
                  f"RMS {audio_utils.rms_energy(cloned):.5f}")

        result = {
            "embedding": embedding,
            "embedding_adv": embedding_adv,
            "stolen_embedding": stolen,
            "protected_waveform": protected,
            "clone_baseline_waveform": baseline,
            "clone_protected_waveform": cloned,
            "shapes": {key: list(value) for key, value in steps.items()},
            "route": "stage1",
        }
        return result

    # -- 单步，便于逐步打印与调试 ----------------------------------------
    def to_mel(self, wav: np.ndarray) -> np.ndarray:
        return self.spec.mel(wav)

    def embedding(self, wav: np.ndarray) -> np.ndarray:
        """波形 -> 说话人嵌入 h (256,)。"""
        if self.speaker is None:
            raise RuntimeError("未加载说话人编码器，无法提取嵌入。")
        return self.speaker.embed(wav, 16000)

    def protect_mel(self, mel: np.ndarray, embedding: np.ndarray,
                    delta: np.ndarray, alpha: float) -> np.ndarray:
        """AutoVC 路线：x_adv = D(c, h_adv)。需要 AutoVC 权重（Stage 1 之后才可用）。"""
        from bridge_autovc import AutoVCBridge

        if getattr(self, "_autovc", None) is None:
            self._autovc = AutoVCBridge(device=self.device)
        embedding_adv = perturb_embedding(embedding, delta, alpha)
        return self._autovc.protect(mel, embedding, embedding_adv)

    # -- 端到端 ----------------------------------------------------------
    def protect(self, wav: np.ndarray, alpha: float = config.STAGE2["alpha"],
                delta: np.ndarray | None = None, seed: int = config.SEED,
                text: str = DEFAULT_REFERENCE_TEXT,
                attack_mode: str = "clone", verbose: bool = True,
                mode: str = "clone") -> dict:
        """一整条链：波形 -> 保护后波形（+ 攻击者克隆对比）。

        参数
        ----
        wav         : 原始波形（16 kHz 单声道）
        alpha       : 扰动强度 α
        text        : 攻击者要念的文本（参考文本）
        attack_mode : "clone" 时额外产出攻击者克隆音；"none" 只产出保护后语音
        mode        : ★ 路线选择
                     "clone"  —— 用克隆机重新合成（默认；Sim_prot 上限 ≈0.82）
                     "stage1" —— ★ 论文原版 D(c, h_adv) → HiFi-GAN（δ 是随机噪声）
                     "stage2" —— ★★ 论文原版 + **训练好的 G**（δ = G(h)，这才是防御）
                     "auto"   —— 有 Stage 2 权重就用 stage2，其次 stage1，最后 clone

        返回 dict，含中间每一步，便于逐步核对形状。
        """
        if mode == "auto":
            if self.has_stage2:
                mode = "stage2"
            elif self.has_stage1:
                mode = "stage1"
            else:
                mode = "clone"
        if mode == "stage2":
            return self.protect_stage2(wav, alpha=alpha, delta=delta,
                                       seed=seed, verbose=verbose)
        if mode == "stage1":
            return self.protect_stage1(wav, alpha=alpha, delta=delta,
                                       seed=seed, verbose=verbose)

        if delta is None:
            delta = random_perturbation(seed=seed)

        steps: dict[str, tuple] = {}

        # 1. 体检：波形 -> 身份卡
        embedding = self.embedding(wav)
        steps["embedding"] = embedding.shape
        if verbose:
            print(f"  [1] 身份卡 h        {tuple(embedding.shape)}  "
                  f"L2 范数 {float(np.linalg.norm(embedding)):.4f}")

        # 2. 打疫苗
        embedding_adv = perturb_embedding(embedding, delta, alpha)
        steps["embedding_adv"] = embedding_adv.shape
        if verbose:
            similarity = float(np.dot(embedding, embedding_adv))
            print(f"  [2] 扰动后身份卡    {tuple(embedding_adv.shape)}  "
                  f"与原卡的余弦相似度 {similarity:.4f}（α={alpha}）")

        # 3. 用被污染的音色念同一句话 -> 保护后语音
        protected = self.cloner.clone(text, embedding_adv)
        steps["protected_waveform"] = protected.shape
        if verbose:
            print(f"  [3] 保护后语音      {tuple(protected.shape)}  "
                  f"RMS {audio_utils.rms_energy(protected):.5f}")

        result = {
            "embedding": embedding,
            "embedding_adv": embedding_adv,
            "protected_waveform": protected,
            "shapes": {key: list(value) for key, value in steps.items()},
        }

        # 4. 攻击者行骗：用原始卡 vs 用保护后语音重新提的卡
        if attack_mode == "clone":
            baseline = self.cloner.clone(text, embedding)
            stolen = self.embedding(protected)
            cloned = self.cloner.clone(text, stolen)
            result["clone_baseline_waveform"] = baseline
            result["clone_protected_waveform"] = cloned
            result["stolen_embedding"] = stolen
            steps["clone_baseline_waveform"] = baseline.shape
            steps["clone_protected_waveform"] = cloned.shape
            if verbose:
                print(f"  [4] 未防御克隆音    {tuple(baseline.shape)}  "
                      f"RMS {audio_utils.rms_energy(baseline):.5f}")
                print(f"  [5] 防御后克隆音    {tuple(cloned.shape)}  "
                      f"RMS {audio_utils.rms_energy(cloned):.5f}")
                print(f"      攻击者从保护后语音重新提的卡与原卡相似度 = "
                      f"{float(np.dot(embedding, stolen)):.4f}")
                print("      （越小 -> 防御越有效；论文用 Sim_clone 衡量）")

        result["shapes"] = {key: list(value) for key, value in steps.items()}
        return result


# ---------------------------------------------------------------------------
# 模块级接口（成员 B 的 scripts/make_sample.py 按约定一调用）
# ---------------------------------------------------------------------------
def protect(wav_path: str | Path, alpha: float = config.STAGE2["alpha"],
            output_path: str | Path | None = None,
            text: str = DEFAULT_REFERENCE_TEXT,
            attack_mode: str = "clone",
            seed: int = config.SEED,
            sample_rate: int = 16000,
            mode: str = "clone",
            stage1_checkpoint: str | Path | None = None,
            stage2_checkpoint: str | Path | None = None) -> dict:
    """一站式保护：读音频 -> 保护 -> 写音频。

    参数
    ----
    wav_path    : 输入音频路径
    alpha       : 扰动强度 α
    output_path : 保护后音频的输出路径（None 时不写文件）
    text        : 攻击者要念的参考文本
    attack_mode : "clone" 时额外把攻击者克隆音写到同目录
    mode        : ★ 路线选择 —— "clone"（默认）/ "stage1"（论文原版，δ 随机）
                  / "stage2"（论文原版 + 训练好的 G）/ "auto"
    stage1_checkpoint : Stage 1 的权重（stage1 / stage2 路线需要）
    stage2_checkpoint : Stage 2 的权重（stage2 路线需要）

    返回
    ----
    dict，至少含 'protected_path' 与 'shapes'（成员 B 的脚本依赖这两个键）。
    """
    wav_path = Path(wav_path)
    sample_rate = sample_rate or config.MEL_PRESETS[config.DEFAULT_PRESET]["sample_rate"]
    wav = audio_utils.load_audio(wav_path, sample_rate)

    protector = Protector(stage1_checkpoint=stage1_checkpoint,
                          stage2_checkpoint=stage2_checkpoint)
    result = protector.protect(
        wav, alpha=alpha, seed=seed, text=text, attack_mode=attack_mode,
        verbose=False, mode=mode,
    )

    written: dict[str, str] = {}
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_utils.save_audio(output_path, result["protected_waveform"], sample_rate)
        written["protected_path"] = str(output_path)

        if attack_mode == "clone":
            baseline_path = output_path.with_name(output_path.stem + "_clone_baseline.wav")
            protected_path = output_path.with_name(output_path.stem + "_clone_protected.wav")
            audio_utils.save_audio(baseline_path, result["clone_baseline_waveform"], sample_rate)
            audio_utils.save_audio(protected_path, result["clone_protected_waveform"], sample_rate)
            written["clone_baseline_path"] = str(baseline_path)
            written["clone_protected_path"] = str(protected_path)

    result.update(written)
    result["input_path"] = str(wav_path)
    result["alpha"] = alpha
    result["text"] = text
    return result


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="声纹保护推理（clone=克隆机路线 / stage1=论文原版 / stage2=论文原版+训练好的 G）")
    parser.add_argument("audio", help="输入音频路径（wav）")
    parser.add_argument("--alpha", type=float, default=config.STAGE2["alpha"],
                        help=f"扰动强度 α，默认 {config.STAGE2['alpha']}")
    parser.add_argument("--seed", type=int, default=config.SEED, help="随机种子")
    parser.add_argument("--text", default=DEFAULT_REFERENCE_TEXT, help="攻击者要念的参考文本")
    parser.add_argument("--output", default=None, help="输出 wav 路径")
    parser.add_argument("--report", default=None, help="输出 JSON 报告路径")
    parser.add_argument("--no-clone", action="store_true", help="不产出攻击者克隆音")
    parser.add_argument("--mode", choices=("clone", "stage1", "stage2", "auto"), default="clone",
                        help="推理路线（默认 clone；stage2 需要 Stage 1 + Stage 2 两份权重）")
    parser.add_argument("--stage1-checkpoint", default=None,
                        help="Stage 1 权重路径（stage1 / stage2 路线需要）")
    parser.add_argument("--stage2-checkpoint", default=None,
                        help="Stage 2 权重路径（stage2 路线需要；默认 weights/stage2/stage2_last.pt）")
    args = parser.parse_args()

    print("=" * 66)
    print(f"  声纹保护推理 · 路线 {args.mode}")
    print("=" * 66)
    print(paths.describe())
    print()

    wav = audio_utils.load_audio(args.audio, 16000)
    print(f"输入: {args.audio}  时长 {audio_utils.duration_seconds(wav, 16000):.2f} 秒")
    print(f"参考文本: {args.text}")
    print()

    protector = Protector(stage1_checkpoint=args.stage1_checkpoint,
                          stage2_checkpoint=args.stage2_checkpoint)
    print("【推理链】")
    result = protector.protect(
        wav, alpha=args.alpha, seed=args.seed, text=args.text,
        attack_mode="none" if args.no_clone else "clone",
        mode=args.mode,
    )

    output = Path(args.output) if args.output else Path(args.audio).with_name(
        Path(args.audio).stem + "_protected.wav"
    )
    audio_utils.save_audio(output, result["protected_waveform"], 16000)
    print(f"\n已保存保护后音频: {output}")
    if "clone_baseline_waveform" in result:
        baseline = output.with_name(output.stem + "_clone_baseline.wav")
        protected = output.with_name(output.stem + "_clone_protected.wav")
        audio_utils.save_audio(baseline, result["clone_baseline_waveform"], 16000)
        audio_utils.save_audio(protected, result["clone_protected_waveform"], 16000)
        print(f"已保存未防御克隆音: {baseline}")
        print(f"已保存防御后克隆音: {protected}")

    if args.report:
        report = {
            "alpha": args.alpha,
            "seed": args.seed,
            "text": args.text,
            "input": str(args.audio),
            "shapes": result["shapes"],
            "note": "L1 阶段使用随机高斯扰动占位；Stage 2 训练后替换为生成器 G(h)。",
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        print(f"已保存报告: {args.report}")
    return 0


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
