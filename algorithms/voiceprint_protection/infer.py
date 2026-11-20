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


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
class Protector:
    """声纹保护器：把"体检 + 打疫苗 + 骗子行骗"封装成一个对象。

    三件套（都是上游 Real-Time-Voice-Cloning，权重全部可下载）：
      · speaker  —— 体检仪：波形 -> 256 维身份卡
      · cloner   —— 骗子：(文本, 身份卡) -> 音频
      · stage1   —— ★ 可选：**论文原版**的重建器（成员 A 训的 Stage 1 + HiFi-GAN）

    【两条路线，差别很大】
    | 路线 | 保护后语音怎么来的 | Sim_prot 上限 |
    |---|---|---|
    | "clone"（路线 B，默认） | `clone(text, h_adv)` —— 用克隆机**重新合成** | **≈0.82**（重新合成本身就改音色/韵律） |
    | "stage1"（★ 论文原版） | `D(c, h_adv)` —— **重建原句**，只换音色 | 论文 0.95 |

    成员 B 在 `docs/experiments.md §3c` 实测过：路线 B 的 Sim_prot 有 **0.82 的
    结构性天花板**，与扰动强弱无关 —— 论文那组数字**必须走 stage1 路线**。
    """

    def __init__(self, use_speaker_bridge: bool = True, device: str | None = None,
                 with_cloner: bool = True, stage1_checkpoint: str | Path | None = None,
                 with_stage1: bool = True):
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
                encoder.model.load_state_dict(state["speaker_encoder"])
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
                     "stage1" —— ★ 论文原版 D(c, h_adv) → HiFi-GAN
                     "auto"   —— 有 Stage 1 权重就用 stage1，否则退回 clone

        返回 dict，含中间每一步，便于逐步核对形状。
        """
        if mode == "auto":
            mode = "stage1" if self.has_stage1 else "clone"
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
            mode: str = "clone") -> dict:
    """一站式保护：读音频 -> 保护 -> 写音频。

    参数
    ----
    wav_path    : 输入音频路径
    alpha       : 扰动强度 α
    output_path : 保护后音频的输出路径（None 时不写文件）
    text        : 攻击者要念的参考文本
    attack_mode : "clone" 时额外把攻击者克隆音写到同目录
    mode        : ★ 路线选择 —— "clone"（默认）/ "stage1"（论文原版）/ "auto"

    返回
    ----
    dict，至少含 'protected_path' 与 'shapes'（成员 B 的脚本依赖这两个键）。
    """
    wav_path = Path(wav_path)
    sample_rate = sample_rate or config.MEL_PRESETS[config.DEFAULT_PRESET]["sample_rate"]
    wav = audio_utils.load_audio(wav_path, sample_rate)

    protector = Protector()
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

    parser = argparse.ArgumentParser(description="声纹保护推理（L1：随机扰动占位，未训练生成器 G）")
    parser.add_argument("audio", help="输入音频路径（wav）")
    parser.add_argument("--alpha", type=float, default=config.STAGE2["alpha"],
                        help=f"扰动强度 α，默认 {config.STAGE2['alpha']}")
    parser.add_argument("--seed", type=int, default=config.SEED, help="随机种子")
    parser.add_argument("--text", default=DEFAULT_REFERENCE_TEXT, help="攻击者要念的参考文本")
    parser.add_argument("--output", default=None, help="输出 wav 路径")
    parser.add_argument("--report", default=None, help="输出 JSON 报告路径")
    parser.add_argument("--no-clone", action="store_true", help="不产出攻击者克隆音")
    args = parser.parse_args()

    print("=" * 66)
    print("  声纹保护推理 · L1（随机扰动占位，未训练生成器 G）")
    print("=" * 66)
    print(paths.describe())
    print()

    wav = audio_utils.load_audio(args.audio, 16000)
    print(f"输入: {args.audio}  时长 {audio_utils.duration_seconds(wav, 16000):.2f} 秒")
    print(f"参考文本: {args.text}")
    print()

    protector = Protector()
    print("【推理链】")
    result = protector.protect(
        wav, alpha=args.alpha, seed=args.seed, text=args.text,
        attack_mode="none" if args.no_clone else "clone",
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
    raise SystemExit(main())
