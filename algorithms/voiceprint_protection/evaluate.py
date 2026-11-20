"""评测脚本 —— Sim_prot / Sim_clone / DSR / WER（任务板 W5–W6，共同区）。

【论文 §3.1 的指标】
| 指标 | 定义 | 论文报告 |
|---|---|---|
| **Sim_prot** ↑ | 保护后语音与原始语音的说话人相似度 cos(E_spk(x_adv), E_spk(x)) | 0.95 |
| **Sim_clone** ↓ | 克隆音与原始语音的说话人相似度 cos(E_spk(clone), E_spk(x)) | 0.13 |
| **DSR** ↑ | Sim_clone < τ 的样本比例（τ = 0.3 / 0.4 / 0.5） | 79.8 / 83.5 / 87.2 |
| **WER** ↓ | 保护后语音的转写错误率（Whisper-large） | 5.30 |
| **MOS** ↑ | 人工听测（50 人） | 4.18 |

【★ Sim_clone 的两种算法，差别很大】
本项目里有两处提到"Sim_clone"，**含义不同**，别混：

1. **论文口径**（本脚本的主指标）
       Sim_clone = cos(E_spk(V(x_clone)), E_spk(V(x)))
   —— "攻击者克隆出来的**声音**有多像本人"。
2. **卡片口径**（`docs/interface-contract.md` §4.3 写的那个）
       cos(h, h_stolen)   其中 h_stolen = E_spk(保护后语音)
   —— "攻击者从保护后语音里重新提的**卡**有多像本人的卡"。

两者的关系：`cos(h, h_stolen)` 就是 **Sim_prot**（保护后语音 vs 原语音）。
所以本脚本里：
    Sim_prot  = cos(h, h_stolen)
    Sim_clone = cos(E_spk(clone_protected), h)     ← 论文口径

【怎么跑】
    # 1) 无防御基线 + 高斯噪声基线（不需要 Stage 2）
    python algorithms/voiceprint_protection/evaluate.py --mode none
    python algorithms/voiceprint_protection/evaluate.py --mode random

    # 2) 训练好的生成器
    python algorithms/voiceprint_protection/evaluate.py \
        --mode trained --generator weights/stage2/stage2_last.pt

    # 3) 加算 WER（需要 faster-whisper，会下载模型）
    python algorithms/voiceprint_protection/evaluate.py --mode trained --wer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

try:
    import numpy as np  # noqa: E402

    import audio_utils  # noqa: E402
    import config  # noqa: E402
    import paths  # noqa: E402
except ImportError as error:  # 依赖缺失（多半是忘了 conda activate vpp）
    import paths as _paths  # noqa: E402

    _paths.print_env_hint(error)
    raise SystemExit(1) from error


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """两个向量的余弦相似度（都已 L2 归一化，点积即可）。"""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


def defense_success_rate(similarities: list[float], tau: float) -> float:
    """DSR = Sim_clone < τ 的样本比例。"""
    if not similarities:
        return 0.0
    return float(np.mean([s < tau for s in similarities]))


def summarize(values: list[float], name: str) -> dict:
    """一组指标 → {均值, 标准差, 中位数}。"""
    if not values:
        return {f"{name}_mean": 0.0, f"{name}_std": 0.0, f"{name}_median": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{name}_mean": float(array.mean()),
        f"{name}_std": float(array.std()),
        f"{name}_median": float(np.median(array)),
    }


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------
class WhisperScorer:
    """用 faster-whisper 算 WER。

    ★ 为什么用 faster-whisper 而不是 openai-whisper
      openai-whisper 的 setup.py 无条件依赖 triton>=2.0.0，
      PyPI 上**没有 Windows 轮子** → Windows 上装不上。
      faster-whisper 是 CTranslate2 实现，转写结果与 openai-whisper 等价，
      而且论文用的正是 Whisper-large（`config.EVAL["whisper_model"] = "large-v3"`）。

    ★ 参考文本从哪来
      优先用 VCTK 自带的转写（`--transcript-dir`）；
      没有的话就用 Whisper 转写**原始**语音当参考（相对 WER），
      这时算出来的是"保护让转写变差多少"，不是绝对 WER —— 报告里要写清楚。
    """

    def __init__(self, model_name: str | None = None, device: str = "cpu",
                 compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self.model_name = model_name or config.EVAL["whisper_model"]
        print(f"加载 Whisper 模型：{self.model_name}（{device}/{compute_type}）…")
        self.model = WhisperModel(self.model_name, device=device, compute_type=compute_type)

    def transcribe(self, wav: np.ndarray, sample_rate: int) -> str:
        segments, _ = self.model.transcribe(wav, language="en", beam_size=1)
        return " ".join(segment.text.strip() for segment in segments).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """词错误率（编辑距离 / 参考词数）。

    自己实现是为了避免多装一个 jiwer；算法是标准的 Levenshtein 对齐。
    """
    reference_words = reference.lower().split()
    hypothesis_words = hypothesis.lower().split()
    if not reference_words:
        return 0.0

    # 编辑距离（滚动数组）
    previous = list(range(len(hypothesis_words) + 1))
    for i, ref_word in enumerate(reference_words, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis_words, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1] / len(reference_words)


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_test_items(split: str = "test", limit_per_speaker: int | None = None,
                    data_root: Path | None = None) -> list[tuple[Path, str]]:
    """读测试集清单，返回 [(音频路径, 说话人)]。

    `limit_per_speaker`：每个说话人取几条（论文是 100 条，课程作业可缩到 10）。
    ★ 取法是**在每个说话人的全部语句里均匀抽样**，不是取清单里的前 N 条 ——
      理由见下面代码处的注释（取前 N 条会全是同一句话）。
    """
    root = data_root or paths.data_root()
    manifest = root / f"{split}.txt"
    if not manifest.exists():
        raise SystemExit(
            f"找不到清单 {manifest}\n"
            "请先跑：python algorithms/voiceprint_protection/scripts/prepare_data.py"
        )

    rows: list[tuple[Path, str]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        path = root / parts[0]
        speaker = parts[1]
        if not path.exists():
            continue
        rows.append((path, speaker))

    if limit_per_speaker is None:
        return rows

    # ★ 每说话人**均匀抽样**，不是取前 N 条。
    #   清单是按文件名排序的，取前 N 条 => 每个说话人拿到的都是
    #   `pXXX_001` —— 而 VCTK 里 pXXX_001 恰好是同一句
    #   "Please call Stella."。20 个说话人 20 条样本全是一句话，
    #   WER 这类指标就没有代表性了。均匀抽样能覆盖整句话表。
    grouped: dict[str, list[tuple[Path, str]]] = {}
    for path, speaker in rows:
        grouped.setdefault(speaker, []).append((path, speaker))

    picked: list[tuple[Path, str]] = []
    for speaker in sorted(grouped):
        entries = grouped[speaker]
        if len(entries) <= limit_per_speaker:
            picked.extend(entries)
            continue
        for i in range(limit_per_speaker):
            picked.append(entries[(i * len(entries)) // limit_per_speaker])
    return picked


def load_transcripts(transcript_dir: Path | None) -> dict[str, str]:
    """读 VCTK 的转写文本（`<spk>-<utt>.txt`）。

    ★ 用 rglob **递归**找：VCTK 的 txt/ 是**按说话人分子目录**的
      （`txt/p225/p225_001.txt`），不是平铺在 txt/ 下面。
      以前用 glob 只扫顶层，一个都找不到 -> 绝对 WER 会**静默退化成**
      「Whisper 转写原始语音」的相对 WER，而且**不报错**。
    """
    transcripts: dict[str, str] = {}
    if transcript_dir is None or not transcript_dir.exists():
        return transcripts
    for path in transcript_dir.rglob("*.txt"):
        transcripts[path.stem] = path.read_text(encoding="utf-8", errors="ignore").strip()
    return transcripts


def transcript_key(path: Path) -> str:
    """音频文件名 -> VCTK 转写键。

    ★ 为什么不能直接用 `path.stem`
      我们把 VCTK 的**两路麦克风**都收进来了，转 16 kHz 时统一改名为
      `<spk>_<utt>_mic1.wav` / `<spk>_<utt>_mic2.wav`（见 prepare_data.py），
      而 VCTK 自带的转写只有**一路**，叫 `<spk>_<utt>.txt`。
      直接拿 `path.stem` 去查 -> **永远查不到** -> 绝对 WER 静默退化成相对 WER。
      所以要去掉 `_mic1` / `_mic2` 后缀（两路麦克风读的是同一句话，参考文本通用）。
    """
    stem = path.stem
    for suffix in ("_mic1", "_mic2"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


# ---------------------------------------------------------------------------
# 评测主流程
# ---------------------------------------------------------------------------
def evaluate(args) -> dict:
    import infer

    sample_rate = config.MEL_PRESETS[config.DEFAULT_PRESET]["sample_rate"]
    items = load_test_items(args.split, args.limit_per_speaker)
    if not items:
        raise SystemExit("测试集里没有可用样本")

    transcripts = load_transcripts(Path(args.transcript_dir) if args.transcript_dir else None)
    scorer = None
    if args.wer:
        scorer = WhisperScorer(model_name=args.whisper_model, device=args.whisper_device)

    print("=" * 68)
    print(f"  评测 · mode={args.mode}  split={args.split}  α={args.alpha}")
    print("=" * 68)
    print(f"  样本数        : {len(items)}")
    print(f"  说话人        : {len({s for _, s in items})}")
    print(f"  DSR 阈值      : {args.taus}")
    if scorer is not None:
        print(f"  WER 参考      : {'VCTK 原始转写（保护时念同一句 -> 绝对 WER）' if transcripts else 'Whisper 转写原始语音（相对 WER）'}")
    print()

    # -- 生成器（如果用训练好的）------------------------------------------
    generator = None
    if args.mode == "trained":
        if not args.generator_checkpoint:
            raise SystemExit("--mode trained 必须给 --generator-checkpoint")
        from stage2.generator import PerturbationGenerator

        checkpoint = Path(args.generator_checkpoint)
        if not checkpoint.exists():
            raise SystemExit(f"找不到 checkpoint：{checkpoint}")
        generator = PerturbationGenerator()
        state = torch_load(checkpoint)
        generator.load_state_dict(state["generator"])
        generator.eval()
        print(f"  生成器       : {checkpoint}（step {state.get('step', '?')}）")

    # -- 保护器（★ 整个评测只加载一次权重）--------------------------------
    print("\n加载模型（encoder + synthesizer + vocoder，只加载一次）…")
    protector = infer.Protector()
    alpha = 0.0 if args.mode == "none" else args.alpha
    if args.mode == "none":
        print("  mode=none：α=0，即不做任何防御（基线）")
    elif args.mode == "random":
        print("  mode=random：随机高斯扰动（论文 Table 1 的 Gaussian Noise 基线）")
    print()

    sim_prot_values: list[float] = []
    sim_clone_values: list[float] = []
    sim_clone_baseline_values: list[float] = []
    wer_values: list[float] = []
    wer_original_values: list[float] = []
    transcript_miss = 0
    per_speaker: dict[str, list[float]] = {}

    started = time.time()
    for index, (path, speaker) in enumerate(items):
        try:
            wav = audio_utils.load_audio(path, sample_rate)

            # -- 扰动：训练好的生成器 vs 随机高斯 --------------------------
            delta = None
            if generator is not None:
                import torch

                embedding = protector.embedding(wav)
                with torch.no_grad():
                    tensor = torch.from_numpy(embedding).float()[None, :]
                    delta = generator(tensor).numpy()[0]

            # ★ 先把参考文本取出来：**保护时念的就该是这一句**。
            #   protect() 默认念固定的 "The quick brown fox jumps over the lazy dog."，
            #   拿它跟「原句的转写」比，WER 恒等于 ~100%，一点意义都没有。
            #   改成念原句，转写出来才能和论文的 WER 对得上。
            vctk_reference = transcripts.get(transcript_key(path)) if transcripts else None
            clone_text = vctk_reference or infer.DEFAULT_REFERENCE_TEXT

            result = protector.protect(
                wav,
                alpha=alpha,
                delta=delta,
                seed=args.seed,
                text=clone_text,
                attack_mode="clone",
                verbose=False,
            )
        except Exception as error:  # noqa: BLE001
            print(f"  [跳过] {path.name}：{error}")
            continue

        embedding = result["embedding"]
        stolen = result["stolen_embedding"]
        protected = result["protected_waveform"]
        cloned = result["clone_protected_waveform"]
        baseline = result["clone_baseline_waveform"]

        # Sim_prot：保护后语音 vs 原始语音（stolen 就是 E_spk(protected)）
        sim_prot = cosine(stolen, embedding)

        # Sim_clone（论文口径）：攻击者克隆出来的**声音** vs 原始说话人
        sim_clone = cosine(_embed(cloned, sample_rate), embedding)

        # 未防御时攻击者克隆音 vs 原始说话人（应当很接近 1）
        sim_clone_baseline = cosine(_embed(baseline, sample_rate), embedding)

        sim_prot_values.append(sim_prot)
        sim_clone_values.append(sim_clone)
        sim_clone_baseline_values.append(sim_clone_baseline)
        per_speaker.setdefault(speaker, []).append(sim_clone)

        # -- WER ----------------------------------------------------------
        if scorer is not None:
            hypothesis = scorer.transcribe(protected, sample_rate)
            if vctk_reference is not None:
                # 绝对 WER：参考 = VCTK 自带的原始转写
                wer_values.append(word_error_rate(vctk_reference, hypothesis))
                # 原音的 WER = ASR 在这段语音上的**下限**（不可能比它更低）
                wer_original_values.append(
                    word_error_rate(vctk_reference, scorer.transcribe(wav, sample_rate))
                )
            else:
                # 没有原始转写 -> 只能用「Whisper 转写原音」当参考 = **相对** WER
                transcript_miss += 1
                reference = scorer.transcribe(wav, sample_rate)
                wer_values.append(word_error_rate(reference, hypothesis))

        if (index + 1) % 10 == 0 or index == len(items) - 1:
            print(f"  [{index + 1}/{len(items)}] "
                  f"Sim_prot={np.mean(sim_prot_values):.4f} "
                  f"Sim_clone={np.mean(sim_clone_values):.4f} "
                  f"[{time.time() - started:.0f}s]")

    if scorer is not None and transcripts and transcript_miss:
        print(f"  [注意] 有 {transcript_miss}/{len(items)} 条没找到 VCTK 原始转写，"
              "这些改用「Whisper 转写原始语音」当参考 —— **不是绝对 WER**。")

    # -- 汇总 --------------------------------------------------------------
    report = {
        "mode": args.mode,
        "split": args.split,
        "alpha": alpha,
        "seed": args.seed,
        "num_samples": len(sim_clone_values),
        "num_speakers": len(per_speaker),
        "sim_prot": summarize(sim_prot_values, "sim_prot"),
        "sim_clone": summarize(sim_clone_values, "sim_clone"),
        "sim_clone_no_defense": summarize(sim_clone_baseline_values, "sim_clone_no_defense"),
        "dsr": {f"tau_{tau}": defense_success_rate(sim_clone_values, tau)
                for tau in args.taus},
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if wer_values:
        report["wer"] = summarize(wer_values, "wer")
    if wer_original_values:
        report["wer_original"] = summarize(wer_original_values, "wer_original")

    print("\n" + "=" * 68)
    print(f"  结果（mode={args.mode}）")
    print("=" * 68)
    print(f"  Sim_prot            {report['sim_prot']['sim_prot_mean']:.4f}"
          f"  （论文 0.95，越高越好）")
    print(f"  Sim_clone           {report['sim_clone']['sim_clone_mean']:.4f}"
          f"  （论文 0.13，越低越好）")
    print(f"  Sim_clone(未防御克隆) {report['sim_clone_no_defense']['sim_clone_no_defense_mean']:.4f}"
          f"  （应当接近 1）")
    for tau in args.taus:
        print(f"  DSR(τ={tau})          {report['dsr'][f'tau_{tau}'] * 100:.1f}%"
              f"  （论文 79.8 / 83.5 / 87.2）")
    if wer_values:
        print(f"  WER(保护后)         {report['wer']['wer_mean'] * 100:.2f}%"
              f"  （论文 5.30%）")
    if wer_original_values:
        print(f"  WER(原音·ASR 下限)  {report['wer_original']['wer_original_mean'] * 100:.2f}%"
              f"  （原始语音的转写错误率 —— 防御不可能比它更低）")
    print()

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  报告已保存：{output}")
    return report


def torch_load(path: Path):
    import torch

    return torch.load(path, map_location="cpu")


_EMBEDDER = None


def _embed(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    """算一条波形的说话人嵌入（进程内缓存一个编码器）。"""
    global _EMBEDDER
    if _EMBEDDER is None:
        from bridge_speaker import SpeakerBridge

        _EMBEDDER = SpeakerBridge(device="cpu")
    return _EMBEDDER.embed(wav, sample_rate)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="声纹保护评测（Sim_prot / Sim_clone / DSR / WER）")
    parser.add_argument("--mode", choices=("none", "random", "trained"), default="random",
                        help="none=无防御基线，random=高斯噪声基线，trained=训练好的生成器")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--limit-per-speaker", type=int,
                        default=config.EVAL["utterances_per_speaker"],
                        help="每个说话人取几条（论文 100，作业可缩到 10）")
    parser.add_argument("--alpha", type=float, default=config.STAGE2["alpha"])
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--taus", type=float, nargs="+", default=list(config.EVAL["taus"]))
    parser.add_argument("--generator-checkpoint", default=None)
    parser.add_argument("--wer", action="store_true", help="算 WER（需要 faster-whisper）")
    parser.add_argument("--whisper-device", default="cpu")
    parser.add_argument("--whisper-model", default=None,
                        help="Whisper 模型名（默认 config.EVAL['whisper_model']，即 large-v3）。CPU 上想省时间可换 small/medium，但要写明偏离了论文设定")
    parser.add_argument("--transcript-dir", default=None,
                        help="VCTK 转写文本目录（有它才能算绝对 WER）")
    parser.add_argument("--output", default=None, help="报告 JSON 的输出路径")
    args = parser.parse_args()

    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
