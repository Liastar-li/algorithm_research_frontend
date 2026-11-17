"""L1 冒烟测试 —— 一条命令跑完整条推理链。

用法：
    conda activate vpp
    python algorithms/voiceprint_protection/scripts/smoke_test.py --audio path/to/input.wav

它做四件事：
  0. 检查三个权重是否就位（缺了就明确告诉你去哪儿下）
  1. 单独测 **GE2E 说话人分支**（只需 `encoder.pt`）—— 论文路线的核心
  2. 跑通"重建"（不加扰动）—— 这验证 AutoVC 权重 + 我们的 mel 是否配套
  3. 跑通"保护"（加随机扰动）—— 这验证整条链

【为什么要分 2 和 3 两步】
重建输出是噪音 -> 问题在 mel 或权重（跟保护逻辑无关）
重建正常但保护后是噪音 -> 问题在扰动或解码路径
分两步测，能立刻定位是哪一环坏了。

【为什么第 1 步单独拎出来】
论文路线的说话人嵌入来自 **上游 GE2E**（`encoder.pt`，17 MB），
而第 2/3 步要的是 **AutoVC 全家桶**（`autovc.ckpt` / `3000000-BL.ckpt` / WaveNet 声码器）。
两套权重独立，缺哪套就跳过对应步骤，**不要因为 AutoVC 权重没下就连说话人分支都测不了**。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import audio_utils  # noqa: E402
    import config  # noqa: E402
    import paths  # noqa: E402
except ImportError as error:  # 依赖缺失时给个能看懂的原因（多半是忘了 activate vpp）
    import paths as _paths  # noqa: E402

    _paths.print_env_hint(error)
    raise SystemExit(1) from error

METHOD_ID = "group_n005_voiceprint_protection"


def pick_input(explicit: str | None) -> Path | None:
    """挑一条测试音频：命令行 > 平台样例 > 清单里的第一条。"""
    if explicit:
        return Path(explicit)

    sample = paths.sample_dir(METHOD_ID) / "input.wav"
    if sample.exists():
        return sample

    manifest = paths.data_root() / "test.txt"
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if parts:
                    candidate = paths.data_root() / parts[0]
                    if candidate.exists():
                        return candidate
    return None


def check_speaker_branch(wav_path: Path | None, sample_rate: int) -> bool:
    """第 1 步：单独测 GE2E 说话人分支（只需 encoder.pt）。

    这是**论文路线**的核心：h 从哪来、长什么样、归一化了没有。
    """
    print("\n【1】GE2E 说话人分支（论文路线，只需 encoder.pt）")
    try:
        from bridge_speaker import SpeakerBridge, SpeakerBridgeError
    except ImportError as error:
        print(f"  [跳过] 无法导入 bridge_speaker：{error}")
        return False

    try:
        bridge = SpeakerBridge()
    except SpeakerBridgeError as error:
        print(f"  [跳过] {error}")
        return False

    info = bridge.describe()
    print(f"  上游布局  : {info.get('上游布局', '?')}")
    print(f"  权重      : {info.get('权重文件', '?')}")
    print(f"  采样率    : {info.get('采样率 (Hz)', '?')} Hz")
    print(f"  mel       : {info.get('mel 维数', '?')} 维，"
          f"n_fft/hop = {info.get('mel n_fft / hop', '?')}")
    print(f"  嵌入维度  : {info.get('嵌入维度', '?')}")

    if wav_path is None:
        print("  [提示] 没有可用的测试音频，跳过嵌入测试")
        return True

    embedding = bridge.embed_path(wav_path)
    norm = float(np.linalg.norm(embedding))
    print(f"  嵌入形状  : {tuple(embedding.shape)}  L2 范数 {norm:.6f}")
    if abs(norm - 1.0) > 1e-3:
        print("  [失败] 嵌入没有归一化到单位长度")
        return False
    print("  [OK] 嵌入已 L2 归一化")
    return True


def check_cloner_branch() -> bool:
    """第 2 步：测克隆机分支（合成器 + 声码器）。

    这是**论文 Figure 1 里那个"攻击者"**：给它一段文本 + 一张身份卡，
    它就念出那段话。没有它就没法演示"防御让克隆失效"。
    """
    print("\n【2】克隆机分支（合成器 + 声码器，即论文里的「攻击者」）")
    try:
        from bridge_cloner import ClonerBridge
    except ImportError as error:
        print(f"  [跳过] 无法导入 bridge_cloner：{error}")
        print("         上游合成器需要 Unidecode / inflect：")
        print("           pip install Unidecode==1.3.2 inflect==5.3.0")
        return False

    try:
        bridge = ClonerBridge()
    except Exception as error:  # noqa: BLE001
        print(f"  [跳过] {error}")
        return False

    for line in bridge.describe().splitlines():
        print(f"  {line}")
    print("  [OK] 克隆机就绪")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="L1 冒烟测试")
    parser.add_argument("--audio", default=None, help="输入音频（wav）。不给则用平台样例/清单第一条")
    parser.add_argument("--alpha", type=float, default=config.STAGE2["alpha"])
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--outdir", default=str(paths.MODULE_DIR / "results"))
    args = parser.parse_args()

    print("=" * 66)
    print("  L1 冒烟测试 · 声纹保护推理链")
    print("=" * 66)

    # -- 步骤 0：组件与权重 ----------------------------------------------
    print("\n【0】组件与权重检查")
    print(paths.describe())

    # 论文主线的三个权重（上游 RTC：encoder / synthesizer / vocoder）
    # 注意：download_* 脚本会把它们放到 <数据根>/models/default/ 下
    def _find_weight(name: str) -> Path | None:
        for candidate in (
            paths.models_dir() / "default" / name,
            paths.models_dir() / name,
            paths.RTC_DIR / name,
        ):
            if candidate.exists() and candidate.stat().st_size > 1024:
                return candidate
        return None

    missing = [name for name in ("encoder.pt", "synthesizer.pt", "vocoder.pt")
               if _find_weight(name) is None]
    if missing:
        print(f"\n  [注意] 缺少论文主线的权重：{', '.join(missing)}")
        print("         下载：")
        print("           python algorithms/voiceprint_protection/scripts/download_encoder_weights.py   # encoder.pt")
        print("           python algorithms/voiceprint_protection/scripts/download_rtc_weights.py      # synthesizer + vocoder")
        print("         → 第 1 步（GE2E）不受影响，第 2/3 步会跳过。")
    else:
        print("  [OK] 论文主线权重齐备（encoder / synthesizer / vocoder）")

    # AutoVC 全家桶（可选路线：Stage 1 之后才用得上）
    autovc_missing = [name for name in paths.WEIGHT_FILES.values()
                      if not paths.weight_path(name).exists()]
    if autovc_missing:
        print(f"  [提示] AutoVC 路线权重缺失：{', '.join(autovc_missing)}")
        print("         AutoVC 官方链接实测已 404 失效；该路线改为「Stage 1 的架构参考」，")
        print("         不影响当前 L1 演示。")

    # -- 准备输入 --------------------------------------------------------
    sample_rate = config.MEL_PRESETS[config.DEFAULT_PRESET]["sample_rate"]
    audio_path = pick_input(args.audio)
    if audio_path is not None:
        wav = audio_utils.load_audio(str(audio_path), sample_rate)
        print(f"\n输入: {audio_path}")
    else:
        audio_path = None
        print("\n[提示] 找不到可用的测试音频，使用一段合成信号"
              "（仅用于验证链路，不用于效果评估）")
        t = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
        wav = (0.25 * np.sin(2 * np.pi * 140 * t)
               + 0.12 * np.sin(2 * np.pi * 430 * t) * np.sin(2 * np.pi * 2.7 * t)).astype(np.float32)
    print(f"  时长 {audio_utils.duration_seconds(wav, sample_rate):.2f} 秒  "
          f"采样率 {sample_rate} Hz")

    # -- 步骤 1：GE2E 说话人分支 -----------------------------------------
    speaker_ok = check_speaker_branch(audio_path, sample_rate)

    if missing:
        print("\n" + "=" * 66)
        print("  论文主线权重不齐，已跳过克隆机与端到端两步。")
        print("  已验证：GE2E 说话人分支 =", "通过" if speaker_ok else "未通过")
        print("  补齐权重后重跑本脚本即可完成整链验证。")
        print("=" * 66)
        return 0 if speaker_ok else 1

    # -- 步骤 2：克隆机分支 -----------------------------------------------
    cloner_ok = check_cloner_branch()
    if not cloner_ok:
        print("\n" + "=" * 66)
        print("  克隆机权重不齐，无法验证整条链。")
        print("  下载：")
        print("    python algorithms/voiceprint_protection/scripts/download_rtc_weights.py")
        print("  已验证：GE2E 说话人分支 =", "通过" if speaker_ok else "未通过")
        print("=" * 66)
        return 0 if speaker_ok else 1

    # -- 步骤 3：端到端 --------------------------------------------------
    from infer import DEFAULT_REFERENCE_TEXT, Protector

    print("\n【3】端到端：体检 -> 打疫苗 -> 骗子行骗")
    print(f"  参考文本: {DEFAULT_REFERENCE_TEXT}")
    try:
        protector = Protector()
    except Exception as error:  # noqa: BLE001
        print(f"  [失败] 加载模型出错：{error}")
        return 1

    result = protector.protect(
        wav, alpha=args.alpha, seed=args.seed, text=DEFAULT_REFERENCE_TEXT, verbose=False
    )

    similarity = float(np.dot(result["embedding"], result["embedding_adv"]))
    protected_rms = audio_utils.rms_energy(result["protected_waveform"])
    baseline_rms = audio_utils.rms_energy(result["clone_baseline_waveform"])
    cloned_rms = audio_utils.rms_energy(result["clone_protected_waveform"])
    stolen_similarity = float(np.dot(result["embedding"], result["stolen_embedding"]))

    print(f"  α = {args.alpha}")
    print(f"  原始卡 vs 扰动卡 余弦相似度        {similarity:.4f}（1.0 = 没扰动；越小扰动越强）")
    print(f"  保护后语音 RMS                    {protected_rms:.5f}")
    print(f"  未防御克隆音 RMS                  {baseline_rms:.5f}")
    print(f"  防御后克隆音 RMS                  {cloned_rms:.5f}")
    print(f"  攻击者从保护后语音提的卡 vs 原卡   {stolen_similarity:.4f}")
    print("    （越小 -> 防御越有效；论文用 Sim_clone 衡量，目标是 < 0.5）")

    if protected_rms < 1e-4:
        print("  [失败] 保护后音频接近静音")
        return 1
    if baseline_rms < 1e-4:
        print("  [失败] 克隆机输出接近静音 —— 合成器/声码器可能没配套")
        return 1
    print("  [OK] 保护后语音与克隆音都有声")

    # -- 输出 ------------------------------------------------------------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_path).stem if audio_path else "synthetic"
    protected_path = outdir / f"{stem}_protected.wav"
    baseline_path = outdir / f"{stem}_clone_baseline.wav"
    cloned_path = outdir / f"{stem}_clone_protected.wav"
    audio_utils.save_audio(protected_path, result["protected_waveform"], sample_rate)
    audio_utils.save_audio(baseline_path, result["clone_baseline_waveform"], sample_rate)
    audio_utils.save_audio(cloned_path, result["clone_protected_waveform"], sample_rate)
    print(f"\n  已保存保护后语音     {protected_path}")
    print(f"  已保存未防御克隆音   {baseline_path}")
    print(f"  已保存防御后克隆音   {cloned_path}")

    report = {
        "alpha": args.alpha,
        "seed": args.seed,
        "shapes": result["shapes"],
        "embedding_cosine": similarity,
        "stolen_embedding_cosine": stolen_similarity,
        "rms_protected": protected_rms,
        "rms_clone_baseline": baseline_rms,
        "rms_clone_protected": cloned_rms,
        "note": "L1 阶段：δ 为随机高斯噪声占位，尚未训练生成器 G。"
                "stolen_embedding_cosine 即论文的 Sim_clone；随机扰动下它不会很低，"
                "要等 Stage 2 训练出生成器 G 才会显著下降（论文报告 0.13）。",
    }
    report_path = outdir / f"{stem}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已保存报告         {report_path}")

    print("\n" + "=" * 66)
    print("  冒烟测试通过。")
    print("  ★ 听一下这三个文件：")
    print("      protected.wav       —— 保护后的语音（人耳听感应与原音频基本一致）")
    print("      clone_baseline.wav  —— 未防御时攻击者克隆出来的声音（应很像本人）")
    print("      clone_protected.wav —— 防御后攻击者克隆出来的声音（应明显不像）")
    print("  ★ 上面的 stolen_embedding_cosine 就是论文的 Sim_clone。")
    print("    L1 用的是随机扰动，所以它还很高（接近 1 = 防御几乎无效）。")
    print("    这是**预期**的：Stage 2 训练出生成器 G 之后才会显著下降。")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
