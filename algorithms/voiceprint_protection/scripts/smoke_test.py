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
    missing = [name for name in paths.WEIGHT_FILES.values()
               if not paths.weight_path(name).exists()]
    if missing:
        print(f"\n  [注意] 缺少 AutoVC 路线所需的权重：{', '.join(missing)}")
        print("         下载地址见 third_party/README.md；放到 algorithms/voiceprint_protection/weights/")
        print("         → 第 2/3 步会跳过；第 1 步（GE2E）不受影响。")
    else:
        print("  [OK] AutoVC 路线权重齐备")

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
        print("  AutoVC 权重不齐，已跳过重建/保护两步。")
        print("  已验证：GE2E 说话人分支 =", "通过" if speaker_ok else "未通过")
        print("  补齐权重后重跑本脚本即可完成整链验证。")
        print("=" * 66)
        return 0 if speaker_ok else 1

    # -- 步骤 2：加载模型 ------------------------------------------------
    from infer import Protector, perturb_embedding, random_perturbation

    print("\n【2】加载模型")
    try:
        protector = Protector()
    except Exception as error:  # noqa: BLE001
        print(f"  [失败] 加载模型出错：{error}")
        return 1
    print("  [OK] AutoVC / 说话人编码器 / 声码器 全部就绪")

    # -- 步骤 3：只重建 --------------------------------------------------
    print("\n【3】重建测试（不加扰动）—— 验证 mel 与权重是否配套")
    mel = protector.to_mel(wav)
    embedding = protector.embedding(mel)
    print(f"  mel 形状        {tuple(mel.shape)}")
    print(f"  嵌入形状        {tuple(embedding.shape)}  "
          f"L2 范数 {float(np.linalg.norm(embedding)):.4f}")
    reconstructed_mel = protector.autovc.reconstruct(mel, embedding)
    print(f"  重建 mel 形状   {tuple(reconstructed_mel.shape)}")
    print(f"  重建 mel 范围   [{reconstructed_mel.min():.3f}, {reconstructed_mel.max():.3f}]")

    reconstructed_wav = protector.to_waveform(reconstructed_mel)
    rms = audio_utils.rms_energy(reconstructed_wav)
    original_rms = audio_utils.rms_energy(wav)
    print(f"  重建波形 RMS    {rms:.5f}（原音频 {original_rms:.5f}）")
    if rms < 1e-4:
        print("  [失败] 重建音频接近静音 —— mel 或权重不配套，先别往下走")
        return 1
    print("  [OK] 重建音频有声")

    # -- 步骤 4：加扰动 --------------------------------------------------
    print("\n【4】保护测试（加随机扰动）")
    delta = random_perturbation(seed=args.seed)
    result = protector.protect(wav, alpha=args.alpha, seed=args.seed, verbose=False)

    protected_rms = audio_utils.rms_energy(result["waveform"])
    similarity = float(np.dot(result["embedding"], result["embedding_adv"]))
    mel_difference = float(np.mean(np.abs(result["protected_mel"] - mel)))
    print(f"  α = {args.alpha}")
    print(f"  嵌入余弦相似度  {similarity:.4f}（1.0 = 没扰动；越小扰动越强）")
    print(f"  mel 平均绝对差  {mel_difference:.4f}")
    print(f"  保护后波形 RMS  {protected_rms:.5f}")
    if protected_rms < 1e-4:
        print("  [失败] 保护后音频接近静音")
        return 1
    print("  [OK] 保护后音频有声")

    # -- 输出 ------------------------------------------------------------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_path).stem if audio_path else "synthetic"
    protected_path = outdir / f"{stem}_protected.wav"
    reconstructed_path = outdir / f"{stem}_reconstructed.wav"
    audio_utils.save_audio(protected_path, result["waveform"], sample_rate)
    audio_utils.save_audio(reconstructed_path, reconstructed_wav, sample_rate)
    print(f"\n  已保存保护后音频   {protected_path}")
    print(f"  已保存重建音频     {reconstructed_path}")

    report = {
        "alpha": args.alpha,
        "seed": args.seed,
        "shapes": result["shapes"],
        "embedding_cosine": similarity,
        "mel_mean_abs_diff": mel_difference,
        "rms_original": original_rms,
        "rms_reconstructed": rms,
        "rms_protected": protected_rms,
        "note": "L1 阶段：δ 为随机高斯噪声占位，尚未训练生成器 G。",
    }
    report_path = outdir / f"{stem}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已保存报告         {report_path}")

    print("\n" + "=" * 66)
    print("  冒烟测试通过。")
    print("  ★ 下一步：对比「重建音频」与「原始音频」，确认听感一致；")
    print("    然后再对比「保护后音频」，确认听感几乎无差别。")
    print("  ★ 如果听感不对，先怀疑 mel 对照，而不是保护逻辑。")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
