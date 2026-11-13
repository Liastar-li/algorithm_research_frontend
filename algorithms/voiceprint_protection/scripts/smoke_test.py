"""L1 冒烟测试 —— 一条命令跑完整条推理链。

用法：
    conda activate vpp
    python algorithms/voiceprint_protection/scripts/smoke_test.py --audio path/to/input.wav

它做四件事：
  1. 检查三个权重是否就位（缺了就明确告诉你去哪儿下）
  2. 跑通"重建"（不加扰动）—— 这验证 AutoVC 权重 + 我们的 mel 是否配套
  3. 跑通"保护"（加随机扰动）—— 这验证整条链
  4. 输出音频 + 报告，并打印关键诊断量

【为什么要分 2 和 3 两步】
重建输出是噪音 -> 问题在 mel 或权重（跟保护逻辑无关）
重建正常但保护后是噪音 -> 问题在扰动或解码路径
分两步测，能立刻定位是哪一环坏了。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import audio_utils  # noqa: E402
import config  # noqa: E402
import paths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="L1 冒烟测试")
    parser.add_argument("--audio", default=None, help="输入音频（wav）。不给则用合成信号")
    parser.add_argument("--alpha", type=float, default=config.STAGE2["alpha"])
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--outdir", default=str(paths.MODULE_DIR / "results"))
    args = parser.parse_args()

    print("=" * 66)
    print("  L1 冒烟测试 · 声纹保护推理链")
    print("=" * 66)

    print("\n【0】组件与权重检查")
    print(paths.describe())
    missing = [name for name in paths.WEIGHT_FILES.values()
               if not paths.weight_path(name).exists()]
    if missing:
        print(f"\n  [中止] 缺少权重：{', '.join(missing)}")
        print("  下载地址见 third_party/README.md；放到 algorithms/voiceprint_protection/weights/")
        return 1
    print("  [OK] 三个权重齐备")

    # -- 准备输入 --------------------------------------------------------
    sample_rate = config.MEL_PRESETS[config.DEFAULT_PRESET]["sample_rate"]
    if args.audio:
        wav = audio_utils.load_audio(args.audio, sample_rate)
        print(f"\n输入: {args.audio}")
    else:
        print("\n[提示] 未指定 --audio，使用一段合成信号（仅用于验证链路，不用于效果评估）")
        t = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
        wav = (0.25 * np.sin(2 * np.pi * 140 * t)
               + 0.12 * np.sin(2 * np.pi * 430 * t) * np.sin(2 * np.pi * 2.7 * t)).astype(np.float32)
    print(f"  时长 {audio_utils.duration_seconds(wav, sample_rate):.2f} 秒  "
          f"采样率 {sample_rate} Hz")

    from infer import Protector, perturb_embedding, random_perturbation

    print("\n【1】加载模型")
    try:
        protector = Protector()
    except Exception as error:  # noqa: BLE001
        print(f"  [失败] 加载模型出错：{error}")
        return 1
    print("  [OK] AutoVC / 说话人编码器 / 声码器 全部就绪")

    # -- 步骤 1：只重建 --------------------------------------------------
    print("\n【2】重建测试（不加扰动）—— 验证 mel 与权重是否配套")
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

    # -- 步骤 2：加扰动 --------------------------------------------------
    print("\n【3】保护测试（加随机扰动）")
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
    stem = Path(args.audio).stem if args.audio else "synthetic"
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
