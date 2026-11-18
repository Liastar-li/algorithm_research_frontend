"""生成一份**假数据**，用来冒烟测试训练主循环（不依赖真实的 VCTK）。

【为什么要它】
真实 VCTK 有 11 GB、下载要 55 分钟。但训练脚本里可能有各种只在真数据上
才暴露的问题（清单格式、帧数、说话人 ID、collate 的堆叠……）。
用假数据先跑通，能把"代码 bug"和"数据还没下好"这两件事分开。

★ 假数据放在**仓库外**（默认 D:/vpp_dummy_data），通过环境变量 VPP_DATA_ROOT 指定，
  不会污染仓库，也不会让人误以为它是真数据。

【用法】
    conda activate vpp
    python algorithms/voiceprint_protection/scripts/make_dummy_data.py
    $env:VPP_DATA_ROOT = "D:/vpp_dummy_data"     # PowerShell
    python algorithms/voiceprint_protection/stage1/train.py --stage classifier --iterations 3 --smoke
    python algorithms/voiceprint_protection/stage1/train.py --stage model --iterations 3 --smoke
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import audio_utils  # noqa: E402
import config  # noqa: E402

DEFAULT_ROOT = Path("D:/vpp_dummy_data")
# 拿仓库里的平台样例当底料（有真实语音内容，比纯噪声更容易看出"重建是否正常"）
DEFAULT_SEED_WAV = (Path(__file__).resolve().parent.parent.parent.parent
                     / "public" / "samples" / "group_n005_voiceprint_protection" / "input.wav")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成训练冒烟用的假数据")
    parser.add_argument("--out", default=str(DEFAULT_ROOT))
    parser.add_argument("--speakers", type=int, default=8)
    parser.add_argument("--utterances", type=int, default=4)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)
    audio_root = out_root / "audio16k"
    audio_root.mkdir(parents=True, exist_ok=True)

    sample_rate = config.MEL_PRESETS[config.DEFAULT_PRESET]["sample_rate"]
    if DEFAULT_SEED_WAV.exists():
        base = audio_utils.load_audio(DEFAULT_SEED_WAV, sample_rate)
    else:
        print("  [提示] 找不到平台样例，用随机噪声代替。")
        base = np.random.RandomState(0).randn(sample_rate * 3).astype(np.float32) * 0.1

    generator = np.random.RandomState(args.seed)
    rows: list[str] = []
    for speaker_index in range(args.speakers):
        speaker = f"p{225 + speaker_index:03d}"
        speaker_dir = audio_root / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        for utterance_index in range(args.utterances):
            start = generator.randint(0, max(1, len(base) - sample_rate))
            segment = base[start:start + sample_rate]
            if len(segment) < sample_rate:
                segment = np.pad(segment, (0, sample_rate - len(segment)))
            # 加一点随机扰动，别让所有片段完全一样
            segment = segment + generator.randn(len(segment)).astype(np.float32) * 0.005
            name = f"{speaker}_{utterance_index + 1:03d}.wav"
            audio_utils.save_audio(speaker_dir / name, segment, sample_rate)
            rows.append(f"audio16k/{speaker}/{name} {speaker}")

    for split, count in (("train", args.speakers), ("val", 2), ("test", 2)):
        subset = rows[: count * args.utterances]
        (out_root / f"{split}.txt").write_text(
            "\n".join(subset) + "\n", encoding="utf-8")

    (out_root / "spk2id.txt").write_text(
        "\n".join(f"p{225 + index:03d} {index}" for index in range(args.speakers)) + "\n",
        encoding="utf-8")

    print("=" * 66)
    print("  假数据已生成")
    print("=" * 66)
    print(f"  根目录   : {out_root}")
    print(f"  说话人数 : {args.speakers}（train / val / test 各取前 N 条）")
    print(f"  样本条数 : {len(rows)}")
    print("\n  下一步（PowerShell）：")
    print(f'    $env:VPP_DATA_ROOT = "{out_root.as_posix()}"')
    print("    python algorithms/voiceprint_protection/stage1/train.py --stage classifier --iterations 3 --smoke")
    print("    python algorithms/voiceprint_protection/stage1/train.py --stage model --iterations 3 --smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
