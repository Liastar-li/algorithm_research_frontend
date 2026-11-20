"""从「说话人名单 + 种子」复现 80/10/20 划分 —— **不需要任何音频**。

【为什么需要这个脚本】
两个人各自下载 VCTK（各 1.36 GiB / 55 分钟），最怕的就是**划分不一致**：
训练集说话人不同 -> 指标没法对比 -> 整个实验作废。

而划分**只取决于两样东西**：说话人名单 + 种子，与音频内容**完全无关**。
`prepare_data.py::split_speakers()` 做的就一件事：

    random.Random(seed).shuffle(名单) -> 按 80/10/20 切 -> 各自排序

所以只要两人用的是**同一份名单 + 同一个 seed**，划分必然逐字一致。
本脚本把这条约定变成**可执行的**：对方不用再下一份 1.36 GiB 的 VCTK
就能对账（也能直接生成 spk2id.txt 给 Stage 1 训练用）。

【★ 第 110 个说话人叫 s5】
它不遵循 `p+数字` 的命名规律。按 `p` 前缀筛说话人会只剩 109 人，
80/10/20 的划分直接失败。名单文件里已经包含它。

【用法】
    conda activate vpp

    # 1) 复现划分并打印（用仓库里的名单 docs/notes/vctk_speakers.txt）
    python algorithms/voiceprint_protection/scripts/split_speakers.py

    # 2) 写到某个目录（生成 split_speakers.json + spk2id.txt）
    python algorithms/voiceprint_protection/scripts/split_speakers.py --out-dir data

    # 3) 跟已有的划分对账（对方跑完 prepare_data.py 之后）
    python algorithms/voiceprint_protection/scripts/split_speakers.py --check <路径>/split_speakers.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR.parent))
sys.path.insert(0, str(MODULE_DIR))

try:
    import config  # noqa: E402
except ImportError as error:  # 依赖缺失（多半是忘了 conda activate vpp）
    import paths as _paths  # noqa: E402

    _paths.print_env_hint(error)
    raise SystemExit(1) from error

from prepare_data import split_speakers, write_spk2id  # noqa: E402

DEFAULT_SPEAKERS = Path(__file__).resolve().parents[3] / "docs" / "notes" / "vctk_speakers.txt"


def read_speakers(path: Path) -> list[str]:
    """读名单文件（`#` 开头的是注释）。"""
    names = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line.split()[0])
    return sorted(set(names))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从说话人名单 + 种子复现 80/10/20 划分（不需要音频）")
    parser.add_argument("--speakers-file", default=str(DEFAULT_SPEAKERS),
                        help="说话人名单（每行一个名字，# 开头是注释）")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--train", type=int, default=80)
    parser.add_argument("--val", type=int, default=10)
    parser.add_argument("--test", type=int, default=20)
    parser.add_argument("--out-dir", default=None,
                        help="写 split_speakers.json + spk2id.txt 到这个目录")
    parser.add_argument("--check", default=None,
                        help="跟已有的 split_speakers.json 对账")
    args = parser.parse_args()

    speakers_file = Path(args.speakers_file)
    if not speakers_file.exists():
        print(f"[失败] 找不到名单文件：{speakers_file}")
        return 1
    speakers = read_speakers(speakers_file)
    print("=" * 68)
    print("  VCTK 说话人划分复现（不需要音频）")
    print("=" * 68)
    print(f"  名单    : {speakers_file}")
    print(f"  人数    : {len(speakers)}")
    print(f"  seed    : {args.seed}")

    splits = split_speakers(speakers, args.train, args.val, args.test, args.seed)
    for name in ("train", "val", "test"):
        people = splits[name]
        print(f"  {name:<6}: {len(people):>3} 人  {people[0]} … {people[-1]}")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {"seed": args.seed, "preset": "paper16k", "splits": splits}
        (out / "split_speakers.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        write_spk2id(out / "spk2id.txt", splits["train"])
        print(f"\n  已写 {out / 'split_speakers.json'}")
        print(f"  已写 {out / 'spk2id.txt'}（K = {len(splits['train'])}）")

    if args.check:
        other = json.loads(Path(args.check).read_text(encoding="utf-8"))
        same = other.get("splits") == splits
        print(f"\n  对账 {args.check}：{'✅ 完全一致' if same else '❌ 不一致'}")
        if not same:
            for name in ("train", "val", "test"):
                mine = set(splits[name])
                theirs = set(other.get("splits", {}).get(name, []))
                print(f"    {name}: 我方独有 {sorted(mine - theirs)[:5]}"
                      f" / 对方独有 {sorted(theirs - mine)[:5]}")
            return 1

    print("\n  ★ 只要两人的名单与 seed 相同，划分就必然一致 —— 不需要比对音频。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
