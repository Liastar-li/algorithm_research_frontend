"""VCTK 预处理与划分 —— 成员 B 主责（任务板 W2-D2-5 / W2-D3-5）。

【本脚本负责什么】
把解压后的 VCTK 变成 Stage 1 / Stage 2 能直接吃的数据：

    data/raw/...                  （VCTK 解压后的原始目录）
        │
        ├─► data/audio16k/<spk>/<utt>.wav   16 kHz / 单声道 / 16-bit PCM
        ├─► data/train.txt                  训练清单（80 说话人）
        ├─► data/val.txt                    验证清单（10 说话人）
        ├─► data/test.txt                   测试清单（20 说话人）
        ├─► data/spk2id.txt                 训练说话人 -> 0..79 的整数 ID
        └─► data/split_speakers.json        三个划分的说话人名单（便于机器校验）

【三条硬要求（任务板 §1.3）】
1. **排除 p*_eval 目录** —— 那是刻意加噪的评测版本，混进训练集会污染实验；
2. **训练 / 验证 / 测试说话人完全不重叠** —— 泄漏会让指标虚高；
3. **spk2id.txt 的 K 必须等于训练集说话人数 = 80** ——
   L_dis 的说话人分类器 Ws 形状是 K×256，K 错了分类器根本建不起来。

【清单格式】
每行两列，空格分隔：<相对 data/ 的音频路径> <说话人名>

    audio16k/p225/p225_001.wav p225

第二列用**说话人名**（p225 / p292 …）而不是整数，是为了让三份清单能做
集合运算（任务板给的 `cut -d' ' -f2 | sort -u | comm -12` 就是按名字比对的）；
整数 ID 由 spk2id.txt 提供（只覆盖训练集说话人，因为 Ws 只在训练集上工作）。

【用法】
    conda activate vpp
    # 先干跑一遍，看看会划分出哪些说话人、要做多少转换
    python algorithms/voiceprint_protection/scripts/prepare_data.py --dry-run
    # 正式执行
    python algorithms/voiceprint_protection/scripts/prepare_data.py

【关于"唯一真源"】
本脚本不自己写重采样/写文件的代码，一律调用 audio_utils
（采样率、mel 参数的唯一真源所在）。采样率取自 config.MEL_PRESETS。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import audio_utils  # noqa: E402
    import config  # noqa: E402
    import paths  # noqa: E402
except ImportError as error:  # 依赖缺失时给个能看懂的原因（多半是忘了 activate vpp）
    import paths as _paths  # noqa: E402

    _paths.print_env_hint(error)
    raise SystemExit(1) from error

# VCTK 的音频目录名（按优先级）。wav48_silence_trimmed 已裁掉首尾静音。
AUDIO_DIR_NAMES = ("wav48_silence_trimmed", "wav")
# 压缩包解压后，音频目录外面可能还套一层目录（官方 zip 解出来是 VCTK-Corpus-0.92/）
WRAPPER_DIRS = ("", "VCTK", "VCTK_2", "VCTK-Corpus-0.92", "VCTK-Corpus-0.92/VCTK-Corpus-0.92")
AUDIO_SUFFIXES = (".flac", ".wav")


# ---------------------------------------------------------------------------
# 目录勘察
# ---------------------------------------------------------------------------
def _is_speaker_dir(path: Path) -> bool:
    """是否是 VCTK 的说话人目录。

    ⚠️ **别只认 `p` + 数字**！
    VCTK 的第 110 个说话人叫 **s5**（见数据里的 speaker-info.txt 第 111 行）：

        s5    22  F    British
             (new speaker, more data would be released for training a speaker-dependent model)

    它是**合法的说话人**，只是命名不遵循 `pXXX` 的规律。
    漏掉它 → 只有 109 人 → 80/10/20 的划分直接失败。
    （2026-11-17 实际踩到过这个坑。）
    """
    if not path.is_dir():
        return False
    name = path.name
    if name == "s5":
        return True
    if not name.startswith("p"):
        return False
    return name[1:].split("_")[0].isdigit()


def _looks_like_audio_root(path: Path) -> bool:
    """该目录下是否有说话人子目录。"""
    if not path.is_dir():
        return False
    return any(_is_speaker_dir(child) for child in path.iterdir())


def find_audio_root(raw_root: Path) -> Path:
    """在 raw_root 下找到真正装着说话人目录的那一层。

    支持三种常见摆放方式：
        data/raw/wav48_silence_trimmed/p225/...
        data/raw/wav/p225/...
        data/raw/VCTK/wav48_silence_trimmed/p225/...
    """
    raw_root = raw_root.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"原始数据目录不存在：{raw_root}")

    if _looks_like_audio_root(raw_root):
        return raw_root

    for wrapper in WRAPPER_DIRS:
        for name in AUDIO_DIR_NAMES:
            candidate = raw_root / wrapper / name if wrapper else raw_root / name
            if _looks_like_audio_root(candidate):
                return candidate

    # 兜底：深度优先找第一个像样的音频目录
    for candidate in sorted(raw_root.rglob("*")):
        if candidate.is_dir() and candidate.name in AUDIO_DIR_NAMES:
            if _looks_like_audio_root(candidate):
                return candidate

    raise FileNotFoundError(
        f"在 {raw_root} 下找不到 VCTK 的音频目录。\n"
        "请确认 VCTK 已解压，且目录里能看到 p225 / p226 这样的说话人文件夹。\n"
        "参考：https://datashare.ed.ac.uk/handle/10283/3191"
    )


def discover_speakers(audio_root: Path) -> tuple[list[str], list[str]]:
    """返回 (正常说话人列表, 加噪评测说话人列表)，均按名字排序。"""
    clean: list[str] = []
    noisy: list[str] = []
    for child in sorted(audio_root.iterdir()):
        if not _is_speaker_dir(child):
            continue
        if child.name.endswith("_eval"):
            noisy.append(child.name)
        else:
            clean.append(child.name)
    return clean, noisy


def list_utterances(speaker_dir: Path) -> list[Path]:
    """列出该说话人目录下的音频（按文件名排序，保证可复现）。"""
    files: list[Path] = []
    for suffix in AUDIO_SUFFIXES:
        files.extend(sorted(speaker_dir.glob(f"*{suffix}")))
    return sorted(files)


# ---------------------------------------------------------------------------
# 音频转换
# ---------------------------------------------------------------------------
def _trim_silence(wav, sample_rate: int):
    """裁掉首尾静音（与 resemblyzer 的 trim_long_silences 同一思路）。

    只有在使用 VCTK 的 `wav/`（未裁静音）目录时才需要；
    `wav48_silence_trimmed/` 本身已经裁过了。
    """
    import librosa

    trimmed, _ = librosa.effects.trim(wav, top_db=30)
    return trimmed if len(trimmed) >= sample_rate // 10 else wav


def convert_one(
    src: Path,
    dst: Path,
    sample_rate: int,
    trim_silence: bool = False,
) -> tuple[bool, str]:
    """把一条原始音频转成 16 kHz 单声道 16-bit PCM。"""
    try:
        wav = audio_utils.load_audio(src, sample_rate=sample_rate)
    except Exception as error:  # noqa: BLE001
        return False, f"读取失败 {src.name}：{error}"

    if len(wav) < sample_rate // 10:  # 短于 0.1 秒，视为无效音频
        return False, f"过短（{len(wav) / sample_rate:.3f}s）{src.name}"

    if trim_silence:
        wav = _trim_silence(wav, sample_rate)

    try:
        audio_utils.save_audio(dst, wav, sample_rate)
    except Exception as error:  # noqa: BLE001
        return False, f"写入失败 {dst.name}：{error}"
    return True, ""


# ---------------------------------------------------------------------------
# 划分
# ---------------------------------------------------------------------------
def split_speakers(
    speakers: list[str], train_n: int, val_n: int, test_n: int, seed: int
) -> dict[str, list[str]]:
    """按种子随机划分说话人。同一 seed 必然得到同一结果（可复现）。"""
    needed = train_n + val_n + test_n
    if len(speakers) < needed:
        raise RuntimeError(
            f"说话人不够：需要 {needed} 个（{train_n} 训练 + {val_n} 验证 + "
            f"{test_n} 测试），实际只有 {len(speakers)} 个。\n"
            "请确认 VCTK 解压完整，且 p*_eval 目录已按设计排除。"
        )

    shuffled = list(speakers)
    random.Random(seed).shuffle(shuffled)

    train = sorted(shuffled[:train_n])
    val = sorted(shuffled[train_n:train_n + val_n])
    test = sorted(shuffled[train_n + val_n:needed])
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# 清单读写
# ---------------------------------------------------------------------------
def write_manifest(path: Path, rows: list[tuple[str, str]]) -> None:
    """写清单：每行 "<相对路径> <说话人名>"。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for rel_path, speaker in rows:
            handle.write(f"{rel_path} {speaker}\n")


def write_spk2id(path: Path, train_speakers: list[str]) -> None:
    """写 spk2id.txt：训练集说话人 -> 0..K-1 的整数 ID（K = 训练说话人数）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, speaker in enumerate(sorted(train_speakers)):
            handle.write(f"{speaker} {index}\n")


def read_manifest_speakers(path: Path) -> set[str]:
    """从清单里取出说话人集合。"""
    speakers: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{path} 里有格式不对的行：{line!r}")
            speakers.add(parts[1])
    return speakers


def read_spk2id(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            name, index = line.split()
            mapping[name] = int(index)
    return mapping


# ---------------------------------------------------------------------------
# 自检（任务板 D3-5 的验收命令，脚本内自动跑一遍）
# ---------------------------------------------------------------------------
def verify(out_root: Path, splits: dict[str, list[str]], sample_rate: int) -> bool:
    print("\n【自检】")
    ok = True

    manifest_paths = {name: out_root / f"{name}.txt" for name in ("train", "val", "test")}
    for name, path in manifest_paths.items():
        if not path.exists():
            print(f"  [失败] 缺少清单 {path}")
            ok = False

    if not ok:
        return False

    speakers = {name: read_manifest_speakers(path) for name, path in manifest_paths.items()}

    # 硬要求 1：p*_eval 一个都不能出现
    for name, spk_set in speakers.items():
        bad = sorted(s for s in spk_set if s.endswith("_eval"))
        if bad:
            print(f"  [失败] {name}.txt 里混入了加噪评测说话人：{bad}")
            ok = False
    if ok:
        print("  [通过] 三份清单都不含 p*_eval 加噪说话人")

    # 硬要求 2：两两不相交
    names = list(speakers)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = speakers[names[i]] & speakers[names[j]]
            if overlap:
                print(f"  [失败] {names[i]} 与 {names[j]} 的说话人重叠：{sorted(overlap)}")
                ok = False
    if ok:
        print("  [通过] 训练 / 验证 / 测试说话人两两不相交")

    # 附加：清单里的说话人必须与划分的名单完全一致（防止清单写串了）
    for name in ("train", "val", "test"):
        expected = set(splits[name])
        if speakers[name] != expected:
            print(f"  [失败] {name}.txt 的说话人与划分名单不一致")
            print(f"          清单缺: {sorted(expected - speakers[name])}")
            print(f"          清单多: {sorted(speakers[name] - expected)}")
            ok = False
        elif expected:
            print(f"  [通过] {name}.txt 的说话人与划分名单一致（{len(expected)} 个）")

    # 硬要求 3：spk2id 的 K = 训练说话人数，且 ID 连续覆盖 0..K-1
    spk2id = read_spk2id(out_root / "spk2id.txt")
    train_count = len(speakers["train"])
    if len(spk2id) != train_count:
        print(f"  [失败] spk2id.txt 有 {len(spk2id)} 条，训练集说话人有 {train_count} 个")
        ok = False
    elif sorted(spk2id.values()) != list(range(train_count)):
        print(f"  [失败] spk2id.txt 的整数 ID 不是 0..{train_count - 1} 的连续区间")
        ok = False
    elif set(spk2id) != speakers["train"]:
        print("  [失败] spk2id.txt 的说话人与 train.txt 对不上")
        ok = False
    else:
        print(f"  [通过] spk2id.txt 的 K = {train_count}，ID 连续覆盖 0..{train_count - 1}")

    # 附加：抽样核对音频确实是 16 kHz 单声道 16-bit
    import soundfile as sf

    rows = []
    with manifest_paths["test"].open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if parts:
                rows.append(parts[0])
    sample = rows[: min(20, len(rows))]
    bad_audio = []
    for rel in sample:
        info = sf.info(str(out_root / rel))
        if info.samplerate != sample_rate or info.channels != 1 or info.subtype != "PCM_16":
            bad_audio.append((rel, info.samplerate, info.channels, info.subtype))
    if bad_audio:
        for item in bad_audio:
            print(f"  [失败] {item[0]} 的格式不对：{item[1]} Hz / {item[2]} ch / {item[3]}")
        ok = False
    elif sample:
        print(f"  [通过] 抽查 {len(sample)} 条测试音频：{sample_rate} Hz / 单声道 / PCM_16")

    return ok


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="VCTK 预处理与划分（成员 B）")
    parser.add_argument("--raw-root", default=str(paths.raw_root()),
                        help="VCTK 解压后的根目录（默认 <数据根>/raw，可用 VPP_DATA_ROOT 指定数据根）")
    parser.add_argument("--out-root", default=str(paths.data_root()),
                        help="输出根目录（默认 <数据根>）")
    parser.add_argument("--preset", default=config.DEFAULT_PRESET,
                        choices=list(config.MEL_PRESETS), help="采样率取自该预设")
    parser.add_argument("--train-spk", type=int, default=80)
    parser.add_argument("--val-spk", type=int, default=10)
    parser.add_argument("--test-spk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--limit-per-speaker", type=int, default=0,
                        help="每个说话人最多取几条（0 = 全部）。调试时可用 5")
    parser.add_argument("--workers", type=int, default=4, help="并发转换线程数")
    parser.add_argument("--trim-silence", action="store_true",
                        help="裁掉首尾静音（仅当用的是未裁剪的 wav/ 目录时需要）")
    parser.add_argument("--skip-audio", action="store_true",
                        help="只写清单，不做音频转换（音频已备好时用）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已转换的音频")
    parser.add_argument("--dry-run", action="store_true", help="只勘察与打印，不落盘")
    args = parser.parse_args()

    sample_rate = config.MEL_PRESETS[args.preset]["sample_rate"]
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    print("=" * 64)
    print("  VCTK 预处理与划分（成员 B 主责）")
    print("=" * 64)
    print("\n  路径解析：")
    for line in paths.describe().splitlines():
        print(f"    {line}")

    try:
        audio_root = find_audio_root(raw_root)
    except FileNotFoundError as error:
        print(f"\n[失败] {error}")
        return 1

    clean, noisy = discover_speakers(audio_root)
    print(f"\n  音频目录   : {audio_root}")
    print(f"  说话人总数 : {len(clean)} 个正常 + {len(noisy)} 个 p*_eval（已排除）")
    if noisy:
        print(f"  加噪评测集 : {', '.join(noisy)}")

    if args.limit_per_speaker:
        print(f"  每说话人上限: {args.limit_per_speaker} 条（调试模式）")

    try:
        splits = split_speakers(
            clean, args.train_spk, args.val_spk, args.test_spk, args.seed
        )
    except RuntimeError as error:
        print(f"\n[失败] {error}")
        return 1

    for name in ("train", "val", "test"):
        print(f"  {name:<5} : {len(splits[name])} 个说话人  "
              f"({splits[name][0]} … {splits[name][-1]})")

    # -- 收集要转换的音频 ------------------------------------------------
    jobs: list[tuple[Path, Path, str]] = []
    total_utterances = 0
    for name in ("train", "val", "test"):
        for speaker in splits[name]:
            speaker_dir = audio_root / speaker
            if not speaker_dir.is_dir():
                print(f"  [警告] 说话人目录不存在，已跳过：{speaker_dir}")
                continue
            utterances = list_utterances(speaker_dir)
            if args.limit_per_speaker:
                utterances = utterances[: args.limit_per_speaker]
            total_utterances += len(utterances)
            for utterance in utterances:
                rel = Path("audio16k") / speaker / utterance.name.replace(".flac", ".wav")
                jobs.append((utterance, out_root / rel, name))

    print(f"  待处理语句 : {total_utterances} 条 -> {out_root / 'audio16k'}")

    if args.dry_run:
        print("\n  [干跑] 未写入任何文件。去掉 --dry-run 即可正式执行。")
        return 0

    # -- 转换音频 --------------------------------------------------------
    if not args.skip_audio:
        audio_root_out = out_root / "audio16k"
        audio_root_out.mkdir(parents=True, exist_ok=True)

        todo = [
            job for job in jobs
            if args.overwrite or not job[1].exists()
        ]
        print(f"\n  开始转换（{len(todo)} 条待转，{len(jobs) - len(todo)} 条已存在）...")

        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [
                pool.submit(convert_one, src, dst, sample_rate, args.trim_silence)
                for src, dst, _ in todo
            ]
            for index, future in enumerate(futures, start=1):
                ok, message = future.result()
                if not ok:
                    failures.append(message)
                if index % 500 == 0 or index == len(futures):
                    print(f"    {index}/{len(futures)}")

        if failures:
            print(f"\n  [警告] {len(failures)} 条转换失败，前 5 条：")
            for item in failures[:5]:
                print(f"    - {item}")

    # -- 写清单 ----------------------------------------------------------
    rows_by_split: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
    for src, dst, split in jobs:
        if not dst.exists():
            continue
        rel = dst.relative_to(out_root).as_posix()
        rows_by_split[split].append((rel, src.parent.name))

    for name, rows in rows_by_split.items():
        rows.sort()
        write_manifest(out_root / f"{name}.txt", rows)
        print(f"  已写 {out_root / (name + '.txt')}（{len(rows)} 行）")

    write_spk2id(out_root / "spk2id.txt", splits["train"])
    print(f"  已写 {out_root / 'spk2id.txt'}（K = {len(splits['train'])}）")

    with (out_root / "split_speakers.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"seed": args.seed, "preset": args.preset, "splits": splits},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  已写 {out_root / 'split_speakers.json'}")

    if not verify(out_root, splits, sample_rate):
        print("\n  自检未通过，请按上面的提示排查。")
        return 1

    print("\n  全部完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
