"""VCTK 目录勘察 —— 成员 B 主责（任务板 W2-D1-5）。

【产出】
把 data/raw 下的 VCTK 勘察一遍，生成 docs/notes/vctk_layout.md，
内容包含任务板 D1-5 要求的四项：
    1. 说话人总数
    2. 每个说话人的句子数
    3. 有多少个 p*_eval 目录
    4. 音频格式 / 采样率 / 时长统计

【为什么做成脚本而不是手写笔记】
因为这份笔记要写进 README 的"数据来源"一节，也要在组内被复核。
手写的数字迟早会和数据对不上；脚本跑出来的数字不会。
**评审时直接重跑一遍即可验证。**

【用法】
    conda activate vpp
    python algorithms/voiceprint_protection/scripts/survey_vctk.py
    python algorithms/voiceprint_protection/scripts/survey_vctk.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # config / audio_utils / paths
sys.path.insert(0, str(Path(__file__).resolve().parent))          # prepare_data

import paths  # noqa: E402
from prepare_data import (  # noqa: E402
    discover_speakers,
    find_audio_root,
    list_utterances,
)

NOTE_PATH = Path("docs/notes/vctk_layout.md")

DATASET_INFO = """\
| 项 | 值 |
|---|---|
| 数据集 | CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit（version 0.92） |
| **DOI（权威标识）** | **<https://doi.org/10.7488/ds/2645>** → <https://datashare.ed.ac.uk/handle/10283/3443> |
| 直接下载 | <https://datashare.ed.ac.uk/server/api/core/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/content>（`VCTK-Corpus-0.92.zip`，11,747,302,977 字节） |
| 许可 | **CC BY 4.0**（见数据自带的 `VCTK.txt`：“licensed under the Creative Commons License: Attribution 4.0 International”）→ **对外发布样例时必须署名来源** |
| 说话人数 | **110**（109 个 `pXXX` + 1 个 `s5`），与论文一致 |
| 论文中的用途 | 训练 80 人 / 验证 10 人 / 测试 20 人，全部重采样到 16 kHz |
| 引用格式 | Veaux, C., Yamagishi, J., & MacDonald, K. (2017). *CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit* (version 0.92). The Centre for Speech Technology Research (CSTR), Edinburgh. <https://doi.org/10.7488/ds/2645> |
| 论文引用 | Yamagishi, J., et al. (2020). *The VCTK Corpus: A multi-speaker English dataset for speaker adaptation and multi-speaker TTS.* Zenodo. <https://doi.org/10.5281/zenodo.3235654> |

> ⚠️ **引用勘误（2026-11-17 核实）**：
> 最后一行的 Zenodo DOI `10.5281/zenodo.3235654` **实测 404** —— Zenodo 上并没有
> VCTK 的主记录。**要引用 VCTK 请用 DataShare 的 DOI：<https://doi.org/10.7488/ds/2645>**
> （已验证它指向我们下载的同一个 item，handle `10283/3443`）。
> 另外，最后一行那篇 Zenodo 论文其实是 **AutoVC 的论文**（它用了 VCTK 当训练数据），
> 不是 VCTK 本身的引用 —— 写 README 时别把两者混用。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="勘察 VCTK 目录并生成勘察笔记")
    parser.add_argument("--raw-root", default=str(paths.raw_root()),
                        help="VCTK 原始数据目录（默认 <数据根>/raw，可用 VPP_DATA_ROOT 指定数据根）")
    parser.add_argument("--out", default=str(NOTE_PATH), help="输出的 markdown 路径")
    parser.add_argument("--probe", type=int, default=5,
                        help="抽样读取多少条音频来确认格式与时长")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    print("=" * 64)
    print("  VCTK 目录勘察（成员 B 主责）")
    print("=" * 64)
    print("\n  路径解析：")
    for line in paths.describe().splitlines():
        print(f"    {line}")

    try:
        audio_root = find_audio_root(raw_root)
    except FileNotFoundError as error:
        print(f"\n[失败] {error}")
        print("\n提示：先按 docs/notes/download_vctk.md 里的命令把 VCTK 下载到 <数据根>/raw。")
        print("      数据根由环境变量 VPP_DATA_ROOT 决定（当前：", paths.data_root(), "）")
        return 1

    clean, noisy = discover_speakers(audio_root)

    # -- 每个说话人的句子数 ------------------------------------------------
    counts: dict[str, int] = {}
    for speaker in clean:
        counts[speaker] = len(list_utterances(audio_root / speaker))

    total = sum(counts.values())
    if total:
        minimum = min(counts.items(), key=lambda item: item[1])
        maximum = max(counts.items(), key=lambda item: item[1])
        median = statistics.median(counts.values())
    else:
        minimum = maximum = ("-", 0)
        median = 0

    print(f"\n  音频目录   : {audio_root}")
    print(f"  正常说话人 : {len(clean)}")
    print(f"  加噪说话人 : {len(noisy)}  -> {', '.join(noisy) if noisy else '（无）'}")
    print(f"  总语句数   : {total}")
    print(f"  每说话人句数: 最少 {minimum[0]}={minimum[1]}，最多 {maximum[0]}={maximum[1]}，中位数 {median}")

    # -- 抽样看格式 --------------------------------------------------------
    probe_rows = []
    try:
        import soundfile as sf

        sampled = 0
        for speaker in clean:
            if sampled >= args.probe:
                break
            for utterance in list_utterances(audio_root / speaker)[:1]:
                info = sf.info(str(utterance))
                probe_rows.append(
                    (speaker, utterance.name, info.samplerate, info.channels,
                     info.subtype, info.frames / info.samplerate)
                )
                sampled += 1
    except Exception as error:  # noqa: BLE001
        print(f"  [警告] 抽样读取失败（不影响勘察结论）：{error}")

    for row in probe_rows:
        print(f"    {row[0]:<8} {row[1]:<24} {row[2]} Hz / {row[3]} ch / {row[4]} / {row[5]:.2f}s")

    # -- 找许可文件 --------------------------------------------------------
    license_files = []
    for name in ("LICENSE.txt", "LICENSE", "COPYING", "README.txt"):
        for candidate in raw_root.rglob(name):
            license_files.append(candidate)
            break
        if license_files:
            break

    # -- 生成笔记 ----------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# VCTK 目录勘察记录")
    lines.append("")
    lines.append("> 由 `scripts/survey_vctk.py` 自动生成（成员 B，任务板 W2-D1-5）。")
    lines.append("> 数据变了就重跑脚本，别手改本文件。")
    lines.append("")
    lines.append("## 1. 数据来源")
    lines.append("")
    lines.append(DATASET_INFO)
    lines.append("## 2. 目录结构")
    lines.append("")
    lines.append("```text")
    lines.append(f"{audio_root}/")
    lines.append("├── p225/            正常说话人目录（p + 数字）")
    lines.append("│   ├── p225_001_mic1.flac      mic1/mic2 = 录音麦克风编号")
    lines.append("│   ├── p225_003_mic2.flac")
    lines.append("│   └── ...")
    lines.append("├── s5/              ★ 第 110 个说话人（名字不遵循 p+数字，筛选时别漏）")
    lines.append("│   └── s5_001_mic1.flac")
    if noisy:
        lines.append(f"├── {noisy[0]}/   加噪评测版本（训练时必须排除）")
        lines.append("│   └── ...")
    lines.append("└── ...")
    lines.append("```")
    lines.append("")
    lines.append(f"实际使用的音频目录：`{audio_root}`")
    lines.append("")
    lines.append("> 说明：`mic1` / `mic2` 是录音用的麦克风编号（VCTK 每个句子通常有两次录音），")
    lines.append("> 不是说话人编号。")
    lines.append("")
    lines.append("## 3. 说话人与句子数统计")
    lines.append("")
    lines.append(f"- 正常说话人总数：**{len(clean)}**")
    lines.append(f"- 加噪说话人（`p*_eval`）总数：**{len(noisy)}**")
    lines.append(f"- 语料总句数：**{total}**")
    lines.append(f"- 每个说话人的句子数：最少 `{minimum[0]}` = {minimum[1]}，"
                 f"最多 `{maximum[0]}` = {maximum[1]}，中位数 {median}")
    lines.append("")
    lines.append("### 3.1 每个说话人的句子数明细")
    lines.append("")
    lines.append("| 说话人 | 句子数 | 说话人 | 句子数 |")
    lines.append("|---|---|---|---|")
    items = sorted(counts.items())
    half = (len(items) + 1) // 2
    for index in range(half):
        left = items[index]
        right = items[index + half] if index + half < len(items) else None
        right_cell = f"{right[0]} | {right[1]}" if right else " | "
        lines.append(f"| {left[0]} | {left[1]} | {right_cell} |")
    lines.append("")
    lines.append("### 3.2 加噪评测说话人（`p*_eval`）清单")
    lines.append("")
    if noisy:
        lines.append("以下目录是**刻意加噪**的评测版本，训练/验证/测试划分都必须排除：")
        lines.append("")
        for speaker in noisy:
            lines.append(f"- `{speaker}`")
    else:
        lines.append("本次勘察未发现 `p*_eval` 目录。若确认数据来自官方完整包，")
        lines.append("请核对压缩包内容是否完整。")
    lines.append("")
    lines.append("## 4. 音频格式抽样")
    lines.append("")
    if probe_rows:
        lines.append("| 说话人 | 文件 | 采样率 | 声道 | 编码 | 时长(s) |")
        lines.append("|---|---|---|---|---|---|")
        for row in probe_rows:
            lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]:.2f} |")
        lines.append("")
        lines.append("> VCTK 原生是 48 kHz，**必须重采样到 16 kHz** 才能喂给 GE2E 与本项目的 mel 契约。")
    else:
        lines.append("（本次未能抽样读取音频文件）")
    lines.append("")
    lines.append("## 5. 本项目的划分方案")
    lines.append("")
    lines.append("| 集合 | 说话人数 | 用途 |")
    lines.append("|---|---|---|")
    lines.append("| train | 80 | Stage 1 + Stage 2 训练 |")
    lines.append("| val | 10 | 超参网格搜索（λ_cd / λ_dis / λ_per / λ_de / λ_mask） |")
    lines.append("| test | 20 | 评测（DSR / Sim_prot / Sim_clone / WER） |")
    lines.append("")
    lines.append("划分由 `scripts/prepare_data.py` 用固定 seed 生成，")
    lines.append("说话人名单记录在 `data/split_speakers.json`，**训练/验证/测试三集合严格不相交**。")
    lines.append("")
    if license_files:
        lines.append("## 6. 许可文件")
        lines.append("")
        for path in license_files:
            lines.append(f"- `{path}`")
        lines.append("")
        lines.append("> 对外发布样例音频（`public/samples/`）时必须保留 VCTK 来源署名。")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  已生成 {out_path}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
