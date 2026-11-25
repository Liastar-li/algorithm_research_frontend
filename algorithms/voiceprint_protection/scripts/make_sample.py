"""生成平台样例 —— 成员 B 主责（任务板 W2-D4-4 / W3-P4）。

【要解决什么】
平台接入要求 `public/samples/<方法ID>/` 下有一组**真实跑出来的**输入/输出，
且样例素材必须**具有公开展示权限**。

本脚本做三件事：
  1. 从 `data/test.txt` 里挑一条**测试集**说话人的语音，复制到
     `public/samples/<方法ID>/input.wav`（16 kHz 单声道 16-bit PCM）；
  2. 写一份 `SOURCES.md`，写明这条音频的来源（VCTK 说话人/句子编号）与许可，
     **保证对外发布时能署名**；
  3. 如果 `infer.py` 已经就绪，再跑一次真实推理，生成 `output.wav`
     （没有 infer.py 时只出输入，并明确提示，不伪造输出）。

【为什么必须来自测试集】
测试集说话人从未出现在训练集里，样例才有演示价值；
而且样例必须"真实跑出来"，不能拿手搓的波形凑数。

【VCTK 还没下好时：从上游样例取】
上游 Real-Time-Voice-Cloning 仓库自带两个 **VCTK 真实语音**的压缩版
（`samples/p240_00000.mp3`、`samples/p260_00000.mp3`），
并且 `samples/VCTK.txt` 里带着**完整的 CC BY 4.0 许可声明与引用格式**。
所以 VCTK 还在下载时也能把样例的**输入**先准备好：

    python algorithms/voiceprint_protection/scripts/make_sample.py ^
        --from-wav D:\\vpp\\third_party\\RTC\\samples\\p240_00000.mp3

【用法】
    conda activate vpp
    python algorithms/voiceprint_protection/scripts/make_sample.py
    python algorithms/voiceprint_protection/scripts/make_sample.py --utterance 3 --alpha 0.1
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import audio_utils  # noqa: E402
    import config  # noqa: E402
    import paths  # noqa: E402
except ImportError as error:  # 依赖缺失时给个能看懂的原因（多半是忘了 activate vpp）
    import paths as _paths  # noqa: E402

    _paths.print_env_hint(error)
    raise SystemExit(1) from error

# ⚠️ 必须全小写：平台契约的标识符正则 /^[a-z][a-z0-9_]{1,63}$/ 不接受大写字母。
METHOD_ID = os.environ.get("VPP_METHOD_ID", "group_n005_voiceprint_protection")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# 样例目录由 paths.sample_dir() 统一解析（必须留在仓库的 public/ 下，前端按相对路径取）

VCTK_LICENSE_NOTE = """\
## 样例来源与许可

- **数据集**：CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning
  Toolkit（version 0.92），The Centre for Speech Technology Research (CSTR), Edinburgh
- **DOI（权威标识）**：<https://doi.org/10.7488/ds/2645>
  → <https://datashare.ed.ac.uk/handle/10283/3443>
- **许可**：CC BY 4.0（见数据自带的 `VCTK.txt`：“licensed under the Creative Commons
  License: Attribution 4.0 International”）
  → **对外展示本样例时必须保留上述署名**。
- **引用格式**：
  Veaux, C., Yamagishi, J., & MacDonald, K. (2017). *CSTR VCTK Corpus: English
  Multi-speaker Corpus for CSTR Voice Cloning Toolkit* (version 0.92). The Centre
  for Speech Technology Research (CSTR), Edinburgh.
  <https://doi.org/10.7488/ds/2645>
- **处理过程**：原始 48 kHz FLAC → 重采样到 16 kHz / 单声道 / 16-bit PCM；
  保护后音频由本项目的推理链实际生成。
"""


def pick_utterance(manifest: Path, index: int) -> tuple[str, str]:
    """从清单里取第 index 条，返回 (相对 data/ 的路径, 说话人名)。"""
    rows = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if parts:
                rows.append((parts[0], parts[1]))
    if not rows:
        raise RuntimeError(f"{manifest} 是空的，先跑 prepare_data.py")
    index = index % len(rows)
    return rows[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="生成平台样例（成员 B）")
    parser.add_argument("--data-root", default=str(paths.data_root()))
    parser.add_argument("--utterance", type=int, default=0, help="取测试集第几条")
    parser.add_argument("--alpha", type=float, default=config.STAGE2["alpha"])
    parser.add_argument("--no-output", action="store_true", help="只出输入，不跑推理")
    parser.add_argument("--from-wav", default=None,
                        help="直接用这条音频做样例输入（VCTK 未下好时用上游 samples 里的 mp3）")
    parser.add_argument("--source-note", default=None, help="来源说明里要写的来源描述")
    parser.add_argument("--method-id", default=METHOD_ID)
    parser.add_argument("--mode", choices=("clone", "stage1", "stage2"), default="clone",
                        help="★ 保护链：clone=用克隆机重新合成（默认）；"
                             "stage1=论文原版 D(c,h_adv)→HiFi-GAN（δ 随机）；"
                             "stage2=论文原版 + 训练好的 G（**样例要展示防御效果就用这条**）")
    parser.add_argument("--stage1-checkpoint", default=None,
                        help="Stage 1 的权重（stage1 / stage2 路线需要）")
    parser.add_argument("--stage2-checkpoint", default=None,
                        help="Stage 2 的权重（stage2 路线需要）")
    args = parser.parse_args()

    print("=" * 64)
    print("  生成平台样例（成员 B）")
    print("=" * 64)

    data_root = Path(args.data_root)
    sample_dir = paths.sample_dir(args.method_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    input_path = sample_dir / "input.wav"

    if args.from_wav:
        # 直接用给定的音频（统一转成 16 kHz 单声道 16-bit PCM）
        source = Path(args.from_wav)
        if not source.exists():
            print(f"\n[失败] 音频不存在：{source}")
            return 1

        wav = audio_utils.load_audio(source, sample_rate=16000)
        audio_utils.save_audio(input_path, wav, 16000)
        speaker = "未知"
        rel_path = str(source)
        origin = args.source_note or f"上游仓库样例 {source.name}"
        print(f"\n  输入样例 : {input_path}")
        print(f"  来源     : {origin}")
        print(f"  已转成   : 16 kHz / 单声道 / PCM_16（{len(wav) / 16000:.2f}s）")
    else:
        manifest = data_root / "test.txt"
        if not manifest.exists():
            print(f"\n[失败] 找不到 {manifest}，请先跑 prepare_data.py。")
            print("       （VCTK 还没下好时可以用 --from-wav 指定一条音频）")
            return 1

        rel_path, speaker = pick_utterance(manifest, args.utterance)
        source = data_root / rel_path
        if not source.exists():
            print(f"\n[失败] 音频不存在：{source}")
            return 1

        shutil.copyfile(source, input_path)
        origin = f"VCTK 测试集（说话人 {speaker}）"
        print(f"\n  输入样例 : {input_path}")
        print(f"  来源     : {rel_path}（说话人 {speaker}，来自测试集）")

    # -- 尝试跑真实推理，生成输出 ------------------------------------------
    output_path = sample_dir / "output.wav"
    produced_output = False
    if not args.no_output:
        sys.path.insert(0, str(REPO_ROOT / "algorithms" / "voiceprint_protection"))
        try:
            import infer

            sample_rate = config.MEL_PRESETS[config.DEFAULT_PRESET]["sample_rate"]

            if hasattr(infer, "protect"):
                # 约定一：模块级 protect(wav_path, alpha=..., output_path=...)
                result = infer.protect(
                    str(input_path), alpha=args.alpha, output_path=str(output_path),
                    mode=args.mode,
                    stage1_checkpoint=args.stage1_checkpoint,
                    stage2_checkpoint=args.stage2_checkpoint,
                )
                produced_output = True
                print(f"  输出样例 : {result.get('protected_path', output_path)}")
                shapes = result.get("shapes", {})
                if shapes:
                    print(f"  张量形状 : {shapes}")
            elif hasattr(infer, "Protector"):
                # 约定二（成员 A 当前的实现）：Protector.protect(wav, alpha=...)
                protector = infer.Protector()
                wav = audio_utils.load_audio(str(input_path), sample_rate)
                result = protector.protect(wav, alpha=args.alpha, verbose=False)
                audio_utils.save_audio(output_path, result["waveform"], sample_rate)
                produced_output = True
                print(f"  输出样例 : {output_path}")
                shapes = result.get("shapes", {})
                if shapes:
                    print(f"  张量形状 : {shapes}")
            else:
                print("  [提示] infer.py 还没有 protect()/Protector，跳过生成输出")
        except ImportError:
            print("  [提示] infer.py 还不存在（成员 A 主责），跳过生成输出")
        except Exception as error:  # noqa: BLE001
            print(f"  [警告] 推理失败：{error}")

    # -- 写来源说明 --------------------------------------------------------
    lines = [
        f"# 样例说明 · {args.method_id}",
        "",
        "## 文件",
        "",
    ]
    lines.append("| 文件 | 说明 |")
    lines.append("|---|---|")
    lines.append("| `input.wav` | 原始语音（VCTK 测试集说话人） |")
    if produced_output:
        lines.append(f"| `output.wav` | 保护后语音，由本项目推理链实际生成"
                     f"（α = {args.alpha}，保护链 = {args.mode}） |")
    else:
        lines.append("| `output.wav` | **待生成** —— 需要成员 A 的 `infer.py` 就位后重跑本脚本 |")
    lines.append("")
    lines.append("## 参数")
    lines.append("")
    lines.append("```json")
    lines.append(f'{{ "alpha": {args.alpha} }}')
    lines.append("```")
    lines.append("")
    lines.append(f"保护链：`{args.mode}`"
                 + ("（论文原版：x_adv = D(c, h_adv) → HiFi-GAN，"
                    "δ 来自训练好的生成器 G）"
                    if args.mode == "stage2" else ""))
    lines.append("")
    lines.append(VCTK_LICENSE_NOTE)
    lines.append("## 具体条目")
    lines.append("")
    lines.append(f"- 来源：{origin}")
    lines.append(f"- 原始文件：`{rel_path}`")
    if speaker != "未知":
        lines.append(f"- 说话人：`{speaker}`（测试集，未参与训练）")
    lines.append("")
    (sample_dir / "SOURCES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  来源说明 : {sample_dir / 'SOURCES.md'}")

    if not produced_output:
        print("\n  [待补] output.wav 还没生成 —— 平台样例要求输入输出真实对应，")
        print("         等 infer.py 就绪后重跑本脚本。")
        return 2

    print("\n  样例已就绪，可写进方法 JSON 的 examples 字段。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
