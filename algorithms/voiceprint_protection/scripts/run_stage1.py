"""Stage 1 一条龙：数据对账 → 训 W_s → 训解耦网络 → 验收（M3）。

【为什么要有这个脚本】
Stage 1 有 5 个步骤，每步都有"看什么、什么算对"。
新手最容易犯的错是**跳步**（比如没训 W_s 就训解耦网络），
而这类错误的症状是"loss 在降但指标是假的"，**不报错**。
所以这里把它做成一条命令，每步都打印**该看哪个数**、**什么范围算对**。

【前置：数据】
```powershell
node scripts/download_vctk.mjs --out <仓库根>/data/raw --per-speaker 100 --concurrency 48
python algorithms/voiceprint_protection/scripts/prepare_data.py
```
（`prepare_data.py` 会自跑三条硬要求自检：划分不重叠 / K=80 / 无 p*_eval）

【用法】
```powershell
conda activate vpp

# 先冒烟（几分钟，确认全链路通）
python algorithms/voiceprint_protection/scripts/run_stage1.py --smoke

# 正式训练（先 20k 看曲线，再拉长）
python algorithms/voiceprint_protection/scripts/run_stage1.py --iterations 20000

# 只训到某一步（调试用）
python algorithms/voiceprint_protection/scripts/run_stage1.py --steps prepare,train_classifier
```
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ALGO_DIR = REPO_ROOT / "algorithms" / "voiceprint_protection"
sys.path.insert(0, str(ALGO_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config  # noqa: E402
import paths  # noqa: E402

PYTHON = sys.executable


def run(command: list[str], title: str) -> int:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print("  $ " + " ".join(str(part) for part in command) + "\n")
    start = time.time()
    result = subprocess.run(command, cwd=str(REPO_ROOT))
    print(f"\n  [{title}] 退出码 {result.returncode}，耗时 {(time.time() - start) / 60:.1f} 分钟")
    return result.returncode


def banner(text: str) -> None:
    print()
    print("#" * 72)
    print(f"#  {text}")
    print("#" * 72)


def check_data() -> bool:
    """检查清单齐不齐，并顺便对账划分。"""
    banner("第 0 步 · 检查数据")
    data_root = paths.data_root()
    print(f"  数据根目录：{data_root}")

    missing = [name for name in ("train.txt", "val.txt", "test.txt", "spk2id.txt")
               if not (data_root / name).exists()]
    if missing:
        print(f"  [失败] 缺清单：{missing}")
        print("\n  先做这两步：")
        print(f"    node scripts/download_vctk.mjs --out \"{data_root / 'raw'}\" "
              f"--per-speaker 100 --concurrency 48")
        print("    python algorithms/voiceprint_protection/scripts/prepare_data.py")
        return False

    for name in ("train.txt", "val.txt", "test.txt"):
        count = sum(1 for line in (data_root / name).read_text(encoding="utf-8").splitlines()
                    if line.strip())
        print(f"  [OK] {name:<12} {count} 行")
    speakers = (data_root / "spk2id.txt").read_text(encoding="utf-8").splitlines()
    speakers = [line for line in speakers if line.strip()]
    print(f"  [OK] spk2id.txt   K = {len(speakers)}"
          f"（L_dis 的分类器 Ws 就是 K×256，K 错了建不起来）")

    # 划分对账：与另一台机器的划分逐字对比（名单 + seed 就能复现）
    split_script = ALGO_DIR / "scripts" / "split_speakers.py"
    if split_script.exists():
        print("\n  ── 划分对账（与另一台机器比对）──")
        result = subprocess.run(
            [PYTHON, str(split_script), "--check", str(data_root / "split_speakers.json")],
            cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
        for line in (result.stdout or "").splitlines()[-8:]:
            print("  " + line)
        if result.returncode != 0:
            print("  [警告] 划分对账没过 —— 先看上面的输出。")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1 一条龙")
    parser.add_argument("--steps", default="prepare,train_classifier,train_model,evaluate",
                        help="要跑哪几步（逗号分隔）")
    parser.add_argument("--iterations", type=int, default=20000,
                        help="第 2 步（解耦网络）的训练步数")
    parser.add_argument("--classifier-iterations", type=int, default=2000,
                        help="第 1 步（W_s）的训练步数")
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟模式：几步就跑完，只验证链路通不通")
    parser.add_argument("--eval-num", type=int, default=20, help="验收时评测多少条")
    args = parser.parse_args()

    steps = {name.strip() for name in args.steps.split(",") if name.strip()}
    iterations = 20 if args.smoke else args.iterations
    classifier_iterations = 3 if args.smoke else args.classifier_iterations
    train_script = ALGO_DIR / "stage1" / "train.py"

    print("=" * 72)
    print("  Stage 1 一条龙（成员 A 主责）")
    print("=" * 72)
    print(f"  步骤      : {', '.join(sorted(steps))}")
    print(f"  迭代      : W_s {classifier_iterations} 步 / 解耦网络 {iterations} 步")
    print(f"  冒烟模式  : {'是' if args.smoke else '否'}")

    if "prepare" in steps:
        if not check_data():
            return 1

    # ---- 第 1 步：训说话人分类器 W_s ------------------------------------
    if "train_classifier" in steps:
        banner("第 1 步 · 训练说话人分类器 W_s（★ 顺序不能反）")
        print("  【看什么】最后一行「训练集准确率」")
        print("  【什么算对】越高越好（VCTK 80 类、GE2E 特征，预期 90%+）")
        print("  【为什么必须先做】W_s 是「听力正常的人」。随机初始化的话，")
        print("                后面的熵最大化会退化成「让一个瞎子输出均匀」，")
        print("                约束形同虚设 —— 而且不报错。")
        code = run([PYTHON, str(train_script), "--stage", "classifier",
                    "--iterations", str(classifier_iterations)],
                   "第 1 步 · 训练 W_s")
        if code != 0:
            return code

    # ---- 第 2 步：训解耦-重建网络 --------------------------------------
    if "train_model" in steps:
        banner("第 2 步 · 训练解耦-重建网络（Stage 1 主体）")
        print("  【看什么】rec / cd / dis 三条数")
        print("  【什么算对】")
        print("    · rec ↓ 稳定下降（重建在变好）")
        print("    · cd  ↓ 下降（内容没丢）")
        print("    · dis 稳在 log(80) = 4.38 附近")
        print("      （= 分类器完全猜不出说话人 = ★解耦成功）")
        print("  【别直接上 50 万步】先用 5k~20k 看曲线形状，形状对了再拉长。")
        print("  【断了怎么办】重跑加 --resume，从最近的断点接着跑。")
        code = run([PYTHON, str(train_script), "--stage", "model",
                    "--iterations", str(iterations),
                    "--save-every", "2000", "--val-every", "2000"],
                   "第 2 步 · 训练解耦-重建网络")
        if code != 0:
            return code

    # ---- 第 3 步：验收（M3）--------------------------------------------
    if "evaluate" in steps:
        banner("第 3 步 · 验收（M3 的三条标准）")
        print("  【M3-1】有 checkpoint      → weights/stage1_model.pt 存在")
        print("  【M3-2】重建音听感正常    → ★**必须真的听一遍** results/stage1_eval/ 里的")
        print("                              *_original.wav 与 *_rebuilt.wav")
        print("                              （指标好看 ≠ 听感正常）")
        print("  【M3-3】Sim_prot > 0.9    → 脚本会直接打印「★ M3 验收：通过/未通过」")
        print("  【辅助】内容码泄露率越低越好（= 解耦成功）；")
        print("         重建音说话人准确率越高越好（= 音色保真）")
        code = run([PYTHON, str(ALGO_DIR / "scripts" / "eval_stage1.py"),
                    "--split", "test", "--num", str(args.eval_num), "--save-audio"],
                   "第 3 步 · 验收")
        if code != 0:
            return code

    banner("全部完成")
    print("  ✔ Stage 1 跑完了。")
    print()
    print("  接下来：")
    print("    ① ★ **听一遍** results/stage1_eval/ 里的重建音频（M3-2，人耳验收）")
    print("    ② 契约 §7.5 说 Stage 2 要吃 Stage 1 的 c / x / h ——")
    print("       现在 checkpoint 有了，Stage 2 的阻塞点②也解掉了：")
    print("         python algorithms/voiceprint_protection/stage2/train.py \\")
    print("             --vocoder hifigan --stage1-checkpoint "
          "<weights/stage1_model.pt> --steps 200000")
    print("    ③ 要跑到论文量级（Stage 1 500k / Stage 2 200k）就得上云，")
    print("       本机 4060 按 0.3 s/step 算：Stage 1 50 万步 ≈ 42 小时")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
