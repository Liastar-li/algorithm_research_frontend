"""路径解析 —— 第三方组件 + 数据集 + 权重的统一入口。

【为什么要单独一个文件】
两件互相独立又都很容易踩坑的事，都放这里：

1. **第三方官方代码**（`third_party/`）里放的是**别人写的代码**，我们不能改动它们
   （改动就要记录 diff，而且升级时会被覆盖）。所以统一在这里
   (a) 把官方仓库目录加进 sys.path，好让 `import model_vc` / `import wavenet_vocoder` 生效
   (b) 集中管理权重文件的位置

2. **数据集与权重的位置**。VCTK 有 11 GB，仓库本身只有 77 MB，
   所以"数据放别的盘、仓库留原盘"是本项目的常态（见 docs/notes/data-location.md），
   由下面的环境变量决定。

【形象比喻】
third_party 像"图书馆里的借来的书"—— 我们在书旁边贴便签做笔记，
但绝不直接在书上划线改字。

【环境变量】
| 变量 | 作用 | 默认 |
|---|---|---|
| `VPP_DATA_ROOT`   | 数据根（VCTK、清单、16 kHz 副本） | `<仓库根>/data` |
| `VPP_THIRD_PARTY_DIR` | 第三方官方代码的根 | `<仓库根>/third_party` |
| `VPP_RTC_DIR`     | Real-Time-Voice-Cloning 仓库 | `<第三方根>/Real-Time-Voice-Cloning` |
| `VPP_MODELS_DIR`  | 外部预训练权重（GE2E 的 encoder.pt 等） | `<数据根>/models` |
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import config

# algorithms/voiceprint_protection/paths.py -> algorithms/voiceprint_protection
MODULE_DIR = Path(__file__).resolve().parent
# -> algorithms
ALGORITHMS_DIR = MODULE_DIR.parent
# -> 仓库根
REPO_ROOT = ALGORITHMS_DIR.parent

# --- 第三方组件的根 -------------------------------------------------------
# 可用环境变量挪到别的盘（仓库在 C 盘、数据在 D 盘是本项目的常态）
THIRD_PARTY_DIR = Path(
    os.environ.get("VPP_THIRD_PARTY_DIR") or (REPO_ROOT / "third_party")
)

# AutoVC 官方仓库（MIT）。含 model_vc.py / model_bl.py / hparams.py / synthesis.py
AUTOVC_DIR = THIRD_PARTY_DIR / "AutoVC"

# WaveNet 声码器（r9y9）。★ 必须用 commit d4333546（2019-08-25）——
#   这是**最后一个还含 builder.py 的提交**。AutoVC 的 synthesis.py 依赖
#   `from wavenet_vocoder import builder`，而新版（2019-08-26 起）删掉了 builder.py
#   且模型结构有变，会导致权重加载失败。
#   ★ 该 commit 的 wavenet_vocoder/version.py 在 git 里不存在 —— 那个年代它是
#     由 setup.py 在打包时自动生成的（见其 build_py.create_version_file）。
#     我们按同样方式生成了一份，否则 `import wavenet_vocoder` 会直接报
#     "No module named 'wavenet_vocoder.version'"。
WAVENET_DIR = THIRD_PARTY_DIR / "wavenet_vocoder"
# ★ 只把**包含 package 的这一层**放进 sys.path，不能放仓库根目录！
#   因为 wavenet_vocoder 仓库根目录也有一个 hparams.py，
#   若它先进了 sys.path，AutoVC 的 `from hparams import hparams` 就会拿到
#   声码器的 hparams（完全不同的配置）-> 模型结构错、权重加载失败、且不报错。
WAVENET_PACKAGE_PARENT = WAVENET_DIR

# Real-Time-Voice-Cloning（GE2E 说话人编码器，论文脚注引用）
# ★ 目录名两种都认：clone 时可能叫 RTC，也可能叫 Real-Time-Voice-Cloning
_RTC_CANDIDATES = ("Real-Time-Voice-Cloning", "RTC")


def _resolve_rtc_dir() -> Path:
    env = os.environ.get("VPP_RTC_DIR")
    if env:
        return Path(env).expanduser()
    for name in _RTC_CANDIDATES:
        candidate = THIRD_PARTY_DIR / name
        if candidate.exists():
            return candidate
    return THIRD_PARTY_DIR / _RTC_CANDIDATES[0]


RTC_DIR = _resolve_rtc_dir()

# 权重目录。★ 不提交到 git（.gitignore 已忽略 *.pt / *.pth / *.safetensors）
WEIGHTS_DIR = MODULE_DIR / "weights"

# 期望的权重文件（AutoVC 官方 README 提供的 IBM Box 链接）
WEIGHT_FILES = config.WEIGHT_FILES

# 组件清单的唯一真源在 config.THIRD_PARTY_COMPONENTS，
# scripts/setup_third_party.ps1 / .sh 也读同一份（见那两个脚本里的同名表）。
COMPONENTS = config.THIRD_PARTY_COMPONENTS


# ---------------------------------------------------------------------------
# 数据集与权重的位置
# ---------------------------------------------------------------------------
def data_root() -> Path:
    """数据根目录（train.txt / test.txt / audio16k / raw 都在它下面）。"""
    env = os.environ.get("VPP_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    candidate = Path(config.DATA_ROOT)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def raw_root() -> Path:
    """VCTK 原始数据目录。"""
    return data_root() / "raw"


def models_dir() -> Path:
    """外部预训练权重目录（GE2E 的 encoder.pt 等）。

    权重加起来有 1 GB 级别（Whisper 更大），放 D 盘更合适。

    ★ **默认不跟随 VPP_DATA_ROOT**：模型权重和数据是**两回事**。
      把 VPP_DATA_ROOT 指到别处（冒烟测试的假数据、云机的数据盘）时，
      预训练权重不应该跟着搬家 —— 否则 `find_weights` 会突然找不到 encoder.pt。
      要换权重目录请用 `VPP_MODELS_DIR`。
    """
    env = os.environ.get("VPP_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    return REPO_ROOT / "data" / "models"


def sample_dir(method_id: str) -> Path:
    """平台样例目录（必须留在仓库里，前端要按 public/ 相对路径访问）。"""
    return REPO_ROOT / "public" / "samples" / method_id


# ---------------------------------------------------------------------------
# 第三方组件
# ---------------------------------------------------------------------------
def ensure_import_paths() -> None:
    """把第三方官方代码目录加入 sys.path（可重复调用）。

    ⚠️ **顺序很重要**：AutoVC 目录必须排在 WaveNet 目录**前面**。
    原因：AutoVC 的 synthesis.py 里 `from hparams import hparams`，
    而 wavenet_vocoder 仓库根目录也有一个 hparams.py（配置完全不同）。
    顺序反了就会加载错的那份 -> 模型结构不对、权重加载失败、**且不报错**。
    """
    for directory in (WAVENET_PACKAGE_PARENT, AUTOVC_DIR, RTC_DIR):
        if not directory.exists():
            continue
        text = str(directory)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    # 保证 AutoVC 在最前
    if AUTOVC_DIR.exists():
        text = str(AUTOVC_DIR)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


def check_version_file() -> str:
    """检查 wavenet_vocoder/version.py 是否已生成。

    那个年代它是 setup.py 打包时自动生成的，git 里没有。
    缺了它 `import wavenet_vocoder` 会直接失败。
    """
    target = WAVENET_DIR / "wavenet_vocoder" / "version.py"
    if target.exists():
        return "[OK]   " + str(target)

    script = (
        "import os, subprocess; "
        "cwd = 'third_party/wavenet_vocoder'; "
        "sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=cwd)"
        ".decode('ascii').strip(); "
        "v = '0.1.1+' + sha[:7]; "
        "p = os.path.join(cwd, 'wavenet_vocoder', 'version.py'); "
        "open(p, 'w').write(\"__version__ = '\" + v + \"'\\n\"); "
        "print('已生成', p)"
    )
    return (
        "[缺失] " + str(target) + "\n"
        "         生成方式（与官方 setup.py 的 build_py.create_version_file 一致）：\n"
        "         python -c \"" + script + "\""
    )


def weight_path(key: str) -> Path:
    """按名字取权重文件路径。

    查找顺序：
      1. algorithms/voiceprint_protection/weights/<名字>
      2. 仓库根 weights/<名字>       （方便集中放）
      3. third_party/AutoVC/<名字>    （AutoVC 官方 README 的原始目录结构）
      4. <外部权重目录>/<名字>        （VPP_MODELS_DIR）
    """
    name = WEIGHT_FILES.get(key, key)
    candidates = [
        WEIGHTS_DIR / name,
        REPO_ROOT / "weights" / name,
        AUTOVC_DIR / name,
        models_dir() / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # 都没找到时返回首选路径，让调用方给出明确的"缺失"提示
    return WEIGHTS_DIR / name


def describe() -> str:
    """生成一段人类可读的组件与路径状态说明，用于 README 与自检输出。"""
    lines = ["第三方组件状态："]
    for component in COMPONENTS:
        directory = THIRD_PARTY_DIR / component["name"]
        label = component["name"]
        mark = "[OK]  " if directory.exists() else "[缺失]"
        lines.append(f"  {mark} {label:<26} {directory}")
        if directory.exists():
            head = directory / ".git" / "HEAD"
            if head.exists():
                lines.append(f"         HEAD: {head.read_text(encoding='utf-8').strip()}")
    lines.append("权重文件：")
    for key, name in WEIGHT_FILES.items():
        path = weight_path(key)
        mark = "[OK]  " if path.exists() else "[缺失]"
        size = f"{path.stat().st_size / 1024 ** 2:.1f} MB" if path.exists() else ""
        lines.append(f"  {mark} {name:<38} {size}")
    lines.append("wavenet_vocoder/version.py（打包时生成）：")
    lines.append("  " + check_version_file())
    lines.append("")
    lines.append("数据与路径：")
    lines.append(f"  仓库根      : {REPO_ROOT}")
    lines.append(f"  数据根      : {data_root()}"
                 + ("（来自 VPP_DATA_ROOT）" if os.environ.get("VPP_DATA_ROOT") else ""))
    lines.append(f"  原始数据    : {raw_root()}")
    lines.append(f"  外部权重    : {models_dir()}"
                 + ("（来自 VPP_MODELS_DIR）" if os.environ.get("VPP_MODELS_DIR") else ""))
    lines.append(f"  第三方根    : {THIRD_PARTY_DIR}")
    lines.append(f"  RTC 仓库    : {RTC_DIR}")
    lines.append("")
    lines.append("缺少组件时运行：powershell -ExecutionPolicy Bypass -File scripts/setup_third_party.ps1")
    return "\n".join(lines)


def print_env_hint(error: BaseException | None = None) -> None:
    """依赖缺失时，给出"忘了激活 conda 环境"的人话提示。

    这是本项目**最高频的一个坑**：新开终端忘记 `conda activate vpp`，
    于是跑的是系统 Python，`import numpy` 直接炸。
    """
    print("", file=sys.stderr)
    print("=" * 64, file=sys.stderr)
    print("  [环境不对] 依赖缺失", file=sys.stderr)
    if error is not None:
        print(f"  {error}", file=sys.stderr)
    print("", file=sys.stderr)
    print("  最常见的原因：**新开的终端没有激活 conda 环境**，", file=sys.stderr)
    print("  于是跑的是系统 Python（没装 numpy/torch），而不是我们的 vpp 环境。", file=sys.stderr)
    print("", file=sys.stderr)
    print("  先执行：", file=sys.stderr)
    print("      conda activate vpp", file=sys.stderr)
    print("", file=sys.stderr)
    print("  如果 conda activate 报错（cmd 里要初始化一次）：", file=sys.stderr)
    print("      conda init cmd.exe        # 之后重开终端", file=sys.stderr)
    print("      conda run -n vpp python <脚本>", file=sys.stderr)
    print("      D:\\conda\\envs\\vpp\\python.exe <脚本>   # 直接用绝对路径", file=sys.stderr)
    print("", file=sys.stderr)
    print("  怎么确认当前用的是哪个 python：", file=sys.stderr)
    print("      python -c \"import sys, numpy, torch; print(sys.executable); print(numpy.__version__, torch.__version__)\"", file=sys.stderr)
    print("=" * 64, file=sys.stderr)
