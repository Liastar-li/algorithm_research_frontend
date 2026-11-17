"""路径解析 —— 统一处理"数据到底放在哪个盘"的问题。

【为什么需要这个文件】
本项目要吃 **VCTK 数据集**（压缩包 11 GB + 解压 11 GB + 16 kHz 副本约 5 GB ≈ 27 GB）。
C 盘往往放不下，而**代码仓库本身只有 77 MB**，没必要搬。

所以约定：**仓库留在原盘，数据放另一个盘**，由本模块统一解析，
脚本里不再各写各的 `Path("data")`。

【优先级】
1. 环境变量 `VPP_DATA_ROOT`  —— 显式指定数据根目录（推荐，跨盘最干净）
2. `config.DATA_ROOT`        —— 默认为 `data`，相对仓库根解析
3. 两者都不设时，第 2 条等价于 `<仓库根>/data`

【两种把数据放到 D 盘的办法】

A. 环境变量（推荐，路径写在文档里，一眼可见）
   ```powershell
   setx VPP_DATA_ROOT "D:\\vpp\\data"        # 永久生效，新开终端
   $env:VPP_DATA_ROOT = "D:\\vpp\\data"       # 只在当前窗口生效
   ```
   原始数据放 `D:\\vpp\\data\\raw\\`，转换结果也落在 `D:\\vpp\\data\\`。

B. 目录联接（junction，零代码改动，脚本里的 `data/` 照常能用）
   ```powershell
   cmd //c mklink /J "C:\\Users\\z\\Desktop\\26秋\\语音识别\\语音识别code\\data" "D:\\vpp\\data"
   ```
   `data/` 这个路径实际指向 D 盘，Python / git / VSCode 全都透明跟随。
   （**注意**：junction 只在本地 NTFS 之间有效；D 盘不是 NTFS 的话用办法 A。）

【同理，第三方组件也可以挪走】
`bridge_speaker.py` 已经支持环境变量 `VPP_RTC_DIR`（见该文件的 `find_rtc_dir`）。

【不要改 config.py / audio_utils.py 的约定】
这两个是"唯一真源"，改动需要两人同意。本模块只做路径解析，不碰超参与 mel 约定。
"""

from __future__ import annotations

import os
from pathlib import Path

# algorithms/voiceprint_protection/  ->  algorithms/  ->  仓库根
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent


def data_root() -> Path:
    """数据根目录（train.txt / test.txt / audio16k / raw 都在它下面）。"""
    env = os.environ.get("VPP_DATA_ROOT")
    if env:
        return Path(env).expanduser()

    import config

    candidate = Path(config.DATA_ROOT)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def raw_root() -> Path:
    """VCTK 解压后的原始数据目录。"""
    return data_root() / "raw"


def sample_dir(method_id: str) -> Path:
    """平台样例目录（必须留在仓库里，前端要按 public/ 相对路径访问）。"""
    return REPO_ROOT / "public" / "samples" / method_id


def models_dir() -> Path:
    """第三方预训练权重目录（GE2E 的 encoder.pt 等）。

    优先级：环境变量 `VPP_MODELS_DIR` > `<数据根>/models` > `<仓库根>/models`。
    权重加起来有 1 GB 级别（Whisper 更大），放 D 盘更合适。
    """
    env = os.environ.get("VPP_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    return data_root() / "models"


def weights_dir() -> Path:
    """checkpoint 目录。默认在仓库里；放不下时可改指向 D 盘。"""
    import config

    candidate = Path(config.WEIGHTS_DIR)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def describe() -> str:
    """打印当前解析结果，方便两个人比对环境。"""
    return (
        f"仓库根 : {REPO_ROOT}\n"
        f"数据根 : {data_root()}"
        f"{'（来自 VPP_DATA_ROOT）' if os.environ.get('VPP_DATA_ROOT') else ''}\n"
        f"原始数据: {raw_root()}\n"
        f"外部权重: {models_dir()}"
        f"{'（来自 VPP_MODELS_DIR）' if os.environ.get('VPP_MODELS_DIR') else ''}\n"
        f"自带权重: {weights_dir()}"
    )


def print_env_hint(error: BaseException | None = None) -> None:
    """依赖缺失时，给出"忘了激活 conda 环境"的人话提示。

    这是本项目**最高频的一个坑**：新开终端忘记 `conda activate vpp`，
    于是跑的是系统 Python，`import numpy` 直接炸。
    """
    import sys

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
