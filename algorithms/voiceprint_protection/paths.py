"""第三方组件的路径与权重管理。

【为什么要单独一个文件】
`third_party/` 里放的是**别人写的官方代码**，我们不能改动它们（改动就要记录 diff，
而且升级时会被覆盖）。所以统一在这里做两件事：
  1. 把官方仓库目录加进 sys.path，好让 `import model_vc` / `import wavenet_vocoder` 生效
  2. 集中管理权重文件的位置

【形象比喻】
third_party 像"图书馆里的借来的书"—— 我们在书旁边贴便签做笔记，
但绝不直接在书上划线改字。
"""

from __future__ import annotations

import sys
from pathlib import Path

import config

# algorithms/voiceprint_protection/paths.py -> algorithms/voiceprint_protection
MODULE_DIR = Path(__file__).resolve().parent
# -> algorithms
ALGORITHMS_DIR = MODULE_DIR.parent
# -> 仓库根
REPO_ROOT = ALGORITHMS_DIR.parent

THIRD_PARTY_DIR = REPO_ROOT / "third_party"

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
RTC_DIR = THIRD_PARTY_DIR / "Real-Time-Voice-Cloning"

# 权重目录。★ 不提交到 git（.gitignore 已忽略 *.pt / *.pth / *.safetensors）
WEIGHTS_DIR = MODULE_DIR / "weights"

# 期望的权重文件（AutoVC 官方 README 提供的 IBM Box 链接）
WEIGHT_FILES = config.WEIGHT_FILES

# 组件清单的唯一真源在 config.THIRD_PARTY_COMPONENTS，
# scripts/setup_third_party.ps1 / .sh 也读同一份（见那两个脚本里的同名表）。
COMPONENTS = config.THIRD_PARTY_COMPONENTS


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
    """
    name = WEIGHT_FILES.get(key, key)
    candidates = [
        WEIGHTS_DIR / name,
        REPO_ROOT / "weights" / name,
        AUTOVC_DIR / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # 都没找到时返回首选路径，让调用方给出明确的"缺失"提示
    return WEIGHTS_DIR / name


def describe() -> str:
    """生成一段人类可读的组件状态说明，用于 README 与自检输出。"""
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
    lines.append("缺少组件时运行：powershell -ExecutionPolicy Bypass -File scripts/setup_third_party.ps1")
    return "\n".join(lines)
