"""下载 GE2E 说话人编码器的预训练权重（成员 B 主责）。

【为什么需要这个脚本】
2026-03 之后的上游 Real-Time-Voice-Cloning **不再把权重放在仓库里**了，
改成运行时从 Hugging Face 自动下载：

    repo: CorentinJ/SV2TTS
    file: encoder.pt   （17,090,379 字节）

我们不想在推理时联网（服务部署环境常常没有外网），所以提前下好、校验、存到本地。

【它只下 encoder.pt】
上游的 `ensure_default_models()` 会把 encoder(17MB) + synthesizer(353MB) +
vocoder(51MB) 全下下来。本项目**只需要说话人编码器**，
所以这里做定点下载，只拿 17 MB。

【国内网络】
Hugging Face 在国内可能很慢或连不上，用镜像即可：

    set HF_ENDPOINT=https://hf-mirror.com

【用法】
    conda activate vpp
    python algorithms/voiceprint_protection/scripts/download_encoder_weights.py

    # 放到 D 盘
    set VPP_MODELS_DIR=D:\\vpp\\models
    python algorithms/voiceprint_protection/scripts/download_encoder_weights.py

【校验】
下载完会核对文件大小必须**精确等于** 17,090,379 字节，
不一致说明下到了错误页面/截断文件，会明确报错而不是留个坏文件。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: E402

HF_REPO = "CorentinJ/SV2TTS"
HF_FILENAME = "encoder.pt"
# 上游 utils/default_models.py 里写死的期望大小
EXPECTED_SIZE = 17_090_379
DEFAULT_MIRROR = "https://hf-mirror.com"


def _candidate_dirs(explicit: str | None) -> list[Path]:
    dirs: list[Path] = []
    if explicit:
        dirs.append(Path(explicit))
    dirs.append(paths.models_dir() / "default")
    dirs.append(paths.models_dir())
    return dirs


def download_with_hub(target_dir: Path) -> Path | None:
    """方式一：用 huggingface_hub（推荐，支持断点续传与镜像）。"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  [跳过] 没装 huggingface_hub")
        return None

    print(f"  用 huggingface_hub 下载（endpoint = {os.environ.get('HF_ENDPOINT', '默认')}）...")
    try:
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_FILENAME,
            local_dir=str(target_dir),
        )
        return Path(path)
    except Exception as error:  # noqa: BLE001
        print(f"  [失败] {error}")
        return None


def download_with_urllib(target_dir: Path) -> Path | None:
    """方式二：直链下载（不需要 huggingface_hub，认 HF_ENDPOINT 镜像）。"""
    import urllib.error
    import urllib.request

    endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    base = endpoint if endpoint else "https://huggingface.co"
    url = f"{base}/{HF_REPO}/resolve/main/{HF_FILENAME}"

    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / HF_FILENAME
    print(f"  直链下载：{url}")

    try:
        with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as handle:
            total = 0
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                total += len(chunk)
                print(f"\r    已下载 {total / 1024 / 1024:.1f} MB", end="", flush=True)
        print()
        return target
    except (urllib.error.URLError, OSError) as error:
        print(f"  [失败] {error}")
        return None


def verify(path: Path) -> bool:
    """核对大小与文件头 —— 防止把错误页面当成权重存下来。"""
    if not path.exists():
        print(f"  [失败] 文件不存在：{path}")
        return False

    size = path.stat().st_size
    print(f"  文件大小：{size:,} 字节（期望 {EXPECTED_SIZE:,}）")
    if size != EXPECTED_SIZE:
        print("  [失败] 大小不符 —— 多半是下载被截断或拿到了错误页面。")
        print("          请删除该文件后重试；国内网络建议先设：")
        print("            set HF_ENDPOINT=https://hf-mirror.com")
        return False

    with path.open("rb") as handle:
        head = handle.read(4)
    if head[:1] == b"<":
        print("  [失败] 文件开头是 '<'，这是 HTML 错误页，不是权重。")
        return False

    print("  [通过] 大小与文件头都正常")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 GE2E 权重 encoder.pt（成员 B）")
    parser.add_argument("--dir", default=None, help="下载目录（默认 <数据根>/models/default）")
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = parser.parse_args()

    print("=" * 64)
    print("  下载 GE2E 权重 encoder.pt")
    print("=" * 64)
    print("\n  路径解析：")
    for line in paths.describe().splitlines():
        print(f"    {line}")

    target_dir = _candidate_dirs(args.dir)[0]
    target = target_dir / HF_FILENAME

    if target.exists() and not args.force:
        print(f"\n  已存在：{target}")
        if verify(target):
            print("\n  无需下载。")
            return 0
        print("  将重新下载。")

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  目标目录：{target_dir}")

    downloaded = download_with_hub(target_dir)
    if downloaded is None:
        print("  改用直链下载…")
        downloaded = download_with_urllib(target_dir)

    if downloaded is None:
        print("\n  [失败] 两种方式都没成功。可以手动下载：")
        print(f'    浏览器打开 https://hf-mirror.com/{HF_REPO}/resolve/main/{HF_FILENAME}')
        print(f"    存成 {target}")
        print(f"    文件大小应为 {EXPECTED_SIZE:,} 字节")
        return 1

    print(f"\n  已下载：{downloaded}")
    if not verify(downloaded):
        return 1

    print("\n  下一步：")
    print("    python algorithms/voiceprint_protection/bridge_speaker.py --wav <一条音频.wav>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
