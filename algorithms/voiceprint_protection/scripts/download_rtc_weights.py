"""下载上游 Real-Time-Voice-Cloning 的合成器与声码器权重（成员 A 主责）。

【为什么需要】
AutoVC 官方权重链（IBM Box）实测 **404 失效**，拿不到主模型与配套声码器。
于是 L1 推理链改用上游 RTC 的合成器 + 声码器（HF `CorentinJ/SV2TTS`）：
    encoder.pt      17 MB   GE2E 说话人编码器  （成员 B 的 download_encoder_weights.py 负责）
    synthesizer.pt 371 MB   Tacotron 合成器     ← 本脚本
    vocoder.pt      54 MB   WaveRNN 声码器     ← 本脚本

【为什么这两个是必需的】
没有它们就没有"攻击者"：
论文的核心主张是"扰动加在说话人嵌入上 -> 攻击者克隆出的声音就不像本人了"。
没有合成器就没法克隆，没有声码器就没法把克隆结果变成能听的音频。

【★ 校验方式：不写死文件大小，改为向 HF 查询】
上游权重会更新，写死的字节数迟早误报（我们就被坑过一次：
写死 353,426,445 而实际是 370,554,559，文件明明下好了却被判成"失败"）。
所以这里：
  1. 先问 HF："这个文件现在多大、sha256 是多少"
  2. 下载后比对**大小和 sha256**
  3. 问不到（离线）时退回"确认不是 HTML 错误页 + 大小合理"

【国内网络】
Hugging Face 在国内可能很慢或连不上，用镜像即可：
    set HF_ENDPOINT=https://hf-mirror.com

【用法】
    conda activate vpp
    python algorithms/voiceprint_protection/scripts/download_rtc_weights.py

    # 放到 D 盘（数据目录）
    set VPP_MODELS_DIR=D:\\vpp\\models
    python algorithms/voiceprint_protection/scripts/download_rtc_weights.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: E402

HF_REPO = "CorentinJ/SV2TTS"

# 本脚本负责的两个文件（encoder.pt 由成员 B 的脚本负责）
FILES = ["synthesizer.pt", "vocoder.pt"]

# 兜底：问不到 HF 时的"最小合理大小"（用于排除 HTML 错误页 / 空文件）
MIN_PLAUSIBLE_SIZE = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_remote_info(filename: str) -> tuple[int | None, str | None]:
    """向 HF 查询该文件的真实大小与 sha256。问不到就返回 (None, None)。"""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(HF_REPO, files_metadata=True)
    except Exception as error:  # noqa: BLE001
        print(f"  [提示] 向 HF 查询元信息失败（将退回宽松校验）：{error}")
        return None, None

    for sibling in getattr(info, "siblings", []) or []:
        if getattr(sibling, "rfilename", None) != filename:
            continue
        lfs = getattr(sibling, "lfs", None) or {}
        size = getattr(lfs, "size", None) or getattr(sibling, "size", None)
        sha = getattr(lfs, "sha256", None) or getattr(sibling, "sha256", None)
        return (int(size) if size else None), (sha.lower() if sha else None)
    return None, None


def download_with_hub(target_dir: Path, filename: str) -> Path | None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  [跳过] 没装 huggingface_hub")
        return None

    endpoint = os.environ.get("HF_ENDPOINT", "默认")
    print(f"  用 huggingface_hub 下载 {filename}（endpoint = {endpoint}）...")
    try:
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            local_dir=str(target_dir),
        )
        return Path(path)
    except Exception as error:  # noqa: BLE001
        print(f"  [失败] {error}")
        return None


def download_with_urllib(target_dir: Path, filename: str) -> Path | None:
    """直链下载（不依赖 huggingface_hub，认 HF_ENDPOINT 镜像）。"""
    import urllib.error
    import urllib.request

    endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    base = endpoint if endpoint else "https://huggingface.co"
    url = f"{base}/{HF_REPO}/resolve/main/{filename}"

    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    print(f"  直链下载：{url}")

    try:
        with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as handle:
            total = 0
            while True:
                chunk = response.read(1024 * 512)
                if not chunk:
                    break
                handle.write(chunk)
                total += len(chunk)
                print(f"\r    已下载 {total / 1024 / 1024:.1f} MB", end="", flush=True)
        print()
        return target
    except (urllib.error.URLError, OSError) as error:
        print(f"  [失败] {error}")
        if target.exists():
            target.unlink()
        return None


def verify(path: Path, filename: str) -> bool:
    """核对文件是不是真的那个权重。

    优先用 HF 上报的 size + sha256；问不到就退回宽松校验。
    """
    if not path.exists():
        print(f"  [失败] 文件不存在：{path}")
        return False

    size = path.stat().st_size
    print(f"  本地大小：{size:,} 字节（{size / 1024 / 1024:.1f} MB）")

    remote_size, remote_sha = fetch_remote_info(filename)
    if remote_size is not None:
        print(f"  远端大小：{remote_size:,} 字节")
        if size != remote_size:
            print("  [失败] 大小与远端不符 —— 多半是下载被截断或拿到了错误页面。")
            print("          请删除该文件后重试；国内网络建议先设：")
            print("            set HF_ENDPOINT=https://hf-mirror.com")
            return False
    else:
        print("  [提示] 拿不到远端大小，退回宽松校验")
        if size < MIN_PLAUSIBLE_SIZE:
            print(f"  [失败] 文件太小（< {MIN_PLAUSIBLE_SIZE:,} 字节），多半是错误页面")
            return False

    with path.open("rb") as handle:
        head = handle.read(4)
    if head[:1] == b"<":
        print("  [失败] 文件开头是 '<'，这是 HTML 错误页，不是权重。")
        return False

    if remote_sha:
        local_sha = _sha256(path)
        print(f"  sha256   : {local_sha[:16]}…")
        if local_sha != remote_sha:
            print("  [失败] sha256 与远端不符 —— 文件已损坏，请删除后重试")
            return False

    print("  [通过] 大小、文件头与 sha256 均正常")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="下载上游合成器/声码器权重（成员 A）")
    parser.add_argument("--dir", default=None, help="下载目录（默认 <数据根>/models/default）")
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    parser.add_argument("--only", nargs="*", choices=FILES, help="只下其中几个")
    args = parser.parse_args()

    print("=" * 66)
    print("  下载上游合成器 / 声码器权重")
    print("=" * 66)
    print("\n  路径解析：")
    for line in paths.describe().splitlines():
        print(f"    {line}")

    target_dir = Path(args.dir) if args.dir else paths.models_dir() / "default"
    wanted = args.only or FILES
    failed: list[str] = []

    for filename in wanted:
        print(f"\n--- {filename} ---")
        target = target_dir / filename
        if target.exists() and not args.force:
            print(f"  已存在：{target}")
            if verify(target, filename):
                print("  无需下载。")
                continue
            print("  将重新下载。")

        target_dir.mkdir(parents=True, exist_ok=True)
        downloaded = download_with_hub(target_dir, filename)
        if downloaded is None:
            print("  改用直链下载…")
            downloaded = download_with_urllib(target_dir, filename)
        if downloaded is None:
            print("  [失败] 两种方式都没成功。可以手动下载：")
            print(f'    浏览器打开 https://hf-mirror.com/{HF_REPO}/resolve/main/{filename}')
            print(f"    存成 {target}")
            failed.append(filename)
            continue
        print(f"\n  已下载：{downloaded}")
        if not verify(downloaded, filename):
            failed.append(filename)

    print()
    print("=" * 66)
    if failed:
        print(f"  [失败] 未能下载：{', '.join(failed)}")
        print("  其余权重不受影响，可以先继续。")
        print("=" * 66)
        return 1

    print("  全部就绪。还缺 GE2E 的话再跑一次：")
    print("    python algorithms/voiceprint_protection/scripts/download_encoder_weights.py")
    print("  然后：")
    print("    python algorithms/voiceprint_protection/scripts/smoke_test.py")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
