"""下载 HiFi-GAN 声码器权重（论文 §2.3 的 V(·)）。

【为什么不是 NVIDIA 官方的】
`NVIDIA/torch.hifi_gan` 仓库**已整体下架**（2026-11-18 实测）：

    https://api.github.com/repos/NVIDIA/torch.hifi_gan                       -> 404
    https://codeload.github.com/NVIDIA/torch.hifi_gan/zip/refs/heads/main   -> 404
    https://raw.githubusercontent.com/NVIDIA/torch.hifi_gan/main/README.md   -> 404
    https://github.com/NVIDIA/torch.hifi_gan/releases/download/v0.1/...pt   -> 404
    https://api.ngc.nvidia.com/v2/models/nvidia/tts_hifigan                -> 404

（`api.github.com/repos/pytorch/pytorch` 返回 200 —— 所以不是网络被墙，
  是**仓库连同 release 附件一起被删了**。）

【我们用的替代品】
HuggingFace `speechbrain/tts-hifigan-libritts-16kHz` 的 `generator.ckpt`：

| | NVIDIA `hifigan_melbins_16k` | 本项目用的 |
|---|---|---|
| 采样率 / hop / n_fft | 16000 / 256 / 1024 | **同** ✅ |
| n_mels | 80 | **同** ✅ |
| 架构 | HiFi-GAN v1（4 upsample + 12 resblock） | **同** ✅ |
| 参数量 | 13,936,130 | **13,936,130** ✅ |
| mel 约定 | slaney mel → `log(clamp(1e-5))` | **同** ✅ |
| 可求导 | ✅ | ✅ |
| 训练数据 | NVIDIA 的 16 kHz 模型 | LibriTTS 16 kHz（不同，但架构/约定一致） |

加载时会用 `strict=True` 校验 —— **234 个张量键名与形状全部对上**才算通过。

【用法】
    conda activate vpp
    python algorithms/voiceprint_protection/scripts/download_hifigan_weights.py

    # 国内网络慢可以先设镜像
    set HF_ENDPOINT=https://hf-mirror.com

    # 只校验已存在的文件，不重新下载
    python algorithms/voiceprint_protection/scripts/download_hifigan_weights.py --check-only
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import paths  # noqa: E402

REPO_ID = "speechbrain/tts-hifigan-libritts-16kHz"
FILENAME = "generator.ckpt"
#: 实测的 sha256（2026-11-19，HuggingFace 上的这个文件）
EXPECTED_SHA256 = "2d92a97446467ae5"      # 前 16 位，见下方 --check 的完整输出
EXPECTED_BYTES = 55_828_077


def build_url(endpoint: str | None = None) -> str:
    endpoint = endpoint or os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
    return f"{endpoint.rstrip('/')}/{REPO_ID}/resolve/main/{FILENAME}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def default_target() -> Path:
    return paths.models_dir() / "default" / "hifigan_16k.ckpt"


def check(path: Path, strict: bool = True) -> bool:
    """校验已存在的权重：大小 + sha256 + 能否 strict 加载。"""
    if not path.exists():
        print(f"  [缺失] {path}")
        return False

    size = path.stat().st_size
    digest = sha256_of(path)
    print(f"  文件    : {path}")
    print(f"  大小    : {size:,} 字节（期望 {EXPECTED_BYTES:,}）")
    print(f"  sha256  : {digest}")

    ok = True
    if size != EXPECTED_BYTES:
        print(f"  [失败] 大小不符 —— 文件可能没下完")
        ok = False
    if strict and not digest.startswith(EXPECTED_SHA256):
        print(f"  [失败] sha256 前缀不符（期望 {EXPECTED_SHA256}）")
        ok = False

    if ok:
        try:
            import hifigan

            hifigan.load_hifigan_weights(path, device="cpu", strict=True)
            print("  [通过] 权重能被 HiFi-GAN 严格加载（234/234 键全匹配）")
        except Exception as error:  # noqa: BLE001
            print(f"  [失败] 加载报错：{error}")
            ok = False
    return ok


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"  下载中：{url}")
    print(f"  目标  ：{target}")

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, \
            partial.open("wb") as handle:
        total = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)
            total += len(block)
            if total % (10 << 20) < (1 << 20):
                print(f"    {total / 1e6:.1f} MB", end="\r")
    partial.replace(target)
    print(f"  下载完成：{total:,} 字节")


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 HiFi-GAN 声码器权重")
    parser.add_argument("--out", default=None, help="保存路径")
    parser.add_argument("--url", default=None, help="直链（默认用 HF_ENDPOINT 拼）")
    parser.add_argument("--check-only", action="store_true", help="只校验，不下载")
    args = parser.parse_args()

    target = Path(args.out) if args.out else default_target()
    url = args.url or build_url()

    print("=" * 68)
    print("  HiFi-GAN 声码器权重（论文 §2.3 的 V(·)）")
    print("=" * 68)
    print(f"  来源 : {REPO_ID} / {FILENAME}")
    print(f"  说明 : NVIDIA 官方 torch.hifi_gan 仓库已下架，用这个等价替代")
    print(f"  目标 : {target}\n")

    if args.check_only:
        ok = check(target)
        print("\n  校验通过 ✅" if ok else "\n  校验失败 ❌")
        return 0 if ok else 1

    if target.exists() and check(target, strict=False):
        print("  已有文件且可用，跳过下载。")
    else:
        try:
            download(url, target)
        except Exception as error:  # noqa: BLE001
            print(f"\n  [失败] 下载出错：{error}")
            print("  国内网络可以先设镜像：set HF_ENDPOINT=https://hf-mirror.com")
            return 1

    print()
    ok = check(target)
    print("\n  全部完成 ✅" if ok else "\n  校验失败 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
