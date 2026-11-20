#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VCTK 定向下载器（Python 版）—— 只取需要的目录，并发 range 请求。

【为什么要有 Python 版】
scripts/download_vctk.mjs 是 Node 版，云机（Linux 轻量镜像）常常没装 Node；
而 curl/wget 只能整包下 11.75 GiB。这个脚本只用 Python 标准库，
云机上 `python3 scripts/download_vctk.py --out <dir>` 就能定向拉数据。

【它做什么】
  1) 解析远端 zip 的中央目录（支持 ZIP64），列出全部条目
  2) 只下载需要的顶层目录（默认 wav48_silence_trimmed/）
  3) 并发 HTTP range 请求，逐条目直接解压落盘（不落地整个 zip 容器）

【用法】
  python3 scripts/download_vctk.py --out /root/vpp/data/raw
  python3 scripts/download_vctk.py --out /root/vpp/data/raw --concurrency 64
  python3 scripts/download_vctk.py --out /root/vpp/data/raw --include wav48_silence_trimmed --include txt
  python3 scripts/download_vctk.py --out /root/vpp/data/raw --dry-run
  python3 scripts/download_vctk.py --out /root/vpp/data/raw --limit 20
  python3 scripts/download_vctk.py --out /root/vpp/data/raw --per-speaker 5

【断点续传】已落盘且大小正确的条目会跳过，中断后重跑同一条命令即可。
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import threading
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

DEFAULT_URL = ("https://datashare.ed.ac.uk/server/api/core/bitstreams/"
               "535f4286-e54c-4038-838c-a02285e32cb2/content")

MB = 1024 * 1024
GB = 1024 * MB
# 配额只算这些扩展名的文件（其余文件照下，但不占名额）
AUDIO_EXTS = (".flac", ".wav", ".mp3")


# ---------------------------------------------------------------------------
# HTTP range 辅助
# ---------------------------------------------------------------------------
class RangeFetcher:
    """按字节区间拉取远端 zip；带重试与并行分块。"""

    def __init__(self, url: str, total: int = 0, retries: int = 4):
        self.url = url
        self.total = total
        self.retries = retries
        self._lock = threading.Lock()
        self.bytes_down = 0

    def head(self) -> int:
        request = urllib.request.Request(self.url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            if length is None:
                raise RuntimeError("拿不到 Content-Length")
            return int(length)

    def get_range(self, start: int, end: int) -> bytes:
        end = min(end, self.total - 1)
        if end < start:
            return b""
        headers = {"Range": "bytes={}-{}".format(start, end)}
        last_error = None
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(self.url, headers=headers)
                with urllib.request.urlopen(request, timeout=180) as response:
                    data = response.read()
                if len(data) != end - start + 1:
                    if start == 0 and len(data) == self.total:
                        return data
                    raise IOError("区间长度不对：{} != {}".format(len(data), end - start + 1))
                with self._lock:
                    self.bytes_down += len(data)
                return data
            except Exception as error:
                last_error = error
                time.sleep(0.8 * (attempt + 1))
        raise RuntimeError("下载区间失败 {}..{}：{}".format(start, end, last_error))

    def get_range_parallel(self, start: int, end: int, workers: int = 16) -> bytes:
        length = end - start + 1
        if length < 4 * MB:
            return self.get_range(start, end)
        chunk = -(-length // workers)
        parts = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for i in range(workers):
                s = start + i * chunk
                if s > end:
                    break
                e = min(end, s + chunk - 1)
                futures.append(pool.submit(self.get_range, s, e))
            for future in futures:
                parts.append(future.result())
        return b"".join(parts)


# ---------------------------------------------------------------------------
# zip 中央目录解析（含 ZIP64）
# ---------------------------------------------------------------------------
def read_central_directory(fetcher: RangeFetcher) -> list:
    tail_len = min(65536, fetcher.total)
    tail = fetcher.get_range(fetcher.total - tail_len, fetcher.total - 1)

    eocd = -1
    for i in range(len(tail) - 22, -1, -1):
        if struct.unpack_from("<I", tail, i)[0] == 0x06054B50:
            eocd = i
            break
    if eocd < 0:
        raise RuntimeError("不是有效的 zip（找不到 EOCD）")

    entries, cd_size, cd_offset = struct.unpack_from("<HII", tail, eocd + 10)

    if cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF or entries == 0xFFFF:
        loc = -1
        for i in range(eocd - 20, -1, -1):
            if struct.unpack_from("<I", tail, i)[0] == 0x07064B50:
                loc = i
                break
        if loc < 0:
            raise RuntimeError("需要 ZIP64 但找不到 locator")
        z64_offset = struct.unpack_from("<Q", tail, loc + 8)[0]
        z64 = fetcher.get_range(z64_offset, z64_offset + 55)
        if struct.unpack_from("<I", z64, 0)[0] != 0x06064B50:
            raise RuntimeError("ZIP64 EOCD 签名不对")
        entries = struct.unpack_from("<Q", z64, 32)[0]
        cd_size = struct.unpack_from("<Q", z64, 40)[0]
        cd_offset = struct.unpack_from("<Q", z64, 48)[0]

    print("  中央目录：{} 个条目，{:.1f} MB".format(entries, cd_size / MB))
    cd_buffer = fetcher.get_range_parallel(cd_offset, cd_offset + cd_size - 1, 16)
    return parse_central_directory(cd_buffer)


def parse_central_directory(cd: bytes) -> list:
    result = []
    p = 0
    while p + 46 <= len(cd) and struct.unpack_from("<I", cd, p)[0] == 0x02014B50:
        method = struct.unpack_from("<H", cd, p + 10)[0]
        comp_size = struct.unpack_from("<I", cd, p + 20)[0]
        uncomp_size = struct.unpack_from("<I", cd, p + 24)[0]
        name_len = struct.unpack_from("<H", cd, p + 28)[0]
        extra_len = struct.unpack_from("<H", cd, p + 30)[0]
        local_offset = struct.unpack_from("<I", cd, p + 42)[0]
        name = cd[p + 46:p + 46 + name_len].decode("utf8", "replace")

        if uncomp_size == 0xFFFFFFFF or comp_size == 0xFFFFFFFF or local_offset == 0xFFFFFFFF:
            e = p + 46 + name_len
            end = e + extra_len
            while e + 4 <= end:
                field_id, field_len = struct.unpack_from("<HH", cd, e)
                if field_id == 0x0001:
                    q = e + 4
                    if uncomp_size == 0xFFFFFFFF:
                        uncomp_size = struct.unpack_from("<Q", cd, q)[0]
                        q += 8
                    if comp_size == 0xFFFFFFFF:
                        comp_size = struct.unpack_from("<Q", cd, q)[0]
                        q += 8
                    if local_offset == 0xFFFFFFFF:
                        local_offset = struct.unpack_from("<Q", cd, q)[0]
                        q += 8
                e += 4 + field_len
        if not name.endswith("/"):
            result.append({
                "name": name,
                "method": method,
                "compSize": comp_size,
                "uncompSize": uncomp_size,
                "localOffset": local_offset,
            })
        p += 46 + name_len + extra_len
    return result


# ---------------------------------------------------------------------------
# 条目下载
# ---------------------------------------------------------------------------
def download_entry(fetcher: RangeFetcher, entry: dict, out_dir: str) -> int:
    target = os.path.join(out_dir, *entry["name"].split("/"))
    if os.path.exists(target) and os.path.getsize(target) == entry["uncompSize"]:
        return 0

    probe_extra = 512
    header_end = entry["localOffset"] + 30 + 255 + probe_extra
    buffer = fetcher.get_range(entry["localOffset"], min(fetcher.total - 1, header_end))
    if struct.unpack_from("<I", buffer, 0)[0] != 0x04034B50:
        raise RuntimeError("本地文件头签名不对：" + entry["name"])
    name_len = struct.unpack_from("<H", buffer, 26)[0]
    extra_len = struct.unpack_from("<H", buffer, 28)[0]
    data_start = 30 + name_len + extra_len
    data_end = data_start + entry["compSize"]
    if data_end > len(buffer):
        buffer = fetcher.get_range(entry["localOffset"], entry["localOffset"] + data_end - 1)

    raw = buffer[data_start:data_start + entry["compSize"]]
    if entry["method"] == 0:
        content = raw
    elif entry["method"] == 8:
        content = zlib.decompress(raw, -15)
    else:
        raise RuntimeError("不支持的压缩方式 {}：{}".format(entry["method"], entry["name"]))

    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = target + ".part"
    with open(tmp, "wb") as handle:
        handle.write(content)
    os.replace(tmp, target)
    return len(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="VCTK 定向下载器（只取需要的目录 + 并发 range）")
    parser.add_argument("--out", default="D:/vpp/data/raw")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--include", action="append", default=None,
                        help="要下载的顶层目录，可重复；默认 wav48_silence_trimmed")
    parser.add_argument("--limit", type=int, default=0, help="最多下载几个条目（0 = 不限）")
    parser.add_argument("--per-speaker", type=int, default=0, help="每个说话人最多取几句（0 = 全部）")
    parser.add_argument("--speakers", default=None, help="只下这些说话人，逗号分隔")
    parser.add_argument("--dry-run", action="store_true", help="只看清单不下载")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    includes = args.include or ["wav48_silence_trimmed"]

    print("=" * 64)
    print("  VCTK 定向下载器（Python 版）")
    print("=" * 64)
    print("  源      : " + args.url)
    print("  输出目录: " + out_dir)
    print("  只要    : " + ", ".join(includes))
    print("  并发数  : {}".format(args.concurrency))

    fetcher = RangeFetcher(args.url)
    fetcher.total = fetcher.head()
    print("  总大小  : {:.2f} GiB ({} 字节)".format(fetcher.total / GB, fetcher.total))

    cache_path = os.path.join(out_dir, ".vctk_entries.json")
    entries = None
    try:
        with open(cache_path, "r", encoding="utf8") as handle:
            cached = json.load(handle)
        if cached.get("total") == fetcher.total and cached.get("entries"):
            entries = cached["entries"]
            print("  用本地缓存的中央目录（{} 个条目），跳过下载".format(len(entries)))
    except Exception:
        entries = None
    if entries is None:
        entries = read_central_directory(fetcher)
        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(cache_path, "w", encoding="utf8") as handle:
                json.dump({"total": fetcher.total, "entries": entries}, handle)
            print("  已缓存到 " + cache_path)
        except Exception as error:
            print("  [提示] 缓存写入失败（不影响下载）：{}".format(error))

    speaker_filter = None
    if args.speakers:
        speaker_filter = [s.strip() for s in args.speakers.split(",") if s.strip()]

    by_speaker = {}
    extras = []
    for entry in entries:
        if "/" not in entry["name"]:
            if entry["name"].endswith(".txt") and entry["compSize"] < MB:
                extras.append(entry)
            continue
        if not any(inc in entry["name"] for inc in includes):
            continue
        parts = entry["name"].split("/")
        speaker = parts[1] if len(parts) > 1 else parts[0]
        if speaker_filter is not None and speaker not in speaker_filter:
            continue
        by_speaker.setdefault(speaker, []).append(entry)

    wanted = []
    for speaker in sorted(by_speaker):
        items = sorted(by_speaker[speaker], key=lambda e: e["name"])
        # ★ 配额只算**音频文件**：wav48_silence_trimmed/ 里混着一个 log.txt，
        #   它不是音频，不该占掉一个名额（prepare_data 也只认 .flac/.wav）；
        #   但它要一起下载（那是 VCTK 的日志，README 要引）。
        audio = []
        others = []
        for entry in items:
            if entry["name"].lower().endswith(AUDIO_EXTS):
                audio.append(entry)
            else:
                others.append(entry)
        if args.per_speaker > 0 and len(audio) > args.per_speaker:
            for i in range(args.per_speaker):
                wanted.append(audio[(i * len(audio)) // args.per_speaker])
        else:
            wanted.extend(audio)
        wanted.extend(others)

    total_bytes = sum(e["compSize"] for e in wanted)
    print("")
    print("  说话人 {} 个".format(len(by_speaker)))
    print("  选中 {} 个条目，共 {:.2f} GiB（整包的 {:.0f}%）".format(
        len(wanted), total_bytes / GB, 100.0 * total_bytes / max(fetcher.total, 1)))
    if args.dry_run:
        print("  [干跑] 没有下载任何东西。")
        return 0
    if not wanted:
        print("  [失败] 没有匹配的条目，检查 --include 的写法。")
        return 1

    todo = wanted[:args.limit] if args.limit > 0 else wanted + extras
    print("  开始下载（{} 个条目，并发 {}）".format(len(todo), args.concurrency))
    print("")

    done = [0]
    failed = [0]
    started = time.time()
    lock = threading.Lock()

    def worker(entry):
        try:
            download_entry(fetcher, entry, out_dir)
            with lock:
                done[0] += 1
                if done[0] % 100 == 0 or done[0] == len(todo):
                    seconds = max(time.time() - started, 0.001)
                    rate = fetcher.bytes_down / 1024 / seconds
                    eta = (total_bytes - fetcher.bytes_down) / 1024 / max(rate, 0.001)
                    print("  {}/{}  {:.2f} GiB  {:.0f} KiB/s  ETA {:.1f}h".format(
                        done[0], len(todo), fetcher.bytes_down / GB, rate, eta / 3600))
        except Exception as error:
            with lock:
                failed[0] += 1
                print("  [失败] {}：{}".format(entry["name"], error))

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        list(pool.map(worker, todo))

    print("")
    print("  完成，用时 {:.1f} 分钟（失败 {} 个）。".format(
        (time.time() - started) / 60, failed[0]))
    print("  输出目录：" + out_dir)
    print("  接下来：")
    print("    python algorithms/voiceprint_protection/scripts/survey_vctk.py")
    print("    python algorithms/voiceprint_protection/scripts/prepare_data.py")
    return 0 if failed[0] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
