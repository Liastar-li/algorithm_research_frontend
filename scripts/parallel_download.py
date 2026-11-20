#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""并发 range 下载器 —— 给「单连接被 CDN 限速」的大文件用。

【为什么需要它】
HuggingFace 的 CDN（xet-bridge）从国内**单连接只有 ~350 KB/s**，
而 faster-whisper 的 `large-v3` 权重 `model.bin` 有 **3.09 GB** ⇒ 单连接要 2.4 小时，
而且连接会被中途 reset（`ChunkedEncodingError`）。
实测**并发 16~24 个 range 请求**能快一个数量级。

【用法】
python scripts/parallel_download.py <url> <输出路径> [并发数]
python scripts/parallel_download.py <url> <输出路径> 24

【三个必须注意的坑（都实测踩过）】
1. **User-Agent**：HuggingFace 对 urllib 默认的 `Python-urllib/3.x` 直接回
   **403 Forbidden**（curl 不受影响）。表现上像"下载卡住"，其实请求根本没成功。
2. **不能靠 HEAD 探长度**：CDN 对 HEAD 不响应，会挂住。用 `Range: bytes=0-0` 探。
3. **服务端可能不支持 Range**：hf-mirror 对**大文件** 302 到 CDN（支持 Range），
   对**小文件** 307 到自己的 `resolve-cache`（**忽略 Range，直接返回整个文件**）。
   所以要先探一次：拿到 206 才能并发；拿到 200 就退回单流下载。
   —— 否则每个分块都会把整个文件下一遍，磁盘和时间双输。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

# ★ 必须带一个像浏览器的 User-Agent，否则 HuggingFace 回 403
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """阻止 urllib 自动跳转 —— 我们要自己跟，才能判断 Range 有没有生效。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _headers(extra: dict | None = None) -> dict:
    merged = {"User-Agent": USER_AGENT}
    if extra:
        merged.update(extra)
    return merged


def resolve_final_url(url: str, max_hops: int = 6) -> str:
    """手动跟完重定向链，返回最终的 URL。"""
    opener = urllib.request.build_opener(_NoRedirect)
    current = url
    for _ in range(max_hops):
        try:
            with opener.open(urllib.request.Request(current, headers=_headers()),
                             timeout=60) as response:
                return current
        except urllib.error.HTTPError as error:
            location = error.headers.get("Location")
            if error.code not in (301, 302, 303, 307, 308) or not location:
                return current
            current = urljoin(current, location)
        except Exception:  # noqa: BLE001
            return current
    return current


def probe(url: str) -> tuple[int, str]:
    """探（总长度, 用于分块的 URL）。

    返回的总长度 <= 0 表示服务端**不支持 Range** —— 这时只能单流下载。
    """
    final = resolve_final_url(url)
    request = urllib.request.Request(final, headers=_headers({"Range": "bytes=0-0"}))
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = response.status
            content_range = response.headers.get("Content-Range")
            content_length = response.headers.get("Content-Length")
            response.read(1)
    except urllib.error.HTTPError as error:
        if error.code != 206:
            return 0, final
        content_range = error.headers.get("Content-Range")
        content_length = None
    if status == 206 and content_range and "/" in content_range:
        return int(content_range.split("/")[1]), final
    # 200 = 服务端忽略了 Range，返回的是整个文件
    if content_length:
        return -int(content_length), final
    return 0, final


def fetch_range(url: str, start: int, end: int, retries: int = 12) -> bytes:
    want = end - start + 1
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers=_headers({"Range": "bytes={}-{}".format(start, end)}))
            with urllib.request.urlopen(request, timeout=600) as response:
                data = response.read()
            if len(data) != want:
                raise IOError("长度不对：{} != {}".format(len(data), want))
            return data
        except Exception as error:  # noqa: BLE001
            last_error = error
            print("  区间 {}-{} 第 {} 次失败：{}".format(start, end, attempt + 1, error))
            time.sleep(2)
    raise RuntimeError("区间 {}-{} 连续失败：{}".format(start, end, last_error))


def download_single(url: str, output: str) -> int:
    print("  服务端不支持 Range —— 退回单流下载")
    request = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(request, timeout=1800) as response:
        with open(output, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    return os.path.getsize(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="并发 range 下载器")
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("parts", nargs="?", type=int, default=16)
    args = parser.parse_args()

    total, final_url = probe(args.url)
    if total <= 0:
        size = download_single(args.url, args.output)
        print("已写入 {}：{} 字节".format(args.output, size))
        return 0

    parts = max(1, args.parts)
    chunk = -(-total // parts)
    print("总大小 {} 字节，切成 {} 块，每块 {} 字节".format(total, parts, chunk))

    blobs: dict[int, bytes] = {}
    started = time.time()

    def worker(index: int):
        start = index * chunk
        end = min(total - 1, start + chunk - 1)
        if start > end:
            return index, b""
        return index, fetch_range(final_url, start, end)

    with ThreadPoolExecutor(max_workers=parts) as pool:
        for index, data in pool.map(worker, range(parts)):
            blobs[index] = data
            done = sum(len(v) for v in blobs.values())
            print("  块 {:>3} 完成  {}/{} 字节  {:.0f} KB/s".format(
                index, done, total,
                done / 1024 / max(time.time() - started, 0.001)))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "wb") as handle:
        # ★ 必须按**数字顺序**拼（p0 p1 p2 ... p10），不是字典序 ——
        #   `cat p*` 是字典序，拼出来是乱码，而且大小刚好对得上，**不报错**。
        for index in range(parts):
            handle.write(blobs.get(index, b""))
    print("已写入 {}：{} 字节（应为 {}）".format(
        args.output, os.path.getsize(args.output), total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
