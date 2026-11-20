"""平台端到端自检：上传 -> 处理 -> 预览 -> 下载（任务板 W3-P5）。

【为什么要有这个脚本】
`scripts/test_service.py` 验的是**服务协议本身**；
这个脚本验的是**前端真正会走的那条路** ——
浏览器在 src/services/algorithms.js::runAlgorithm 里做的事：
  1. FormData 里 append 每个文件字段（字段名取自方法 JSON 的 inputs[].name）
  2. append `method_id`
  3. append `parameters`（JSON **字符串**）
  4. POST 到方法 JSON 的 `adapter.endpoint`
  5. 期望响应 { artifacts: [{kind,label,filename,url}], message }
  6. 每个 artifact.url 用 safeUrl(url, endpoint) 解析后**下载**
这条链以前只**人工**点过一遍，没留下可重复执行的自检 ⇒ M2 的验收没有凭证。

【★ 一切配置都从方法 JSON 读】（唯一真源）
    endpoint  <- public/methods/<方法ID>.json 的 adapter.endpoint
    样例输入  <- 同一份 JSON 的 examples[0].input.src
    参数      <- 同一份 JSON 的 examples[0].parameters
改了方法 JSON，这个脚本自动跟着变。

【用法】
    # 1) 先起服务（另一个终端）
    conda activate vpp
    uvicorn algorithms.voiceprint_protection.service:app --host 127.0.0.1 --port 8000
    # 2) 再跑自检
    python algorithms/voiceprint_protection/scripts/test_platform_e2e.py
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = MODULE_DIR.parent
REPO_ROOT = PACKAGE_DIR.parent.parent

METHOD_ID = "group_n005_voiceprint_protection"
METHOD_JSON = REPO_ROOT / "public" / "methods" / f"{METHOD_ID}.json"
PUBLIC_DIR = REPO_ROOT / "public"

PASSED = 0
FAILED = 0
ALLOWED_KINDS = {"image", "audio", "video", "json", "text"}


def check(ok: bool, name: str, extra: str = "") -> bool:
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  [通过] {name}" + (f"  {extra}" if extra else ""))
    else:
        FAILED += 1
        print(f"  [失败] {name}  {extra}")
    return ok


def post_multipart(endpoint: str, origin: str, fields: dict, files: dict):
    """按浏览器的方式发 multipart/form-data。"""
    boundary = "----vppE2Eboundary7f3a91c"
    buf = io.BytesIO()
    for name, value in fields.items():
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        buf.write(str(value).encode("utf-8") + b"\r\n")
    for name, (filename, data, ctype) in files.items():
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode())
        buf.write(f"Content-Type: {ctype}\r\n\r\n".encode())
        buf.write(data)
        buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        endpoint, data=buf.getvalue(), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Origin": origin})
    return urllib.request.urlopen(request, timeout=900)


def resolve_artifact_url(url: str, endpoint: str) -> str:
    """复刻前端 `safeUrl(url, endpoint)` 的解析规则。"""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    origin = endpoint.split("/", 3)
    base = f"{origin[0]}//{origin[2]}"
    if url.startswith("/"):
        return base + url
    return endpoint.rsplit("/", 1)[0] + "/" + url


def main() -> int:
    parser = argparse.ArgumentParser(description="平台端到端自检（上传 -> 处理 -> 预览 -> 下载）")
    parser.add_argument("--endpoint", default=None, help="覆盖方法 JSON 里的 adapter.endpoint")
    parser.add_argument("--origin", default="http://127.0.0.1:5173",
                        help="模拟的前端来源（用于 CORS 检查）")
    args = parser.parse_args()

    method = json.loads(METHOD_JSON.read_text(encoding="utf-8"))
    endpoint = args.endpoint or method["adapter"]["endpoint"]
    example = method["examples"][0]
    sample = PUBLIC_DIR / example["input"]["src"]
    parameters = {k: v for k, v in example["parameters"].items() if k != "report"}
    origin = args.origin
    base = endpoint.split("/", 3)
    health_url = f"{base[0]}//{base[2]}/healthz"

    print("=" * 70)
    print("  平台端到端自检 · 上传 -> 处理 -> 预览 -> 下载")
    print("=" * 70)
    print(f"  方法      : {method['id']}")
    print(f"  端点      : {endpoint}")
    print(f"  样例输入  : {example['input']['src']} ({sample.stat().st_size:,} 字节)")
    print(f"  参数      : {json.dumps(parameters, ensure_ascii=False)}")

    # --- 0) 平台样例本身可达（清单里的"样例可访问"）----------------
    out_src = example.get("output", {}).get("src")
    if out_src:
        out_path = PUBLIC_DIR / out_src
        check(out_path.exists() and out_path.stat().st_size > 1024,
              "方法 JSON 里登记的输出样例存在且非空", out_src)

    # --- 1) 健康检查 -------------------------------------------------
    try:
        with urllib.request.urlopen(health_url, timeout=60) as response:
            health = json.loads(response.read())
        check(response.status == 200, "GET /healthz 返回 200",
              json.dumps(health, ensure_ascii=False)[:100])
    except Exception as error:  # noqa: BLE001
        check(False, "GET /healthz", f"{type(error).__name__}: {error}")
        print("\n  服务没起来？先跑：")
        print("    uvicorn algorithms.voiceprint_protection.service:app "
              "--host 127.0.0.1 --port 8000")
        return 1

    # --- 2) CORS 预检（浏览器会先发 OPTIONS）------------------------
    try:
        request = urllib.request.Request(endpoint, method="OPTIONS", headers={
            "Origin": origin, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"})
        with urllib.request.urlopen(request, timeout=60) as response:
            allow = response.headers.get("access-control-allow-origin", "")
            methods = response.headers.get("access-control-allow-methods", "")
        check(allow == origin or allow == "*", "CORS 预检允许前端来源", f"allow-origin={allow!r}")
        check("POST" in methods.upper(), "CORS 允许 POST", f"methods={methods!r}")
    except Exception as error:  # noqa: BLE001
        check(False, "CORS 预检", f"{type(error).__name__}: {error}")

    # --- 3) 上传 + 处理 ----------------------------------------------
    file_field = method["inputs"][0]["name"]
    started = time.time()
    try:
        with post_multipart(endpoint, origin,
                            {"method_id": method["id"],
                             "parameters": json.dumps(parameters)},
                            {file_field: (sample.name, sample.read_bytes(), "audio/wav")}) as response:
            status = response.status
            payload = json.loads(response.read())
        elapsed = time.time() - started
        check(status == 200, "POST /process 返回 200", f"耗时 {elapsed:.1f}s")
    except urllib.error.HTTPError as error:
        check(False, "POST /process", f"HTTP {error.code}: {error.read()[:200]}")
        return 1

    artifacts = payload.get("artifacts")
    check(isinstance(artifacts, list) and 1 <= len(artifacts) <= 20,
          "artifacts 是 1–20 个结果",
          f"实际 {len(artifacts) if isinstance(artifacts, list) else artifacts}")
    check(isinstance(payload.get("message"), str) and payload["message"].strip(),
          "返回了可显示的 message", str(payload.get("message"))[:80])

    # --- 4) 预览 + 下载 ----------------------------------------------
    import soundfile as sf

    for artifact in artifacts:
        label = artifact.get("label") or artifact.get("filename")
        check(artifact.get("kind") in ALLOWED_KINDS,
              f"结果类型合法：{label}", f"kind={artifact.get('kind')}")
        check(bool(artifact.get("label")), f"结果有中文标签：{label}")
        url = artifact.get("url", "")
        check(bool(url), f"结果有下载地址：{label}", url)
        full = resolve_artifact_url(url, endpoint)
        try:
            with urllib.request.urlopen(full, timeout=180) as response:
                data = response.read()
            check(len(data) > 256, f"结果可下载且非空：{label}", f"{len(data):,} 字节")
            if artifact.get("kind") == "audio":
                wav, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
                rms = float((wav ** 2).mean() ** 0.5) if len(wav) else 0.0
                check(rms > 0.005, f"结果是有效音频（不是静音）：{label}",
                      f"{len(wav)/sample_rate:.2f}s @ {sample_rate}Hz, RMS={rms:.4f}")
            elif artifact.get("kind") == "json":
                report = json.loads(data.decode("utf-8"))
                check(isinstance(report, dict) and report,
                      f"结果是合法 JSON：{label}", f"{len(data):,} 字节")
        except Exception as error:  # noqa: BLE001
            check(False, f"结果可下载：{label}", f"{type(error).__name__}: {error}")

    # --- 5) 错误路径（前端要能显示人话）------------------------------
    try:
        with post_multipart(endpoint, origin,
                            {"method_id": method["id"],
                             "parameters": json.dumps(parameters)}, {}):
            check(False, "缺文件时返回错误", "竟然返回 200")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        try:
            message = json.loads(body).get("error")
        except Exception:  # noqa: BLE001
            message = None
        check(error.code >= 400, "缺文件时返回 4xx", f"HTTP {error.code}")
        check(isinstance(message, str) and message.strip(),
              "错误体是 {'error': '...'}（前端读的就是它）", str(message)[:100])

    print("\n" + "=" * 70)
    print(f"  通过 {PASSED} 项，失败 {FAILED} 项")
    print("=" * 70)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
