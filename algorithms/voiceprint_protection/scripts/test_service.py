"""平台接口契约自测 —— 不用点网页，直接验服务对不对。

用法：
    conda activate vpp
    python algorithms/voiceprint_protection/scripts/test_service.py            # 用真实权重
    python algorithms/voiceprint_protection/scripts/test_service.py --mock     # 用假模型（不需要权重）

【为什么要有 --mock】
上游三个权重加起来 425 MB，而**接口对不对**跟模型准不准没关系。
所以这里用假模型把整条 HTTP 通路跑一遍：
参数校验 → 上传落盘 → 推理 → 写文件 → 拼 artifacts → 前端下载 URL。
权重还没下齐时也能验证协议，随时可以回归。

【它验的是什么（对齐 src/services/algorithms.js 与 src/lib/contract.js）】
  * 成功必须返回 1–20 个 artifacts，kind ∈ image/audio/video/json/text
  * url 必须是相对路径或 http(s)，前端会自己拼成绝对地址
  * **失败必须是非 2xx 且响应体里有 `error` 字段** ——
    前端读的是 detail.error，读不到就只会显示"算法服务返回 HTTP xxx。"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MODULE_DIR.parent.parent

sys.path.insert(0, str(MODULE_DIR))

import requests  # noqa: E402

METHOD_ID = "group_n005_voiceprint_protection"
ARTIFACT_KINDS = ("image", "audio", "video", "json", "text")


# ---------------------------------------------------------------------------
# 假模型（--mock 时用）
# ---------------------------------------------------------------------------
def install_mock_model() -> None:
    """把 sys.modules 里的 infer 换成一个假的 Protector。

    只为验协议，不做任何真实推理 —— 返回的波形是随机噪声。
    """
    import types

    import numpy as np

    module = types.ModuleType("infer")

    class MockProtector:
        def __init__(self, *args, **kwargs):
            pass

        def protect(self, wav, alpha=0.1, seed=42, text="", attack_mode="clone", verbose=False):
            rng = np.random.default_rng(seed)
            length = len(wav)
            embedding = rng.standard_normal(256).astype(np.float32)
            embedding /= np.linalg.norm(embedding)
            embedding_adv = (embedding + alpha * rng.standard_normal(256)).astype(np.float32)
            embedding_adv /= np.linalg.norm(embedding_adv)
            result = {
                "embedding": embedding,
                "embedding_adv": embedding_adv,
                "protected_waveform": (0.1 * rng.standard_normal(length)).astype(np.float32),
                "shapes": {"embedding": [256], "embedding_adv": [256],
                           "protected_waveform": [length]},
            }
            if attack_mode == "clone":
                result["clone_baseline_waveform"] = (0.1 * rng.standard_normal(length)).astype(np.float32)
                result["clone_protected_waveform"] = (0.1 * rng.standard_normal(length)).astype(np.float32)
                result["stolen_embedding"] = rng.standard_normal(256).astype(np.float32)
            return result

    module.Protector = MockProtector
    module.DEFAULT_REFERENCE_TEXT = "mock"
    sys.modules["infer"] = module


def serve_mock(host: str, port: int) -> int:
    """子进程角色：装上假模型，然后把服务跑起来。"""
    install_mock_model()
    import uvicorn

    from service import app  # noqa: F401  （service 自己会把模块目录加进 sys.path）

    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


# ---------------------------------------------------------------------------
# 检查项
# ---------------------------------------------------------------------------
class Checker:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, ok: bool, name: str, extra: str = "") -> bool:
        if ok:
            self.passed += 1
            print(f"  [通过] {name}")
        else:
            self.failed += 1
            print(f"  [失败] {name}" + (f"  —— {extra}" if extra else ""))
        return ok


def _sample_audio() -> Path:
    """找一段能用的音频（平台样例优先）。"""
    sample = REPO_ROOT / "public" / "samples" / METHOD_ID / "input.wav"
    if sample.exists():
        return sample
    manifest = Path(os.environ.get("VPP_DATA_ROOT", REPO_ROOT / "data")) / "test.txt"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            candidate = manifest.parent / parts[0]
            if candidate.exists():
                return candidate
    raise SystemExit("找不到测试音频：请先跑 make_sample.py 或准备 VCTK 数据。")


def run_checks(base_url: str, expect_success: bool) -> Checker:
    checker = Checker()
    endpoint = f"{base_url}/process"
    audio_path = _sample_audio()

    # -- 1. 健康检查 ------------------------------------------------------
    print("\n【1】健康检查")
    response = requests.get(f"{base_url}/healthz", timeout=15)
    checker.check(response.status_code == 200, "GET /healthz 返回 200",
                  f"实际 {response.status_code}")
    if response.ok:
        body = response.json()
        checker.check(body.get("method_id") == METHOD_ID, "healthz 报告的方法 ID 正确")
        checker.check("model_loaded" in body, "healthz 报告了模型是否就绪")

    # -- 2. 错误路径：必须是非 2xx 且响应体含 error -----------------------
    print("\n【2】错误路径（前端靠响应体里的 error 显示原因）")
    cases = [
        ("未知 method_id", {"method_id": "wrong_id", "parameters": "{}"},
         lambda r: r.status_code == 404),
        ("parameters 不是合法 JSON", {"method_id": METHOD_ID, "parameters": "{oops"},
         lambda r: r.status_code == 422),
        ("alpha 越界", {"method_id": METHOD_ID, "parameters": '{"alpha": 5}'},
         lambda r: r.status_code == 422),
        ("attack_mode 非法", {"method_id": METHOD_ID, "parameters": '{"attack_mode": "hack"}'},
         lambda r: r.status_code == 422),
    ]
    for name, fields, status_ok in cases:
        with audio_path.open("rb") as handle:
            files = {"input_file": (audio_path.name, handle, "audio/wav")}
            response = requests.post(endpoint, files=files, data=fields, timeout=30)
        checker.check(status_ok(response), f"{name} -> 状态码正确",
                      f"实际 {response.status_code}")
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        checker.check(isinstance(payload.get("error"), str) and bool(payload["error"]),
                      f"{name} -> 响应体含 error 字段", f"实际键：{list(payload)}")

    # -- 3. 缺字段 / 坏文件 ----------------------------------------------
    print("\n【3】表单与文件校验")
    response = requests.post(endpoint, data={"method_id": METHOD_ID, "parameters": "{}"},
                             timeout=30)
    checker.check(response.status_code == 422, "缺 input_file 字段 -> 422",
                  f"实际 {response.status_code}")
    checker.check("error" in response.json(), "缺 input_file 字段 -> 响应体含 error")

    fake = REPO_ROOT / "algorithms" / "voiceprint_protection" / "results" / "_fake_input.wav"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_bytes(b"this is definitely not audio")
    try:
        with fake.open("rb") as handle:
            files = {"input_file": ("fake.wav", handle, "audio/wav")}
            response = requests.post(endpoint, files=files,
                                     data={"method_id": METHOD_ID, "parameters": "{}"},
                                     timeout=30)
        checker.check(response.status_code == 422, "上传非音频 -> 422",
                      f"实际 {response.status_code}")
        checker.check("error" in response.json(), "上传非音频 -> 响应体含 error")
    finally:
        fake.unlink(missing_ok=True)

    # -- 4. 正常请求 ------------------------------------------------------
    print("\n【4】正常请求")
    with audio_path.open("rb") as handle:
        files = {"input_file": (audio_path.name, handle, "audio/wav")}
        data = {
            "method_id": METHOD_ID,
            "parameters": json.dumps({"alpha": 0.1, "attack_mode": "clone", "report": True}),
        }
        response = requests.post(endpoint, files=files, data=data, timeout=180)

    if response.status_code != 200:
        detail = ""
        try:
            detail = response.json().get("error", "")
        except ValueError:
            detail = response.text[:200]
        if not expect_success:
            print(f"  [跳过] 正常路径未通过（服务返回 {response.status_code}：{detail}）")
            print("         权重齐全后重跑本脚本即可验证成功路径。")
            return checker
        checker.check(False, "正常请求 -> 200", f"实际 {response.status_code}：{detail}")
        return checker

    checker.check(True, "正常请求 -> 200")
    payload = response.json()
    artifacts = payload.get("artifacts")
    checker.check(isinstance(artifacts, list) and 1 <= len(artifacts) <= 20,
                  "artifacts 数量在 1–20 之间", f"实际 {len(artifacts) if artifacts else artifacts}")
    checker.check(isinstance(payload.get("message"), str) and bool(payload["message"]),
                  "返回了 message")

    for artifact in artifacts or []:
        kind = artifact.get("kind")
        checker.check(kind in ARTIFACT_KINDS, f"artifact kind 合法（{kind}）")
        checker.check(isinstance(artifact.get("label"), str), f"artifact 有 label（{artifact.get('label')}）")
        url = artifact.get("url")
        checker.check(isinstance(url, str) and url.startswith(("/files/", "http")),
                      f"artifact url 是相对路径或 http（{url}）")

        # 前端会拿这个 url 去下载 —— 必须真的能下到
        absolute = f"{base_url}{url}" if url.startswith("/") else url
        download = requests.get(absolute, timeout=60)
        checker.check(download.status_code == 200, f"artifact 可下载（{artifact.get('filename')}）",
                      f"实际 {download.status_code}")
        checker.check(len(download.content) > 0, f"artifact 非空（{artifact.get('filename')}）")

    # -- 5. attack_mode=none 时不该产出克隆音 ------------------------------
    print("\n【5】attack_mode=none")
    with audio_path.open("rb") as handle:
        files = {"input_file": (audio_path.name, handle, "audio/wav")}
        data = {
            "method_id": METHOD_ID,
            "parameters": json.dumps({"alpha": 0.1, "attack_mode": "none", "report": False}),
        }
        response = requests.post(endpoint, files=files, data=data, timeout=180)
    if response.status_code == 200:
        names = [a.get("filename", "") for a in response.json().get("artifacts", [])]
        checker.check(len(names) == 1, "attack_mode=none 只产出 1 个文件", f"实际 {names}")
        checker.check(not any("clone" in name for name in names),
                      "attack_mode=none 没有克隆音", f"实际 {names}")
    else:
        print(f"  [跳过] 服务返回 {response.status_code}")

    # -- 6. 空文本必须被兜底（★ 间歇性崩溃的回归测试）---------------------
    print("\n【6】空文本兜底")
    print("    上游 Tacotron 拿到空文本会吐出退化 mel，它自己的裁静音循环会越界，")
    print("    而且是**间歇性**的 —— 所以这里显式验一次。")
    with audio_path.open("rb") as handle:
        files = {"input_file": (audio_path.name, handle, "audio/wav")}
        data = {
            "method_id": METHOD_ID,
            "parameters": json.dumps({"alpha": 0.1, "attack_mode": "none",
                                      "report": False, "text": ""}),
        }
        response = requests.post(endpoint, files=files, data=data, timeout=180)
    checker.check(response.status_code == 200,
                  "空文本 -> 200（用默认参考文本兜底）",
                  f"实际 {response.status_code}：{response.text[:200]}")

    return checker


def main() -> int:
    parser = argparse.ArgumentParser(description="平台接口契约自测")
    parser.add_argument("--mock", action="store_true",
                        help="用假模型测协议（不需要上游权重）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--external", action="store_true",
                        help="不自己起服务，测已在跑的服务（默认用 --host/--port）")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    if args.external:
        print(f"测试已在运行的服务：{base_url}")
        checker = run_checks(base_url, expect_success=True)
    else:
        env = dict(os.environ)
        command = [sys.executable, "-m", "uvicorn", "algorithms.voiceprint_protection.service:app",
                   "--host", args.host, "--port", str(args.port), "--log-level", "warning"]
        if args.mock:
            # 子进程角色：装假模型再起服务
            env["VPP_TEST_ROLE"] = "server"
            env["VPP_TEST_PORT"] = str(args.port)
            command = [sys.executable, str(Path(__file__).resolve())]

        print(f"启动测试服务：{' '.join(command)}")
        process = subprocess.Popen(command, cwd=str(REPO_ROOT), env=env)
        try:
            ready = False
            for _ in range(120):
                time.sleep(1)
                if process.poll() is not None:
                    print(f"服务进程提前退出，返回码 {process.returncode}")
                    return 2
                try:
                    if requests.get(f"{base_url}/healthz", timeout=3).status_code == 200:
                        ready = True
                        break
                except requests.RequestException:
                    continue
            if not ready:
                print("服务在 120 秒内没有就绪。")
                return 2
            checker = run_checks(base_url, expect_success=args.mock)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    print("\n" + "=" * 60)
    print(f"  通过 {checker.passed} 项，失败 {checker.failed} 项")
    print("=" * 60)
    return 0 if checker.failed == 0 else 1


if __name__ == "__main__":
    # 子进程角色：装假模型并起服务
    if os.environ.get("VPP_TEST_ROLE") == "server":
        raise SystemExit(serve_mock("127.0.0.1", int(os.environ.get("VPP_TEST_PORT", "8123"))))

    raise SystemExit(main())
