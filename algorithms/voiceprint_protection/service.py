"""FastAPI 服务：把声纹保护推理链包装成平台约定的 HTTP 接口。

【平台协议（前端已定死，见 docs/算法接入规范.md 与 src/services/algorithms.js）】
  请求：POST <endpoint>，multipart/form-data
    - 素材文件：字段名 = 方法配置里 inputs[].name（本方法为 input_file）
    - method_id：字符串
    - parameters：JSON **字符串**（不是对象）
  成功：HTTP 200 + {"message": str, "artifacts": [{kind, label, filename, url}, ...]}
    - kind ∈ image / audio / video / json / text，1–20 个
    - url 可以是相对路径，前端按接口地址解析 → 这里用 /files/<名字>
  失败：非 2xx + {"error": "用户能看懂的说明"}（前端会原样显示，超过 500 字截断）

【几个容易踩的点，这里都处理了】
1. **模型只加载一次**。平台的请求是**同步等待**的（建议 < 60 s），
   而 `infer.protect()` 每次调用都会新建 `Protector()` → 每次请求重载
   400 MB 权重（encoder + synthesizer + vocoder），必然超时。
   所以这里在进程内缓存一个 `Protector` 实例，懒加载 + 只建一次。
2. **临时文件用 uuid 命名**，不把用户上传的文件名拼进任何路径或命令
   （文件名可能含中文、斜杠、还有 `..`）。
3. **结果文件由本服务自己托管**（挂载在 /files），返回相对地址，
   不返回服务器磁盘路径（平台要求可跨域下载的真实地址）。
4. **错误要给人看**：解码失败、参数非法、权重缺失，都返回
   非 2xx + 一句中文说明，而不是 500 + traceback。
5. 每次处理完把上传的临时文件删掉；结果文件保留（前端要能下载），
   并在启动时清空上一次的残留。

【启动】
    conda activate vpp
    uvicorn algorithms.voiceprint_protection.service:app --host 127.0.0.1 --port 8000

【自检】
    curl -X POST http://127.0.0.1:8000/process ^
      -F "input_file=@public/samples/group_n005_voiceprint_protection/input.wav" ^
      -F "method_id=group_n005_voiceprint_protection" ^
      -F "parameters={\"alpha\":0.1,\"attack_mode\":\"clone\",\"report\":true}"
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audio_utils  # noqa: E402
import config  # noqa: E402
import paths  # noqa: E402

METHOD_ID = "group_n005_voiceprint_protection"

# 结果文件目录（对外由 /files 提供）
# C 盘空间紧张时可用 VPP_RESULTS_DIR 指到 D 盘（与 VPP_DATA_ROOT 同一套环境变量风格）
RESULTS_DIR = Path(os.environ.get("VPP_RESULTS_DIR") or config.RESULTS_DIR)
UPLOAD_DIR = RESULTS_DIR / "_uploads"

# 允许的 artifacts 类型（前端只认这几个）
ARTIFACT_KINDS = ("image", "audio", "video", "json", "text")

app = FastAPI(
    title="语音声纹主动防御 · 算法服务",
    version="1.0.0",
    description="基于人机感知差异解耦的隐式声纹保护（Interspeech 2026 复现）",
)

# 跨域：本地演示放开所有来源；上线请改成具体域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 统一错误格式
#
# ★ 前端读的是 `detail.error`（见 src/services/algorithms.js:104）：
#   const detail = await response.json();
#   throw new Error(typeof detail.error === 'string' ? detail.error.slice(0,500) : ...);
# 而 FastAPI 默认返回的是 `{"detail": "..."}` —— 那样前端只会显示
# "算法服务返回 HTTP 422。"，把我们的中文说明全丢掉。
# 所以这里覆盖掉默认处理器，**所有**错误一律用 {"error": "..."}。
# ---------------------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, str):
        message = detail
    else:
        message = json.dumps(detail, ensure_ascii=False)
    return JSONResponse(status_code=exc.status_code, content={"error": message})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError) -> JSONResponse:
    """表单字段缺失/类型不对（例如没传 input_file）也要用 {"error": ...}。"""
    problems = []
    for error in exc.errors():
        location = " -> ".join(str(part) for part in error.get("loc", []) if part != "body")
        problems.append(f"{location}：{error.get('msg')}" if location else str(error.get("msg")))
    return JSONResponse(
        status_code=422,
        content={"error": "请求参数不对 —— " + "；".join(problems) if problems else "请求参数不对。"},
    )


# ---------------------------------------------------------------------------
# 模型缓存：进程内只建一次
# ---------------------------------------------------------------------------
_protector: Any = None
_protector_error: str | None = None


def get_protector():
    """懒加载并缓存 Protector（模型只加载一次，见文件头说明）。"""
    global _protector, _protector_error
    if _protector is not None:
        return _protector
    if _protector_error is not None:
        raise HTTPException(status_code=503, detail=_protector_error)

    try:
        from infer import Protector

        started = time.time()
        _protector = Protector()
        print(f"[service] 模型已加载，用时 {time.time() - started:.1f}s")
    except FileNotFoundError as error:
        # 权重缺失：给出"去哪儿下"的明确指引
        _protector_error = str(error)
        raise HTTPException(status_code=503, detail=_protector_error) from error
    except Exception as error:  # noqa: BLE001
        _protector_error = f"模型加载失败：{error}"
        traceback.print_exc()
        raise HTTPException(status_code=503, detail=_protector_error) from error
    return _protector


# ---------------------------------------------------------------------------
# 参数解析与校验
# ---------------------------------------------------------------------------
def parse_parameters(raw: str | None) -> dict:
    """把 parameters 的 JSON 字符串解析成 dict，并按方法配置校验。"""
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=422, detail=f"parameters 不是合法的 JSON：{error}") from error
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail="parameters 必须是一个 JSON 对象。")
    return parsed


def validate_parameters(params: dict) -> dict:
    """按方法 JSON 的定义校验参数（与前端 contract.js 的规则一致）。"""
    cleaned: dict[str, Any] = {}

    alpha = params.get("alpha", config.STAGE2["alpha"])
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="alpha 必须是数字。") from error
    if not 0.0 <= alpha <= 1.0:
        raise HTTPException(status_code=422, detail="alpha 的取值范围是 0 到 1。")
    cleaned["alpha"] = alpha

    attack_mode = params.get("attack_mode", "clone")
    if attack_mode not in ("clone", "none"):
        raise HTTPException(status_code=422, detail="attack_mode 只能是 clone 或 none。")
    cleaned["attack_mode"] = attack_mode

    report = params.get("report", True)
    cleaned["report"] = bool(report)

    text = params.get("text")
    if text is not None:
        text = str(text).strip()
        if len(text) > 500:
            raise HTTPException(status_code=422, detail="text 太长了（最多 500 字）。")
    cleaned["text"] = text or None

    seed = params.get("seed", config.SEED)
    try:
        cleaned["seed"] = int(seed)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="seed 必须是整数。") from error

    return cleaned


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _cosine(a, b) -> float:
    """两个向量的余弦相似度（音频里全是非负分量，用点积即可）。"""
    import numpy as np

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


def _artifact(kind: str, label: str, filename: str) -> dict:
    """拼一个 artifact（url 用相对路径，前端按接口地址解析）。"""
    if kind not in ARTIFACT_KINDS:
        raise ValueError(f"不支持的 artifact 类型：{kind}")
    return {
        "kind": kind,
        "label": label,
        "filename": filename,
        "url": f"/files/{filename}",
    }


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.get("/healthz")
def healthz() -> dict:
    """健康检查：顺带告诉用户模型装好了没、缺什么。"""
    ready = _protector is not None
    return {
        "ok": True,
        "method_id": METHOD_ID,
        "model_loaded": ready,
        "model_error": _protector_error,
        "results_dir": str(RESULTS_DIR),
        "data_root": str(paths.data_root()),
    }


@app.post("/process")
async def process(
    input_file: UploadFile = File(...),
    method_id: str = Form(...),
    parameters: str = Form(...),
) -> JSONResponse:
    """平台的主接口：一段语音 → 保护后语音（+ 攻击者克隆对比 + 评测报告）。"""
    if method_id != METHOD_ID:
        return JSONResponse(
            status_code=404,
            content={"error": f"未知的方法 ID：{method_id}。本服务只提供 {METHOD_ID}。"},
        )

    params = validate_parameters(parse_parameters(parameters))

    # -- 上传落盘（uuid 命名，不用用户提供的文件名）------------------------
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    suffix = Path(input_file.filename or "").suffix.lower()
    if suffix not in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
        suffix = ".wav"
    upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    try:
        content = await input_file.read()
    except Exception as error:  # noqa: BLE001
        return JSONResponse(status_code=422, content={"error": f"读取上传文件失败：{error}"})
    if not content:
        return JSONResponse(status_code=422, content={"error": "上传的文件是空的。"})

    max_bytes = int(config.SERVICE["max_upload_mb"]) * 1024 * 1024
    if len(content) > max_bytes:
        return JSONResponse(
            status_code=413,
            content={"error": f"文件超过 {config.SERVICE['max_upload_mb']} MB 上限。"},
        )

    upload_path.write_bytes(content)

    try:
        # -- 解码 -------------------------------------------------------
        sample_rate = config.MEL_PRESETS[config.DEFAULT_PRESET]["sample_rate"]
        try:
            wav = audio_utils.load_audio(upload_path, sample_rate)
        except Exception as error:  # noqa: BLE001
            return JSONResponse(
                status_code=422,
                content={"error": f"无法解码音频（请上传 wav/flac/mp3）：{error}"},
            )

        duration = audio_utils.duration_seconds(wav, sample_rate)
        if duration <= 0:
            return JSONResponse(status_code=422, content={"error": "音频长度为 0，无法处理。"})
        if duration > float(config.SERVICE["max_seconds"]):
            return JSONResponse(
                status_code=422,
                content={"error": f"音频过长（{duration:.1f} 秒），本方法只处理 "
                                  f"{config.SERVICE['max_seconds']} 秒以内的语音。"},
            )

        # -- 推理 -------------------------------------------------------
        protector = get_protector()
        started = time.time()
        result = protector.protect(
            wav,
            alpha=params["alpha"],
            seed=params["seed"],
            text=params["text"] or "",
            attack_mode=params["attack_mode"],
            verbose=False,
        )
        elapsed = time.time() - started

        # -- 落盘 + 组装 artifacts --------------------------------------
        artifacts: list[dict] = []
        stem = uuid.uuid4().hex[:12]

        protected_name = f"protected_{stem}.wav"
        audio_utils.save_audio(RESULTS_DIR / protected_name, result["protected_waveform"], sample_rate)
        artifacts.append(_artifact("audio", "保护后语音", protected_name))

        if params["attack_mode"] == "clone":
            baseline_name = f"clone_no_defense_{stem}.wav"
            defended_name = f"clone_with_defense_{stem}.wav"
            audio_utils.save_audio(
                RESULTS_DIR / baseline_name, result["clone_baseline_waveform"], sample_rate
            )
            audio_utils.save_audio(
                RESULTS_DIR / defended_name, result["clone_protected_waveform"], sample_rate
            )
            artifacts.append(_artifact("audio", "未防御时攻击者的克隆音", baseline_name))
            artifacts.append(_artifact("audio", "防御后攻击者的克隆音", defended_name))

        # -- 评测报告 ----------------------------------------------------
        if params["report"]:
            report = {
                "method_id": method_id,
                "alpha": params["alpha"],
                "seed": params["seed"],
                "attack_mode": params["attack_mode"],
                "duration_seconds": round(duration, 3),
                "elapsed_seconds": round(elapsed, 3),
                "shapes": result.get("shapes", {}),
                "note": (
                    "L1 阶段：扰动 δ 为随机方向占位，尚未训练防御生成器 G，"
                    "因此这里的指标反映的是链路是否正确，不代表论文水平。"
                ),
            }

            embedding = result.get("embedding")
            embedding_adv = result.get("embedding_adv")
            stolen = result.get("stolen_embedding")
            if embedding is not None and embedding_adv is not None:
                report["sim_embedding_perturbed"] = round(_cosine(embedding, embedding_adv), 4)
            if embedding is not None and stolen is not None:
                # 攻击者从保护后语音重新提的卡 vs 原卡 —— 越小说明防御越有效
                report["sim_clone"] = round(_cosine(embedding, stolen), 4)
                report["sim_clone_threshold_0.5"] = bool(_cosine(embedding, stolen) < 0.5)

            report_name = f"report_{stem}.json"
            (RESULTS_DIR / report_name).write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            artifacts.append(_artifact("json", "评测报告", report_name))

        return JSONResponse(
            status_code=200,
            content={
                "message": f"防御处理完成（用时 {elapsed:.1f} 秒）",
                "artifacts": artifacts,
            },
        )

    except HTTPException as error:
        return JSONResponse(status_code=error.status_code, content={"error": str(error.detail)})
    except KeyError as error:
        return JSONResponse(
            status_code=500,
            content={"error": f"推理结果缺少字段 {error}，请检查 infer.py 的实现。"},
        )
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"处理失败：{error}"},
        )
    finally:
        # 上传的临时文件用完即删；结果文件保留供前端下载
        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 静态托管：结果文件
# ---------------------------------------------------------------------------
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(RESULTS_DIR)), name="files")


@app.on_event("startup")
def cleanup_stale_uploads() -> None:
    """启动时清掉上一次的残留（上传目录与结果目录里的旧文件）。"""
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # 结果文件只保留最近 50 个，避免磁盘无限增长
    try:
        files = sorted(
            (p for p in RESULTS_DIR.glob("*") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in files[50:]:
            stale.unlink(missing_ok=True)
    except OSError:
        pass

    print(f"[service] 结果目录：{RESULTS_DIR}")
    print(f"[service] 数据根  ：{paths.data_root()}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.SERVICE["host"],
        port=int(config.SERVICE["port"]),
    )
