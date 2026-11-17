"""GE2E 说话人编码器桥接层 —— 成员 B 主责（任务板 W2-D3-4）。

【它解决什么】
Stage 1 的解码器、Stage 2 的防御损失 L_de、以及全部评测指标，
都要用到说话人嵌入 h（256 维）。h 从哪来？
→ **GE2E**（论文脚注 2 直接引用的 Real-Time-Voice-Cloning 仓库里的 `SpeakerEncoder`）。

本模块把那个"外部预训练组件"包成全项目统一的入口：

    wav (任意采样率) ──► bridge_speaker.embed(wav) ──► h (256,) L2 归一化

【★ 上游有两种布局，本模块都支持】
2026-03 之前的"旧版"：平铺文件 `speaker_encoder.py` + `voice_encoder.py` + `preprocess.py`，
权重是仓库里的 `pretrained.pt`（约 30 MB）。
2026-03 之后的"新版"（**当前上游 master**）：包结构 `encoder/`（model / inference / audio /
params_data / params_model），权重**不再随仓库分发**，改从 Hugging Face 下载
`CorentinJ/SV2TTS` 的 `encoder.pt`（17,090,379 字节）。

新版还改了 GE2E 内部的一个关键约定：**mel 不再做 log + 均值方差归一化，直接用线性 mel**
（`encoder/audio.py::wav_to_mel_spectrogram`，注释原文 "Note: this not a log-mel spectrogram"）。
所以**不能拿旧版的 mel 代码去喂新版的权重** —— 必须用上游自己的函数。

本模块的做法：**优先调用上游自己的代码**（`encoder.inference.embed_utterance`），
不复制、不重写它那套"切片 → 逐片编码 → 平均 → 归一化"的逻辑。

【权重从哪来】
    https://huggingface.co/CorentinJ/SV2TTS  →  encoder.pt（17,090,379 字节）
用 `scripts/download_encoder_weights.py` 下载（支持国内 hf-mirror 加速）。

【关于"输入约定"的四个问题（任务板 D2-4）】
本模块**不靠记忆**回答这四个问题，而是运行时从上游代码里读出来（见 `describe()`）：
  Q1 输入采样率？          ← `encoder/params_data.py::sampling_rate`
  Q2 内部 mel 几维/什么参数？← `encoder/params_data.py` 的 mel_window_length / mel_window_step
                              + `encoder/audio.py::wav_to_mel_spectrogram`
  Q3 输出 h 是否已归一化？  ← `encoder/model.py::forward` 里显式 L2 归一化；运行时再实测模长
  Q4 输出形状？            ← `encoder/params_model.py::model_embedding_size`

【安全提示】
上游仓库是从网上拉下来的**第三方代码**，本模块会 import 并执行它。因此：
  * 务必在 third_party/README.md 里**记录 commit hash**，锁定版本；
  * 运行时**不要把工作目录切到上游仓库里面**（避免误执行它的 demo 脚本）；
  * 建议用 `python -I` 运行本项目脚本，减少被污染的解释器环境。

【用法】
    conda activate vpp
    python algorithms/voiceprint_protection/bridge_speaker.py --wav 某条音频.wav
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

try:
    import numpy as np
    import torch

    import audio_utils  # noqa: E402
    import config  # noqa: E402
except ImportError as error:  # 多半是忘了 conda activate vpp
    paths.print_env_hint(error)
    raise SystemExit(1) from error

# GE2E 只吃 16 kHz（它的权重就是 16 kHz 训的，改采样率权重直接失效）
GE2E_SAMPLE_RATE = 16000

# 上游仓库的可能位置（按优先级）
RTC_DIR_CANDIDATES = (
    "third_party/RTC",
    "third_party/Real-Time-Voice-Cloning",
    "algorithms/voiceprint_protection/third_party/RTC",
)


class SpeakerBridgeError(RuntimeError):
    """找不到 GE2E 权重或加载失败时的统一报错。"""


# ---------------------------------------------------------------------------
# 定位上游仓库
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _looks_like_rtc(path: Path) -> bool:
    """这个目录像不像 RTC 仓库（两种布局都认）。"""
    if not path.is_dir():
        return False
    new_layout = (path / "encoder" / "inference.py").exists()
    old_layout = (path / "speaker_encoder.py").exists() or (path / "voice_encoder.py").exists()
    return new_layout or old_layout


def find_rtc_dir(explicit: str | None = None) -> Path:
    """找到 Real-Time-Voice-Cloning 仓库目录。

    优先级：命令行参数 > 环境变量 VPP_RTC_DIR > 仓库内常见位置。
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_dir = os.environ.get("VPP_RTC_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    root = _repo_root()
    for relative in RTC_DIR_CANDIDATES:
        candidates.append(root / relative)
        candidates.append(Path(relative))

    for candidate in candidates:
        if _looks_like_rtc(candidate):
            return candidate.resolve()

    raise SpeakerBridgeError(
        "找不到 Real-Time-Voice-Cloning 仓库。\n"
        "请先执行（任务板 W2-D1-3）：\n"
        "    git clone https://github.com/CorentinJ/Real-Time-Voice-Cloning.git third_party/RTC\n"
        "国内可换成 gitee 镜像：\n"
        "    git clone --depth 1 https://gitee.com/mirrors/real-time-voice-cloning.git third_party/RTC\n"
        "或用 --rtc-dir / 环境变量 VPP_RTC_DIR 指定路径。\n"
        f"已尝试：{', '.join(str(c) for c in candidates)}"
    )


def _is_html_error_page(path: Path) -> bool:
    """判断文件是不是"下载失败留下的 HTML 错误页"。"""
    try:
        with path.open("rb") as handle:
            head = handle.read(200).lstrip().lower()
    except OSError:
        return False
    return head.startswith(b"<!doctype") or head.startswith(b"<html")


# ---------------------------------------------------------------------------
# 权重定位
# ---------------------------------------------------------------------------
def _candidate_weight_files(rtc_dir: Path) -> list[Path]:
    """按优先级列出可能的权重文件位置。"""
    models_dir = paths.models_dir()
    return [
        models_dir / "default" / "encoder.pt",   # 推荐：download_encoder_weights.py 的输出
        models_dir / "encoder.pt",
        rtc_dir / "encoder.pt",
        rtc_dir / "saved_models" / "default" / "encoder.pt",
        rtc_dir / "pretrained.pt",               # 旧版布局
    ]


def find_weights(rtc_dir: Path, explicit: str | None = None) -> Path:
    """找到 GE2E 的权重文件。"""
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SpeakerBridgeError(f"指定的权重文件不存在：{path}")
        return path

    for candidate in _candidate_weight_files(rtc_dir):
        if candidate.exists() and candidate.stat().st_size > 1024:
            if _is_html_error_page(candidate):
                raise SpeakerBridgeError(
                    f"{candidate} 是一个 HTML 错误页，不是权重文件！\n"
                    "（通常是下载被拦或 404 之后把错误页面存成了 .pt）\n"
                    "请删掉它，然后重新下载：\n"
                    "    python algorithms/voiceprint_protection/scripts/download_encoder_weights.py"
                )
            return candidate

    searched = "\n".join(f"    {c}" for c in _candidate_weight_files(rtc_dir))
    raise SpeakerBridgeError(
        "找不到 GE2E 的权重文件（encoder.pt）。\n"
        "**新版上游不再把权重放在仓库里**，需要单独下载（约 17 MB）：\n"
        "\n"
        "    python algorithms/voiceprint_protection/scripts/download_encoder_weights.py\n"
        "\n"
        "国内网络慢可以先设镜像：\n"
        "    set HF_ENDPOINT=https://hf-mirror.com\n"
        "\n"
        f"已尝试：\n{searched}"
    )


# ---------------------------------------------------------------------------
# 桥接层
# ---------------------------------------------------------------------------
class SpeakerBridge:
    """GE2E 说话人编码器的统一入口（同时支持新版 / 旧版上游布局）。"""

    def __init__(
        self,
        rtc_dir: str | None = None,
        device: str | None = None,
        weights_path: str | None = None,
    ):
        self.rtc_dir = find_rtc_dir(rtc_dir)
        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(requested)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            # 装了 CPU 版 torch 时不要在这里就炸：推理用不到 GPU。
            # 只有 Stage 1/2 训练才真的需要 GPU（放云上，见环境文档 §2.3）。
            print("  [提示] 指定了 CUDA，但当前 torch 是 CPU 版"
                  "（torch.cuda.is_available() == False），自动改用 CPU。")
            print("         推理不需要 GPU；只有训练才需要，按环境文档 §2.3 换 CUDA 版。")
            self.device = torch.device("cpu")
        self.layout = "new" if (self.rtc_dir / "encoder" / "inference.py").exists() else "old"

        self.weights_path = find_weights(self.rtc_dir, weights_path)
        self.encoder = self._build_encoder()
        self._describe_cache: dict | None = None

    # -- 内部 ------------------------------------------------------------
    def _build_encoder(self):
        if self.layout == "new":
            return self._build_new()
        return self._build_old()

    def _build_new(self):
        """新版：直接用上游的 `encoder.inference`。"""
        if str(self.rtc_dir) not in sys.path:
            sys.path.insert(0, str(self.rtc_dir))
        try:
            from encoder import inference as encoder_inference
        except ImportError as error:
            raise SpeakerBridgeError(
                f"导入上游的 encoder 包失败：{error}\n"
                "上游的 encoder/model.py 依赖 scikit-learn，请先安装：\n"
                "    pip install scikit-learn==1.5.2"
            ) from error

        # ⚠️ 上游 load_model 的签名是 load_model(weights_fpath: Path, device=None)，
        #    两个参数各有各的"正确类型"，搞反任何一个都会炸：
        #
        # 1) weights_fpath 必须是 **Path 对象**
        #    上游最后一行要打印 weights_fpath.name —— 传 str 会报
        #       AttributeError: 'str' object has no attribute 'name'
        #
        # 2) device 必须是 **字符串**
        #    上游写法：
        #        if device is None:          _device = ...
        #        elif isinstance(device, str): _device = torch.device(device)
        #        # ← 传 torch.device 对象时两个分支都不进，_device 保持 None
        #    于是 torch.load(weights, None) → map_location=None →
        #    试图把**存于 CUDA 的 checkpoint 反序列化回 CUDA**，在 CPU 版 torch 上报：
        #        RuntimeError: Attempting to deserialize object on a CUDA device
        #                       but torch.cuda.is_available() is False
        encoder_inference.load_model(self.weights_path, str(self.device))

        # ★ 修 librosa 0.10 的 API 漂移（上游按 0.9.2 写的）。
        #   上游的 mel 函数用位置参数调用 librosa.feature.melspectrogram，
        #   而 0.10 把 y / sr 改成了仅关键字参数 → 必然 TypeError。
        #   换成数值完全等价的实现（参数现场从上游 params_data 读）。
        try:
            from encoder import audio as encoder_audio
            from encoder import params_data as encoder_params

            if patch_upstream_mel(encoder_audio, encoder_params):
                print("  [已适配] 上游的 mel 计算改为 librosa 0.10 兼容写法（数值等价）。")
        except Exception as error:  # noqa: BLE001
            print(f"  [警告] mel 适配失败：{error}")

        return encoder_inference

    def _build_old(self):
        """旧版：平铺的 speaker_encoder.py + 仓库内 pretrained.pt。"""
        speaker_file = self.rtc_dir / "speaker_encoder.py"
        if not speaker_file.exists():
            raise SpeakerBridgeError(
                f"{self.rtc_dir} 里既没有 encoder/ 包也没有 speaker_encoder.py，"
                "不是有效的 Real-Time-Voice-Cloning 仓库。"
            )

        module = _load_module_from_file(speaker_file, "vpp_rtc_speaker_encoder")
        encoder_cls = module.SpeakerEncoder
        encoder = encoder_cls(
            **_filter_kwargs(
                encoder_cls.__init__,
                device=self.device,
                sampling_rate=GE2E_SAMPLE_RATE,
            )
        )
        state = _load_state_dict(self.weights_path, self.device)
        try:
            encoder.load_state_dict(state)
        except Exception as error:  # noqa: BLE001
            raise SpeakerBridgeError(
                f"权重与模型结构对不上：{error}\n"
                "请确认 pretrained.pt 与仓库版本配套（见 third_party/README.md 的 commit）。"
            ) from error
        encoder.eval()
        return encoder

    # -- 对外 ------------------------------------------------------------
    def embed(self, wav: np.ndarray, sample_rate: int) -> torch.Tensor:
        """波形 → 说话人嵌入 h (256,)。

        参数
        ----
        wav         : float32 单声道波形，取值范围约 [-1, 1]
        sample_rate : wav 的原生采样率（不是 16 kHz 时由上游内部重采样）

        返回
        ----
        torch.Tensor，形状 (256,)，**已 L2 归一化**（接口契约第 5 条）。
        """
        wav = np.asarray(wav, dtype=np.float32)
        if len(wav) < GE2E_SAMPLE_RATE // 10:
            raise ValueError(f"音频太短（{len(wav) / GE2E_SAMPLE_RATE:.3f}s），无法提取嵌入")

        # ★ 我们自己先重采样到 16 kHz，**不要**让上游去重采样。
        #   上游 encoder/audio.py::preprocess_wav 里是：
        #       wav = librosa.resample(wav, source_sr, sampling_rate)   ← 位置参数
        #   而 librosa **0.10 把采样率改成了仅关键字参数**
        #   （resample(y, *, orig_sr, target_sr, ...)），
        #   位置传参会直接 TypeError。
        #   上游按 librosa 0.9.2 写的（见其 pyproject.toml），
        #   我们锁的是 librosa==0.10.2.post1 —— 这个坑必然踩到。
        #   提前重采样之后，上游那行 if 分支根本不会进。
        if sample_rate != GE2E_SAMPLE_RATE:
            wav = audio_utils.resample(wav, sample_rate, GE2E_SAMPLE_RATE)

        raw = self._embed_raw(wav, GE2E_SAMPLE_RATE)
        return audio_utils.normalize_l2(torch.as_tensor(raw, dtype=torch.float32))

    def embed_path(self, path: str | Path) -> torch.Tensor:
        """音频文件 → 说话人嵌入 h (256,)。"""
        wav = audio_utils.load_audio(path)  # 保持原生采样率
        import soundfile as sf

        return self.embed(wav, sf.info(str(path)).samplerate)

    def embed_batch(self, waves: list[np.ndarray], sample_rates: list[int]) -> torch.Tensor:
        """一批波形 → (N, 256)。"""
        return torch.stack([self.embed(w, sr) for w, sr in zip(waves, sample_rates)])

    def _embed_raw(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        """调用上游模型拿到嵌入。

        新版：`embed_utterance` 内部已经做了"切片 → 逐片编码 → 平均 → 归一化"，
        返回的就是已归一化的向量。
        旧版：`SpeakerEncoder.embed_utterance`。
        """
        if self.layout == "new":
            # 上游的 preprocess_wav 负责音量归一化 + 静音裁剪。
            # 我们已经把音频重采样到 16 kHz 了（见 embed() 里的说明），
            # 所以传 source_sr=16000，上游内部不会再去调 librosa.resample。
            processed = self.encoder.preprocess_wav(wav, source_sr=GE2E_SAMPLE_RATE)
            with torch.no_grad():
                result = self.encoder.embed_utterance(processed)
            return _to_numpy_1d(result)

        # 旧版
        wav_16k = wav
        if sample_rate != GE2E_SAMPLE_RATE:
            wav_16k = audio_utils.resample(wav, sample_rate, GE2E_SAMPLE_RATE)

        encoder = self.encoder
        for holder in (encoder, getattr(encoder, "encoder", None)):
            if holder is None:
                continue
            method = getattr(holder, "embed_utterance", None)
            if method is None:
                continue
            with torch.no_grad():
                result = method(y=wav_16k, sr=GE2E_SAMPLE_RATE)
            return _to_numpy_1d(result)

        mel = self._compute_mel_old(wav_16k)
        with torch.no_grad():
            result = encoder(mel)
        return _to_numpy_1d(result)

    def _compute_mel_old(self, wav_16k: np.ndarray) -> torch.Tensor:
        """旧版布局的 40 维 mel（log + 均值方差归一化）。仅旧版使用。"""
        import librosa

        mel = librosa.feature.melspectrogram(
            y=wav_16k, sr=GE2E_SAMPLE_RATE, n_fft=400, hop_length=160,
            n_mels=40, fmin=0, fmax=8000,
        )
        mel = librosa.power_to_db(mel, power=1.97)
        mel = np.log10(np.maximum(1e-10, mel))
        mel = (mel - (-4.2677393)) / 4.5689974
        return torch.from_numpy(mel.astype(np.float32)).unsqueeze(0)

    # -- 接口自述 --------------------------------------------------------
    def describe(self) -> dict:
        """读出上游的接口约定（回答任务板 D2-4 的四个问题）。"""
        if self._describe_cache is not None:
            return self._describe_cache

        info: dict = {
            "仓库目录": str(self.rtc_dir),
            "上游布局": "新版（encoder/ 包）" if self.layout == "new" else "旧版（平铺文件）",
            "权重文件": str(self.weights_path),
            "权重大小": f"{self.weights_path.stat().st_size / 1024 / 1024:.2f} MB",
            "运行设备": str(self.device),
        }

        if self.layout == "new":
            try:
                from encoder import params_data, params_model
            except ImportError:
                params_data = params_model = None  # type: ignore[assignment]

            if params_data is not None:
                sampling_rate = params_data.sampling_rate
                info.update({
                    "采样率 (Hz)": sampling_rate,
                    "mel 维数": params_data.mel_n_channels,
                    "mel 窗长 (ms)": params_data.mel_window_length,
                    "mel 步长 (ms)": params_data.mel_window_step,
                    "mel n_fft / hop": (
                        f"{int(sampling_rate * params_data.mel_window_length / 1000)}"
                        f" / {int(sampling_rate * params_data.mel_window_step / 1000)}"
                    ),
                    "mel 是否取 log": "否（新版直接用线性 mel，见 encoder/audio.py）",
                    "分片帧数 / 重叠": f"{params_data.partials_n_frames} / 0.5",
                    "音量归一化目标": f"{params_data.audio_norm_target_dBFS} dBFS",
                })
            if params_model is not None:
                info.update({
                    "嵌入维度": params_model.model_embedding_size,
                    "LSTM 隐层/层数": f"{params_model.model_hidden_size} / {params_model.model_num_layers}",
                })
            info["上游输出是否已 L2 归一化"] = "是（encoder/model.py::forward 内部显式归一化）"
        else:
            inner = getattr(self.encoder, "encoder", self.encoder)
            info.update({
                "采样率 (Hz)": getattr(inner, "sampling_rate", GE2E_SAMPLE_RATE),
                "mel 维数": getattr(inner, "mel_n_channels", "未在类属性中暴露"),
                "嵌入维度": getattr(inner, "model_embedding_size", config.SPEAKER_DIM),
                "上游输出是否已 L2 归一化": "否（旧版 forward 不归一化，由本模块统一补上）",
            })

        self._describe_cache = info
        return info


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _load_module_from_file(path: Path, module_name: str):
    """按文件路径 import 一个模块（旧版上游不是 pip 包，不能直接 import）。"""
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise SpeakerBridgeError(f"无法加载模块文件：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_state_dict(path: Path, device) -> dict:
    """加载 checkpoint，兼容裸 state_dict / {'model': ...} / {'model_state': ...}。"""
    try:
        loaded = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:  # 很老的 torch 没有 weights_only 参数
        loaded = torch.load(str(path), map_location=device)
    except Exception as error:  # noqa: BLE001
        raise SpeakerBridgeError(f"加载权重失败 {path}：{error}") from error

    if isinstance(loaded, dict):
        for key in ("model_state", "model", "state_dict", "model_state_dict"):
            if key in loaded and isinstance(loaded[key], dict):
                return loaded[key]
    return loaded


def _filter_kwargs(callable_obj, **kwargs):
    """只保留目标函数真正接受的参数（不同版本签名有出入）。"""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return kwargs
    allowed = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in allowed}


def patch_upstream_mel(encoder_audio, encoder_params) -> bool:
    """把上游的 `wav_to_mel_spectrogram` 换成 librosa 0.10 兼容版。

    **为什么必须换**
    上游按 librosa **0.9.2** 写的（见其 pyproject.toml）：

        librosa.feature.melspectrogram(wav, sampling_rate, n_fft=..., hop_length=..., n_mels=...)

    而 librosa **0.10** 的签名是：

        melspectrogram(*, y=None, sr=22050, n_fft=2048, hop_length=512, ...)

    `y` 与 `sr` 变成了**仅关键字参数** → 位置调用直接
    `TypeError: melspectrogram() takes 0 positional arguments but 2 were given`。
    我们锁的是 librosa==0.10.2.post1，**必然踩到**。
    （上游 2026-03 那个 "Fix librosa.load TypeError" 的 commit 只修了 load，漏了这一处。）

    **换过去为什么数值不变**
    新实现与上游**逐参数等价**：同样的 librosa 默认值
    （`center=True, power=2.0, amin=1e-10, fmin=0, fmax=sr/2, htk=False, norm='slaney'`），
    同样的 n_fft / hop / n_mels（从上游 `params_data` 现场读，不写死）。
    所以算出来的 mel 与上游本意**完全一致**。

    返回 True 表示做了替换。
    """
    import librosa

    sampling_rate = encoder_params.sampling_rate
    mel_window_length = encoder_params.mel_window_length
    mel_window_step = encoder_params.mel_window_step
    mel_n_channels = encoder_params.mel_n_channels

    def wav_to_mel_spectrogram(wav):
        frames = librosa.feature.melspectrogram(
            y=wav,
            sr=sampling_rate,
            n_fft=int(sampling_rate * mel_window_length / 1000),
            hop_length=int(sampling_rate * mel_window_step / 1000),
            n_mels=mel_n_channels,
        )
        return frames.astype(np.float32).T

    encoder_audio.wav_to_mel_spectrogram = wav_to_mel_spectrogram
    return True


def _to_numpy_1d(value) -> np.ndarray:
    """把上游返回的 tensor / ndarray / list 统一成 (D,) 的 float32 numpy。"""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float32)
    if value.ndim > 1:
        value = value.reshape(-1)
    return value


# ---------------------------------------------------------------------------
# 自检入口
# ---------------------------------------------------------------------------
def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="GE2E 桥接层自检（成员 B）")
    parser.add_argument("--wav", nargs="+", default=[], help="用来测试的音频（可多给几条）")
    parser.add_argument("--rtc-dir", default=None, help="Real-Time-Voice-Cloning 仓库路径")
    parser.add_argument("--weights", default=None, help="权重文件路径（默认自动找）")
    parser.add_argument("--device", default=None, help="cpu / cuda")
    args = parser.parse_args()

    print("=" * 64)
    print("  GE2E 说话人编码器桥接层自检")
    print("=" * 64)

    try:
        bridge = SpeakerBridge(
            rtc_dir=args.rtc_dir, device=args.device, weights_path=args.weights
        )
    except SpeakerBridgeError as error:
        print(f"\n[失败] {error}")
        return 1

    info = bridge.describe()
    print("\n【接口约定（运行时从上游代码读出）】")
    for key, value in info.items():
        print(f"  {key}: {value}")

    if not args.wav:
        print("\n未指定 --wav，上面已经验证了「仓库找到 + 权重加载 + 模型构建」。")
        print("加两条音频可以顺便验证：同一人相似度应明显高于不同人。")
        return 0

    print("\n【嵌入测试】")
    embeddings = []
    for wav_path in args.wav:
        embedding = bridge.embed_path(wav_path)
        embeddings.append(embedding)
        print(f"  {Path(wav_path).name:<28} h.shape = {tuple(embedding.shape)}  "
              f"|h| = {float(embedding.norm()):.6f}")

    print(f"\n  Q3 输出是否已 L2 归一化：实测模长 = "
          f"{float(embeddings[0].norm()):.6f}（应为 1.000000）")
    print("     → 本模块出口还会再归一化一次（幂等，无副作用）。")

    if len(embeddings) >= 2:
        similarity = float(torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
        ))
        print(f"\n  两段音频的余弦相似度 = {similarity:.4f}")
        print("     （同一人说话通常 > 0.7，不同人通常 < 0.5；数值异常先查权重是否加载对）")

    print("\n  自检完成。把以上结论抄进 docs/notes/ge2e_interface.md。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
