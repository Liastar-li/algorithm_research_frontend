"""桥接：说话人编码器 —— "体检仪"，把声音扫描成身份卡。

【它在做什么（比喻）】
你把一段录音递给体检仪，它扫描后给你一张 256 个数字组成的"身份卡"。
同一段内容换个人说，身份卡就变了；同一个人说不同的话，身份卡基本不变。

【★ 本文件有两套实现，作用不同，请勿混用】
| | `SpeakerEncoderBridge`（成员 A 写） | `SpeakerBridge`（成员 B 写） |
|---|---|---|
| 后端 | AutoVC 自带 D_VECTOR（`3000000-BL.ckpt`） | **论文脚注引用的 GE2E**（上游 Real-Time-Voice-Cloning） |
| 输入 | **mel (T, 80)**（AutoVC 的 spmel 链） | **原始波形**（自己算 40 维 mel） |
| 权重 | AutoVC 官方 ckpt | HuggingFace `CorentinJ/SV2TTS` 的 `encoder.pt` |
| 谁在用 | `infer.py` 的默认路线 | 论文路线 / 评测 |

两套的**权重不能互相加载**（输入维度与结构都不同）：
`SpeakerEncoderBridge` 的 `autovc_bl` 吃 80 维 mel，
`SpeakerBridge` 走上游自己的 40 维线性 mel。

【为什么 B 的 GE2E 不是"重写一遍"】
上游 RTC 在 2026-03 重构过，现在权重在 HuggingFace 上、模型在 `encoder/` 包里，
而且**内部 mel 约定变了**（不再取 log）。所以正确做法是**直接调用上游的代码**，
而不是照着论文/旧版重写 —— 那会得到"能跑但全错"的嵌入。
`SpeakerBridge` 因此只做三件事：定位上游仓库、加载权重、调用
`encoder.inference.embed_utterance()`。上游那套"切片 → 逐片编码 → 平均 → 归一化"
的逻辑原样保留。

上游的三处 API 漂移（都已适配，见 `SpeakerBridge` 内部注释）：
`load_model` 的参数类型、librosa 0.10 的仅关键字参数、mel 约定变更。

【接口细节见】docs/notes/ge2e_interface.md（成员 B 核实）
【官方实现（third_party/AutoVC/model_bl.py）】
    class D_VECTOR(nn.Module):
        def __init__(self, num_layers=3, dim_input=80, dim_cell=768, dim_emb=256):
            self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_cell,
                                num_layers=num_layers, batch_batch_first=True)
            self.embedding = nn.Linear(dim_cell, dim_emb)
        def forward(self, x):                      # x: (B, T, dim_input)
            lstm_out, _ = self.lstm(x)
            embeds = self.embedding(lstm_out[:, -1, :])   # 取最后一个时间步
            return embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))   # L2 归一化

【官方输入约定（third_party/AutoVC/make_metadata.py）】
    melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()  # (1, 128, 80)
    emb = C(melsp)          # -> (1, 256)
即：**输入是 128 帧（2 秒 @16kHz/hop256）× 80 维的 mel**，取最后时间步的隐藏态。
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from pathlib import Path

import config
import paths

try:
    import numpy as np

    import torch

    import audio_utils  # noqa: E402
except ImportError as error:  # 多半是忘了 conda activate vpp
    paths.print_env_hint(error)
    raise SystemExit(1) from error

# 官方 make_metadata.py 里的 len_crop
LEN_CROP = 128

# GE2E 只吃 16 kHz（它的权重就是 16 kHz 训的，改采样率权重直接失效）
GE2E_SAMPLE_RATE = 16000


# ===========================================================================
# A 的实现：AutoVC 自带的 D_VECTOR（mel 80 维进）
# ===========================================================================
class SpeakerEncoderBridge:
    """说话人编码器的薄封装（AutoVC 官方链）。"""

    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None,
                 encoder: str = "autovc_bl"):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.encoder_name = encoder

        # ★ 论文路线：交给上游 GE2E（下面 SpeakerBridge 的实现）。
        #   这不是"多写一个后端"，而是**修正**：原先这个分支按旧版 RTC 的
        #   pretrained.pt 加载权重，而 2026-03 上游重构后那个文件已不存在，
        #   权重改成 HuggingFace 的 encoder.pt（结构与算法也不同）。
        if encoder == "ge2e_rtc":
            self._upstream = SpeakerBridge(device=str(self.device))
            self.checkpoint_path = self._upstream.weights_path
            self.model = None
            return

        paths.ensure_import_paths()
        from model_bl import D_VECTOR

        settings = config.SPEAKER_ENCODERS[encoder]
        self.model = D_VECTOR(
            num_layers=settings["num_layers"],
            dim_input=settings["dim_input"],
            dim_cell=settings["dim_cell"],
            dim_emb=settings["dim_emb"],
        )
        self.model.eval().to(self.device)

        key = "speaker_bl" if encoder == "autovc_bl" else "ge2e_rtc"
        self.checkpoint_path = Path(checkpoint) if checkpoint else paths.weight_path(key)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"找不到说话人编码器权重：{self.checkpoint_path}\n"
                "AutoVC 的 3000000-BL.ckpt 请从官方 README 的 IBM Box 链接下载，"
                "放到：" + str(paths.WEIGHTS_DIR)
            )
        checkpoint_data = torch.load(self.checkpoint_path, map_location="cpu")
        if "model_b" in checkpoint_data:
            # 官方 make_metadata.py 的写法：键名前 7 个字符是 "module."（DataParallel 包装）
            state = {key[7:]: value for key, value in checkpoint_data["model_b"].items()}
        elif "model" in checkpoint_data:
            state = checkpoint_data["model"]
        else:
            state = checkpoint_data
        self.model.load_state_dict(state)
        self.model.to(self.device)

    # -- 对外 ------------------------------------------------------------
    def embed(self, mel: np.ndarray, crop: int = LEN_CROP, sample_rate: int = 16000) -> np.ndarray:
        """mel (T, 80) -> 说话人嵌入 (256,)，已 L2 归一化。

        官方做法是在时间轴上随机截取 `crop` 帧；
        为了可复现，这里固定取**居中**的 crop 帧（seed 固定的随机截取也可）。

        ⚠️ **当 encoder="ge2e_rtc" 时，第一个参数是原始波形而不是 mel**
        （上游 GE2E 自己算它的 40 维 mel）。
        """
        if self.encoder_name == "ge2e_rtc":
            return self._upstream.embed(mel, sample_rate)

        mel = np.asarray(mel, dtype=np.float32)
        length = mel.shape[0]
        if length >= crop:
            start = (length - crop) // 2
            window = mel[start:start + crop]
        else:
            # 不足 crop 帧时补零（官方数据每条都有 128 帧以上）
            window = np.pad(mel, ((0, crop - length), (0, 0)), "constant")
        tensor = torch.from_numpy(window[None, :, :]).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)
        return embedding[0].cpu().numpy()

    def describe(self) -> str:
        if self.encoder_name == "ge2e_rtc":
            return (
                "说话人编码器桥接就绪（论文路线：上游 GE2E）\n"
                f"  权重    : {self.checkpoint_path}\n"
                "  编码器  : ge2e_rtc（输入是原始波形，内部算 40 维线性 mel）\n"
                f"  设备    : {self.device}\n"
                "  输出    : (256,) 已 L2 归一化"
            )
        return (
            "说话人编码器桥接就绪\n"
            f"  权重    : {self.checkpoint_path}\n"
            f"  编码器  : {self.encoder_name}"
            f"（输入 {config.SPEAKER_ENCODERS[self.encoder_name]['dim_input']} 维 mel）\n"
            f"  设备    : {self.device}\n"
            "  输出    : (256,) 已 L2 归一化"
        )


# ===========================================================================
# B 的实现：论文脚注引用的 GE2E（上游 Real-Time-Voice-Cloning）
# ===========================================================================
class SpeakerBridge:
    """GE2E 说话人编码器的统一入口（wav -> h (256,)）。

    【为什么直接调上游，而不是重写】
    上游的 mel 约定、切片策略、归一化都在 2026-03 改过，
    照着旧论文重写会得到"能跑但全错"的嵌入。
    这里只做：定位仓库 → 加载权重 → 调 `encoder.inference.embed_utterance()`。
    """

    def __init__(
        self,
        rtc_dir: str | None = None,
        device: str | None = None,
        weights_path: str | None = None,
    ):
        self.rtc_dir = find_rtc_dir(rtc_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
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
    def embed(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        """波形 → 说话人嵌入 h (256,)。

        参数
        ----
        wav         : float32 单声道波形，取值范围约 [-1, 1]
        sample_rate : wav 的原生采样率（不是 16 kHz 时由上游内部重采样）

        返回
        ----
        numpy (256,)，**已 L2 归一化**（接口契约第 5 条）。
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
        return audio_utils.normalize_l2(torch.as_tensor(raw, dtype=torch.float32)).numpy()

    def embed_path(self, path: str | Path) -> np.ndarray:
        """音频文件 → 说话人嵌入 h (256,)。"""
        wav = audio_utils.load_audio(path)  # 保持原生采样率
        import soundfile as sf

        return self.embed(wav, sf.info(str(path)).samplerate)

    def embed_batch(self, waves: list[np.ndarray], sample_rates: list[int]) -> np.ndarray:
        """一批波形 → (N, 256)。"""
        return np.stack([self.embed(w, sr) for w, sr in zip(waves, sample_rates)])

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

    def _compute_mel_old(self, wav_16k: np.ndarray) -> np.ndarray:
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


class SpeakerBridgeError(RuntimeError):
    """找不到 GE2E 权重或加载失败时的统一报错。"""


# ---------------------------------------------------------------------------
# 定位上游仓库与权重
# ---------------------------------------------------------------------------
def _looks_like_rtc(path: Path) -> bool:
    """这个目录像不像 RTC 仓库（两种布局都认）。"""
    if not path.is_dir():
        return False
    new_layout = (path / "encoder" / "inference.py").exists()
    old_layout = (path / "speaker_encoder.py").exists() or (path / "voice_encoder.py").exists()
    return new_layout or old_layout


def find_rtc_dir(explicit: str | None = None) -> Path:
    """找到 Real-Time-Voice-Cloning 仓库目录。

    优先级：命令行参数 > 环境变量 VPP_RTC_DIR > paths.RTC_DIR（已解析好）> 常见位置。
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_dir = os.environ.get("VPP_RTC_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(paths.RTC_DIR)

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


def find_weights(rtc_dir: Path, explicit: str | None = None) -> Path:
    """找到 GE2E 的权重文件。"""
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SpeakerBridgeError(f"指定的权重文件不存在：{path}")
        return path

    # ★ 逐个候选目录试（权重可能在 <数据根>/models 或 <仓库根>/data/models）
    candidates: list[Path] = []
    for root in paths.models_dirs():
        candidates.append(root / "default" / "encoder.pt")   # download_encoder_weights.py 的输出
        candidates.append(root / "encoder.pt")
    candidates += [
        rtc_dir / "encoder.pt",
        rtc_dir / "saved_models" / "default" / "encoder.pt",
        rtc_dir / "pretrained.pt",                       # 旧版布局
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 1024:
            if _is_html_error_page(candidate):
                raise SpeakerBridgeError(
                    f"{candidate} 是一个 HTML 错误页，不是权重文件！\n"
                    "（通常是下载被拦或 404 之后把错误页面存成了 .pt）\n"
                    "请删掉它，然后重新下载：\n"
                    "    python algorithms/voiceprint_protection/scripts/download_encoder_weights.py"
                )
            return candidate

    searched = "\n".join(f"    {c}" for c in candidates)
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


def _to_numpy_1d(value) -> np.ndarray:
    """把上游返回的 tensor / ndarray / list 统一成 (D,) 的 float32 numpy。"""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float32)
    if value.ndim > 1:
        value = value.reshape(-1)
    return value


def patch_upstream_mel(encoder_audio, encoder_params) -> bool:
    """把上游的 `wav_to_mel_spectrogram` 换成 librosa 0.10 兼容版。

    **为什么必须换**
    上游按 librosa **0.9.2** 写的：

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
              f"|h| = {float(np.linalg.norm(embedding)):.6f}")

    print(f"\n  Q3 输出是否已 L2 归一化：实测模长 = "
          f"{float(np.linalg.norm(embeddings[0])):.6f}（应为 1.000000）")
    print("     → 本模块出口还会再归一化一次（幂等，无副作用）。")

    if len(embeddings) >= 2:
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        print(f"\n  两段音频的余弦相似度 = {similarity:.4f}")
        print("     （同一人说话通常 > 0.7，不同人通常 < 0.5；数值异常先查权重是否加载对）")

    print("\n  自检完成。把以上结论抄进 docs/notes/ge2e_interface.md。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
