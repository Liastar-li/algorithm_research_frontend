"""全局配置 —— 所有超参的唯一真源。

【铁律】本文件是两人共享的"契约文件"：
  * 任何超参改动，必须两人一致同意后同时修改这里，然后一起提交。
  * 训练脚本、推理脚本、评测脚本**一律从本文件读取**，不允许各自写死数值。
  * 第 8 章"双人接口契约"中列出的张量形状约定，在这里能找到对应来源。

对应论文：Imperceptible Voiceprint Protection via Human-Machine Perception
Discrepancy Feature Disentanglement (Interspeech 2026)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent          # 仓库根
DATA_ROOT = Path("data")                    # VCTK 处理后的清单与音频
WEIGHTS_DIR = ROOT / "weights"              # checkpoint（不提交到 git）
RESULTS_DIR = ROOT / "results"              # 推理产物（不提交到 git）

# ---------------------------------------------------------------------------
# 第三方组件清单（唯一真源）
#
# ★ 为什么不把 third_party/ 提交进仓库：
#   那里是**第三方官方仓库**，提交进来会把别人的代码和许可证混进我们的成果仓库。
#   代价是每个人/每台机器要自己拉一次 —— 所以用 scripts/setup_third_party.ps1
#   （Linux 用 .sh）一键拉取，版本由下面的 SHA 钉死，保证两人完全一致。
#
#   B 需要的"信息"（mel 约定、张量形状）都在本仓库里：
#     · algorithms/voiceprint_protection/config.py      ← 本文件
#     · algorithms/voiceprint_protection/audio_utils.py ← mel 实现
#     · algorithms/voiceprint_protection/docs/notes/autovc_alignment.md ← 实测记录
# ---------------------------------------------------------------------------
THIRD_PARTY_COMPONENTS = [
    {
        "name": "AutoVC",
        "url": "https://github.com/auspicious3000/autovc.git",
        "sha": "c1827a60e83e6d26030b8953843b2f7cf2bd36db",
        "license": "MIT",
        "usage": "Stage 1 模型（Generator）+ 说话人嵌入 D_VECTOR + 官方权重 + 声码器封装",
    },
    {
        # ★ 必须是 d4333546：最后一个还含 builder.py 的提交。
        #   AutoVC 的 synthesis.py 依赖 `from wavenet_vocoder import builder`，
        #   而 2019-08-26 之后的新版删掉了它且模型结构有变 -> 权重加载失败。
        "name": "wavenet_vocoder",
        "url": "https://github.com/r9y9/wavenet_vocoder.git",
        "sha": "d4333546eccd3b681b7092e4f52160ad7e62dcc9",
        "license": "MIT",
        "usage": "WaveNet 声码器（与 AutoVC 的 mel 格式配套）",
    },
    {
        "name": "Real-Time-Voice-Cloning",
        "url": "https://github.com/CorentinJ/Real-Time-Voice-Cloning.git",
        "sha": "",     # 用默认分支
        "license": "MIT",
        "usage": "GE2E 说话人编码器（论文脚注引用），最终系统使用",
    },
]

# 期望的权重文件（AutoVC 官方 README 提供的 IBM Box 链接）
WEIGHT_FILES = {
    "autovc": "autovc.ckpt",                        # 主模型：Generator + 说话人嵌入
    "speaker_bl": "3000000-BL.ckpt",                # D_VECTOR 说话人编码器（80 维输入）
    "vocoder": "checkpoint_step001000000_ema.pth",  # WaveNet 声码器
}

# ---------------------------------------------------------------------------
# 随机性：固定 seed 才能复现
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# 音频 / mel 参数
#
# ★ 2026-11 已按 AutoVC 官方仓库（third_party/AutoVC, commit c1827a6）实测对齐。
#   AutoVC 官方 make_spect.py 的预处理链是自定义的，不是 librosa.feature.melspectrogram：
#     30Hz 高通 -> ×0.96 + 1e-6 噪声 -> STFT(reflect 填充, 幅度)
#     -> librosa.filters.mel(16000, 1024, fmin=90, fmax=7600, 80)
#     -> 20*log10(max(1e-5, mel)) - 16 -> clip((x+100)/100, 0, 1)
#   实现见 audio_utils.autovc_mel()。
#
#   ⚠️ AutoVC 内部不一致：hparams.py 写 fmin=125，而真正生成训练数据的
#      make_spect.py 用 fmin=90。以 make_spect.py 为准（它才是生效的那个）。
#
#   关键结论：**AutoVC 官方权重本身就是 16 kHz 的**，与论文一致，
#   因此全程用 16000 Hz，不需要 22.05 kHz 预设。
#
#   采样率 22050 的 HiFi-GAN 官方权重只属于"声码器侧"约定，
#   见 audio_utils.vocoder_mel()。
# ---------------------------------------------------------------------------
MEL_PRESETS = {
    # 模型侧（AutoVC 链）：喂给 AutoVC / 我们自己的 Stage 1
    "model16k": {
        "sample_rate": 16000,
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mels": 80,
        "fmin": 90,          # ★ 取自 make_spect.py（不是 hparams.py 的 125）
        "fmax": 7600,
        "chain": "autovc",   # 30Hz 高通 + reflect 幅度 STFT + dB + clip[0,1]
    },
    # ★ 第二套模型侧 mel：**上游 WaveRNN 声码器**那套约定。
    #   来源：third_party/Real-Time-Voice-Cloning/synthesizer/audio.py::melspectrogram
    #     preemphasis(0.97) -> librosa.stft(n_fft=800, hop=200, win=800, center=True)
    #     -> 幅度 -> librosa.filters.mel(16000, 800, n_mels=80, fmin=55, fmax=7600)
    #     -> 20*log10(max(1e-5,.)) - 20
    #     -> 对称归一化 clip(8*(S+100)/100 - 4, -4, 4)
    #     -> 再 /4 -> **[-1, 1]**（WaveRNN 训练时的量纲，见 vocoder_dataset.py:28）
    #   ★ 存在的意义：WaveRNN 只吃自己那套 mel 约定。
    #     喂 model16k（AutoVC 链）的 mel 进去，出来是噪音（且不报错）。
    #   ★ 取值范围是 [-1, 1]，与 model16k 的 [0, 1] 不同 —— 别混。
    "rtc16k": {
        "sample_rate": 16000,
        "n_fft": 800,
        "hop_length": 200,
        "win_length": 800,
        "n_mels": 80,
        "fmin": 55,
        "fmax": 7600,             # ★ synthesizer 的 hparams 写的是 7600（不是 8000）
        "preemphasis": 0.97,
        "ref_level_db": 20,
        "min_level_db": -100,
        "max_abs_value": 4.0,     # 对称归一化的半幅；WaveRNN 训练时除以它 -> [-1,1]
        "chain": "rtc",           # preemph + librosa STFT + 幅度 + dB + 对称归一化
    },
    # ★ 第三套模型侧 mel：**HiFi-GAN 声码器**那套约定（★ 当前默认）
    #   来源：HuggingFace `speechbrain/tts-hifigan-libritts-16kHz` 的 README
    #   给出的确切参数 + speechbrain `lobes/models/FastSpeech2.py::mel_spectrogram`
    #   的实现：
    #     torchaudio Spectrogram(hop=256, win=1024, n_fft=1024, power=1,
    #                             normalized=False, center=True, pad_mode='reflect')
    #     -> MelScale(sr=16000, n_mels=80, f_min=0, f_max=8000,
    #                 norm='slaney', mel_scale='slaney')
    #     -> log(clamp(mel, min=1e-5))            ← **自然对数**，不是 log10
    #
    #   ★ **实测验证过**（`scripts/verify_hifigan.py`）：
    #     重建音与原音的说话人嵌入余弦 = **0.9746**（约定错的话 ≈ 0）
    #     原音 RMS 0.0539 -> 重建 RMS 0.0585，长度只差 1 帧
    #   （NVIDIA 那条 log10 链喂进去是 2512% 的相对误差，完全是噪音。）
    "hifigan16k": {
        "sample_rate": 16000,
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mels": 80,
        "fmin": 0,
        "fmax": 8000,
        "power": 1.0,            # 幅度（不是功率）
        "normalized": False,     # 不做逐句能量归一化
        "clip_val": 1e-5,        # log 之前的下限
        "norm": "slaney",        # librosa 兼容的 mel 滤波器组
        "mel_scale": "slaney",
        "chain": "hifigan",     # STFT 幅度 -> slaney mel -> log(clamp(1e-5))
        # ★ 归一化：把上面的原始 mel 做一次**固定仿射变换**，让网络侧的 mel
        #   与 AutoVC 的 mel **同量纲**（都是 ~[0,1] 宽）。
        #   a = scale * s + offset
        #   （s = ln(m)；AutoVC 那边是 a = 0.2*log10(m) + 0.84 = 0.08686*s + 0.84）
        #   为什么要这一步：论文的 λ_cd=1.0、λ_dis=0.01 是在 AutoVC 量纲下
        #   网格搜索出来的。若直接用原始的 s（值域约 [-11.5, 0]），
        #   L_rec 会比论文大 ~100 倍，λ_cd 的相对权重会被稀释 40 倍
        #   ⇒ 论文的超参就失效了。
        #   ★ 仿射**可逆**，所以送进声码器前反变换回去即可，**零损失**。
        "norm_scale": 0.0868577,   # = 0.2 / ln(10)
        "norm_offset": 0.84,
    },
}

DEFAULT_PRESET = "model16k"

# 声码器侧（HiFi-GAN）单独一组，不与模型侧混用
VOCODER_MEL = {
    "sample_rate": 22050,   # HiFi-GAN 官方权重
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mels": 80,
    "mel_basis_sr": 8000,   # ★ HiFi-GAN 的怪癖：mel 基用 sr=8000
    "log_floor": 1e-5,
    "offset": 4.5,
    "scale": 5.0,
}

# ---------------------------------------------------------------------------
# 张量形状约定（与第 8 章"双人接口契约"逐条对应）
#
# ⚠️ content code 的实际维度：AutoVC 的 Encoder 用 dim_neck=32 的**双向** LSTM，
#    前向 32 + 后向 32 拼接 -> **64 维**。论文写的 "c ∈ R^(T'×32)" 指的是
#    LSTM 每个方向的维度（dim_neck=32），不是拼接后的宽度。
#    论文的"瓶颈维度 32"说的也是 dim_neck。
#    下采样因子 freq=32 -> T' = T // 32。
# ---------------------------------------------------------------------------
FRAME_DIM = 80          # x / x_adv : (T, 80)
CONTENT_DIM = 32        # dim_neck（论文"瓶颈维度 32"）
CONTENT_ACTUAL_DIM = 64 # 拼接后实际维度 = dim_neck * 2
CONTENT_DOWNSAMPLE = 32  # freq，T' = T // 32
SPEAKER_DIM = 256       # h / h_adv : (256,)  已 L2 归一化

# AutoVC 官方模型的构造参数（见 third_party/AutoVC/conversion.ipynb）
AUTOVC_MODEL = {
    "dim_neck": 32,       # LSTM 每方向维度
    "dim_emb": 256,       # 说话人嵌入维度
    "dim_pre": 512,       # Decoder 中间维度
    "freq": 32,           # 时间下采样因子
    "pad_base": 32,       # 推理时 mel 需 pad 到 32 的倍数
}

# 说话人编码器的两套权重（★ 不可混用，输入维度不同）
SPEAKER_ENCODERS = {
    # AutoVC 官方链：3000000-BL.ckpt，模型是 model_bl.D_VECTOR
    #   3 层单向 LSTM(dim_input=80) + Linear(768->256) + L2 归一化
    #   输入是 80 维 mel（AutoVC 的 spmel）
    "autovc_bl": {
        "source": "AutoVC 官方 3000000-BL.ckpt",
        "dim_input": 80,
        "dim_cell": 768,
        "dim_emb": 256,
        "num_layers": 3,
        "bidirectional": False,
    },
    # 论文脚注明确引用的 GE2E（Real-Time-Voice-Cloning）
    #   输入是 40 维 mel（16kHz / hop160 / win400）
    "ge2e_rtc": {
        "source": "CorentinJ/Real-Time-Voice-Cloning pretrained.pt",
        "dim_input": 40,
        "dim_cell": 256,
        "dim_emb": 256,
        "num_layers": 3,
        "bidirectional": False,
    },
}

# ---------------------------------------------------------------------------
# Stage 1 超参（论文 §2.2 / §3.1）
#   L_S1 = L_rec + λ_cd·L_cd + λ_dis·L_dis
# ---------------------------------------------------------------------------
STAGE1 = {
    "iterations": 500_000,     # 论文值；算力不足时按第 4 章分档缩减
    "lambda_cd": 1.0,          # 论文值
    "lambda_dis": 0.01,        # 论文值
    "content_dim": 32,         # AutoVC 瓶颈维度 dim_neck（双向 LSTM -> content code 实际 64 维）
    "content_window": 4,       # ★ L_dis 里沿时间维拼接的 content code 个数
                                #   4 x 64 = 256 维，正好与论文的 W_s ∈ R^{K×256} 对上
                                #   （见 stage1/losses.py 文件头的说明）
    "num_speakers": 80,        # 训练集说话人数 K，用于 L_dis 的分类器 Ws
    # ★ batch 16 / accum 1（2026-11-18 实测后调整）
    #   原先按"8 GB 显存要省着用"定的 8 + 累积 2，实测**过于保守**：
    #   batch 16 只用 **1.01 GB**、batch 32 只用 **1.34 GB**（共 8 GB）。
    #   而且 batch 8 时 GPU 严重欠利用（吞吐 69 条/秒），
    #   改成 batch 16 吞吐直接翻倍到 149 条/秒，而每步耗时几乎不变。
    #   有效 batch 仍是 16（8×累积2），只是不再用"小 batch + 累积"去凑 ——
    #   同样的有效 batch，累积要多跑一次前向，反而更慢。
    "batch_size": 16,
    "accum_steps": 1,
    "max_seconds": 3.0,        # 训练片段截断长度（秒）
    "amp": True,               # 混合精度（fp16）：省显存 + 快 1.5~2 倍
    "lr": 1e-4,
    # ★ 训练步数少于这个值的 checkpoint **不许当正式权重用**
    #   （冒烟测试 `--iterations 3 --smoke` 也会写 `stage1_model.pt`，
    #     指标全是垃圾，但会被 Stage 2 静默捡去用，**而且不报错**）。
    #   见 `stage2/models.py::TorchAutoVCStage1._guess_checkpoint()`。
    "min_steps_for_real_use": 1000,
    # ★ 模型侧用哪套 mel 约定（Ec/D 的输入）。
    #   "hifigan16k" = **HiFi-GAN 声码器那套**（n_fft 1024 / hop 256 / fmin 0 /
    #                fmax 8000 / log(clamp(1e-5))）—— ★ 当前默认
    #   "model16k"   = AutoVC 官方链（n_fft 1024 / hop 256 / fmin 90 / 取值 [0,1]）
    #   "rtc16k"     = 上游 WaveRNN 声码器那套（preemph / 对称归一化 / 取值 [-1,1]）
    #
    #   【为什么默认是 hifigan16k —— 2026-11-18/19 定】
    #   论文 §2.3 原文：
    #       "The protected mel-spectrogram is reconstructed as x_adv = D(c, h_adv),
    #        and converted to waveform using **HiFi-GAN vocoder V(·)**."
    #   ⇒ **Stage 1 的 mel 必须就是 HiFi-GAN 的约定**，否则论文自己也要过一个
    #     "近似换算"，而那个换算在滤波器组层面不严格成立。
    #
    #   ★ 而且 HiFi-GAN 与 AutoVC 链**结构完全一样**
    #     （n_fft 1024 / hop 256 / 80 mels / 16 kHz），只差滤波器组与归一化
    #     ⇒ **AutoVC 的网络架构一字不改**（论文 §3.1 要求的是架构）。
    #
    #   ★ HiFi-GAN 是纯卷积网络、**可求导** —— Stage 2 的梯度必须穿过 V(·)
    #     （G → h_adv → D(c,h_adv) → V(·) → E_spk(·) → L_de）。
    #     上游 WaveRNN 是自回归生成器（采样 9 bit 离散码），**不可求导** ❌。
    #
    #   ★ 这是**两人共同区**（config.py），改动需双方同意，见 docs/interface-contract.md。
    "mel_preset": "hifigan16k",
}

# ---------------------------------------------------------------------------
# Stage 2 超参（论文 §2.3 / §3.1）
#   L_G = L_GAN + λ_de·L_de + λ_per·L_per
#   L_per = L_mel + λ_mask·L_mask
# ---------------------------------------------------------------------------
STAGE2 = {
    "iterations": 200_000,     # 论文值
    "alpha": 0.10,             # 扰动强度：h_adv = h + α·δ   ← 平台暴露的参数
    "lambda_per": 15.0,        # 论文值
    "lambda_de": 5.0,          # 论文值
    "lambda_mask": 0.1,        # 论文值
    "lambda_gan": 1.0,         # LSGAN 的 λ_GAN，论文未单列
    "batch_size": 8,
    "accum_steps": 1,
    "max_seconds": 3.0,
    "amp": True,
    "lr": 1e-4,
    "clip_grad": 5.0,
}

# 论文的消融默认值（Table 2 用），评测脚本读这里
ABLATION = {
    "full": {},
    "without_defense_loss": {"lambda_de": 0.0},   # w/o L_de  -> DSR 12.5
    "without_mel_loss": {"lambda_mask": 0.0, "use_mel_loss": False},  # w/o L_mel
    "without_mask_loss": {"lambda_mask": 0.0},    # w/o L_mask -> DSR 84.3
}

# ---------------------------------------------------------------------------
# 评测设置（论文 §3.1）
# ---------------------------------------------------------------------------
EVAL = {
    "tau": 0.5,                # DSR 的阈值；论文同时报 0.3 / 0.4 / 0.5
    "taus": [0.3, 0.4, 0.5],
    "test_speakers": 20,
    "utterances_per_speaker": 100,   # 课程作业可缩到 10，并写明
    "whisper_model": "large-v3",    # 论文用 Whisper-large 算 WER
    "mos_listeners": 10,            # 论文是 50 人；如实写明实际人数
}

# ---------------------------------------------------------------------------
# 服务
# ---------------------------------------------------------------------------
SERVICE = {
    "host": "127.0.0.1",
    "port": 8000,
    "endpoint": "http://127.0.0.1:8000/process",
    "max_upload_mb": 50,        # 需与 public/methods/*.json 里的 max_mb 一致
    "max_seconds": 60,         # 平台同步等待，超过会超时
}
