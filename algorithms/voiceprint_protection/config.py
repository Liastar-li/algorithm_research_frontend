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
WEIGHTS_DIR = ROOT / "weights"              # checkpoint
RESULTS_DIR = ROOT / "results"              # 评测输出

# ---------------------------------------------------------------------------
# 随机性：固定 seed 才能复现
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# 音频 / mel 参数（两个预设，见 audio_utils.py）
#
#   paper16k   —— 论文配置：16 kHz / 80 mel / hop 256 / n_fft 1024 / 40~7600
#                 用于最终系统（Stage 1 自己训练、评测、平台演示）
#   autovc22k  —— AutoVC + HiFi-GAN 官方预训练权重的原生配置：22.05 kHz
#                 用于 L1 阶段快速跑通推理链（能直接加载现成权重）
#
# 两者除采样率外完全一致，方便切换时只改一行。
# ---------------------------------------------------------------------------
MEL_PRESETS = {
    "paper16k": {
        "sample_rate": 16000,
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mels": 80,
        "fmin": 40,
        "fmax": 7600,
        "power": 2.0,       # 与 librosa.feature.melspectrogram 默认一致（幅度平方）
        "log_base": 10.0,   # AutoVC 使用 log10；置 None 则输出线性幅度
        "log_floor": 1e-10,
    },
    "autovc22k": {
        "sample_rate": 22050,
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mels": 80,
        "fmin": 40,
        "fmax": 7600,
        "power": 2.0,
        "log_base": 10.0,
        "log_floor": 1e-10,
    },
}

DEFAULT_PRESET = "paper16k"

# ---------------------------------------------------------------------------
# 张量形状约定（与第 8 章"双人接口契约"逐条对应）
# ---------------------------------------------------------------------------
FRAME_DIM = 80          # x / x_adv : (T, 80)
CONTENT_DIM = 32        # c         : (T', 32)，T' = T // 2
SPEAKER_DIM = 256       # h / h_adv : (256,)

# ---------------------------------------------------------------------------
# Stage 1 超参（论文 §2.2 / §3.1）
#   L_S1 = L_rec + λ_cd·L_cd + λ_dis·L_dis
# ---------------------------------------------------------------------------
STAGE1 = {
    "iterations": 500_000,     # 论文值；算力不足时按第 4 章分档缩减
    "lambda_cd": 1.0,          # 论文值
    "lambda_dis": 0.01,        # 论文值
    "content_dim": 32,         # AutoVC 瓶颈维度
    "num_speakers": 80,        # 训练集说话人数 K，用于 L_dis 的分类器 Ws
    "batch_size": 8,           # 8 GB 显存下的建议值（第 5 章 2.4）
    "accum_steps": 2,          # 梯度累积
    "max_seconds": 3.0,        # 训练片段截断长度（秒）
    "amp": True,               # 混合精度
    "lr": 1e-4,
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
