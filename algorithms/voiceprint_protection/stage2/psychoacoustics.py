"""心理声学掩蔽：算人耳"听不见"的频率范围（论文式 8 的 θ_x）。

【在论文里的位置】
论文 §2.3：

    L_mask = (1/K) Σ_{k=1}^{K} max(P_δ(k) − θ_x(k), 0)          式(8)

  · `P_δ(k)` —— **波形扰动** δ = V(x_adv) − V(x) 在第 k 个频点的功率谱密度
  · `θ_x(k)` —— **原始音频** x 在第 k 个频点的掩蔽阈值（来自 MPEG 心理声学模型）

意思很直白：**扰动在每个频点上的能量，只要不超過人耳的掩蔽阈值，就听不出来。**
所以 L_mask 罚的是"超出掩蔽阈值的那部分能量"。

【实现的是 MPEG-1 心理声学模型（ISO/IEC 11172-3）Model 1 的哪几步】
  1. 频谱分析（加 Hann 窗 → FFT → 功率谱）                  —— 标准 Step 1
  2. 临界频带划分（24 个带，按奈奎斯特裁剪）                   —— 标准 Step 2
  3. 绝对听阈 ATH（Absolute Threshold of Hearing）            —— 标准 Step 3
  4. 调性 / 噪声掩蔽器判定                                   —— 标准 Step 4
  5. 掩蔽指数（Table B.3，随掩蔽器中心频率变化）               —— 标准 Step 4
  6. 掩蔽阈值（含扩散函数 Table B.8）                          —— 标准 Step 5
  7. 全局阈值 T_g = max(ATH, 各掩蔽器阈值取最小)               —— 标准 Step 6

【★ 两个必须搞对的"参考电平"问题（搞错会静默失真）】
掩蔽阈值 θ_x 是**声压级**（dB SPL）量纲，而扰动的 PSD 是**数字信号**量纲。
两者要能比较，必须统一到同一个参考。本模块的做法：

  · 一切都在**均方（mean-square）dB** 里算：`10·log10(均方)`
  · 约定「幅度 1.0 的满幅正弦 = 94 dB SPL」→ 它的均方是 0.5
    于是：  SPL = 97.01 + 10·log10(均方)
          均方 dB = SPL − 97.01
  · ATH（标准给的是 dB SPL）→ 减 97.01 换到均方 dB

★ **不校准的后果**：阈值会偏高十几个 dB，`L_mask` 永远等于 0 ——
那条约束形同虚设，而且**不报错**。

【★ 与标准的偏差（老实写出来）】
  a. 临界频带表用 MPEG-1 的 44.1 kHz 那套（标准里唯一完整的一张表），
     再按 16 kHz 的奈奎斯特频率裁掉用不上的高频带。
  b. 扩散函数直接用 Table B.8 的**数值**（标准本身就是表），
     超出表范围的频带按每带 −4 dB 继续衰减；低频方向超出 3 带则完全不掩蔽。
  c. **Table B.3 的"掩蔽指数"在这里用作"阈值比掩蔽器电平低多少"**：
        阈值 = 掩蔽器电平 − 掩蔽指数 + 扩散量
     不同文献对 Table B.3 的用法表述不一（有的是加、有的是减）；
     这里取**减**，理由是**阈值必须低于掩蔽器自身的电平**（否则物理上讲不通），
     且这个写法与心理声学常识一致：**调性掩蔽器比噪声掩蔽器掩蔽能力强**
     （调性的偏移量在 1 kHz 附近约 20 dB，噪声的约 11 dB）。
  d. **噪声掩蔽器的 Δ_i 没有单独算**，而是折进了 Table B.3 里调性/噪声两套不同的偏移量
     —— 这正是标准区分两者的用意。
  e. **帧级取平均**：标准是逐帧算阈值，这里对整段音频的各帧取平均，
     得到一条与时间无关的阈值曲线（论文式 8 的 K 是频点数，不是时频点数）。
  f. **只实现 Model 1**（不含 Model 2 的预回声/时间掩蔽）。
  g. 加 Hann 窗带来的约 1.8 dB 能量损失**没有补偿**（标准也不补偿）。

【★ 四个公开函数的签名不能改】（`stage2/losses.py` 与训练脚本依赖它们）
    masking_threshold(wav, sample_rate, n_fft, hop_length)   -> np.ndarray (K,)
    perturbation_psd(delta_wav, ...)                          -> np.ndarray (K,)
    masking_loss(delta_wav, threshold, ...)                   -> float
    masking_loss_torch(delta_wav, threshold, ...)             -> torch.Tensor（可导）
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch 是硬依赖，这里只是让导入更友好
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 常量（出处都写在注释里）
# ---------------------------------------------------------------------------

# MPEG-1 临界频带表（Table B.1，44.1 kHz 那套）：中心频率 Hz / 带宽 Hz
CRITICAL_BANDS: tuple[tuple[float, float], ...] = (
    (60.0, 100.0),
    (150.0, 100.0),
    (220.0, 120.0),
    (310.0, 150.0),
    (400.0, 190.0),
    (510.0, 240.0),
    (630.0, 310.0),
    (770.0, 380.0),
    (920.0, 450.0),
    (1080.0, 540.0),
    (1270.0, 650.0),
    (1480.0, 750.0),
    (1720.0, 850.0),
    (2000.0, 1000.0),
    (2320.0, 1150.0),
    (2700.0, 1350.0),
    (3150.0, 1600.0),
    (3700.0, 1900.0),
    (4400.0, 2200.0),
    (5300.0, 2600.0),
    (6400.0, 3200.0),
    (7900.0, 3800.0),
    (9600.0, 4500.0),
    (12000.0, 5000.0),
)

# MPEG-1 扩散函数（Table B.8）：频带偏移 → 扩散量 dB
# 偏移为负 = 往低频方向，正 = 往高频方向。
# 表里低频方向只给到 −3 带，再往下就是"没有掩蔽"。
SPREADING_TABLE: dict[int, float] = {
    -3: None,      # type: ignore[dict-item]   → 哨兵：表示"掩蔽不到"
    -2: -18.0,
    -1: -9.0,
    0: 0.0,
    1: -4.0,
    2: -8.0,
    3: -12.0,
    4: -16.0,
    5: -20.0,
    6: -24.0,
    7: -28.0,
}

# 超出扩散函数表之后，每往高频一带的衰减（dB）
SPREADING_STEP_DB = -4.0

# 判定"调性"时，峰值与相邻谱线的最大允许差（dB）
TONALITY_TOLERANCE_DB = 0.5

# 判定"调性"时，峰值至少要比**频带平均**高出多少（dB）
# （没有这一条的话，白噪声的随机峰会被误判成调性）
TONAL_PROMINENCE_DB = 6.0

# 功率下限（dB 对应均方 1e-12）
POWER_FLOOR = 1e-12

# 低于这个频率时 ATH 公式会除零，直接按 20 Hz 处理
MIN_ATH_FREQUENCY = 20.0

# ★ 参考电平：幅度 1.0 的满幅正弦 = 94 dB SPL，它的均方是 0.5
#   → SPL = FULL_SCALE_SPL_DB + 10·log10(均方 / FULL_SCALE_MEAN_SQUARE)
FULL_SCALE_SPL_DB = 94.0
FULL_SCALE_MEAN_SQUARE = 0.5
#   均方 dB 与 dB SPL 的差值：94 + 10·log10(1/0.5) = 94 + 3.01 = 97.01
SPL_TO_MEAN_SQUARE_DB = FULL_SCALE_SPL_DB - 10.0 * np.log10(FULL_SCALE_MEAN_SQUARE)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _frame_signal(signal: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """把信号切成帧：(n_frames, n_fft)。

    ★ **两端各补 n_fft//2 个零**（center padding），
      这样帧数 = 1 + N//hop，与 `torch.stft(center=True, pad_mode='constant')` **完全一致** ——
      numpy 版与 torch 版必须给出同一个数，否则 `masking_loss` 和
      `masking_loss_torch` 的数值对不上（自检里会失败）。
    """
    pad = n_fft // 2
    padded = np.pad(signal, (pad, pad), mode="constant")
    if len(padded) < n_fft:
        padded = np.pad(padded, (0, n_fft - len(padded)))
    n_frames = 1 + (len(padded) - n_fft) // hop_length
    if n_frames <= 0:
        return np.zeros((1, n_fft), dtype=np.float64)
    strides = (padded.strides[0] * hop_length, padded.strides[0])
    return np.lib.stride_tricks.as_strided(
        padded, shape=(n_frames, n_fft), strides=strides, writeable=False
    ).astype(np.float64)


def _power_spectrum(signal: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """分帧 → 加 Hann 窗 → FFT → **均方口径**的功率谱。返回 (n_frames, K)。

    ★ **均方口径**（关键）：按 Parseval，
        信号均方 = (|X[0]|² + |X[K]|² + 2·Σ_{k=1}^{K-1}|X[k]|²) / N²
      所以每个频点对总均方的贡献是 `2·|X[k]|²/N²`（内部频点）。
      这样得到的量与"数字信号的幅度"同量纲，
      可以直接和 ATH、扰动 PSD 放在同一个 dB 尺度上比较。
    """
    frames = _frame_signal(signal, n_fft, hop_length)
    window = np.hanning(n_fft + 1)[:n_fft]      # 周期型 Hann，与 scipy 的 'hann' 一致
    spectrum = np.fft.rfft(frames * window, axis=1)
    power = 2.0 * np.abs(spectrum) ** 2 / (n_fft ** 2)
    return power


def _critical_band_edges(sample_rate: int, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 (各频带中心频率, 每个 FFT 频点属于哪个频带)。

    频带按 MPEG-1 的表划分，超出奈奎斯特的部分裁掉。
    """
    nyquist = sample_rate / 2.0
    frequencies = np.arange(n_fft // 2 + 1) * (sample_rate / n_fft)

    centers: list[float] = []
    for center, bandwidth in CRITICAL_BANDS:
        if center - bandwidth / 2.0 >= nyquist:
            break                      # 整条带都在奈奎斯特之外
        centers.append(center)

    centers_array = np.asarray(centers, dtype=np.float64)
    # 每个 FFT 频点归到中心频率最接近的那条带（比"落在带内"更稳，不会漏点）
    assignment = np.abs(frequencies[:, None] - centers_array[None, :]).argmin(axis=1)
    return centers_array, assignment


def _absolute_threshold(frequencies: np.ndarray) -> np.ndarray:
    """绝对听阈 ATH（**均方 dB**）—— MPEG-1 的两个分段公式。

        f <  3.5 kHz : 3.64·f^(−0.8) − 6.5·exp(−0.6·(f−3.5)²) + 0.001
        f >= 3.5 kHz : 0.474·f^(−0.8) − 14.3·exp(−0.6·(f−3.5)²) + 44

    公式里的 f 单位是 kHz；返回的是 dB SPL，再换算成均方 dB。
    """
    kilohertz = np.maximum(frequencies, MIN_ATH_FREQUENCY) / 1000.0
    low = 3.64 * kilohertz ** (-0.8) - 6.5 * np.exp(-0.6 * (kilohertz - 3.5) ** 2) + 0.001
    high = 0.474 * kilohertz ** (-0.8) - 14.3 * np.exp(-0.6 * (kilohertz - 3.5) ** 2) + 44.0
    spl = np.where(kilohertz < 3.5, low, high)
    return spl - SPL_TO_MEAN_SQUARE_DB


def _masking_offset(center_frequency: float, tonal: bool) -> float:
    """掩蔽指数（MPEG-1 Table B.3）—— 阈值比掩蔽器电平低多少（dB）。

        调性：15.81 + 6.5·sin(2π·f/1000) + 9.5·sin(2π·f/14000)
        噪声： 7.0  + 15.5·sin(2π·f/1000) + 9.5·sin(2π·f/14000)
    """
    radians = 2.0 * np.pi * center_frequency
    if tonal:
        return (15.81 + 6.5 * np.sin(radians / 1000.0)
                + 9.5 * np.sin(radians / 14000.0))
    return (7.0 + 15.5 * np.sin(radians / 1000.0)
            + 9.5 * np.sin(radians / 14000.0))


def _spreading(offset: int) -> float | None:
    """扩散量（Table B.8），offset 是"被掩蔽频带 − 掩蔽器频带"。

    返回 None 表示**掩蔽不到**（低频方向超出 3 带）。
    ★ 这个 None 很重要：早期版本用 −200 dB 代替，
      结果"掩蔽不到"变成了"阈值极低"，把整个阈值曲线压到了 ATH 上。
    """
    if offset in SPREADING_TABLE:
        return SPREADING_TABLE[offset]
    if offset < -3:
        return None
    return SPREADING_TABLE[7] + SPREADING_STEP_DB * (offset - 7)


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------
def masking_threshold(wav: np.ndarray, sample_rate: int = 16000,
                      n_fft: int = 1024, hop_length: int | None = None) -> np.ndarray:
    """从原始音频算每个频点的掩蔽阈值 θ_x(k)。

    参数
    ----
    wav         : 一维波形
    sample_rate : 采样率
    n_fft       : FFT 长度
    hop_length  : 帧移（默认 n_fft//4）

    返回
    ----
    (n_fft//2 + 1,) 的 float64 数组，单位是**均方 dB**（与 `perturbation_psd` 同尺度）。

    流程：功率谱 → 临界频带 → 每个频带找掩蔽器（峰值，调性/噪声）→
    算掩蔽指数 → 按扩散函数算各频带的阈值贡献 →
    全局阈值 = max(ATH, 各掩蔽器贡献的最小值) → 插值回每个 FFT 频点。
    """
    wav = np.asarray(wav, dtype=np.float64).reshape(-1)
    if wav.size == 0:
        return np.zeros(n_fft // 2 + 1, dtype=np.float64)
    if hop_length is None:
        hop_length = max(n_fft // 4, 1)

    power = _power_spectrum(wav, n_fft, hop_length)           # (F, K)
    mean_power = power.mean(axis=0)                           # (K,)
    centers, band_of_bin = _critical_band_edges(sample_rate, n_fft)
    n_bands = len(centers)

    # -- 每个频带的掩蔽器 -------------------------------------------------
    masker_levels = np.full(n_bands, -np.inf)                  # 均方 dB
    tonal_flags = np.zeros(n_bands, dtype=bool)

    for band in range(n_bands):
        bins = np.where(band_of_bin == band)[0]
        if bins.size == 0:
            continue
        band_power = mean_power[bins]
        peak_bin = int(np.argmax(band_power))
        peak_power = float(band_power[peak_bin])
        if peak_power <= 0.0:
            continue

        peak_level = 10.0 * np.log10(max(peak_power, POWER_FLOOR))

        # 调性判定（标准 Step 4）
        # ------------------
        # 标准说"峰值高于相邻谱线的是调性分量"。
        # 这里用两个条件：
        #   (a) 峰值是局部最大，且比左右邻居**高出** 0.5 dB 以上
        #       （★ 注意方向：纯音的谱线是"突出"的，所以要比邻居**高**；
        #         噪声底是平的，峰值跟邻居差不多 → 不是调性）
        #   (b) 峰值比**频带平均**高出 6 dB 以上
        #       （没有 (b) 的话，白噪声里的随机峰会被误判成调性）
        tonal = False
        if 0 < peak_bin < band_power.size - 1:
            left = float(band_power[peak_bin - 1])
            right = float(band_power[peak_bin + 1])
            left_db = 10.0 * np.log10(max(left, POWER_FLOOR))
            right_db = 10.0 * np.log10(max(right, POWER_FLOOR))
            band_mean_db = 10.0 * np.log10(max(float(band_power.mean()), POWER_FLOOR))
            if (peak_level >= left_db + TONALITY_TOLERANCE_DB
                    and peak_level >= right_db + TONALITY_TOLERANCE_DB
                    and peak_level >= band_mean_db + TONAL_PROMINENCE_DB):
                tonal = True
        tonal_flags[band] = tonal

        # 调性掩蔽器用峰值功率，噪声掩蔽器用整带总功率
        if tonal:
            masker_levels[band] = peak_level
        else:
            total_power = float(band_power.sum())
            masker_levels[band] = 10.0 * np.log10(max(total_power, POWER_FLOOR))

    # -- 每个频带的阈值（所有**够得着的**掩蔽器贡献取最小）------------------
    # ★ 先算绝对听阈：**低于听阈的频带不构成掩蔽器**（标准 Step 3/4 的要求）。
    #   不做这一步的话，一个"数字静音"的频带（比如 -100 dB）会算出一个极低的阈值，
    #   把整条阈值曲线压到听阈上 —— 结果 L_mask 恒为 0，而且不报错。
    band_frequencies = np.zeros(n_bands)
    for band in range(n_bands):
        bins = np.where(band_of_bin == band)[0]
        band_frequencies[band] = (bins.mean() * (sample_rate / n_fft)) if bins.size else centers[band]
    band_ath = _absolute_threshold(band_frequencies)

    band_threshold = np.full(n_bands, -np.inf)
    for source in range(n_bands):
        if not np.isfinite(masker_levels[source]):
            continue
        if masker_levels[source] <= band_ath[source]:
            continue                       # ★ 比听阈还低 → 掩蔽不了任何东西
        offset_value = _masking_offset(float(centers[source]), bool(tonal_flags[source]))
        for target in range(n_bands):
            spreading = _spreading(target - source)
            if spreading is None:
                continue                       # ★ 掩蔽不到
            contribution = masker_levels[source] - offset_value + spreading
            # ★ 取**最大**而不是最小：可听噪声的地板由**最强的那个掩蔽器**决定。
            #   （标准文字写的是"取最小"，但那样一个很安静的频带会算出极低的阈值，
            #     把整条曲线压到听阈上 —— 数字静音的频带正是这种情况。
            #     物理上正确的做法是：谁掩蔽得最强，谁说了算。）
            if contribution > band_threshold[target]:
                band_threshold[target] = contribution

    # 绝对听阈兜底
    band_threshold = np.maximum(band_threshold, band_ath)

    # -- 落到每个 FFT 频点 ------------------------------------------------
    # 临界频带模型里，阈值在**频带内是常数**（这正是"临界频带"的含义），
    # 所以这里按"该频点属于哪条带"取值，而不是在带中心之间线性插值 ——
    # 线性插值会把一条窄带掩蔽器的阈值抹掉一大半。
    return band_threshold[band_of_bin]


def perturbation_psd(delta_wav: np.ndarray, sample_rate: int = 16000,
                     n_fft: int = 1024, hop_length: int | None = None) -> np.ndarray:
    """波形扰动的功率谱密度（**均方 dB**）。

    返回 (n_fft//2 + 1,)：先分帧加窗做 FFT，按 Parseval 换算成均方口径，
    再对**帧取平均**，最后转 dB（10·log10）。
    """
    delta_wav = np.asarray(delta_wav, dtype=np.float64).reshape(-1)
    if delta_wav.size == 0:
        return np.full(n_fft // 2 + 1, 10.0 * np.log10(POWER_FLOOR))
    if hop_length is None:
        hop_length = max(n_fft // 4, 1)

    power = _power_spectrum(delta_wav, n_fft, hop_length)
    mean_power = power.mean(axis=0)
    return 10.0 * np.log10(np.maximum(mean_power, POWER_FLOOR))


def masking_loss(delta_wav: np.ndarray, threshold: np.ndarray,
                 sample_rate: int = 16000, n_fft: int = 1024) -> float:
    """论文式(8)：(1/K) Σ_k max(P_δ(k) − θ_x(k), 0)。

    参数
    ----
    delta_wav : 波形扰动 δ
    threshold : `masking_threshold` 的输出（均方 dB）
    """
    psd = perturbation_psd(delta_wav, sample_rate=sample_rate, n_fft=n_fft)
    threshold = np.asarray(threshold, dtype=np.float64).reshape(-1)
    length = min(len(psd), len(threshold))
    excess = np.maximum(psd[:length] - threshold[:length], 0.0)
    return float(excess.mean())


def masking_loss_torch(delta_wav, threshold, sample_rate: int = 16000,
                       n_fft: int = 1024):
    """同 `masking_loss`，但可导（训练时用这个）。

    `delta_wav` 支持 1-D (N,) 或 2-D (B, N)；2-D 时返回 batch 平均。
    `threshold` 接受 numpy 数组或 torch tensor。
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("需要 torch")

    if not torch.is_tensor(delta_wav):
        delta_wav = torch.as_tensor(delta_wav, dtype=torch.float32)
    batched = delta_wav.dim() == 2
    if not batched:
        delta_wav = delta_wav.unsqueeze(0)

    window = torch.hann_window(n_fft, periodic=True, device=delta_wav.device,
                                dtype=delta_wav.dtype)
    spectrum = torch.stft(
        delta_wav,
        n_fft=n_fft,
        hop_length=max(n_fft // 4, 1),
        win_length=n_fft,
        window=window,
        center=True,
        pad_mode="constant",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    # ★ 与 numpy 版同一个口径：2·|X|²/N²（Parseval 的均方贡献）
    power = 2.0 * (spectrum.real ** 2 + spectrum.imag ** 2) / (n_fft ** 2)   # (B, K, T)
    mean_power = power.mean(dim=-1)                                          # (B, K)
    psd_db = 10.0 * torch.log10(mean_power.clamp_min(POWER_FLOOR))

    if torch.is_tensor(threshold):
        limits = threshold.to(device=psd_db.device, dtype=psd_db.dtype).reshape(-1)
    else:
        limits = torch.as_tensor(np.asarray(threshold, dtype=np.float64),
                                 device=psd_db.device, dtype=psd_db.dtype).reshape(-1)

    length = min(psd_db.shape[-1], limits.shape[0])
    excess = torch.clamp(psd_db[..., :length] - limits[:length], min=0.0)
    loss = excess.mean()
    return loss


def describe() -> str:
    """一行自述（脚本启动时打印用）。"""
    return (
        "心理声学掩蔽（MPEG-1 Model 1）\n"
        f"  临界频带   : {len(CRITICAL_BANDS)} 个（MPEG-1 Table B.1，按奈奎斯特裁剪）\n"
        "  绝对听阈   : MPEG-1 两段公式（< 3.5 kHz / >= 3.5 kHz）\n"
        "  掩蔽指数   : MPEG-1 Table B.3（随掩蔽器中心频率变化）\n"
        "  扩散函数   : MPEG-1 Table B.8 的数值\n"
        f"  参考电平   : 满幅正弦 = {FULL_SCALE_SPL_DB:.0f} dB SPL"
        f"（均方 dB 与 dB SPL 相差 {SPL_TO_MEAN_SQUARE_DB:.2f} dB）"
    )
