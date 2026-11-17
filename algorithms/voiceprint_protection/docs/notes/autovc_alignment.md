# AutoVC 官方仓库实测对齐记录（D1-2 / D2-1 交付物）

> 记录时间：2026-11-13
> 仓库：`third_party/AutoVC` ← <https://github.com/auspicious3000/autovc>
> Commit：`c1827a60e83e6d26030b8953843b2f7cf2bd36db`（2024-10-23）
> 许可证：MIT
>
> ⚠️ **原 `cyqxiaozhen/AutoVC` 已失效（404）**，官方地址是 `auspicious3000/autovc`（1099 stars）。
> 论文脚注只写了"AutoVC [14]"，检索时要认准作者名。

---

## 一、AutoVC 官方参数（逐项实测，非转述）

### 1.1 `hparams.py`（注意：这份主要是给 WaveNet 声码器用的）

| 参数 | 值 |
|---|---|
| `sample_rate` | **16000** |
| `num_mels` | 80 |
| `fmin` | 125 |
| `fmax` | 7600 |
| `fft_size` | 1024 |
| `hop_size` | 256 |
| `upsample_scales` | `[4, 4, 4, 4]`（乘积 256 = hop ✅） |
| `cin_channels` | 80 |
| `builder` | `wavenet` |

### 1.2 `make_spect.py`（★ 真正生成训练数据的地方，以此为准）

```python
mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))        # = 1e-5
b, a = butter_highpass(30, 16000, order=5)

x = np.pad(x, int(fft_length//2), mode='reflect')  # ← reflect 填充
fft_window = get_window('hann', fft_length, fftbins=True)
result = np.fft.rfft(fft_window * result, n=fft_length).T
D = np.abs(result)                                 # ← 幅度，不是功率

y = signal.filtfilt(b, a, x)                        # 30 Hz 高通
wav = y * 0.96 + (prng.rand(...)-0.5)*1e-06         # 缩放 + 微噪声
D_mel = np.dot(D, mel_basis)
D_db  = 20*np.log10(np.maximum(min_level, D_mel)) - 16
S     = np.clip((D_db + 100)/100, 0, 1)             # ← 归一化到 [0,1]
```

> ⚠️ **AutoVC 内部不一致**：`hparams.py` 写 `fmin=125`，而真正生成训练数据的
> `make_spect.py` 用 **`fmin=90`**。**以 make_spect.py 为准**（它才是生效的那个）。

### 1.3 ★ 官方代码在 librosa ≥ 0.10 下跑不起来（实测踩到）

AutoVC 原文：

```python
from librosa.filters import mel
mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
```

**librosa 0.10 起，`filters.mel` 的所有参数都变成关键字专用**，直接抄会报：

```text
TypeError: mel() takes 0 positional arguments but 2 positional arguments (and 3 keyword-only arguments) were given
```

**等价改写**（数值完全一致）：

```python
mel_basis = mel(sr=16000, n_fft=1024, fmin=90, fmax=7600, n_mels=80).T
```

> 影响：想直接跑 AutoVC 官方的 `make_spect.py` / `conversion.ipynb`，
> 必须先改这一处（或把 librosa 降到 <0.10）。
> 我们在 `audio_utils.autovc_mel_filterbank()` 里已做等价改写。

---

## 二、★ 三个推翻初始假设的结论

### 结论 1：AutoVC 官方权重是 **16 kHz** 的，不是 22.05 kHz

原以为 AutoVC 是 22.05 kHz（那是很多语音项目的默认）。实测 `hparams.py` 与
`make_spect.py` 都是 16000。

**影响**：
- ✅ **与论文的 16 kHz 完全一致** → 全程 16 kHz，不需要"22.05k 预设"
- ✅ **与 GE2E（RTC）的 16 kHz 一致**
- ⚠️ 只有 **HiFi-GAN 官方权重是 22.05 kHz** → 那是**声码器侧**的约定

### 结论 2：AutoVC 的 mel **不是 librosa.feature.melspectrogram**

它的链是自定义的（见 §1.2），与 librosa 有 **4 处** 差异：

| 环节 | AutoVC 官方 | librosa 默认 |
|---|---|---|
| STFT 填充 | **reflect** | constant |
| 输出 | **幅度** | 功率（power=2） |
| mel fmin | **90** | 125/40 视配置 |
| 后处理 | **dB + clip 到 [0,1]** | 无 |

**如果用 librosa 直算的 mel 去喂 AutoVC 权重 → 输出是噪音，且不报错。**
这就是 D2-1 要抓的东西。已实现于 `audio_utils.autovc_mel()`，
并由 `scripts/verify_autovc_mel.py` 用官方原始代码逐元素对照锁定。

### 结论 3：content code 实际是 **64 维**，不是 32 维

论文写 `c ∈ R^(T'×32)`。AutoVC 官方 `model_vc.py`：

```python
self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)  # dim_neck=32
out_forward  = outputs[:, :, :self.dim_neck]    # 32
out_backward = outputs[:, :, self.dim_neck:]    # 32
codes.append(torch.cat((out_forward[...], out_backward[...]), dim=-1))  # → 64
```

- **双向 LSTM**，前向 32 + 后向 32 拼接 = **64 维**
- 论文说的"瓶颈维度 32"指的是 **每个方向的维度** `dim_neck=32`
- Decoder 输入维度印证：`nn.LSTM(dim_neck*2 + dim_emb, ...)` = 32×2 + 256 = **320** ✅
- 时间下采样 `freq=32` → **T' = T // 32**（不是 T//2）

> 📌 契约文档已同步更正：`CONTENT_DIM=32`（dim_neck）、`CONTENT_ACTUAL_DIM=64`、`CONTENT_DOWNSAMPLE=32`。

---

## 三、AutoVC 官方推理链（`conversion.ipynb`）

```python
G = Generator(32, 256, 512, 32).eval().to(device)
#        dim_neck  dim_emb  dim_pre  freq
g_checkpoint = torch.load('autovc.ckpt')
G.load_state_dict(g_checkpoint['model'])

x_org, len_pad = pad_seq(x_org, base=32)     # mel 需 pad 到 32 的倍数
uttr_org = torch.from_numpy(x_org[np.newaxis, :, :])   # (1, T, 80)
emb_org  = torch.from_numpy(sbmt_i[1][np.newaxis, :])  # (1, 256)

_, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)    # (1, 1, T, 80)
```

- 模型结构：3 层 Conv(512, k=5) + BatchNorm + ReLU → 双向 LSTM → 3 层 Conv → LSTM → Linear(1024→80) → Postnet(5 层 Conv 残差)
- Postnet 输出与主输出**残差相加** → `x̂_psnt = x̂ + postnet(x̂)`
- 声码器：AutoVC 配套的 **WaveNet vocoder**（`synthesis.py`），直接吃模型的 80 维 mel

---

## 四、说话人嵌入：两套权重，**不可混用**

AutoVC 的 256 维嵌入来自 `model_bl.py` 的 `D_VECTOR`：

```python
C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256)
c_checkpoint = torch.load('3000000-BL.ckpt')
```

```python
class D_VECTOR(nn.Module):
    def __init__(self, num_layers=3, dim_input=40, dim_cell=256, dim_emb=64):
        self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_cell, num_layers=num_layers, batch_first=True)
        self.embedding = nn.Linear(dim_cell, dim_emb)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        embeds = self.embedding(lstm_out[:,-1,:])
        return embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))   # ← L2 归一化
```

| | AutoVC 的 `D_VECTOR` | 论文脚注引用的 GE2E (RTC) |
|---|---|---|
| 来源 | AutoVC `3000000-BL.ckpt` | `CorentinJ/Real-Time-Voice-Cloning` `pretrained.pt` |
| 结构 | 3 层单向 LSTM + Linear + L2 归一化 | 同构 |
| **输入 mel 维度** | **80** | **40** |
| 输出 | 256 维，已 L2 归一化 | 256 维，已 L2 归一化 |

**结构相同但第一层输入维度不同 → 权重不能互相加载。**

### 选哪个？

| 阶段 | 选哪个 | 理由 |
|---|---|---|
| **L1（跑通推理链）** | **AutoVC 的 `D_VECTOR`（80 维）** | 与 AutoVC 官方权重链**完全配套**，零转换 |
| **最终系统（论文忠实）** | **GE2E / RTC（40 维）** | 论文脚注明确引用它；且它能处理任意采样率的输入 |

> 论文原文：*Speaker embeddings are extracted using a GE2E model initialized from pretrained weights and fine-tuned end-to-end during Stage 1 training* + 脚注指向 RTC。
> 所以**最终系统用 RTC 的 GE2E**，L1 阶段先用 AutoVC 自带的把链路跑通。

---

## 四之二、GE2E (RTC) 的实测参数（B 的 D2-4 答案）

> 仓库：`third_party/Real-Time-Voice-Cloning`（commit `890f3a0`）
> ⚠️ 新版 RTC 已重构：`params.py` 拆成了 `encoder/params_data.py` 与 `encoder/params_model.py`。

### 4.2.1 音频与 mel 参数（`encoder/params_data.py` + `encoder/audio.py`）

```python
mel_window_length = 25   # 毫秒  -> n_fft       = int(16000 * 25/1000) = 400
mel_window_step   = 10   # 毫秒  -> hop_length = int(16000 * 10/1000) = 160
mel_n_channels    = 40
sampling_rate     = 16000

# encoder/audio.py 的 wav_to_mel_spectrogram()
frames = librosa.feature.melspectrogram(
    wav, sampling_rate,
    n_fft=int(sampling_rate * mel_window_length / 1000),
    hop_length=int(sampling_rate * mel_window_step / 1000),
    n_mels=mel_n_channels,
)
```

| 项 | 值 |
|---|---|
| 采样率 | **16000** |
| n_fft | **400** |
| hop_length | **160** |
| n_mels | **40** |
| fmin / fmax | **0 / 8000**（librosa 默认） |
| power | **2**（librosa 默认，功率谱） |
| 输出 | **线性幅度，不是 log-mel**（源码注释原文：*"Note: this not a log-mel spectrogram."*） |

### 4.2.2 模型结构（`encoder/params_model.py` + `encoder/model.py`）

```python
model_hidden_size    = 256
model_embedding_size = 256
model_num_layers     = 3

class SpeakerEncoder(nn.Module):
    self.lstm   = nn.LSTM(input_size=40, hidden_size=256, num_layers=3, batch_first=True)
    self.linear = nn.Linear(256, 256)
    self.relu   = nn.ReLU()
```

**与 AutoVC 的 `D_VECTOR` 对比：**

| | AutoVC `D_VECTOR` | RTC `SpeakerEncoder` |
|---|---|---|
| 输入维度 | **80** | **40** |
| LSTM | 3 层单向，hidden 768 | 3 层单向，hidden 256 |
| 输出层 | Linear(768→256) | Linear(256→256) + ReLU |
| **L2 归一化** | ✅ 有 | ⚠️ **没有**（需我们自己归一化） |
| 结构本质 | 相同 | 相同 |

> ⚠️ **归一化差异很重要**：AutoVC 的 `D_VECTOR` 内部做了 L2 归一化；
> RTC 的 `SpeakerEncoder` **没有**（它用余弦相似度算损失，天然不需要归一化）。
> 我们在 `bridge_speaker.py` 里统一做 L2 归一化，与接口契约第 5 条一致。
> 这也意味着：**RTC 的权重不能直接塞进 `D_VECTOR`**（hidden 维度都不同：256 vs 768）。

### 4.2.3 推理时的帧数

```python
inference_n_frames = 80    # 800 ms
partials_n_frames  = 160   # 1600 ms
```

即 RTC 推理时把音频切成 **80 帧（0.8 秒）** 的片段，取每段的嵌入。

---

| 权重 | 地址 | 备注 |
|---|---|---|
| AutoVC 主模型 | `https://ibm.box.com/s/upmfbstk8n8ded0xd3aaic4ayclh90ee` | IBM Box，**不是 Google Drive** |
| 另两个 | `https://ibm.box.com/s/k4nyvzkz26r9ux1o8rcq9tov36h69o2e`<br>`https://ibm.box.com/s/pdr8b0z5oim9q0p4be496mt3ue5lb004` | — |
| HiFi-GAN v1（推荐给 conversion.ipynb） | `https://ibm.box.com/s/asvv554v0zd09yipl2qadz49i7jpdhta` | 22.05 kHz |
| WaveNet vocoder | 见 README | 16 kHz，与 AutoVC 配套 |
| GE2E (RTC) | `pretrained.pt`（RTC README） | Google Drive / Dropbox |

**国内网络注意**：Google Drive 直连不通（已实测），需走代理或找镜像；
IBM Box 需实测可达性。HiFi-GAN 另有 GitHub 上的 `NVIDIA/torch.hifi_gan`（含 `hifigan_melbins_16k`），走 GitHub 更稳。

---

## 五、★ AutoVC 官方权重链已失效（2026-11-17 实测）

```
curl -I https://ibm.box.com/s/upmfbstk8n8ded0xd3aaic4ayclh90ee
→ HTTP/1.1 404 Not Found
```

AutoVC 官方 README 给的三件套权重**全部托管在这个 IBM Box 链接上，现已 404**：

| 权重 | 链接 | 状态 |
|---|---|---|
| AutoVC 主模型 | `https://ibm.box.com/s/upmfbstk8n8ded0xd3aaic4ayclh90ee` | **404** |
| 另两个 | `.../s/k4nyvzkz26r9ux1o8rcq9tov36h69o2e`、`.../s/pdr8b0z5oim9q0p4be496mt3ue5lb004` | **404** |
| HiFi-GAN v1 | `https://ibm.box.com/s/asvv554v0zd09yipl2qadz49i7jpdhta` | **404** |
| WaveNet vocoder | 见 README | **404** |

**结论：路线 A（AutoVC 全家桶）走不通 —— 不是"难下"，是"下不到"。**

### 替代方案（都已实测可下载）

| 用途 | 来源 | 实测 |
|---|---|---|
| GE2E 说话人编码器 | HF `CorentinJ/SV2TTS` → `encoder.pt`（17,090,379 字节） | ✅ 200 OK |
| **合成器**（= 论文里的"攻击者"） | 同上 → `synthesizer.pt` | ✅ |
| **声码器** | 同上 → `vocoder.pt` | ✅ |
| AutoVC 的 Ec/D/GE2E 打包权重（MakeItTalk 出品） | HF `marlenezw/AutoVC_Voice_Conversion` → `ckpt_autovc.pth`（171,881,992 字节） | ✅ |

> `marlenezw` 的 README 明确写：*"The speaker encoder is a pre-trained model provided Wan et al. [2018]"*
> —— 正是论文用的 GE2E（论文脚注 2）。

### 由此产生的路线调整

**L1 推理链改用上游 RTC 的合成器 + 声码器**（`bridge_cloner.py`），
因为它们**互相配套**（同一套 mel：16 kHz / 80 mel / n_fft 800 / hop 200 / fmin 55），
我们完全不用自己猜 mel 约定。

**AutoVC 降级为「Stage 1 的架构参考 + 初始化权重来源」**：
它的 `Ec / D / PostNet` 架构、`dim_neck=32`、双向 LSTM、`freq=32`，
以及那条已复刻的 mel 预处理链（`audio_utils.autovc_mel()`，逐元素对照差 0.000e+00），
在 Stage 1 自己训练时全部沿用。

详见任务板 [TASKS.md](../../TASKS.md) §0.8。

---

## 六、给下一阶段的行动项

| # | 事项 | 负责人 | 状态 |
|---|---|---|---|
| 1 | 跑 `scripts/verify_autovc_mel.py`，确认逐元素一致 | A | ⬜ |
| 2 | 下载 AutoVC `autovc.ckpt`（IBM Box） | A | ⬜ |
| 3 | 下载 WaveNet vocoder 权重（16 kHz，配套 AutoVC） | A | ⬜ |
| 4 | 确认 `hifigan_melbins_16k` 是否可用（备选声码器） | A | ⬜ |
| 5 | 写 `bridge_autovc.py`：加载 `Generator(32,256,512,32)` + `autovc.ckpt` | A | ⬜ |
| 6 | 写 `bridge_speaker.py`：加载 `D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256)` + `3000000-BL.ckpt` | A | ⬜ |
| 7 | 写 `bridge_vocoder.py`：WaveNet vocoder，喂模型输出的 80 维 mel | A | ⬜ |
| 8 | 确认 GE2E (RTC) 的 40 维 mel 参数并做对照 | B | ⬜ |
