# 双人接口契约（任务板 W2-D3-1）

> **状态**：草案 v0.1 —— 成员 B 起草，**待成员 A 确认签字**。
> 依据：指南第 8 章 §8.5 的 10 条 + 本项目三个桥接模块的实际签名。
>
> **规矩**：本文件里的任何一条，改动都要**两人同意**并更新本文件。
> 这是两人协作里性价比最高的一件事 —— 没有它，Stage 2 拿到 Stage 1 的输出时
> 形状/归一化对不上，会浪费一整天。

---

## 1. 音频与 mel（不可协商的硬约定）

| # | 约定 | 值 | 真源 |
|---|---|---|---|
| 1 | 采样率 | **16000 Hz**，单声道，16-bit PCM | `config.MEL_PRESETS` |
| 2 | **模型侧 mel（Ec/D 用）★** | 由 `config.STAGE1["mel_preset"]` 决定，当前 = **`hifigan16k`**（见 §3.4）。**统一入口：`audio_utils.model_mel(wav)`** | `audio_utils.model_mel()` |
| 3 | AutoVC 侧 mel（**保留**，非默认） | **不是 librosa 默认链**：30 Hz 高通 → reflect 幅度 STFT → `mel(fmin=90, fmax=7600, n_mels=80, n_fft=1024, hop=256)` → `20·log10(max(1e-5,·)) - 16` → `clip((x+100)/100, 0, 1)` | `audio_utils.autovc_mel()` |
| 4 | 预设 | **三个**：`hifigan16k`（★默认）/ `model16k`（AutoVC 官方链）/ `rtc16k`（上游 WaveRNN）。⚠️ 原 `paper16k` / `autovc22k` 已取消 | `config.MEL_PRESETS` |

> **规则**：**任何地方都不许再手写 mel 参数**，一律调 `audio_utils` 里对应的函数。
> ★ **模型侧 mel 一律用 `audio_utils.model_mel(wav)` 取**，别直接调
> `autovc_mel` / `rtc_mel` / `hifigan_mel` ——
> 否则会出现"训练用一套、评测用另一套"，这种坑会以**"指标莫名不对"**的形式出现，
> 而且**不报错**。
>
> **为什么没有 22.05 kHz 预设**：AutoVC 官方权重本身就是 **16 kHz** 的（成员 A 实测），
> 与论文一致 → 全程 16 kHz。
>
> **特例（唯一允许的例外）**：GE2E 内部的 40 维 mel（`n_fft=400, hop=160, fmin=0, fmax=8000`，
> **线性、不取 log**）是**上游自己的约定，我们不碰**，只通过 `bridge_speaker.py` 使用。
> 详见 [`notes/ge2e_interface.md`](notes/ge2e_interface.md)。

---

## 2. 张量形状（指南 §8.5 第 3–9 条）

| 符号 | 名称 | 形状 | dtype | 归一化 | 说明 |
|---|---|---|---|---|---|
| `x` | 输入 mel | `(T, 80)` | float32 | 见 `audio_utils` | `T` = 帧数 |
| `c` | 内容码 | `(T//32, 64)` | float32 | — | ★ **实测更正**：`freq=32` 降采样、`dim_neck=32` × 双向 LSTM = 64 维。论文写的 `(T', 32)`、`T'=T//2` 与官方代码不符，**以代码为准**（成员 A 实测 `third_party/AutoVC/model_vc.py`） |
| `h` | 说话人嵌入 | `(256,)` | float32 | **L2 归一化** | 由 `bridge_speaker.embed()` 出口保证 |
| `δ` | 扰动 | `(256,)` | float32 | — | Stage 2 生成器 `G(h)` 的输出 |
| `h_adv` | 被污染的嵌入 | `(256,)` | float32 | 继承 h | `h_adv = h + α·δ` |
| `x_adv` | 保护后 mel | `(T, 80)` | float32 | — | `x_adv = D(c, h_adv)` |
| `wav` / `wav_adv` | 波形 | `(samples,)` | float32 | — | `wav_adv = V(x_adv)`，16 kHz |
| `δ_wav` | 波形扰动 | `(samples,)` | float32 | — | `δ_wav = wav_adv − wav`，供 `L_mask` 用 |

**形状不变量**：`x` 与 `x_adv` 的帧数必须**完全相等**；
content code 个数 = `T // 32`（`freq=32`）、每个 64 维；
`L_dis` 用的 `c̃` = **沿时间维拼接 4 个 content code = R^256**（与论文 `W_s ∈ R^{K×256}` 对齐）。
> 不相等时 `L_mel` 会退化成恒定大常数，**而且不报错** —— 这是最难查的一类 bug。

---

## 3. 三个桥接模块的签名

> A 负责 `bridge_autovc.py` / `bridge_vocoder.py`，B 负责 `bridge_speaker.py`（已完成）。
> 三个模块的**调用方式必须一致**：构造时加载权重，方法名统一。

### 3.1 `bridge_speaker.py`（成员 B 维护，**含两套实现**）

```python
from bridge_speaker import SpeakerBridge, SpeakerEncoderBridge

# ---- 路线 B：论文脚注引用的上游 GE2E（成员 B）----
bridge = SpeakerBridge(rtc_dir=None, device=None, weights_path=None)

h    = bridge.embed(wav: np.ndarray, sample_rate: int) -> np.ndarray  # (256,) 已归一化
h    = bridge.embed_path(path: str | Path)              -> np.ndarray  # (256,) 已归一化
H    = bridge.embed_batch(waves: list, srs: list)      -> np.ndarray  # (N, 256)
info = bridge.describe()                               -> dict         # 接口自述

# ---- 路线 A：AutoVC 自带的 D_VECTOR（成员 A，infer.py 默认用这个）----
encoder = SpeakerEncoderBridge(checkpoint=None, device=None, encoder="autovc_bl")
h = encoder.embed(mel: np.ndarray, crop: int = 128)    -> np.ndarray  # (256,) 已归一化
```

| | `SpeakerBridge`（路线 B） | `SpeakerEncoderBridge`（路线 A） |
|---|---|---|
| 输入 | **波形**（自己算 40 维线性 mel） | **mel (T, 80)**（AutoVC 的 spmel 链） |
| 权重 | `encoder.pt`（HuggingFace，17 MB） | `3000000-BL.ckpt`（IBM Box） |
| `embed()` 返回 | numpy `(256,)` | numpy `(256,)` |

- **两套权重不可混用**（输入维度与结构都不同）。
- `SpeakerEncoderBridge(encoder="ge2e_rtc")` 会**委托给 `SpeakerBridge`**，
  此时 `embed()` 的第一个参数是**波形**而不是 mel。
- 两条路线的输出**都已 L2 归一化**，调用方不要再归一化。
- `sample_rate` 是**输入音频的原生采样率**；不是 16 kHz 时桥接层内部自动重采样。

> ✅ **路线取舍已定（成员 A，2026-11-17）**：**走路线 B**（论文脚注 2 的 GE2E）。
> 决定性依据：**AutoVC 官方权重链实测 404 失效**（IBM Box），路线 A 走不通。
> AutoVC 降级为「Stage 1 的架构参考 + 初始化权重来源」。
> 详见 TASKS.md §0.8 与 `notes/autovc_alignment.md`。

### 3.2 `bridge_autovc.py`（成员 A）

> ⚠️ 下面是**按实际代码**（2026-11-17 合并时核对）写的，不是最初草案。

```python
from bridge_autovc import AutoVCBridge

bridge = AutoVCBridge(checkpoint: str | Path | None = None, device: str | None = None)

x_out = bridge.protect(mel, embedding, embedding_adv)  # 加扰动的重建
x_rec = bridge.reconstruct(mel, embedding)              # 不加扰动的重建（自检用）
```

- `mel` 是 **(T, 80)** 的 AutoVC 侧 mel（不是 batch 维），`embedding` 是 **(256,)**。
- `protect()` 与 `reconstruct()` 内部都走 AutoVC 的 `Generator`，
  差别只是传 `embedding_adv` 还是 `embedding`。
- **重建（reconstruct）是自检的关键一步**：重建输出是噪音 → 问题在 mel 或权重；
  重建正常但保护后是噪音 → 问题在扰动或解码路径。分两步测能立刻定位坏在哪一环。

### 3.3 `bridge_vocoder.py`（成员 A，**v0.4 重写 / v0.5 换成 HiFi-GAN**）

```python
from bridge_vocoder import VocoderBridge, WaveRNNBridge

bridge = VocoderBridge(weights_path=None, device=None)

wav = bridge.mel_to_wav(mel)                    # 网络侧 mel (80,T)/(T,80) -> 波形 (N,)
wav = bridge.mel_to_wav(mel, normalized=False)  # 原始 mel（不反归一化）
mel_back = bridge.wav_to_mel(wav)               # 波形 -> 原始 mel (T, 80)
```

- 输入是 **(80, T)** 的 **HiFi-GAN 约定 mel**。
- 内部是 **HiFi-GAN v1 生成器**（torch 原生、**可求导**）—— 论文 §2.3 指定的 V(·)。
- 输出是一维波形，采样率 **16 kHz**。
- `normalized=True`（默认）：输入是**网络侧**（归一化过的）mel，内部反变换成原始量纲。

> ★ **为什么必须可求导**：Stage 2 的梯度要穿过声码器
> ```
> G → h_adv → D(c, h_adv) → V(·) → E_spk(·) → L_de
> ```
> 上游 WaveRNN 是**自回归生成器**（采样 9 bit 离散码），梯度断在那里，
> G 收不到梯度而且**不报错**。所以 WaveRNN 只保留为 `WaveRNNBridge`（对照用）。

> ★ **权重从哪来**：NVIDIA 官方的 `NVIDIA/torch.hifi_gan` 仓库**已整体下架**
> （2026-11-18 实测：GitHub API / codeload / raw / release 附件全部 404）。
> 我们用 HuggingFace `speechbrain/tts-hifigan-libritts-16kHz` 的 `generator.ckpt`：
> **同一套 HiFi-GAN v1 架构、同一套 mel 约定、同样 13,936,130 参数**，
> 加载时 `strict=True`（实测 234/234 键全匹配）。
> 下载：`scripts/download_hifigan_weights.py`

### 3.4 `audio_utils.py` 的 mel 约定（⚠️ v0.5 已变更）

| 用途 | 函数 | 约定 |
|---|---|---|
| **模型侧（Ec/D）★** | `audio_utils.model_mel(wav)` | 走 `config.STAGE1["mel_preset"]`，当前 = `hifigan16k`（**归一化量纲**） |
| HiFi-GAN 原始 | `audio_utils.hifigan_mel(wav)` | preemph 无 / STFT 幅度 / slaney mel(fmin 0, fmax 8000) / **log(clamp(1e-5))**，值域 **[-11.5, 0]**，形状 (T, 80) |
| ★ 网络侧归一化 | `audio_utils.hifigan_mel_norm(wav)` | `0.08686·原始 + 0.84`，值域 **[-0.11, 0.92]**（与 AutoVC 同量纲） |
| ★ 反归一化 | `audio_utils.hifigan_mel_unnorm(mel)` | 上面的逆变换（**可逆，零损失**） |
| 模型侧 · AutoVC 链 | `audio_utils.autovc_mel(wav)` | 30 Hz 高通 → reflect 幅度 STFT → mel(fmin=90) → dB → clip[0,1] |
| 模型侧 · WaveRNN 链 | `audio_utils.rtc_mel(wav)` | preemph → librosa STFT(800/200) → 幅度 → mel(fmin=55,fmax=7600) → 对称归一化 → /4，值域 **[-1, 1]** |
| GE2E 侧 | `audio_utils.ge2e_mel(wav)` | 40 维**线性** mel（上游 `encoder/params_data.py` 约定），喂说话人编码器 |
| 通用 librosa 版（对照用） | `audio_utils.librosa_mel(wav)` | librosa 原生 mel，用于数值对照 |
| 声码器侧（NVIDIA 旧约定） | `audio_utils.vocoder_mel(wav)` | mel 基 sr=8000 + `(x+4.5)/5`（**目前无对应权重，保留备用**） |
| 通用预设 | `audio_utils.MelSpec(name).mel(wav)` | 走 `config.MEL_PRESETS[name]` |

> ★ **代码里请一律用 `audio_utils.model_mel(wav)` 取模型侧 mel，别直接调
> `autovc_mel` / `rtc_mel` / `hifigan_mel`** —— 否则会出现"训练用一套、评测用另一套"，
> 这种坑会以"指标莫名不对"的形式出现，而且**不报错**。
>
> ★ **三套模型侧 mel 的取舍**：
> | 预设 | n_fft/hop | 取值 | 配套声码器 | 可求导 | 状态 |
> |---|---|---|---|---|---|
> | `model16k`（AutoVC 官方） | 1024 / 256 | [0, 1] | ❌ 权重全 404 | — | 保留（已通过官方对照 0.000e+00） |
> | `rtc16k`（WaveRNN） | 800 / 200 | [-1, 1] | ✅ 上游 WaveRNN | ❌ | 保留（对照用） |
> | **`hifigan16k`（HiFi-GAN）★默认** | 1024 / 256 | [-0.11, 0.92]（归一化后） | ✅ **HiFi-GAN 16k** | ✅ | **当前默认** |
>
> **为什么是 hifigan16k**：论文 §2.3 原文
> `"x_adv = D(c, h_adv), and converted to waveform using HiFi-GAN vocoder V(·)"`
> ⇒ **Stage 1 的 mel 必须就是 HiFi-GAN 的约定**。
> ★ 而且我们**实测过**仿射换算不可接受：
> AutoVC mel → HiFi-GAN mel 的仿射换算后，重建音的说话人余弦从 **0.9746 掉到 0.6983**
> （滤波器组不同，仿射补不了）—— 所以必须直接用 HiFi-GAN 的 mel。
>
> ★ **AutoVC 的网络架构一字未改**（3 层卷积 + 双向 LSTM 瓶颈 32 + 解码器 + 5 层 postnet），
> 论文 §3.1 要求的是**架构**。mel 是预处理约定，论文没有规定。
>
> ★ **为什么要归一化**：论文的 `λ_cd=1.0`、`λ_dis=0.01` 是在 AutoVC 量纲下
> 网格搜索出来的。原始 HiFi-GAN mel 值域是 [-11.5, 0]，
> 直接用会让 `L_rec` 比论文大 ~**100 倍**（实测 165 vs 0.86），
> `λ_cd` 的相对权重被稀释 ~40 倍 ⇒ **论文的超参失效**。
> 归一化是**固定仿射变换且可逆**，送进声码器前反变换回去，**零损失**。


## 4. 推理主流程 `infer.py`（成员 A 主责，B 写冒烟测试要用）

> ⚠️ 下面是**按实际代码**（2026-11-17 合并时核对）写的。

```python
from infer import Protector, perturb_embedding, random_perturbation

protector = Protector()                       # 装配 AutoVC / 说话人编码器 / 声码器
result = protector.protect(wav, alpha=0.10, seed=42, verbose=True)
# result 里至少有（★ 2026-11-17 按路线 B 的实际实现更新）：
#   "protected_waveform"     (N,) 保护后语音
#   "embedding"              (256,) 说话人嵌入
#   "embedding_adv"          (256,) 加扰动后的嵌入
#   "clone_baseline_waveform" (N,) 未防御时攻击者的克隆音
#   "clone_protected_waveform" (N,) 防御后攻击者的克隆音
#   "stolen_embedding"       (256,) 攻击者从保护后语音重新提取的嵌入
#   "shapes"                 各步的张量形状
```

- `protect()` 的第一个参数是**波形数组**（不是路径），
  路径 → 波形由 `audio_utils.load_audio(path, sample_rate)` 负责。
- 逐步接口（便于定位问题）：`to_mel()` / `embedding()` / `protect_mel()` / `to_waveform()`。
- **L1 阶段**：`δ` 用**随机方向**（`random_perturbation(seed=...)`，固定 seed）代替训练好的 `G`。
- `scripts/smoke_test.py` 与 `scripts/make_sample.py` 都按这个 API 调用；
  **`make_sample.py` 兼容两种形态**：模块级 `protect(...)` 或 `Protector.protect(...)`，
  谁先存在用谁 —— **签名再变就要同步改这两个脚本**。

> 💡 若希望 `make_sample.py` 这类脚本更省事，可以给 `infer.py` 加一个模块级薄封装：
> ```python
> def protect(wav_path, alpha=0.10, output_path=None) -> dict:
>     """文件进、保护后 wav 落盘、返回带 shapes 的报告。"""
> ```
> 加了之后 `make_sample.py` 会自动走这条更简洁的路径。

---

### 4.5 ★ Stage 2 的接口与形状（2026-11-18 新增，成员 B；v0.5 由成员 A 同步）

> Stage 2 的代码在 `algorithms/voiceprint_protection/stage2/`。
> 这一节只记**两人都会碰到的**部分：形状、接口、冻结约定。

#### 4.5.1 张量形状（续 §2）

| 符号 | 名称 | 形状 | 说明 |
|---|---|---|---|
| `δ` | 扰动 | `(B, 256)` | **单位长度**（‖δ‖=1）→ α 是唯一的强度旋钮 |
| `h_adv` | 被污染的嵌入 | `(B, 256)` | `h_adv = normalize(h + α·δ)`，**重新 L2 归一化** |
| `c` | 内容码 | `(B, T', 64)` | `T' = T // 32`（AutoVC 抽样式降采样） |
| `c_arb` | 打乱配对的内容码 | `(B, T', 64)` | 论文的 "arbitrary content"，batch 内随机置换 |
| `x_adv` | 保护后 mel | `(B, T, 80)` | **与 `x` 帧数必须完全相等** |
| `wav_adv` | 保护后波形 | `(B, N)` | `N = T · 256`（HiFi-GAN 的上采样倍率就是 256） |
| `δ_wav` | 波形扰动 | `(B, N)` | `δ_wav = V(x_adv) − V(x)`，**两条波形要先对齐到等长** |
| `h_stolen` | 攻击者重新提的卡 | `(B, 256)` | `h_stolen = E_spk(wav_adv)` |

#### 4.5.2 ★ 冻结 ≠ 断梯度（最容易踩的一条）

Stage 1 的解码器、声码器、`E_spk` 都**冻结**（不更新参数），
但**梯度必须能穿过它们流回 G**：

```
G → h_adv → D(c, h_adv) → V(·) → E_spk(·) → L_de
```

- **冻结** = `requires_grad_(False)` + `eval()`
- **切断梯度** = `torch.no_grad()` 或 `.detach()` ← 这两个**不要**在前向里用

> 指南第 6 步的伪代码把整段前向包进了 `torch.no_grad()`。
> **照抄会让训练静默失效**（loss 一步不动、不报错）。
> 已由 `stage2/verify_gradients.py` 变成会失败的断言。

#### 4.5.3 可求导组件放哪（v0.5 更新：声码器可以直接用 A 的桥接层）

| 需求 | 用哪个 | 为什么不用现成的 |
|---|---|---|
| `E_spk`（可导） | `stage2/models.py::TorchSpeakerEncoder` | `bridge_speaker.SpeakerBridge` 是 numpy 封装 |
| Stage 1 解码器（可导） | `stage2/models.py::TorchAutoVCStage1` | `bridge_autovc.AutoVCBridge` 内部 `no_grad()` |
| 声码器（可导） | ★ **`bridge_vocoder.VocoderBridge`（v0.5 起本身就是 torch 版、可求导）** | v0.4 时它是 numpy 封装，要另写 `stage2/pipeline.py::HiFiGANVocoder`；**v0.5 之后不需要了** |

> **v0.5 的变化**：`bridge_vocoder.VocoderBridge` 内部换成了 **HiFi-GAN（torch 原生）**，
> 它的 `forward` **梯度可以穿过**（实测 `scripts/verify_hifigan.py` 第 4 层：
> 梯度能回传到 mel，absmax ≈ 0.38）。
> ⇒ `stage2/pipeline.py::HiFiGANVocoder` **可以删掉**，直接用
> `bridge_vocoder.VocoderBridge`（它会自己反归一化网络侧 mel）。
>
> **两套实现并存是刻意的**：`bridge_*` 供推理链（L1/平台）用，
> `stage2/models.py` 供训练用。两者加载**同一个** checkpoint，结构一致。
> `stage2/verify_mel.py` 保证 torch 版 GE2E 与上游 numpy 版**逐值一致**
> （mel 误差 2.3e-05，嵌入余弦 0.99999481）。

#### 4.5.3b ★★ 成员 A 修的 Stage 2 两个**致命 bug**（2026-11-19）

> 这两个 bug 都在 `stage2/models.py::TorchAutoVCStage1`，
> **症状是"训练能跑、loss 在降、但防御机制根本没生效"** —— 最坏的一类。

| # | bug | 原来的写法 | 后果 | 修法 |
|---|---|---|---|---|
| 1 | **目标嵌入没进解码器** | `self.model(mel, embedding, embedding)` | 官方 `Generator.forward(x, c_org, c_trg)` 的**第三个参数才是目标嵌入**。两个都传 embedding ⇒ **`h_adv` 根本没参与重建** ⇒ 扰动嵌入对 `x_adv` 毫无影响，防御机制完全失效，**且不报错** | `decode()` 自己实现摊开 + 解码，**c_trg = 给定的目标嵌入** |
| 2 | **`content` 参数被忽略** | `decode()` 里根本没用 `content` | 论文的 **"arbitrary content"**（`c_arb`，攻击者用**任意内容**克隆）没起作用，`x_clone` 用的还是原句内容 ⇒ `L_de` 测的不是论文那个场景 | `decode(..., content=c)` 支持注入内容码 |

**修复后的实测**（成员 A，`stage2/models.py` 载入 A 训的 Stage 1 checkpoint）：

```
content_code -> (2, 7, 64)                              ✅ 与 §4.5.1 一致
D(c,h) 与 D(c,h_adv) 的平均差 = 0.525248                ✅ 修 bug 前是 0（h_adv 没进去）
梯度能回到 h_adv: ✅ (absmax = 2165)                    ✅ 修 bug 前是 0
content 被真正使用: ✅                                   ✅ 修 bug 前是 0
```

**⇒ 验收 Stage 2 时，请先确认这三行。** 写法错了 loss 照样在降，但指标是假的。

> ★ 顺带：`content_code()` 的返回值从 **list** 改成**张量** `(B, T', 64)`
> —— 官方 AutoVC 的 `Encoder.forward` 返回 list，
> 但 `stage2/train.py` 要写 `content_codes[permutation]`（batch 内随机置换），
> 对 list 索引是错的。

#### 4.5.3c ★ 回归测试：Stage 1 <-> Stage 2 的接口（成员 B，2026-11-20）

> A 修的两个 bug（§4.5.3b）之所以能溜过去，是因为 `stage2/verify_gradients.py`
> **只验 `StubStage1`** —— 而 `StubStage1.decode()` 的逻辑跟官方 `Generator`
> 根本不是一回事，验不到"目标嵌入有没有真的进解码器"。

B 已把这段补成**会失败的断言**（`stage2/verify_gradients.py` 的【6】段，共 7 项）。
它用 `TorchAutoVCStage1(random_init=True)` —— 权重是随机的，**指标毫无意义**，
但验的是**接口**，**不依赖训练好的权重**，所以 Stage 1 还没训完也能验。

| # | 断言 | 实测值（2026-11-20） |
|---|---|---|
| 1 | `content_code()` 返回**张量**（不是 list） | OK |
| 2 | `content_code` 形状 `(B, T', 64)` | `(2, 3, 64)`（T=96） |
| 3 | `T' = T // 32` | 3 |
| 4 | **h_adv 真的进了解码器**：`D(c,h)` 与 `D(c,h_adv)` 的平均差 > 0 | **0.051788**（修 bug 前是 0） |
| 5 | **梯度能回到 h_adv** | OK |
| 6 | **content 参数真的被用上**：换 content 后输出之差 > 0 | **0.005156**（修 bug 前是 0） |
| 7 | `decode()` 与官方 `Generator.forward(x, c_org, c_trg)` **逐值一致** | 最大差 **2.38e-06** |

> 第 7 条是最硬的一条：它把 `decode()` 的语义**钉死**在官方 AutoVC 上 ——
> 以后谁改 `decode()` 改出偏差，这一条立刻会红。

**跑法**：`python algorithms/voiceprint_protection/stage2/verify_gradients.py`
（现在共 **19 项**断言全通过）。

> ★ 顺带修掉的三个 B 自己的坑（都在共同区）：
> (1) `paths.models_dir()` 指向不存在的目录 => **权重找不到、整条链起不来**；
> (2) `Stage2Pipeline.vocode()` 把 mel 预设写死；
> (3) `audio_utils.MelSpec` 写死 `autovc_mel()`。
> 详见 §9 成员 B 签字栏。

#### 4.5.4 损失函数签名（论文式 7–13）
```python
from stage2.losses import (
    mel_loss, mask_loss, perceptual_loss,          # 式 7/8/9
    discriminator_gan_loss, generator_gan_loss,     # 式 10/11（LSGAN，无 sigmoid）
    defense_loss,                                  # 式 12（★ 要最小化）
    generator_loss,                                # 式 13
)
```

超参全部读 `config.STAGE2`（唯一真源）：`α=0.10, λ_per=15.0, λ_de=5.0, λ_mask=0.1, λ_GAN=1.0`。

> ⚠️ `defense_loss` 是**最小化**余弦相似度（论文的 "Minimizing L_de"）。
> 方向反了训练会把音频毁掉。`stage2/train.py` 每步都打印 `L_de`，**它应该下降**。

#### 4.5.5 评测指标的口径

| 指标 | 定义 | 备注 |
|---|---|---|
| `Sim_prot` ↑ | `cos(h_stolen, h)` | 保护后语音 vs 原始语音 |
| `Sim_clone` ↓ | `cos(E_spk(clone), h)` | **论文口径**：克隆音 vs 原始说话人 |
| `DSR` ↑ | `mean(Sim_clone < τ)`，τ ∈ {0.3, 0.4, 0.5} | 论文 79.8 / 83.5 / 87.2 |

---

## 5. 数据清单格式（成员 B）

```text
data/train.txt   每行：<相对 data/ 的音频路径> <说话人名>
data/val.txt     同上
data/test.txt    同上
data/spk2id.txt  每行：<说话人名> <整数 ID>，只含训练集说话人（K = 80）
```

| 硬要求 | 检查方式 |
|---|---|
| 排除 `p*_eval` | `grep -c "_eval" data/*.txt` 必须为 0 |
| 三集合说话人不相交 | `comm -12 <(cut -d' ' -f2 data/train.txt\|sort -u) <(cut -d' ' -f2 data/test.txt\|sort -u)` 必须为空 |
| `spk2id.txt` 的 K = 80 | `cut -d' ' -f2 data/train.txt \| sort -u \| wc -l` → 80，且 ID 覆盖 0..79 |

> `prepare_data.py` 跑完会自动执行这三条检查，不过则返回码非 0。

---

## 6. 随机性与复现

| 项 | 约定 | 出处 |
|---|---|---|
| 全局 seed | `config.SEED = 42` | `config.py` |
| 说话人划分 | `prepare_data.py --seed`，默认取 `config.SEED` | 划分名单记录在 `data/split_speakers.json` |
| L1 随机扰动 | `numpy.random.default_rng(config.SEED)` | `scripts/smoke_test.py` |
| 训练随机性 | Stage 1/2 的 DataLoader、初始化**全部**显式用 `config.SEED` | 待 A/B 各自实现 |

---

## 7. Stage 1 的约定（成员 A，2026-11-18）

> Stage 2 要吃 Stage 1 的输出，所以这一节和 §1–§2 一样重要。

### 7.1 模型侧 mel（★ 由 `config.STAGE1["mel_preset"]` 决定，别写死）

| 用途 | 函数 | 形状 | 取值 | 喂给谁 |
|---|---|---|---|---|
| **模型侧 mel** | `audio_utils.model_mel(wav)` | `(T, 80)` | 当前 `hifigan16k` → **[-0.11, 0.92]**（归一化后） | `Ec` / `D` / 三个损失 |
| HiFi-GAN 原始 mel | `audio_utils.hifigan_mel(wav)` | `(T, 80)` | [-11.5, 0] | 声码器 V(·) |
| GE2E 侧 mel | `audio_utils.ge2e_mel(wav)` | `(T', 40)` | 线性幅度（不取 log） | GE2E 说话人编码器 |

> ⚠️ GE2E 侧是**第三套 mel 约定**（线性幅度、不取 log），
> 与上游 `encoder/params_data.py` 完全一致，属于"上游自己的约定，我们不碰"。
>
> ⚠️ 模型侧当前用 **`hifigan16k`**。切换方式：
> `config.STAGE1["mel_preset"] = "..."`（**共同区，改动需双方同意**）。
> 详见 §3.4 的三套 mel 对比表与选择理由。

### 7.1b ★ 帧数必须是 `freq` 的整数倍（训练时最容易踩的坑）

两套 mel 链给的都是 **`1 + N // hop` 帧**（center 填充的那一帧），
所以要拿到**恰好 `crop_frames` 帧**，采样点数必须是

```
N = (crop_frames - 1) * hop          # 不是 crop_frames * hop（那样会多出 1 帧）
```

且 `crop_frames` 必须是 `freq`（=32）的整数倍。原因：

1. 官方 AutoVC 的 Encoder 取 `out_forward[:, i+freq-1, :]`（`i = 0, freq, 2*freq, …`），
   `T` 不是 `freq` 整数倍时 `i+freq-1` 会**越界**；
2. Decoder 把每个 code 摊回 `int(T / len(codes))` 帧，摊不满就会
   **解码输出与输入帧数不等** → `L_rec` 直接形状报错。

官方 AutoVC 是在 `synthesis.py` 里 pad 到 32 的倍数来解决的；
我们在 `stage1/dataset.py` 里裁剪时解决（并加了断言自检）。

### 7.2 Stage 1 的模型与形状

| 项 | 值 | 出处 |
|---|---|---|
| 架构 | AutoVC 官方 `model_vc.py`（`Ec`/`D`/`Postnet`） | `stage1/model.py` |
| `dim_neck` | 32 | `config.AUTOVC_MODEL` |
| content code | `(T//32, 64)` | 官方代码实测 |
| `freq` | 32 | 官方代码实测 |
| `dim_emb` | 256 | `config.AUTOVC_MODEL` |
| `dim_pre` | 512 | `config.AUTOVC_MODEL` |
| 输入 | `mel80 (B, T, 80)` + `embedding (B, 256)` | — |
| 输出 | `mel_out (B, T, 80)`、`mel_postnet (B, T, 80)` | 与输入**等长** ✅ |
| `c̃`（L_dis 用） | 沿时间维拼接 **4** 个 content code = **R^256** | 与论文 `W_s ∈ R^{K×256}` 对齐 |

### 7.3 三个损失（论文式 3–6）

| 损失 | 公式 | 实现 |
|---|---|---|
| `L_rec` | `‖x−x̂‖² + ‖x−x̂_psnt‖²` | `stage1/losses.py::reconstruction_loss` |
| `L_cd` | `‖E_c(x,h)−E_c(x̂,h)‖₁ + ‖E_c(x,h)−E_c(x̂_psnt,h)‖₁` | `content_consistency_loss` |
| `L_dis` | `Σ_k p_s(k|c̃) log p_s(k|c̃)` | `disentanglement_loss` |
| 总 | `L_rec + λ_cd·L_cd + λ_dis·L_dis`（λ_cd=1.0, λ_dis=0.01） | `stage1_loss` |

### 7.4 两阶段训练的顺序（★ 顺序不能反）

```
第 1 步：训练说话人分类器 W_s
    用真实说话人嵌入 h (R^256) 训一个 K 类分类器（K=80）
    产物：weights/stage1_speaker_classifier.pt
第 2 步：冻结 W_s，训练解耦-重建网络
    L_S1 = L_rec + λ_cd·L_cd + λ_dis·L_dis
    梯度回传到 Ec / D / Postnet **以及 GE2E**（论文说 GE2E 端到端微调）
    产物：weights/stage1_model.pt
```

> ⚠️ **为什么必须先训 W_s**：如果 W_s 随机初始化，熵最大化会退化成
> "让随机 W_s 输出均匀"，而不是"让内容码不含说话人信息"。
> **先训好再冻结**，这个约束才有效。

**可执行命令（唯一推荐形式，两种方式都验证过）：**

```powershell
conda activate vpp
python algorithms/voiceprint_protection/stage1/train.py --stage classifier --iterations 2000
python algorithms/voiceprint_protection/stage1/train.py --stage model --iterations 500000
```

> ★ `--resume` 从 `weights/stage1_last_ckpt.pt` 续训（云上竞价实例被抢占后用）。
> `--save-every` / `--val-every` 控制断点与验证频率。
> 用 `python -m algorithms.voiceprint_protection.stage1.train` 也能跑，但**不推荐**
> （模块路径和脚本路径混用容易出岔子）。

### 7.5 Stage 2 会用到 Stage 1 的什么

| Stage 2 需要 | Stage 1 提供 | 接口 |
|---|---|---|
| `c`（内容码） | `Stage1Model.content_codes(mel, embedding)` | list of `(B, 64)` |
| `x`（原始 mel） | 数据集直接给 | `(B, T, 80)` |
| `h`（说话人嵌入） | GE2E | `(B, 256)` |
| `x_adv = D(c, h_adv)` | `Stage1Model.forward(mel, h, h_adv)` | `(B, T, 80)` |

**Stage 2 的 `x_adv` 与 Stage 1 的 `x` 帧数必须完全相等** —— 由 `freq` 与
固定帧数裁剪共同保证（`stage1/dataset.py` 已把片段裁成 `freq` 的整数倍）。

---

## 8. 变更记录

| 日期 | 版本 | 改动 | 提出 | 同意 |
|---|---|---|---|---|
| 2026-11-17 | v0.1 | 起草：补齐三个桥接模块签名、`infer.protect()` 签名、清单格式 | 成员 B | ⬜ 待成员 A 确认 |
| 2026-11-18 | **v0.3** | **成员 A 签字并更正三处实测错误**：① §2 `c` 的形状改为 `(T//32, 64)`（官方代码实测，非论文的 `(T',32)`），`c̃` = 沿时间维拼接 4 个 content code = R^256；② §4 `infer.py` 返回键名按路线 B 的实际实现更新；③ 路线取舍落定（路线 B）。**新增 §9 Stage 1 的约定** | 成员 A | ✅ 成员 B 待确认 |
| 2026-11-17 | **v0.2** | **按实际代码重写 §1/§3/§4**：预设改为 `model16k`（取消 22.05k）、AutoVC mel 改为自定义链、`AutoVCBridge` 的方法改为 `protect`/`reconstruct`、`VocoderBridge` 改为 `synthesize`、`infer` 改为 `Protector.protect(wav, ...)`；`bridge_speaker.py` 增补路线 B 的 `SpeakerBridge` | 成员 B（合并时核对） | ⬜ 待成员 A 确认 |
| 2026-11-18 | **v0.4** | **成员 A 第二次签字（训练链路审计后）**：① §3.3 `bridge_vocoder.py` 重写为 `mel_to_wav/wav_to_mel`，内部改用**上游 WaveRNN**（HiFi-GAN 官方仓库已下架，实测 404）；② §3.4 新增**第二套模型侧 mel** `rtc16k`，`config.STAGE1["mel_preset"]` 默认改为 `rtc16k`，并新增 `audio_utils.model_mel()` 作为模型侧统一入口；③ §7.1 更新，新增 §7.1b **帧数必须是 `freq` 整数倍**的硬约定；④ §7.4 补上可执行的训练命令与 `--resume` | 成员 A | ✅ 成员 B 待确认 |
| 2026-11-19 | **v0.5** | **成员 A 第三次签字（★ 找到了 16 kHz 的 HiFi-GAN 权重，v0.4 的方案作废）**：<br>① §3.3 `bridge_vocoder.py` **改用 HiFi-GAN**（论文 §2.3 的 V(·)，**可求导**），WaveRNN 降级为 `WaveRNNBridge`（对照用）；<br>② §3.4 新增**第三套模型侧 mel** `hifigan16k`，`config.STAGE1["mel_preset"]` 默认改为它；新增 `hifigan_mel_norm` / `hifigan_mel_unnorm`（可逆仿射归一化，**目的是让论文的 λ_cd/λ_dis 仍然有效**）；<br>③ §7.1 同步更新 | 成员 A | ✅ 成员 B 待确认 |
| 2026-11-20 | **v0.5** | **成员 B 签字并收口**：(1) 新增 §4.5.3c「Stage 1 <-> Stage 2 接口」的**回归测试**（7 项断言，`stage2/verify_gradients.py`【6】段，共 19 项全过）；(2) §4.5.3 的声码器入口**收敛**为 `bridge_vocoder.VocoderBridge`（删掉重复实现 `stage2/models.py::TorchHiFiGANVocoder`）；(3) ★ 修 `paths.models_dir()` 指向不存在目录的 bug（**权重找不到 → 整条训练链起不来**，且报错误导人去重下已下好的文件）；(4) 修 `Stage2Pipeline.vocode()` 写死 mel 预设、`audio_utils.MelSpec` 写死 `autovc_mel()` | 成员 B | ✅ 成员 A 已于 v0.5 签字 |

---

## 9. 签字

| 角色 | 姓名 | 确认日期 | 备注 |
|---|---|---|---|
| 成员 A | 2026-11-18 | ✅ **已签字（v0.3）**。核对结果：<br>· §3.2 `bridge_autovc.py` 的 `protect`/`reconstruct` 签名一致（但 AutoVC 权重链 404，该路线降级为 Stage 1 架构参考）<br>· §3.3 `bridge_vocoder.py` 的 `synthesize` 签名一致（同上，WaveNet 权重不可得）<br>· §4 `infer.py` 已改为**模块级 `protect(wav_path, alpha, output_path)`**（B 的 `make_sample.py` 约定一），并保留 `Protector.protect(wav, ...)`<br>· §2 `c` 的形状**已更正**为 `(T//32, 64)`，`T' = T//2` 与官方代码不符<br>· 同意"只有 GE2E 内部的 40 维 mel 是例外" |
| 成员 A | 2026-11-19 | ✅ **已签字（v0.5）★ 这条最重要，请 B 重点核对**：<br>· §3.3 `bridge_vocoder.py` **换成 HiFi-GAN**（论文 §2.3 的 V(·)）。**B 的 `stage2/models.py::TorchHiFiGANVocoder` 用的是 `torch.hub.load("NVIDIA/torch.hifi_gan", ...)`，那个仓库已下架 → 改成加载 `algorithms/voiceprint_protection/hifigan.py`（内置的生成器）+ `data/models/default/hifigan_16k.ckpt`（HF `speechbrain/tts-hifigan-libritts-16kHz`，**234/234 键 strict 加载通过**）<br>· §3.4 **模型侧 mel 改成 `hifigan16k`**。⇒ **B 的 `stage2/pipeline.py::autovc_mel_to_hifigan()` 可以删掉了** —— 实测那个近似换算会把重建音的说话人余弦从 **0.9746 掉到 0.6983**，不可接受<br>· ★ B 的 `stage2/dataset.py` 若直接调 `audio_utils.autovc_mel()`，请改成 `audio_utils.model_mel()`<br>· §7.1b 的帧数约定对 B 也有影响：**裁片段时必须用 `N = (crop_frames-1)*hop`** |
| 成员 A | 2026-11-18 | ✅ **已签字（v0.4）**。训练链路审计后的第二次签字：<br>· §3.3 `bridge_vocoder.py` **签名变了**：`synthesize` → `mel_to_wav(mel)` / `wav_to_mel(wav)`，输入是 `rtc16k` 侧 mel（`[-1,1]`），内部是上游 WaveRNN<br>· §3.4 **模型侧 mel 改为 `rtc16k`**，理由：HiFi-GAN 官方仓库下架（实测 404），WaveRNN 只吃这套约定；**AutoVC 架构本身没变**<br>· 新增 `audio_utils.model_mel(wav)` 统一入口，请所有人**不要**再直接调 `autovc_mel`/`rtc_mel`<br>· §7.1b 的帧数约定对 B 也有影响：**如果 B 要写自己的评测脚本，裁片段时必须用 `N = (crop_frames-1)*hop`** |
| 成员 B | 2026-11-20 | ✅ **已签字（v0.5）★ 请 A 重点核对下面三条**：<br>· **§3.3 / §4.5.3 声码器**：同意换成 HiFi-GAN。B 已把 `stage2/pipeline.py::HiFiGANVocoder` 改成**包 `bridge_vocoder.VocoderBridge`**（§4.5.3 指定的唯一权重加载入口），并**删掉** `stage2/models.py::TorchHiFiGANVocoder` —— 那是同一份逻辑的第二份拷贝、全仓库没有任何地方 import 它，留着只会让两份实现慢慢跑偏<br>· **§3.4 模型侧 mel = `hifigan16k`**：同意，已生效。B 的 `stage2/dataset.py` 与 `stage1/dataset.py` 都走 `audio_utils.model_mel()`（**不是** `autovc_mel()`）✅<br>· **§7.1b 帧数**：`N = (crop_frames-1)*hop` 已按约定实现，`stage1/dataset.py` 里有断言自检 ✅<br>· **§4.5.3b 两个致命 bug**：B 已逐行对照 `third_party/AutoVC/model_vc.py::Generator.forward` 复核 A 的修法 —— 语义与官方**逐值一致**（实测最大差 2.38e-06）✅；并已加成**会失败的断言**（`stage2/verify_gradients.py` 的【6】段，见 §4.5.3c）<br>· ★ **B 顺手修了三个自己的问题（都在共同区，本次一起提交）**：<br>　① `paths.models_dir()` 硬返回 `<仓库根>/data/models`（那个目录不存在），而四个上游权重实际在 `<数据根>/models/default/` ⇒ **GE2E 权重找不到 → 整条训练链起不来**，而且报错会误导人"去重下一个已经下好的文件"。已改成候选目录列表 `paths.models_dirs()`，逐个试<br>　② `stage2/pipeline.py::Stage2Pipeline.vocode()` 把归一化系数**写死**成 `hifigan16k` ⇒ 换了预设就会送出错的 mel（出来是噪音，**而且不报错**）。已改成跟着 `config.STAGE1['mel_preset']` 走<br>　③ `audio_utils.MelSpec` 写死 `autovc_mel()` ⇒ `MelSpec('hifigan16k')` 也返回 AutoVC 口径的 mel。已改成走 `model_mel(wav, self.name)` |

> **签字前请重点核对：**
> - [x] A 的 `bridge_autovc.py` 签名与 §3.2 一致（`protect` / `reconstruct`）
> - [x] A 的 `bridge_vocoder.py` 签名与 §3.3 一致（`mel_to_wav` / `wav_to_mel`，**v0.5 换成 HiFi-GAN**）
> - [x] A 的 `infer.py` 签名与 §4 一致（`scripts/smoke_test.py` 与 `scripts/make_sample.py` 依赖它）
> - [x] `c` 的帧数确实是 `T // 32`（**不是** `T // 2`）
> - [x] 双方都同意"只有 GE2E 内部的 40 维 mel 是例外"这一条
> - [x] ~~**路线取舍**~~ → 已定：路线 B（论文的 GE2E），见 TASKS.md §0.8
> - [x] **（v0.5）** 同意模型侧 mel 默认用 `hifigan16k`（HiFi-GAN 那套）
> - [x] **（v0.5）** 同意删掉 `stage2/pipeline.py::autovc_mel_to_hifigan()` 那个近似换算
> - [x] **（v0.5）** `stage2/models.py::TorchHiFiGANVocoder` 改成加载内置生成器 + 本地 ckpt
>       → ★ B 的处理：**这个类已删除**（它与 `stage2/pipeline.py::HiFiGANVocoder`
>       是同一份逻辑的第二份拷贝，全仓库无人 import）。现在**只留一个**实现，
>       它内部包 `bridge_vocoder.VocoderBridge`（§4.5.3 指定的唯一入口），
>       两者加载**同一个** checkpoint ⇒ 不会出现"两份实现慢慢跑偏"。
> - [x] **（v0.5）** 双方都同意：Stage 1 训练时，模型侧 mel 的入口统一是
>       `audio_utils.model_mel()`（不要再直接调 `autovc_mel` / `rtc_mel` / `hifigan_mel`）
