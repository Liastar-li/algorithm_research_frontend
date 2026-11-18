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
| 2 | AutoVC 侧 mel | **不是 librosa 默认链**：30 Hz 高通 → reflect 幅度 STFT → `mel(fmin=90, fmax=7600, n_mels=80, n_fft=1024, hop=256)` → `20·log10(max(1e-5,·)) - 16` → `clip((x+100)/100, 0, 1)` | `audio_utils.autovc_mel()` |
| 3 | 预设 | **`model16k`**（唯一预设，16 kHz）。⚠️ 原 `paper16k` / `autovc22k` 已取消 | `config.MEL_PRESETS` |

> **规则**：**任何地方都不许再手写 mel 参数**，一律调 `audio_utils` 里对应的函数。
> 代码里**一律用 `config.DEFAULT_PRESET` 符号引用**，别写死预设名字。
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
| `c` | 内容码 | `(T', 32)` | float32 | — | `T' = T // 2`（AutoVC 瓶颈降采样） |
| `h` | 说话人嵌入 | `(256,)` | float32 | **L2 归一化** | 由 `bridge_speaker.embed()` 出口保证 |
| `δ` | 扰动 | `(256,)` | float32 | — | Stage 2 生成器 `G(h)` 的输出 |
| `h_adv` | 被污染的嵌入 | `(256,)` | float32 | 继承 h | `h_adv = h + α·δ` |
| `x_adv` | 保护后 mel | `(T, 80)` | float32 | — | `x_adv = D(c, h_adv)` |
| `wav` / `wav_adv` | 波形 | `(samples,)` | float32 | — | `wav_adv = V(x_adv)`，16 kHz |
| `δ_wav` | 波形扰动 | `(samples,)` | float32 | — | `δ_wav = wav_adv − wav`，供 `L_mask` 用 |

**形状不变量**：`T' = T // 2`；`x` 与 `x_adv` 的帧数必须**完全相等**。
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

> ✅ **路线取舍已定：走路线 B**（论文的 GE2E）。`infer.py` 现在用的就是 `SpeakerBridge`。
> `SpeakerEncoderBridge`（路线 A）保留在文件里，供 AutoVC 权重链那条备选路线使用。
> 详见 §4.4。

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

### 3.3 `bridge_vocoder.py`（成员 A）

```python
from bridge_vocoder import VocoderBridge

bridge = VocoderBridge(checkpoint: str | Path | None = None, device: str | None = None)

wav = bridge.synthesize(mel)  # (T, 80) 的 AutoVC mel -> 一维波形
```

- 输入是 **(T, 80)** 的 **AutoVC 侧 mel**（由 `audio_utils.autovc_mel()` 产出）。
- 内部走 **wavenet_vocoder**（与 AutoVC 配套），不是 HiFi-GAN。
- 输出是一维波形，采样率与 AutoVC 一致（16 kHz）。

### 3.4 `audio_utils.py` 的 mel 约定（⚠️ 已变更）

| 用途 | 函数 | 约定 |
|---|---|---|
| AutoVC 侧（模型 + 声码器） | `audio_utils.autovc_mel(wav)` | 自定义链：30 Hz 高通 → reflect 幅度 STFT → mel(fmin=90) → dB → clip[0,1] |
| 通用 librosa 版（对照用） | `audio_utils.librosa_mel(wav)` | librosa 原生 mel，用于数值对照 |
| 声码器侧（HiFi-GAN 约定） | `audio_utils.vocoder_mel(wav)` | mel 基 sr=8000 + `(x+4.5)/5` 归一化 |
| 通用预设 | `audio_utils.MelSpec(name).mel(wav)` | 走 `config.MEL_PRESETS["model16k"]` |

> ★ **预设名变了**：原 `paper16k` / `autovc22k` → 现 `model16k`（唯一预设）。
> **AutoVC 官方权重本身就是 16 kHz 的**，不需要 22.05 kHz 预设。
> 代码里请一律用 `config.DEFAULT_PRESET` 符号引用，别写死字符串。


## 4. 推理主流程 `infer.py`（成员 A 主责，B 的脚本依赖它）

> ⚠️ 下面是**按 2026-11-18 合并后的实际代码**写的。

### 4.1 模块级接口（`make_sample.py` 走这条）

```python
from infer import protect

result = protect(
    wav_path,                       # 输入音频路径
    alpha=0.10,
    output_path=None,               # 保护后 wav 的输出路径（None 则不写文件）
    text="...",                     # 攻击者要念的参考文本
    attack_mode="clone",            # "clone" 额外产出攻击者克隆音；"none" 只产保护后语音
    seed=42,
)
# 返回 dict，至少含 'protected_path' 与 'shapes'
```

### 4.2 类接口（逐步调试用）

```python
from infer import Protector, perturb_embedding, random_perturbation

protector = Protector(use_speaker_bridge=True, with_cloner=True)
result = protector.protect(wav, alpha=0.10, seed=42, text=..., attack_mode="clone")
# result 里至少有：
#   "embedding"                (256,) 原始说话人嵌入
#   "embedding_adv"            (256,) 加扰动后的嵌入
#   "protected_waveform"       (N,)  保护后语音
#   "clone_baseline_waveform"  (N,)  未防御时攻击者的克隆音   ← attack_mode="clone" 时才有
#   "clone_protected_waveform" (N,)  防御后攻击者的克隆音     ← 同上
#   "stolen_embedding"         (256,) 攻击者从保护后语音里重新提的卡
#   "shapes"                   各步张量形状
```

### 4.3 ★ L1 走的是"克隆机路线"（不是 AutoVC 重建）

```text
wav ─► GE2E ─► h ─► h_adv = h + α·δ
                  │
                  └─► 克隆机.clone(text, h_adv) ─► 保护后语音 protected.wav
                                                          │
攻击者行骗 ────────────────────────────────────────────────┘
  baseline = clone(text, h)            未防御时攻击者克隆音（很像本人）
  stolen   = GE2E(protected)           攻击者从保护后语音重新提卡
  cloned   = clone(text, stolen)       防御后攻击者克隆音（应不像本人）
```

- 这条路线**不需要 AutoVC 官方权重**（实测 IBM Box 链接已 404 失效）。
- 需要上游 RTC 的三个权重：`encoder.pt`(17 MB) + `synthesizer.pt`(371 MB) + `vocoder.pt`(54 MB)，
  全部在 HuggingFace `CorentinJ/SV2TTS`。
- **`Sim_clone` 的语义就是 `cos(h, stolen)`** —— 越小说明防御越有效。
- AutoVC 路线（`bridge_autovc.protect`）保留在 `protect_mel()` 里**按需加载**，
  等 Stage 1 训完自己的模型再用。

### 4.4 路线取舍：**已解决**

原 §3.1 的两条路线（AutoVC 自带 D_VECTOR vs 论文的 GE2E）**已经收敛到论文路线**：
`infer.py` 现在用的是 `SpeakerBridge`（上游 GE2E），不再用 `SpeakerEncoderBridge`。
`SpeakerEncoderBridge` 仍保留在 `bridge_speaker.py` 里，供 AutoVC 权重链那条备选路线使用。

- `scripts/smoke_test.py` 与 `scripts/make_sample.py` 都按 §4.1 调用；
  `make_sample.py` 兼容两种形态（模块级 `protect(...)` 或 `Protector.protect(...)`）。
  **签名再变就要同步改这两个脚本。**

### 4.5 ★ Stage 2 的接口与形状（2026-11-18 新增，成员 B）

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
| `wav_adv` | 保护后波形 | `(B, N)` | `N = T · 256` |
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

#### 4.5.3 可求导组件放哪（不改动 A 的桥接层）

| 需求 | 用哪个 | 为什么不用现成的 |
|---|---|---|
| `E_spk`（可导） | `stage2/models.py::TorchSpeakerEncoder` | `bridge_speaker.SpeakerBridge` 是 numpy 封装 |
| Stage 1 解码器（可导） | `stage2/models.py::TorchAutoVCStage1` | `bridge_autovc.AutoVCBridge` 内部 `no_grad()` |
| 声码器（可导） | `stage2/pipeline.py::HiFiGANVocoder` | `bridge_vocoder.VocoderBridge` 内部 `no_grad()` |

> **两套实现并存是刻意的**：`bridge_*` 供推理链（L1/平台）用，
> `stage2/models.py` 供训练用。两者加载**同一个** checkpoint，结构一致。
> `stage2/verify_mel.py` 保证 torch 版 GE2E 与上游 numpy 版**逐值一致**
> （mel 误差 2.3e-05，嵌入余弦 0.99999481）。

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
| `WER` ↓ | Whisper 转写 vs 标准转写 | 论文 5.30% |

> ⚠️ `docs/interface-contract.md` §4.3 里说的 "`Sim_clone` 的语义就是 `cos(h, stolen)`"
> **与论文口径不同** —— 那个量其实等于本节的 `Sim_prot`。
> `evaluate.py` 以**论文口径**为主，同时报 `sim_clone_no_defense` 作对照。
> **A 请确认这一条的口径**（见 TASKS.md §9 待决事项）。

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

## 7. 变更记录

| 日期 | 版本 | 改动 | 提出 | 同意 |
|---|---|---|---|---|
| 2026-11-17 | v0.1 | 起草：补齐三个桥接模块签名、`infer.protect()` 签名、清单格式 | 成员 B | ⬜ 待成员 A 确认 |
| 2026-11-17 | **v0.2** | **按实际代码重写 §1/§3/§4**：预设改为 `model16k`（取消 22.05k）、AutoVC mel 改为自定义链、`AutoVCBridge` 的方法改为 `protect`/`reconstruct`、`VocoderBridge` 改为 `synthesize`、`infer` 改为 `Protector.protect(wav, ...)`；`bridge_speaker.py` 增补路线 B 的 `SpeakerBridge` | 成员 B（合并时核对） | ⬜ 待成员 A 确认 |
| 2026-11-18 | **v0.3** | **按 A 的新链路重写 §4**：L1 改为「克隆机路线」（上游合成器+声码器，不用 AutoVC 权重）；`infer.protect()` 有模块级接口；新增 `bridge_cloner.py`；**路线取舍已定 —— 用路线 B（论文 GE2E）** | 成员 B（合并时核对） | ⬜ 待成员 A 确认 |
| 2026-11-18 | **v0.4** | **新增 §4.5 Stage 2 的接口与形状**；写明「冻结 ≠ 断梯度」、可求导组件的分工、损失函数签名、评测指标口径 | 成员 B | ⬜ 待成员 A 确认 |

---

## 8. 签字

| 角色 | 姓名 | 确认日期 | 备注 |
|---|---|---|---|
| 成员 A | | | |
| 成员 B | | | |

> **签字前请重点核对：**
> - [ ] A 的 `bridge_autovc.py` 签名与 §3.2 一致（`protect` / `reconstruct`）
> - [ ] A 的 `bridge_vocoder.py` 签名与 §3.3 一致（`synthesize`）
> - [ ] A 的 `infer.py` 签名与 §4 一致（`scripts/smoke_test.py` 与 `scripts/make_sample.py` 依赖它）
> - [ ] `c` 的帧数确实是 `T // 2`
> - [ ] 双方都同意"只有 GE2E 内部的 40 维 mel 是例外"这一条
> - [x] **路线取舍**：已定为**路线 B（论文的 GE2E）** —— `infer.py` 已改用 `SpeakerBridge`
> - [ ] A 的 `bridge_cloner.py`（攻击者克隆机）依赖的三个上游权重下载方式与 §4.3 一致
> - [ ] `make_sample.py` / `smoke_test.py` 在 A 改签名后仍能跑通（跑一次确认）
> - [ ] **§4.5.5 的 Sim_clone 口径**：以论文口径（克隆音 vs 原说话人）为准？
> - [ ] **§4.5.2 的冻结约定**：Stage 2 用 `stage2/models.py` 的可求导组件，
>       不改动 `bridge_autovc` / `bridge_vocoder`（推理链继续用它们）
> - [ ] **§4.5.4 的 `defense_loss` 是最小化**（论文式 12 的 "Minimizing L_de"）
