# GE2E 说话人编码器 · 接口约定（任务板 W2-D2-4）

> **主责**：成员 B ｜ **被依赖方**：成员 A（Stage 1 训练）、成员 A（Stage 2 的 `L_de`）
> 组件：`third_party/RTC`（CorentinJ/Real-Time-Voice-Cloning），**论文脚注 2 直接引用的仓库**
> 已核实版本：commit `890f3a03187195b9829db2079b75c2ba2ab0405c`（2026-03-09，MIT）
>
> 本文件回答任务板 D2-4 的四个问题：
> ① 采样率是多少？② 内部 mel 是几维、什么参数？③ 输出 `h` 是否已 L2 归一化？④ 输出形状？

---

## 0. 一句话结论

> GE2E 只吃 **16 kHz**，内部用 **40 维 mel**（`n_fft=400, hop=160, fmin=0, fmax=8000`，
> **不取 log、不做均值方差归一化**，直接用线性 mel），
> 输出 **256 维**且**上游已经做过 L2 归一化**。
>
> ⚠️ **我们自己的 80 维 mel（`audio_utils`）和 GE2E 内部的 40 维 mel 是两套并行的约定，
> 不要混用。** 上游换版本时这两套约定都变过，详见第 3 节。

---

## 1. 结论速查表（**已按源码逐条核实 ✅**）

| # | 问题 | 答案 | 上游出处 |
|---|---|---|---|
| ① | 输入采样率？ | **16000 Hz** | `encoder/params_data.py` → `sampling_rate = 16000` |
| ② | 内部 mel？ | **40 维**；`n_fft=400`（窗长 25 ms）、`hop=160`（步长 10 ms）、`fmin=0`、`fmax=8000`（librosa 默认，即 sr/2）；**线性 mel，不取 log、不做均值方差归一化** | `encoder/params_data.py`（`mel_n_channels=40` / `mel_window_length=25` / `mel_window_step=10`）+ `encoder/audio.py::wav_to_mel_spectrogram` |
| ③ | 输出 `h` 是否已 L2 归一化？ | **是**。`encoder/model.py::forward` 内部就做了 `embeds = embeds_raw / (‖embeds_raw‖ + 1e-5)`；`embed_utterance` 返回前又归一化一次 | `encoder/model.py` 第 58–59 行；`encoder/inference.py` 第 150 行 |
| ④ | 输出形状？ | **`(256,)`**（单条语音） | `encoder/params_model.py` → `model_embedding_size = 256`；`embed_utterance` 返回 `(model_embedding_size,)` |

**结构**：`LSTM(40 → 256, 3 层)` → `Linear(256 → 256)` → `ReLU` → **L2 归一化**
（`encoder/params_model.py` + `encoder/model.py`）

---

## 2. 详细说明

### 2.1 输入侧（`encoder/audio.py::preprocess_wav`）

| 步骤 | 做法 | 出处 |
|---|---|---|
| 读文件 | `librosa.load(sr=None)`，支持 wav/flac/mp3 等 | `encoder/audio.py` |
| 重采样 | 统一到 **16 kHz** | `preprocess_wav` |
| 音量归一化 | 目标 **-30 dBFS**，**只调大不调小**（`increase_only=True`） | `normalize_volume` |
| 静音裁剪 | webrtcvad VAD，裁掉长静音 | `trim_long_silences` |

> ⚠️ **webrtcvad 是可选的**：上游用 `try/except` 包着，缺了会 `warn` 并跳过静音裁剪，
> **不影响能否跑通**。我们没把它写进 `requirements.txt`，避免 Windows 编译问题。
> 不装的话嵌入质量略有差别，但推理链完全正常。

> ℹ️ **新版没有 preemphasis**。旧版 resemblyzer 会在 `embed_utterance` 里做 `preemphasis(0.97)`，
> 新版**不做了**。别自己加 —— 加了就和上游训练时的分布对不上。

### 2.2 内部 mel

```python
# encoder/audio.py::wav_to_mel_spectrogram 的全部内容
frames = librosa.feature.melspectrogram(
    wav, sampling_rate,
    n_fft=int(sampling_rate * mel_window_length / 1000),   # 16000*25/1000 = 400
    hop_length=int(sampling_rate * mel_window_step / 1000),  # 16000*10/1000 = 160
    n_mels=mel_n_channels,                                  # 40
)
return frames.astype(np.float32).T                          # (T, 40)
```

| 参数 | GE2E 内部 | 我们（AutoVC 侧 mel） | 一样吗 |
|---|---|---|---|
| 采样率 | 16000 | 16000 | ✅ |
| 维数 | **40** | **80** | ❌ |
| n_fft | 400 | 1024 | ❌ |
| hop | 160 | 256 | ❌ |
| fmin / fmax | 0 / 8000 | 90 / 7600 | ❌ |
| 幅度 | 功率谱（`power=2` 默认），**不取 log** | `power=2` → `log10` | ❌ |
| 归一化 | 无（只有我们外部的 L2） | 无 | ✅ |

> **规则**：**别动 GE2E 内部的 40 维 mel**，也别用我们自己的 `MelSpec` 去喂它。
> 桥接层 `bridge_speaker.py` **直接调用上游的 `wav_to_mel_spectrogram`**，
> 不复制它的实现 —— 这样上游改了我们也不会跟着错。

### 2.3 输出侧

```python
# encoder/model.py::forward 的最后两行
embeds_raw = self.relu(self.linear(hidden[-1]))
embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)   # ← 已归一化

# encoder/inference.py::embed_utterance 的最后两行
raw_embed = np.mean(partial_embeds, axis=0)
embed = raw_embed / np.linalg.norm(raw_embed, 2)                             # ← 又归一化一次
```

**分片（partial）机制**：`embed_utterance` 默认 `using_partials=True`，
把整条语音切成 **160 帧（1.6 秒）**的片段、**重叠 50%**，逐片编码后取平均再归一化。
> 这一整套逻辑**我们不重写**，直接调上游的 `embed_utterance`。

**所以我们的约定：**

| 约定 | 内容 | 出处 |
|---|---|---|
| 接口契约第 5 条 | `h` 一律 **L2 归一化**后再往下传 | 指南第 8 章 §8.5 |
| 落点 | `bridge_speaker.embed()` 出口统一 `audio_utils.normalize_l2()` | `bridge_speaker.py` |
| 为什么再归一化一次 | 上游已经归一化过，我们再归一化是**幂等操作**（不改数值），只是**兜底**：万一换回旧版上游（不归一化），行为依然一致 | `bridge_speaker.py` |

> ⚠️ **成员 A 写 Stage 1 时**：从 `bridge_speaker` 拿到的 `h` **已经是归一化过的**，
> 别再归一化第三次，也别自己加 `preemphasis`。

---

## 3. ★ 版本差异（最容易踩的坑）

上游在 **2026-03** 做过一次大重构，**目录结构、权重来源、mel 约定全变了**：

| 项 | 旧版（2026-03 之前） | 新版（**当前上游 master**） |
|---|---|---|
| 目录 | 平铺 `speaker_encoder.py` / `voice_encoder.py` / `preprocess.py` | 包结构 `encoder/`（model / inference / audio / params_data / params_model） |
| 权重 | 仓库内 `pretrained.pt`（~30 MB） | **不再随仓库分发**，从 HuggingFace 下载 `CorentinJ/SV2TTS` 的 `encoder.pt`（**17,090,379 字节**） |
| checkpoint 结构 | 裸 `state_dict` | `{"model_state": ..., "step": ...}` |
| mel 取 log | **是**（`power_to_db(power=1.97)` → `log10` → 减均值除方差） | **否**（线性 mel） |
| preemphasis | 有（0.97） | **无** |
| 模型内 L2 归一化 | 无 | **有** |
| 依赖 | librosa / torch | 额外需要 **scikit-learn**（`encoder/model.py` import 时就要） |

**⚠️ 最要命的一条：mel 取不取 log 变了。**
拿旧版的 mel 代码去喂新版的权重 → 嵌入完全不对，
但**不会报错**，只会让相似度看起来"都不像"，非常难查。

> `bridge_speaker.py` **同时支持两种布局**，并按权重文件的实际结构自适应，
> 永远调用**当前布局对应的那套上游代码**，不会混用。

### 3.1 已知坑：`np.int`（只在**训练**路径上炸）

上游 `encoder/model.py` 第 88 行与第 127 行用了 `np.int`：

```python
mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)   # ← np.int
```

**`np.int` 在 numpy 1.24 就被移除了**，我们锁的是 `numpy==1.26.4` → 会直接 `AttributeError`。

| 路径 | 会不会踩到 | 说明 |
|---|---|---|
| **推理**（我们现在的 `embed_utterance`） | ❌ 不会 | 只调用 `forward()`，不经过 `similarity_matrix()` / `loss()` |
| **训练 / 微调 E_spk**（指南第 5 章 5.4 步） | ⚠️ **会炸** | `loss()` → `similarity_matrix()` → `np.int` |

**如果成员 A 要微调 E_spk**，两条路选一条：

1. **改上游那两行**（`np.int` → `int`）—— 一行改动，语义完全等价
   （`np.eye(..., dtype=int)` 就是整数矩阵）；
2. 或者**在导入前打补丁**：在 `sys.modules` 里预先塞一个 `np.int = int`，
   属于绕过上游、不改第三方代码 —— 但可读性差，不如直接改。

> 结论：**推理不受影响，可以先跑通**；真要微调时记得处理。
> 这个坑只在训练路径上，不影响 M1 里程碑。

### 3.2 已知坑：上游 `load_model` 的两个参数"类型要求相反"

上游签名：`load_model(weights_fpath: Path, device=None)`。两个参数各有各的坑：

```python
# 上游 encoder/inference.py::load_model
if device is None:                  _device = torch.device("cuda" if cuda else "cpu")
elif isinstance(device, str):      _device = torch.device(device)
# ← 传 torch.device 对象时，两个分支都不进，_device 保持模块级的 None
checkpoint = torch.load(weights_fpath, _device)   # map_location=None !
_model.load_state_dict(checkpoint["model_state"])
print("Loaded encoder \"%s\" trained to step %d" % (weights_fpath.name, checkpoint["step"]))
#                                                          ↑ 要 Path 对象的 .name
```

| 参数 | 传错会怎样 |
|---|---|
| `weights_fpath` 传 **str** | `AttributeError: 'str' object has no attribute 'name'` |
| `device` 传 **torch.device 对象** | `_device` 静默保持 `None` → `map_location=None` → 试图把**存于 CUDA 的 checkpoint 反序列化回 CUDA**，CPU 版 torch 上报 `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False` |

**正确调用**：`load_model(Path(...), "cpu")` —— **路径用 Path，设备用字符串**。

> 顺带说明：上游 `demo_cli.py` 调 `load_model(path)` 不传 device，走的是 `device is None`
> 分支，所以它自己不会碰到第二个坑 —— **只有我们这种"自己指定设备"的调用方式才会中招**。
> 这两个坑都是 `bridge_speaker.py` 已经处理掉的，不用再改上游。

### 3.3 已知坑：上游按 librosa **0.9.2** 写，我们用 **0.10.2**

上游的 `pyproject.toml` 锁的是 `librosa==0.9.2`，我们的 `requirements.txt` 锁的是
`librosa==0.10.2.post1`。**librosa 0.10 把很多函数改成了"仅关键字参数"**，
上游（2026-03 那个 commit 只修了 `librosa.load`）因此踩到两处：

| 上游写法 | librosa 0.10 的签名 | 后果 |
|---|---|---|
| `librosa.resample(wav, source_sr, sampling_rate)` | `resample(y, *, orig_sr, target_sr, ...)` | `TypeError` |
| `librosa.feature.melspectrogram(wav, sampling_rate, n_fft=..., ...)` | `melspectrogram(*, y=None, sr=22050, ...)` | `TypeError` |

**我们的处理（都已在 `bridge_speaker.py` 里做掉，不用改上游）：**

1. **重采样自己做**：`bridge_speaker.embed()` 里先用 `audio_utils.resample`（soxr 高质量）
   把音频转成 16 kHz，再调上游的 `preprocess_wav(wav, source_sr=16000)`，
   于是上游那个 `if source_sr != sampling_rate` 分支**根本不会进**。
2. **mel 函数换掉**：`patch_upstream_mel()` 用**数值完全等价**的实现替换上游的
   `wav_to_mel_spectrogram`（同样的 librosa 默认值，同样的 n_fft/hop/n_mels，
   且参数是**从上游 `params_data` 现场读**的，不写死），所以结果与上游本意一致。

> 我们自己的 `audio_utils.py` **不受影响** —— 里面的 librosa 调用都已按 0.10 写成关键字形式
> （`trim` / `stft` 的 `y` 在 0.10 里仍可位置传，`filters.mel` 我们本来就用关键字）。

---

## 4. 怎么自证（把结论再确认一遍）

```bash
conda activate vpp
cd "c:/Users/z/Desktop/26秋/语音识别/语音识别code"

# ① 只检查接口约定 + 权重能不能加载（不需要音频）
python algorithms/voiceprint_protection/bridge_speaker.py

# ② 带上真实音频，验证嵌入
python algorithms/voiceprint_protection/bridge_speaker.py \
    --wav data/audio16k/p225/p225_001.wav data/audio16k/p292/p292_001.wav
```

预期输出（形态）：

```text
【接口约定（运行时从上游代码读出）】
  上游布局: 新版（encoder/ 包）
  采样率 (Hz): 16000
  mel 维数: 40
  mel n_fft / hop: 400 / 160
  mel 是否取 log: 否（新版直接用线性 mel，见 encoder/audio.py）
  嵌入维度: 256
  上游输出是否已 L2 归一化: 是（encoder/model.py::forward 内部显式归一化）

【嵌入测试】
  p225_001.wav    h.shape = (256,)  |h| = 1.000000
  p292_001.wav    h.shape = (256,)  |h| = 1.000000
```

**判读：**

| 现象 | 结论 |
|---|---|
| `上游布局` 显示"新版"且 `mel 是否取 log: 否` | 与本文档一致 ✅ |
| `上游布局` 显示"旧版" | 说明克隆到的是旧 commit，查 `git -C third_party/RTC log -1` |
| `\|h\|` 不等于 1.000000 | 权重没加载对 → 查权重文件大小是否正好 17,090,379 字节 |
| `ImportError: sklearn` | 装 `scikit-learn==1.5.2` |
| 上游打印 `Loaded encoder "encoder.pt" trained to step N` | 权重加载成功 ✅ |

**关于余弦相似度的判读（重要，别误判）**

| 场景 | 期望 |
|---|---|
| 同一人说**不同**句子 | 通常 > 0.7 |
| 不同人说**不同**句子 | 通常 < 0.5 |
| 不同人说**同一句话** | **可能到 0.6 ~ 0.8** ← 正常 |

> 原因：GE2E 嵌入里**内容（说了什么）占了很大比重**。
> 两个人读**同一段文字**时，音素序列完全一致 → 嵌入相似度天然偏高。
> 上游自带的 `p240_00000.mp3` / `p260_00000.mp3` 正是这种情况
> （SV2TTS 的标准参考音频，所有说话人读同一段 passage，且都只有 ~1.1 秒）。
> 所以**不能拿这两个文件当"不同人应该 < 0.5"的判据** ——
> 要做这种判读，得用**同一说话人的不同句子** vs **不同说话人的不同句子**。

**实测记录（成员 B · 2026-11-17 · ✅ D2-4 收口）：**

```text
Loaded encoder "encoder.pt" trained to step 1564501
  [已适配] 上游的 mel 计算改为 librosa 0.10 兼容写法（数值等价）。

【接口约定（运行时从上游代码读出）】
  仓库目录: D:\vpp\third_party\RTC
  上游布局: 新版（encoder/ 包）
  权重文件: D:\vpp\data\models\default\encoder.pt
  权重大小: 16.30 MB
  运行设备: cpu
  采样率 (Hz): 16000
  mel 维数: 40
  mel 窗长 (ms): 25
  mel 步长 (ms): 10
  mel n_fft / hop: 400 / 160
  mel 是否取 log: 否（新版直接用线性 mel，见 encoder/audio.py）
  分片帧数 / 重叠: 160 / 0.5
  音量归一化目标: -30 dBFS
  嵌入维度: 256
  LSTM 隐层/层数: 256 / 3
  上游输出是否已 L2 归一化: 是（encoder/model.py::forward 内部显式归一化）

【嵌入测试】
  p240_00000.mp3               h.shape = (256,)  |h| = 1.000000
  p260_00000.mp3               h.shape = (256,)  |h| = 1.000000

  Q3 输出是否已 L2 归一化：实测模长 = 1.000000（应为 1.000000）
  两段音频的余弦相似度 = 0.6858
```

**结论**：四个问题的答案**全部与本文档第 1 节一致**，权重加载正常，嵌入已归一化。
相似度 0.6858 属正常（见上面的"同一句话"说明），**不是**权重或 mel 出问题的信号。

> 环境备注：本机 torch 是 **CPU 版**（`torch.cuda.is_available() == False`）。
> 推理不需要 GPU（GE2E 处理 1 秒音频在 CPU 上是毫秒级），符合环境文档 §2.3 的预期状态。

---

## 5. 与本项目其他部分的关系

```text
                ┌──────────── 我们的 80 维 mel（AutoVC 链）────────────┐
                │  用于：L_mel / L_cd / AutoVC 解码器 / wavenet 声码器  │
                │  真源：algorithms/voiceprint_protection/audio_utils.py│
                └───────────────────────┬──────────────────────────────┘
                                        │
  wav (16k) ──► [GE2E 内部 40 维线性 mel] ──► h (256) ──► L2 归一化 ──► h
                                        │
                                        └──► h_adv = h + α·δ ──► Stage 2
```

| 谁用 h | 用途 | 备注 |
|---|---|---|
| 成员 A（Stage 1） | 解码器 `D(c, h)` 的说话人条件 | 用归一化后的 h |
| 成员 B（Stage 2） | `h_adv = h + α·δ`；`L_de = cos(E_spk(wav_clone), E_spk(wav))` | α = 0.10（论文值） |
| 评测 | `Sim_prot` / `Sim_clone` / `DSR` | 全部是余弦相似度 |

---

## 6. 权重与依赖清单

| 项 | 值 |
|---|---|
| 权重来源 | <https://huggingface.co/CorentinJ/SV2TTS> → `encoder.pt` |
| 期望大小 | **17,090,379 字节**（上游 `utils/default_models.py` 里写死的） |
| 下载脚本 | `python algorithms/voiceprint_protection/scripts/download_encoder_weights.py` |
| 国内加速 | `set HF_ENDPOINT=https://hf-mirror.com` |
| 存放位置 | `<VPP_MODELS_DIR>/default/encoder.pt`，默认 `<数据根>/models/default/` |
| 必需依赖 | `scikit-learn==1.5.2`（已加进 `requirements.txt`） |
| 可选依赖 | `webrtcvad==2.0.10`（静音裁剪，缺失不影响运行） |

---

## 7. 待确认清单

- [ ] 跑第 4 节的命令①，确认权重能加载、接口信息与本文档一致
- [ ] 跑第 4 节的命令②，把实测输出贴进第 4 节
- [ ] 成员 A 确认：Stage 1 里 `h` 直接用桥接层输出，不再二次归一化、不加 preemphasis
- [ ] 在 `third_party/README.md` 里把 RTC 的 commit 填上（已核实：`890f3a0`）
- [ ] 两人 `pip install scikit-learn==1.5.2` 后各自跑一次 `check_env.py` 比对指纹
