# 实验记录 · 语音声纹主动防御复现

> **规矩**：**实验记录与代码同 commit 提交**。
> 凡是"跑过了"的东西，必须在这里留下命令 + 结果；没跑过的就写"未做"，不要留想象。
>
> 本文件是共同维护的。W5–W6 的指标表、消融表也会写在这里。

---

## 0. 目标与当前档位

| 项 | 值 |
|---|---|
| 论文 | Imperceptible Voiceprint Protection via Human-Machine Perception Discrepancy Feature Disentanglement（Interspeech 2026, pp. 5259–5263, DOI: 10.21437/Interspeech.2026-105） |
| 复现方式 | **基于公开组件自建**（AutoVC / GE2E / HiFi-GAN），**未使用作者代码**（论文未提供官方代码链接） |
| 当前档位 | **L1**（推理链跑通）→ 目标 L3 |
| 当前里程碑 | **M1**（L1 推理链跑通）—— 差成员 A 的 `infer.py` |
| 组号 | 待定（分支 `group_N005_submission`，方法 ID 暂用 `group_n005_voiceprint_protection`，**必须全小写**，见平台契约） |

---

## 1. 环境

| 项 | 值 |
|---|---|
| conda 环境 | `vpp`（Python 3.9） |
| torch | `2.4.1`（**CPU 版**，本机推理足够；训练放云上 GPU） |
| 数据根 | `D:\vpp\data`（由 `VPP_DATA_ROOT` 指定；仓库留在 C 盘） |
| 第三方组件根 | `D:\vpp\third_party`（由 `VPP_RTC_DIR` 指定） |
| 平台测试 | `npm test` 6/6 通过；`npm run build` 通过 |

> 环境指纹：待两位成员装齐依赖后重跑 `scripts/check_env.py` 比对
> （指纹算法已加入 `sklearn`，与旧指纹不可比）。

---

## 2. W2 已完成的步骤（2026-11）

### 2.1 GE2E 说话人编码器接口核实（D2-4 / D3-4）✅

```bash
python algorithms/voiceprint_protection/bridge_speaker.py \
  --wav D:\vpp\third_party\RTC\samples\p240_00000.mp3 \
         D:\vpp\third_party\RTC\samples\p260_00000.mp3
```

实测输出（节选）：

```text
Loaded encoder "encoder.pt" trained to step 1564501
  上游布局: 新版（encoder/ 包）
  采样率 (Hz): 16000
  mel 维数: 40
  mel n_fft / hop: 400 / 160
  mel 是否取 log: 否（新版直接用线性 mel，见 encoder/audio.py）
  分片帧数 / 重叠: 160 / 0.5
  嵌入维度: 256
  上游输出是否已 L2 归一化: 是（encoder/model.py::forward 内部显式归一化）
  p240_00000.mp3   h.shape = (256,)  |h| = 1.000000
  p260_00000.mp3   h.shape = (256,)  |h| = 1.000000
  两段音频的余弦相似度 = 0.6858
```

**结论**：四个问题的答案与 [`notes/ge2e_interface.md`](notes/ge2e_interface.md) 第 1 节完全一致。
相似度 0.6858 属正常（两条是**同一段文字**的不同说话人读音，GE2E 嵌入里内容占比很大）。

> 完整接口约定、版本差异、踩过的坑 → [`notes/ge2e_interface.md`](notes/ge2e_interface.md)

### 2.2 VCTK 数据下载与划分（D1-4 / D1-5 / D2-5 / D3-5）✅

```bash
node scripts/download_vctk.mjs --out D:\vpp\data\raw --per-speaker 100 --concurrency 48
python algorithms/voiceprint_protection/scripts/survey_vctk.py
python algorithms/voiceprint_protection/scripts/prepare_data.py
```

数据实况：

| 项 | 值 |
|---|---|
| 说话人 | **110**（109 个 `pXXX` + **1 个 `s5`**）—— 与论文一致 |
| 语句数 | 11,000（每人 100 句，**均匀抽样**，不是取前 100 句） |
| 原始格式 | 48 kHz / 单声道 / FLAC（`wav48_silence_trimmed`） |
| 划分 | seed=42 → **80 训练 / 10 验证 / 20 测试** |
| 清单 | `train.txt` 8,004 行 / `val.txt` 1,000 行 / `test.txt` 2,000 行 / `spk2id.txt` 80 行 |
| 划分名单 | `data/split_speakers.json`（同 seed 可完全复现） |

三条硬要求自检结果：

```text
  [通过] 三份清单都不含 p*_eval 加噪说话人
  [通过] 训练 / 验证 / 测试说话人两两不相交
  [通过] train.txt 的说话人与划分名单一致（80 个）
  [通过] val.txt 的说话人与划分名单一致（10 个）
  [通过] test.txt 的说话人与划分名单一致（20 个）
  [通过] spk2id.txt 的 K = 80，ID 连续覆盖 0..79
  [通过] 抽查 20 条测试音频：16000 Hz / 单声道 / PCM_16
```

### 2.3 平台样例输入（D4-4）✅

```bash
python algorithms/voiceprint_protection/scripts/make_sample.py
```

```text
  输入样例 : public/samples/group_n005_voiceprint_protection/input.wav
  来源     : audio16k/p228/p228_001_mic1.wav（说话人 p228，来自测试集）
  [提示] infer.py 还不存在（成员 A 主责），跳过生成输出
```

校验：`input.wav` = RIFF/WAVE、单声道、16000 Hz、16-bit、2.66 s。
来源与许可见同目录 `SOURCES.md`（**CC BY 4.0，必须署名**）。

---

## 3. 与论文的差异（README 必须写明）

| 项 | 论文 | 本组 | 说明 |
|---|---|---|---|
| 训练语句数 | 每人全部（~400–800 句） | **每人 100 句**（均匀抽样） | 为把 10.94 GiB 的下载压到 1.36 GiB / 约 55 分钟 |
| 说话人数 | 110 | **110** | ✅ 一致 |
| 划分 | 80/10/20 | 80/10/20 | ✅ 一致（seed=42） |
| 采样率 | 16 kHz | 16 kHz | ✅ 一致 |
| Stage 1 迭代 | 500k | 待定 | 算力受限，按分档缩减 |
| Stage 2 迭代 | 200k | 待定 | 同上 |
| MOS | 50 人 | 待定 | 课程作业按实际人数写明 |
| 官方代码 | — | **无**（论文未提供） | 基于 AutoVC / GE2E / HiFi-GAN 自行搭建 |

---

## 3b. ★★ M1 达成：L1 推理链在本机跑通（2026-11-18）

三个上游权重（`encoder.pt` 17 MB / `synthesizer.pt` 370 MB / `vocoder.pt` 54 MB）
就位、补装 `Unidecode` + `inflect` 之后，**整条链一次跑通**。

### 实测记录（`scripts/smoke_test.py`）

| 项 | 实测值 |
|---|---|
| 原始卡 vs 扰动卡 余弦相似度 | **0.9950**（α = 0.1；1.0 = 没扰动） |
| 保护后语音 RMS | 0.06780 |
| 未防御克隆音 RMS | 0.07663 |
| 防御后克隆音 RMS | 0.09838 |
| 攻击者重新提的卡 vs 原卡（Sim_clone 的卡片口径） | **0.8971** |
| GE2E 嵌入范数 | 1.000000（L2 归一化 ✓） |

> ★ **Sim_clone = 0.8971 是预期结果**，不是 bug：
> L1 阶段的 δ 是**随机高斯方向**，还没有训练生成器 G，
> 所以防御几乎无效。论文报告 0.13 —— 那要等 Stage 2 训完。

### 平台样例（真实产出，P4）

| 文件 | 时长 | RMS | 说明 |
|---|---|---|---|
| `public/samples/.../input.wav` | 2.66 s | 0.05394 | 原始语音（测试集说话人 p228） |
| `public/samples/.../output.wav` | 3.30 s | 0.07429 | 保护后语音 |
| `public/samples/.../output_clone_baseline.wav` | 3.13 s | — | 未防御时攻击者的克隆音 |
| `public/samples/.../output_clone_protected.wav` | 2.57 s | — | 防御后攻击者的克隆音 |

`node scripts/check_method_config.mjs` → **全部通过**（含"样例引用的文件已存在"）。

### 服务端到端（`scripts/test_service.py`，**真权重**）

**41/41 通过** —— 覆盖参数校验、错误格式、artifacts 结构、文件下载、空文本兜底。

### ★ 这轮踩到的坑：上游 Tacotron 的**空文本**会间歇性崩

| 项 | 内容 |
|---|---|
| 现象 | 服务偶发返回 500：`index -1 is out of bounds for axis 1 with size 0` |
| 根因 | 服务把**空文本**传给了上游合成器。上游 Tacotron 拿到空文本会吐出一段退化的 mel，它自己的"裁尾部静音"循环（`while np.max(m[:, -1]) < stop_threshold: m = m[:, :-1]`）会把这 mel 一路裁空，然后在 `m[:, -1]` 上越界 |
| 为什么难查 | **间歇性**：同一份输入，有时成功（输出 0.3 s、RMS 抖动 10 倍），有时越界 |
| 解法 | ① 服务端把空文本兜底成 `infer.DEFAULT_REFERENCE_TEXT`；② `bridge_cloner.clone()` 显式拒绝空文本并给一句人话；③ `test_service.py` 加了一条**回归断言**（第 41 项） |

> 这条坑值得记：**上游的"自动裁静音"逻辑，在退化输入下会把数组裁空。**
> 任何把外部输入直接喂给上游的地方，都要先挡一道。

---

## 3c. ★★ 基线测量（2026-11-18，40 条测试集样本 = 20 说话人 × 2 句）

用 `evaluate.py` 跑的两条**基线**，这是后面 "Ours" 行的参照系。

| 变体 | Sim_prot ↑ | Sim_clone ↓ | DSR(τ=0.3) | DSR(τ=0.4) | DSR(τ=0.5) |
|---|---|---|---|---|---|
| **无防御**（α=0） | 0.8089 | 0.6967 | 0.0% | 0.0% | 0.0% |
| **高斯噪声扰动**（α=0.1，随机方向） | 0.8171 | 0.6994 | 0.0% | 0.0% | **2.5%** |
| 论文 · Original（无防御） | 0.91 | 0.91 | 0.0% | 0.0% | 0.0% |
| 论文 · Gaussian Noise | 0.85 | 0.35 | 32.1% | 38.6% | 45.2% |
| 论文 · Ours | 0.95 | 0.13 | 79.8% | 83.5% | 87.2% |

**读出来的三件事：**

1. ✅ **无防御基线完全正确**：DSR = 0%（克隆必然像本人）。
2. ⚠️ **我们的高斯噪声扰动比论文的弱得多**（Sim_clone 0.699 vs 论文 0.35），
   随机方向在 256 维球面上几乎不改变"克隆音像不像本人"。
3. ★ **Sim_prot 有结构性天花板 ≈ 0.82**（论文 0.95）——
   **这不是 bug，是路线的必然结果**：

   > 我们的保护后语音是**用克隆机重新合成**的（`clone(text, h_adv)`），
   > 不是"原音频 + 扰动"。重新合成本身就会改变音色/韵律，
   > 所以 E_spk(保护后) 与 E_spk(原音) 的相似度天然封顶在 ~0.82，
   > **与扰动强弱无关**。

   论文的 0.95 需要 `x_adv = D(c, h_adv)` —— Stage 1 的解码器**重建原句**，
   内容码 c 保证"说了什么"不变，只换音色。
   **→ 要拿到论文那组数字，Stage 1 是绕不过去的。**

---

## 3d. ★ 平台接口的耗时优化（2026-11-18）

平台是**同步等待**的，而 L1 的 clone 模式一次请求实测 **201.8 秒** → 必然超时。

### 耗时分解（CPU，12 逻辑核，torch 2.4.1+cpu）

| 环节 | 耗时 |
|---|---|
| GE2E 嵌入 | 1.2 s |
| 上游合成器 | 0.3 – 3.2 s（随文本长度） |
| **上游声码器** | **20.4 – 28.7 s** ← 大头 |

### 关键发现：声码器耗时**几乎与 mel 长度无关**

| mel 帧数 | 45 | 89 | 263 | 507 |
|---|---|---|---|---|
| 声码器耗时 | 16.9 s | 20.8 s | 25.0 s | 28.7 s |

根因：上游 WaveRNN 把 mel 折叠成 `target` 长度的块**并行**生成，
折叠后 Python 层循环次数 = `target − overlap`，**固定**。

### 解法与实测

| target / overlap | 循环次数 | 耗时 | 与默认的波形差异 |
|---|---|---|---|
| 8000 / 800（上游默认） | 7200 | 25.2 s | — |
| 2000 / 200 | 1800 | 9.3 s | 0.08995 |
| **1000 / 100（采用）** | **900** | **6.6 s** | 0.09279 |
| 500 / 50 | 450 | 6.2 s | 0.08892 |

**音质没有损失**：同样参数连跑两次的波形差异 = **0.09441**（声码器本身有随机性，
mu-law 采样），与"默认 vs 快速"的差异 **0.09279 同量级** → 差异落在随机性范围内。

### 效果

| 模式 | 优化前 | 优化后 |
|---|---|---|
| `attack_mode=none` | 201.8 s | **11.4 s** |
| `attack_mode=clone` | 201.8 s | **21.6 s** |

---

## 3e. ★ MakeItTalk 的 `ckpt_autovc.pth` **不能**直接初始化 AutoVC

任务板 §0.8 写的"Stage 1 用 `ckpt_autovc.pth` 初始化"，**实测行不通**：

```
下载：hf-mirror.com/marlenezw/AutoVC_Voice_Conversion → ckpt_autovc.pth（171 MB）
内容：只有 AutoVC Generator 的 encoder(3.56M) + decoder(6.41M) + postnet(4.35M)
      —— **没有声码器，也没有说话人编码器**
```

形状对不上（试过 dim_neck ∈ {32,64} × dim_pre ∈ {512,1024}，全部失败）：

| 张量 | checkpoint | AutoVC 的 Generator |
|---|---|---|
| `encoder.lstm.weight_ih_l0` | (64, 512) | (dim_neck, 512) ✓ 这一项其实对得上 |
| `encoder.lstm.weight_hh_l0` | **(64, 16)** | (dim_neck, dim_neck) ✗ 不是方阵 |
| `decoder.linear_projection.linear_layer.weight` | (80, 512) | (80, dim_pre) |

`weight_hh` 不是方阵 → 它**不是标准 PyTorch LSTM 的权重布局**，
多半是从 Keras/TF 转换来的，需要写转换脚本才能用。

> **→ 结论**：Stage 1 **只能从零训练**（这本来就是 A 的主责），
> 这个 checkpoint 最多当参考，不能当初始化。
> 文件已存到 `algorithms/voiceprint_protection/weights/ckpt_autovc.pth`（gitignored）。

---

## 3f. ★★ 2026-11-20：v0.5 合并后的复核（成员 B）

### 3f.1 复核 A 的三处改动（成员 A 的 v0.5）

| # | A 的改动 | B 的复核结论 |
|---|---|---|
| 1 | HiFi-GAN 16 kHz 权重与内置生成器 | ✅ **同意**。逐项核对：采样率 16000 / hop 256 / n_fft 1024 / 80 mels、HiFi-GAN v1 架构、参数量 13,936,130、`strict=True` 234/234 键全匹配。`hifigan.py` 是 `bshall/hifigan` 的 MIT 代码，已加署名 |
| 2 | 模型侧 mel 改成 `hifigan16k`，删掉近似换算 | ✅ **同意**。实测支持 A 的结论：精确的 HiFi-GAN mel 说话人余弦 **0.9746**，AutoVC mel → 仿射换算只有 **0.6983**（两套 mel 的滤波器组不同，仿射补不回来） |
| 3 | 修 Stage 1 ↔ Stage 2 的两个致命 bug | ✅ **修法正确**。B 逐行对照 `third_party/AutoVC/model_vc.py:181-202`（官方 `Generator.forward`），A 的 `decode()` 与官方语义**逐值一致**（实测最大差 **2.38e-06**） |

> ★ 关于第 3 条要补一句：**这两个 bug 之所以能溜过去，
> 是因为 `verify_gradients.py` 原来只验 `StubStage1`** ——
> 而 `StubStage1.decode()` 的逻辑跟官方 `Generator` 根本不是一回事。
> B 已把这一段补成**会失败的断言**（见 3f.2）。

### 3f.2 新增的回归测试（`stage2/verify_gradients.py`【6】段，7 项）

用 `TorchAutoVCStage1(random_init=True)` —— 权重随机、**指标无意义**，
但验的是**接口**，**不依赖训练好的权重**，所以 Stage 1 还没训完也能验。

| # | 断言 | 实测 |
|---|---|---|
| 1 | `content_code()` 返回**张量**（不是 list） | OK |
| 2 | 形状 `(B, T', 64)` | `(2, 3, 64)`（T=96） |
| 3 | `T' = T // 32` | 3 |
| 4 | **h_adv 真的进了解码器**（`D(c,h)` 与 `D(c,h_adv)` 之差 > 0） | **0.051788**（修 bug 前是 0） |
| 5 | **梯度能回到 h_adv** | OK |
| 6 | **content 参数真的被用上**（换 content 后之差 > 0） | **0.005156**（修 bug 前是 0） |
| 7 | `decode()` 与官方 `Generator.forward(x, c_org, c_trg)` **逐值一致** | 最大差 **2.38e-06** |
| 8 | `decode()` 在 `T` 不是 freq 整数倍 时 仍返回同样帧数 (§3g.2 / §4.5.3d) | 90 -> 90 |

> 第 7 条最硬：它把 `decode()` 的语义**钉死**在官方 AutoVC 上，
> 以后谁改 `decode()` 改出偏差，这一条立刻会红。

`verify_gradients.py` 现在 **20 项全通过**（原 12 项 + 新 7 项）。

### 3f.3 ★ B 顺手修掉的三个自己的坑（都在共同区）

| # | 坑 | 现象（实测） | 修法 |
|---|---|---|---|
| 1 | `paths.models_dir()` 硬返回 `<仓库根>/data/models`（**该目录不存在**），而四个上游权重实际在 `<数据根>/models/default/` | **`SpeakerBridge()` 直接抛"找不到 GE2E 的权重文件"** —— Stage 1 / Stage 2 训练、L1 推理链、平台服务**全部起不来**；而且报错会误导人"去重下一个已经下好的文件" | 改成候选目录列表 `paths.models_dirs()`（`$VPP_MODELS_DIR` → `<数据根>/models` → `<仓库根>/data/models`），逐个试 |
| 2 | `stage2/pipeline.py::Stage2Pipeline.vocode()` 把归一化系数**写死**成 `hifigan16k` | 换了 `config.STAGE1["mel_preset"]` 就会送出**错的 mel** → 声码器出来是噪音，**而且不报错** | 改成跟着当前预设走，预设没有归一化系数时**显式报错** |
| 3 | `audio_utils.MelSpec` 写死 `autovc_mel()` | `MelSpec("hifigan16k")` 也返回 **AutoVC 口径**的 mel —— 名字写着 A、算出来是 B | 改成走 `model_mel(wav, self.name)` |

### 3f.4 本轮回归结果（2026-11-20，vpp 环境，CPU）

| 脚本 | 结果 |
|---|---|
| `scripts/test_audio_utils.py` | ✅ 通过 |
| `stage2/verify_mel.py` | ✅ 通过（mel 误差 2.3e-05，嵌入余弦 0.99999481） |
| `stage2/verify_psychoacoustics.py` | ✅ **20/20** |
| `stage2/verify_gradients.py` | ✅ **20/20**（含新增 7 项） |
| `scripts/verify_rtc_mel.py` | ✅ 通过（与上游 WaveRNN 兼容，最佳 scale=4） |
| `scripts/verify_hifigan.py` | ✅ 通过（余弦 **0.9745**、梯度可回传 absmax 0.8925） |
| `scripts/test_stage1.py` | ✅ 通过 |
| `scripts/test_service.py --mock` | ✅ **41/41** |
| `scripts/split_speakers.py --check` | ✅ 与本机划分**完全一致**（110 人 / 80-10-20 / seed 42） |

### 3f.5 ★ 数据划分对账（省掉对方再下一份 1.36 GiB）

划分**只取决于「说话人名单 + 种子」**，与音频内容无关。因此：

- B 把 110 人的名单**提交进仓库**：`docs/notes/vctk_speakers.txt`（含 `s5`）；
- 新增 `scripts/split_speakers.py`：从名单 + 种子复现划分、写 `split_speakers.json`
  与 `spk2id.txt`、或跟已有的划分**对账**。

对方**不用再下一份 VCTK** 就能确认两边划分一致：

```bash
python algorithms/voiceprint_protection/scripts/split_speakers.py --check <数据根>/split_speakers.json
```


## 3g. ★★ 2026-11-20：Stage 1/2 在**真数据**上跑通，并抓到一个致命 bug（成员 B）

> 这一节记的是"把两个模块接起来"才暴露出来的问题 ——
> **单独测每个模块都是绿的，接起来就崩**。这类 bug 只有真跑才能发现。

### 3g.1 做了什么

A 的 Stage 1 代码之前只在**假数据**上验过（`scripts/test_stage1.py` 19 项），
Stage 2 也只在 `--stub` 上跑过（替身 Stage 1 + 替身声码器）。
这一轮把它们接到**真 VCTK + 真 HiFi-GAN + 真 GE2E** 上跑。

| # | 跑的东西 | 结果 |
|---|---|---|
| 1 | `stage1/train.py --stage model --iterations 3 --smoke`（真 VCTK） | ✅ 通。模型 28,422,464 参数 / GE2E fine_tune=True / **模型侧 mel = hifigan16k**（A 的 v0.5 生效）/ loss 2.4522 → 2.3852 → 2.3757 递减 / **14 s/step**（CPU, batch 4） |
| 2 | `stage2/train.py --stub --vocoder hifigan --steps 6`（真数据 + 真声码器） | ✅ 通。数据 8004 条 / 80 说话人；L_de 0.44~0.50、L_mask 0.29~1.18 dB、L_mel 0.11~0.13、gan ~1.0 全部有限且在动；**45.5 s/step**（CPU, batch 8） |
| 3 | `stage2/train.py --vocoder hifigan --stage1-checkpoint <真 checkpoint>` | ❌ **崩了** —— 见 3g.2 |

> 为此先把 `stage2/train.py` 的 `--stub` 与 `--vocoder` **解耦**：
> 原来 `--stub` 会强制使用替身声码器，**真实声码器根本没机会上场**，
> 于是"真声码器能不能跑通 Stage 2"这件事一直没人验过。
> 解耦后 `--stub --vocoder hifigan` = 替身 Stage 1 + 真声码器（最有用的冒烟组合）。

### 3g.2 ★★ 抓到的 bug：接上真实 Stage 1 就崩（帧数不变式被打破）

```
RuntimeError: x 与 x_adv 的帧数必须相等（契约 §2 的形状不变量），实际 188 vs 192
```

**根因**（两个地方各有一半责任）：

| # | 位置 | 问题 |
|---|---|---|
| 1 | `stage2/models.py::TorchAutoVCStage1.decode()` | AutoVC 的解码器要求 `T` 是 `freq` 的整数倍，**内部会 pad**（188 → 192），但**输出没裁回来** ⇒ 解码结果比输入多 4 帧 |
| 2 | `stage2/dataset.py::VCTKManifest` | Stage 1 的数据集**保证**帧数是 `freq` 的整数倍（契约 §7.1b），**Stage 2 的数据集没有** ⇒ 裁出来的片段帧数是任意的 |

**为什么之前没发现**：
- `verify_gradients.py` 用的是 `StubStage1`，它的 `decode()` 用线性插值补齐长度，**永远等长**；
- `stage2/train.py --stub` 走的是同一条路；
- ⇒ **只要不把真实 Stage 1 接进来，就碰不到**。

**修法**（两处都改，根因 + 防御）：

- `decode()` 记下**原始**帧数，输出裁回该长度 ⇒ "输出与输入等长"这个不变式
  **由构造保证**，不再依赖调用方喂进来的 `T` 恰好是 32 的倍数；
- `stage2/dataset.py` 按 Stage 1 的同一套约定裁
  （`frames = int(max_seconds*sr/hop)` 向下取整到 32 的倍数，`N = (frames-1)*hop`），
  且长度**恒定**（不够就补零）⇒ batch 内帧数一致。

**修完**：

```
stage2/train.py --vocoder hifigan --stage1-checkpoint <真 checkpoint> --steps 3
   0  total 6.4975  gan 0.9965  de 0.4468  per 0.2178  mel 0.1636  mask 0.5416  sim_clone 0.4526   [41s]
   1  total 6.4733  gan 1.0224  de 0.4243  per 0.2220  mel 0.1587  mask 0.6329  sim_clone 0.4272   [70s]
   2  total 7.1484  gan 1.0010  de 0.4528  per 0.2589  mel 0.1535  mask 1.0540  sim_clone 0.4562   [102s]
```

⇒ **Stage 2 现在能接真实 Stage 1 checkpoint 跑了**（30~35 s/step, CPU, batch 8）。
并且验证了 A 说的「我们的 checkpoint 键名与官方 AutoVC 的 `Generator` 完全一致」——
`TorchAutoVCStage1` 直接 `load_state_dict` 成功（28,422,464 参数全对上，**没报错**）。

> ★ 已加成会失败的断言：`stage2/verify_gradients.py`【6】段 (e) 项 ——
> 用 `T = 90`（不是 32 的倍数）喂 `decode()`，断言输出帧数仍是 90。
> 现在共 **20 项**全通过。

### 3g.3 ★ 另一个坑：3 步的冒烟 checkpoint 会被当成正式权重

`stage1/train.py` 训完会把模型存成 `weights/stage1_model.pt`，
而 `stage2/models.py::_guess_checkpoint()` 的候选列表里 **`stage1_model.pt` 排第一** ⇒
一个只训了 3 步的**垃圾权重**会被 Stage 2 静默捡去用，**而且不报错**。

处置：把这次冒烟产出的文件改名成 `stage1_smoke3steps_NOT_TRAINED.pt`，
让 `_guess_checkpoint()` 找不到它。（正式训练出来的 checkpoint 仍用 `stage1_model.pt`。）

> 💡 **建议**（给 A）：正式训练前把 `_guess_checkpoint()` 改成
> "只认显式传入的路径"，或者在文件名里带 step 数 —— 否则这个坑迟早再踩一次。

### 3g.4 云机预算的实测依据

| 场景 | 设备 | 每步耗时 | batch |
|---|---|---|---|
| Stage 1 | CPU（本机） | 14 s | 4 |
| Stage 2（替身 Stage 1 + 真 HiFi-GAN） | CPU（本机） | 45.5 s | 8 |
| Stage 2（真 Stage 1 + 真 HiFi-GAN） | CPU（本机） | 30~35 s | 8 |
| Stage 1 | RTX 4060 Laptop（A 实测，见 TASKS §0.10.4b） | **0.3 s** | 16 |

⇒ 按 A 的 4060 数据推算，Stage 2 的 200k 步约 **17 小时**，一次性开一台卡就够 ——
这正好是任务板 §8「只在真正要训练时才开机」那条规则的依据。


## 3h. ★★ P5 平台端到端联调（2026-11-20，成员 B）

> 任务板 W3 的 P5「平台端到端：上传 -> 处理 -> 预览 -> 下载」原来是 ⏳，
> 因为只**人工**点过一遍，没留下可重复执行的凭证。

### 3h.1 新增 `scripts/test_platform_e2e.py`（可重复执行的自检）

它验的是**前端真正会走的那条路**（照抄 `src/services/algorithms.js::runAlgorithm`）：

```
FormData: <文件字段> + method_id + parameters(JSON 字符串)
   -> POST 方法 JSON 的 adapter.endpoint
   -> { artifacts: [{kind,label,filename,url}], message }
   -> 每个 url 用 safeUrl(url, endpoint) 解析后**下载**
```

★ **一切配置都从方法 JSON 读**（endpoint / 样例输入 / 参数）⇒
改了方法 JSON，这个脚本自动跟着变，不会出现"脚本验的是一套、平台跑的是另一套"。

**跑法**（服务要先起）：

```bash
conda activate vpp
uvicorn algorithms.voiceprint_protection.service:app --host 127.0.0.1 --port 8000
# 另一个终端
python algorithms/voiceprint_protection/scripts/test_platform_e2e.py
```

### 3h.2 实测结果（本机，CPU，2026-11-20）

```
[通过] 方法 JSON 里登记的输出样例存在且非空
[通过] GET /healthz 返回 200                model_loaded: true
[通过] CORS 预检允许前端来源                 allow-origin='*'
[通过] CORS 允许 POST                        methods='... POST ...'
[通过] POST /process 返回 200                耗时 21.6s
[通过] artifacts 是 1-20 个结果               实际 4
[通过] 返回了可显示的 message                防御处理完成（用时 21.6 秒）
[通过] 保护后语音              可下载 103,644 字节 / 3.24s @16kHz / RMS=0.0604
[通过] 未防御时攻击者的克隆音   可下载  98,844 字节 / 3.09s @16kHz / RMS=0.0605
[通过] 防御后攻击者的克隆音     可下载  79,644 字节 / 2.49s @16kHz / RMS=0.0632
[通过] 评测报告                可下载 705 字节，合法 JSON
[通过] 缺文件时返回 4xx        HTTP 422
[通过] 错误体是 {'error': '...'}   请求参数不对 —— input_file：Field required
```

**29 项通过 / 0 项失败。**

评测报告里的关键数字（这是**平台请求的真实产物**，不是脚本造的）：

```json
{ "alpha": 0.1, "seed": 42, "attack_mode": "clone",
  "elapsed_seconds": 21.6,
  "sim_embedding_perturbed": 0.995,
  "sim_clone": 0.8644,
  "sim_clone_threshold_0.5": false }
```

> `sim_clone = 0.8644` 看着很高，但这是**预期的**：现在是 L1 版本，
> δ 是随机高斯噪声、Stage 2 的生成器 G 还没训练 ⇒ 防御基本没起作用。
> 等 Stage 2 训完，这个数应该掉到 0.13 附近（论文报告值），DSR 才会上去。

### 3h.3 前端侧同步确认

| 项 | 结果 |
|---|---|
| `npm run build` | ✅ 通过（1589 modules，4.53s） |
| `npm test` | ✅ **6/6** |
| 方法 JSON 的 `adapter.endpoint` | `http://127.0.0.1:8000/process` —— 与 `service.py` 实际路由一致 ✓ |


## 3i. ★★ 2026-11-20 夜：P2 基线复测 / P3 绝对 WER / P5 云机脚本（成员 B）

### 3i.1 P2：基线复测（用成员 A 的 v0.5 新代码）

§3c 的基线是 2026-11-18 测的，用的是**旧** mel 预设（`autovc`）。
A 的 v0.5 把模型侧 mel 换成了 `hifigan16k`，基线得重测一遍，
确认换预设没把基线带偏。

```
python algorithms/voiceprint_protection/evaluate.py --mode none --split test --limit-per-speaker 1
```

| 变体 | Sim_prot ↑ | Sim_clone ↓ | DSR(tau=0.3) | DSR(tau=0.4) | DSR(tau=0.5) |
|---|---|---|---|---|---|
| **无防御（v0.5 复测，n=20）** | **0.7994** | **0.6659** | **0.0%** | **0.0%** | **5.0%** |
| 无防御（§3c，旧代码，n=40） | 0.8089 | 0.6967 | 0.0% | 0.0% | 0.0% |
| 论文 · Original（无防御） | 0.91 | 0.91 | 0.0% | 0.0% | 0.0% |

**结论**：基线**没有被 v0.5 带偏**（差异在 n=20/40 的抽样噪声范围内），
§3c 的三个结论依然成立 —— 无防御 DSR=0 / 高斯噪声太弱 / Sim_prot 有 ~0.82 的结构性天花板。

**耗时**（本机 CPU，12 逻辑核，torch 2.4.1+cpu）：**22 s/条**（克隆链要跑两遍：保护后 + 未防御）。
→ 完整测试集 2,000 条 ≈ **12 小时**，本机跑不完。

> ★ **最终数字在 §3i.5**（20 个测试说话人 x 5 句 = **100 条**的分层抽样，
> 顺便带绝对 WER）。上面这张 n=20 的表只是"换预设后基线有没有被带偏"的**快速复核** ——
> 它取的是每个说话人的第 1 句，而 VCTK 里每个说话人的第 1 句恰好是同一句
> "Please call Stella."，所以样本本身没有代表性（§3i.2 第 4 条会讲这件事）。

### 3i.2 ★★ P3：绝对 WER —— 链路上有两个「静默失效」的 bug

要算**绝对** WER（对 VCTK 自带的原始转写），得有 VCTK 的 `txt/`。
2026-11-20 夜源站恢复，用新写的 `scripts/download_vctk.py` 定向拉了
测试集 20 个说话人的 8,000 个转写文件（**5.2 分钟，0 失败**）。

拿到转写后才发现，**绝对 WER 之前根本算不出来** —— 有两个 bug 会让它
**无声地**退化成「Whisper 转写原始语音当参考」的相对 WER：

| # | bug | 后果 |
|---|---|---|
| 1 | `load_transcripts()` 用 `glob("*.txt")` 只扫顶层 | VCTK 的 `txt/` 是**按说话人分子目录**的（`txt/p225/p225_001.txt`）⇒ 一个都找不到 |
| 2 | 拿 `path.stem` 查表 | 我们把两路麦克风都收了，转 16 kHz 时改名成 `p228_001_mic1.wav`，而 VCTK 的转写只有一路、叫 `p228_001.txt` ⇒ **永远查不到** |
| 3 | ★★ `protect()` 念的是**固定的一句**，不是原句 | 默认念 `"The quick brown fox jumps over the lazy dog."`，却拿「原句的转写」当参考 ⇒ **WER 恒等于 ~100%，无论语音好坏**。改成：有原始转写时**保护时念同一句**，WER 才有意义、才能和论文的 5.30% 对得上 |
| 4 | ★ `load_test_items()` 取清单里的**前 N 条** | 清单按文件名排序 ⇒ 每说话人拿到的都是 `pXXX_001`，而 VCTK 里 `pXXX_001` **恰好是同一句** "Please call Stella."。20 个说话人 20 条样本全是同一句话，WER 这类指标就没有代表性。改成在每说话人的全部语句里**均匀抽样** |

两个 bug 都是**静默**的：查不到转写时 `evaluate.py` 会退化成
「Whisper 转写原始语音」当参考，算出的是**相对 WER**，但**一个错都不报**。
这类 bug 比崩溃更危险 —— 报告里会写着一个看起来正常的 WER 数字，来源却不对。

**修法**：
1. `rglob` 递归 + 新增 `transcript_key()`（去掉 `_mic1`/`_mic2` 后缀）；
2. 保护时念**原句**（有原始转写时），WER 才有意义；
3. `load_test_items()` 改成**均匀抽样**；
4. 加 `transcript_miss` 计数，有缺失就醒目提示，不再默默算；
5. 报告里把「WER(保护后)」与「**WER(原音·ASR 下限)**」分开列 ——
   后者是 ASR 在原始语音上的错误率，**防御不可能比它更低**，是判断 WER 是否合理的标尺。

**验证**：转写查表命中率 **20/20**（修之前是 0/20）。

### 3i.3 ★ 数据可复现性：两个「同一份命令在两台机器上选出不同数据」的坑

A 的行动项 (6) 要两边用同一份数据划分。查的时候发现下载器本身就埋了两个坑：

| # | 坑 | 说明 |
|---|---|---|
| 1 | `.mjs` 用 `localeCompare` 排序 | 它依赖运行机器的 **locale 与 ICU 版本** —— 两台机器可能挑出**不同的**句子，数据就不可复现了。改成码点序（和 Python 的 `sorted` 一致） |
| 2 | 每说话人的配额把 `wav48_silence_trimmed/log.txt` 也算进去了 | 它不是音频，不该占名额（`prepare_data` 也只认 `.flac`/`.wav`）。改成只按音频文件计配额 |

**顺带查出 B 自己的数据也不干净**：p303 / p304 各多出 2 个"野文件"
（早期分批下载的残留）⇒ 清单 11,004 条 ≠ 规则的 11,000 条。
已删掉这 4 个并重跑 `prepare_data.py`（**0 条重转**，因为已转好的音频会被跳过），
现在清单是 **train 8,000 / val 1,000 / test 2,000**，全部自检通过。

**逐条对账**：把下载器的选择规则在 B 的本地数据上跑一遍 ——

```
selected: 11000 | local: 11000
EXACT MATCH: True
diff: []
```

⇒ **A 只要照下面的命令下，数据与 B 逐条一致**（源站已恢复，2026-11-20 夜实测 HTTP 200）：

```bash
python scripts/download_vctk.py --out <数据根>/raw --include wav48_silence_trimmed --include txt --per-speaker 100
python algorithms/voiceprint_protection/scripts/prepare_data.py
```

应得到 **11,000 条**（train 8,000 / val 1,000 / test 2,000）；
说话人划分用 `scripts/split_speakers.py --check` 对账（名单见 `docs/notes/vctk_speakers.txt`）。

### 3i.4 ★ P5：云机一键脚本

新增 `scripts/cloud_train_stage1.sh`，把「配环境」和「烧钱训练」**拆成两条命令** ——
云机按小时计费，别一开机就跑满：

| 命令 | 做什么 |
|---|---|
| `--setup` | 建 venv、装 CUDA 版 torch + requirements、拉第三方组件（固定 commit）、定向下 VCTK 与四个上游权重、转 16 kHz 并生成 80/10/20 清单 |
| `--start` | 在 tmux 里起 Stage 1 训练（没装 tmux 就退化成 nohup） |
| `--status` / `--stop` | 看进度 / 停 |

数据下载用 `scripts/download_vctk.py`（**纯标准库**，云机没装 Node 也能跑）。

### 3i.5 ★ P3 的实测结果（绝对 WER）

```
python algorithms/voiceprint_protection/evaluate.py --mode none --split test \
       --limit-per-speaker 5 --wer --whisper-model <模型> \
       --transcript-dir <数据根>/raw/txt
```

| 变体 | 样本 | Sim_prot ↑ | Sim_clone ↓ | DSR(tau=0.3) | DSR(tau=0.4) | DSR(tau=0.5) | WER(保护后) ↓ | WER(原音·ASR 下限) |
|---|---|---|---|---|---|---|---|---|
| **无防御**（mode=none） | 100 | 0.8043 | 0.7080 | 0.00% | 0.00% | 0.00% | 5.81% | 3.30% |
| 高斯噪声扰动（mode=random） | 100 | 0.8053 | 0.6995 | 0.00% | 0.00% | 0.00% | — | — |
| **无防御 · 论文同款 Whisper**（`large-v3`） | — | — | — | — | — | — | — | — |

> 高斯噪声那一行**没算 WER**（`mode=random` 那一跑没带 `--wer`）—— 它要回答的是
> 「随机扰动能不能骗过克隆」，WER 对它意义不大；无防御那两行才是 WER 的正主。

**和 §3c 的旧数字对一下**（§3c 是 2026-11-18 用**旧代码**、n=40 测的）：

| | Sim_prot | Sim_clone | DSR(0.3/0.4/0.5) |
|---|---|---|---|
| §3c 无防御（n=40，旧代码） | 0.8089 | 0.6967 | 0% / 0% / 0% |
| **§3i.5 无防御（n=100，v0.5 新代码）** | **0.8043** | **0.7080** | **0% / 0% / 0%** |
| §3c 高斯噪声（n=40） | 0.8171 | 0.6994 | 0% / 0% / 2.5% |
| **§3i.5 高斯噪声（n=100）** | **0.8053** | **0.6995** | **0% / 0% / 0%** |

→ 结论一致，而且更稳：**换 mel 预设（v0.5）没有把基线带偏**；
**高斯噪声几乎与无防御无法区分**（Sim_clone 0.6995 vs 0.7080，DSR 全 0），
这正是论文 Table 1 里「Gaussian Noise」那一行给我们的启示 —— 随机方向的扰动
在 256 维球面上几乎不改变「克隆音像不像本人」。论文那一行能拿到
Sim_clone 0.35 / DSR 32~45%，是因为它扰动的是**声纹空间**而不是重建出来的音色。

**怎么读这两个 WER**：
- 「WER(原音·ASR 下限)」= 原始录音被 Whisper 转写的错误率。**这是地板** ——
  防御不可能比它更低，因为保护后语音是从原音派生的。
- 「WER(保护后)」= 保护后语音的转写错误率（参考 = VCTK 自带的原始转写，**绝对 WER**）。
- 两者之差 = **防御本身引入的转写损伤**。
- 两个数的**中位数都是 0** —— 绝大多数句子被逐字转写对了，均值是被少数几句拉起来的。

> ★★ **我们的 5.81% 和论文的 5.30% 数值上很接近，但成分完全不同，别当成「复现成功」**：
>
> | | ASR 下限（原始录音） | 防御引入的损伤 | 合计 |
> |---|---|---|---|
> | **我们（L1 路线）** | 3.30% | **2.51 个百分点** | 5.81% |
> | 论文（Stage 1 路线） | 论文未单列（更低） | 5.30% 减去它自己的下限 | 5.30% |
>
> 我们的损伤**主要不是扰动带来的** —— L1 路线没有 Stage 1，
> 保护后语音是 `clone(原句文本, h_adv)`，**整句重新合成**了一遍，
> 所以这 2.51 个百分点里混着**「TTS 往返」的损失**（合成器念一遍 + ASR 再听一遍）。
> 论文的 `x_adv = D(c, h_adv)` 是用 Stage 1 的解码器**重建原句**（只换音色），内容几乎不变，
> 那一行里几乎没有「往返」这一项。
>
> **这正是 §3c 说的那条结构性问题**：Sim_prot 的 ~0.80 天花板、WER 的成分差异，
> 都指向同一个结论 —— **要拿到论文那组数字，Stage 1 是绕不过去的**。

> ★ **模型说明**：论文用 Whisper-large（= faster-whisper 的 `large-v3`）。
> 本机到 HuggingFace 的 CDN 只有 ~350 KB/s，`large-v3` 的 `model.bin` 有 3.09 GB
> （单连接 2.4 小时），所以先用 **`small`**（484 MB）跑通链路，
> 再用 `scripts/parallel_download.py` 并发把 `large-v3` 拉下来跑同一组数字。
> **前两行用的是 `small`（偏离论文设定），最后一行是论文同款的 `large-v3` —— 报告里要写明这一点。**
> 云机上重跑：`bash scripts/cloud_train_stage1.sh --setup` 会把 Whisper 一起拉下来
> （或 `VPP_WHISPER_MODEL=large-v3`），然后 `--whisper-model large-v3` 即可。

**耗时**：保护链要在 CPU 上跑两遍克隆（保护后 + 未防御），**27 s/条**（实测 2,720 s / 100 条）；
完整测试集 2,000 条 ≈ **15 小时**，本机跑不完 —— 所以上表是 **20 个测试说话人 x 5 句 = 100 条**
的分层抽样，**样本量写在表里，不能当全集数字用**。

**★ 为什么 DSR 全是 0%（连高斯噪声也救不回来）—— 这是路线的天花板，不是实现 bug**

DSR 的定义是「攻击者克隆出来的**声音**有多不像本人」。在 L1 路线里，攻击者的克隆是
`clone(参考文本, h_stolen)`，其中 `h_stolen = E_spk(保护后语音)`。
而保护后语音本身又是 `clone(原句, h_adv)` 生成的 —— **保护前后都是同一个合成器念的**，
音色天然就贴着原说话人。所以不管 `h_adv` 怎么扰，
`E_spk(克隆音)` 都落不到离原说话人足够远的地方 ⇒ DSR = 0%。

论文之所以能把 Sim_clone 压到 0.13、DSR 拉到 79.8~87.2%，是因为它走的是
`x_adv = D(c, h_adv)`：**内容码 c 保住「说了什么」**，解码器只换音色，
攻击者的克隆用的是**被污染的声纹**，和原说话人拉开了距离。
→ **这与 §3c 的 Sim_prot ~0.80 天花板、WER 的成分差异，是同一个结论的三种表现。**

---

## 4. 待补充（未做的不要填数字）

- [ ] **M1 验收**：成员 A 的 `infer.py` 就位后跑 `scripts/smoke_test.py`，
      产出 `protected.wav` + 打印全部张量形状
      —— ⏳ **代码已就位，只差 `synthesizer.pt` + `vocoder.pt`（425 MB，需手动下载）**
- [ ] **D5-1**：两位成员**各自独立**跑一次 `smoke_test.py`（换环境、不给解释）
- [ ] **指标表**：Sim_prot / Sim_clone / DSR（τ=0.3/0.4/0.5）/ WER
      —— ✅ 评测脚本已写好（`evaluate.py`），⏳ 等权重
- [ ] **朴素基线**：h 上加高斯噪声，同样算一套指标（论文里 DSR 只有 45.2%）
      —— ✅ `evaluate.py --mode random` 已实现
- [ ] **消融**：至少 `w/o L_de`（DSR 应从 87.2% 崩到 12.5%）与 `w/o L_mask`
      —— ⚠️ 消融要等 Stage 2 训完才能测
- [ ] **黑盒迁移**（可选，最难，放最后）
- [ ] **MOS 听测**（记录实际人数）

---

## 4b. ★ 2026-11-18 新增：平台服务化与 Stage 2 骨架（成员 B）

### 4b.1 平台算法服务（任务板 W3-P3，**B 代 A 写**）

| 项 | 结果 |
|---|---|
| `algorithms/voiceprint_protection/service.py` | ✅ FastAPI + CORS + 统一错误格式 + 临时文件清理 + 模型单例缓存 |
| `scripts/test_service.py` | ✅ 契约自测 **40/40 通过**（`--mock` 模式**不需要权重**就能验协议） |
| `public/methods/index.json` | ✅ 已追加本方法 |
| `npm test` / `npm run build` | ✅ 6/6 通过 / 构建通过 |
| 真实请求（权重未到） | 503 + 明确提示"缺哪个文件、去哪儿下"（**不是** 500 traceback） |

**三个踩到的坑**（细节见 TASKS.md §0.4b）：

| # | 坑 | 后果 |
|---|---|---|
| 1 | FastAPI 默认返回 `{"detail":...}`，前端读的是 `detail.error` | 中文说明全丢，只显示"算法服务返回 HTTP 422。" |
| 2 | `infer.protect()` 每次调用新建 `Protector()` | 每次请求重载 400 MB 权重 → 平台同步等待必超时 |
| 3 | 用户上传的文件名不能拼进路径 | 目录穿越 / 中文名炸路径 |

### 4b.2 Stage 2 骨架（任务板 W4–W5，**B 主责**）

| 项 | 结果 |
|---|---|
| `stage2/` 12 个模块 | ✅ 代码写完（generator / discriminator / psychoacoustics / losses / models / pipeline / dataset / train + 3 个 verify 脚本） |
| 训练机制跑通 | ✅ `--stub --steps 5` 正常训练（8004 条 / 80 说话人；L_de / L_per / L_mask / L_GAN 全有值） |
| **梯度通路自检** | ✅ **20/20 通过**（`stage2/verify_gradients.py`；含成员 A 修的两个 bug 的 7 项回归断言） |
| torch GE2E vs 上游数值对照 | ✅ mel 最大绝对误差 **2.3e-05**；嵌入余弦 **0.99999481**（`stage2/verify_mel.py`） |
| **心理声学掩蔽自检** | ✅ **20/20 通过**（`stage2/verify_psychoacoustics.py`） |
| 正式训练 | ⛔ **缺可求导声码器 + 缺 Stage 1 checkpoint** → 见 [notes/stage2_blockers.md](notes/stage2_blockers.md) |

**★ 本轮最重要的四个发现**（都已变成会失败的断言）：

1. **指南伪代码里的 `with torch.no_grad()` 不能照抄** ——
   照抄会让梯度被切断，G 收不到任何梯度，loss 一步不动，**而且不报错**。
   「冻结」= 不更新参数（`requires_grad_(False)` + `eval()`），
   **不等于**切断梯度（那才是 `no_grad()` / `.detach()`）。
2. **torch 版 GE2E 必须与上游 numpy 版逐值一致** ——
   不一致时训练与推理的说话人嵌入不是一回事，**同样不报错**。
3. **心理声学掩蔽的参考电平必须对齐** ——
   掩蔽阈值是 dB SPL，扰动 PSD 是数字信号量纲；不统一的话 `L_mask` **恒为 0**，
   那条约束形同虚设，而且**不报错**。
4. **"掩蔽不到"不能当成"极低阈值"** ——
   扩散函数够不着的频带如果用 −200 dB 代替，会把整条阈值曲线压到绝对听阈上，
   结果 `L_mask` 又是恒为 0。

---

## 4c. ★★ 2026-11-20：Stage 1 第一次真训练（成员 A）—— 抓到一个致命 bug + 一条重要的定量结论

### 4c.1 训练实况（RTX 4060 Laptop，VCTK 真数据）

| 项 | 结果 |
|---|---|
| 数据 | 110 说话人 × 100 句 = 11,000 条（**与成员 B 的下载逐条一致**），train 8000 / val 1000 / test 2000 |
| 划分对账 | `split_speakers.py --check` → ✅ 与 B **完全一致** |
| W_s 分类器 | 3000 步，**训练集准确率 90.5%** |
| 解耦网络 | 20,000 步，**11 it/s**（0.09 s/step），32 分钟 |
| 训练曲线 | `rec` 0.714→**0.0094**，`cd` 0.352→**0.0005**，`dis` −3.72→**−4.3729** |
| 验证曲线 | `rec` **0.0588**、`cd` 0.0032、`dis` **−4.3729**（与训练集几乎相同 ⇒ **解耦能力泛化到没见过的说话人** ✅） |

⇒ **`dis` 的理论极限是 −log 80 = −4.3820，我们到 −4.3729（99.8%）** —— 解耦成功。

### 4c.2 ★★ 抓到的致命 bug：`L_dis` 的符号方向反了

第一次真训练时 `dis` 停在 **+0.0016**，怎么想都不对（那意味着熵≈0，分类器**完全确定**）。
查下来是**实现多了一个负号**：

```
论文式(5)：  L_dis = −H(p_s) = Σ_k p_s(k|c̃) log p_s(k|c̃)
             Σ p log p **本身就是 −H** ⇒ 实现里不能再加负号

   p 均匀    → Σ p log p = −4.38（最负 = 最均匀）⭐
   p one-hot → Σ p log p ≈ 0（分类器很确定）❌

原实现：  -(p * log p).sum()  =  +H(p)
         ⇒ 最小化它 = 最小化熵 = 让分类器「更确定」= 与论文正好相反
```

**修前**：`dis` 3.62 → 0.0016（往"更确定"走）❌
**修后**：`dis` −3.72 → **−4.3729**（往"更均匀"走）✅

> ⚠️ **这个 bug 不会报错、loss 照样下降**，只能靠看 `dis` 的**数值与方向**发现。
> ⚠️ 而且**自检断言当时把错误的符号一起编码进去了**（"均匀分布时 L_dis ≈ log(K) = +4.38"），
> 所以测试是"绿"的 —— **测试和数据可以同时错**。现在断言已改成 `-log(K) = -4.3820`。

### 4c.3 ★★ Sim_prot 只有 0.385 —— 定量定位到"训练量不够"，不是实现 bug

M3 要求 `Sim_prot > 0.9`，第一次测只有 **0.385**。逐层排查：

| 诊断 | 结果 | 结论 |
|---|---|---|
| 纯声码器往返（原 mel → HiFi-GAN → 波形） | **0.9856** | 声码器**无损** ✅ |
| 已见说话人（train）vs 未见说话人（test） | 0.3871 vs 0.3812 | **一样差 ⇒ 不是过拟合** |
| 给模型 mel 加 σ=0.02 噪声（比模型自身误差小 22 倍） | Sim_prot 仍 0.3~0.5 | 误差大小不是关键，**差异是结构性的** |
| 原始 mel std vs 模型 mel std | 0.2659 vs **0.1841** | ⭐ **过度平滑**：高频被削平（−0.10）、低/中频被抬高（+0.14/+0.20） |
| **过拟合 8 个样本 3000 步** | `L_rec` 1.306→**0.258 仍在降**、`mel std` 1.00→0.55 仍在往目标走、**Sim_prot 0.394→0.536 在涨** | ⭐ **实现没问题，是训练量不够** |

⇒ 论文训 **500k 步**，我们只训了 **20k**。本机 11 it/s ⇒ 500k ≈ **12.6 小时**。

> ★ 顺带核对过：**AutoVC 官方训练用的也是 MSE**
> （`solver_encoder.py`：`g_loss_id = F.mse_loss(x_real, x_identic)`），
> 所以 `L_rec` 的形式**不是**问题所在。

---

## 4d. ★★ 2026-11-21：Stage 2 开训前的准备（成员 B）—— 训练器补齐 + 一个"隐藏前提"

### 4d.1 先说清楚：**本机跑不了论文量级**

| 项 | 实测值 |
|---|---|
| 显卡 | **Intel Iris Xe 核显，没有 NVIDIA GPU**（`nvidia-smi` 不存在，torch `cuda.is_available()=False`） |
| torch | 2.4.1**+cpu** |
| 内存 | 16 GB，**空闲只剩 2.2 GB** |
| 后果 | Stage 2 用真 Stage 1 + 真 HiFi-GAN ≈ **30 s/step**（§0.12 的数）⇒ 200k 步要跑**两个月** |

⇒ 本机的定位只能是**验证链路 + 测耗时**，正式训练必须上 GPU。
（成员 A 的 4060 那台能跑 Stage 1 到 500k，我们的云机方案见 §4d.5。）

### 4d.2 合并成员 A 的 7 个 commit 之后的回归

`git pull --ff-only`（A 已经 rebase 过，本地 0 ahead / 7 behind，干净快进）⇒ **7 个自检脚本全绿**：

```
verify_gradients 20/20 · verify_psychoacoustics 20/20 · verify_mel 4/4
verify_hifigan ✅ · verify_rtc_mel ✅ · test_stage1 ✅ · test_audio_utils ✅
```

### 4d.3 ★★ Stage 2 训练器补齐六件事（把 A 在 Stage 1 上的审计照搬过来）

A 2026-11-18 做过一次"训练到底能不能跑起来"的审计（§0.10.1），
当时 Stage 2 **一件都没有**。这次补齐：

| # | 缺什么 | 不补的后果 |
|---|--------|-----------|
| 1 | **断点续训 `--resume`** | 200k 步在竞价实例上被抢占 = 白跑 |
| 2 | **AMP** | `config.STAGE2["amp"]` 写了但**代码里一个字都没用** ⇒ 8 GB 显存会 OOM |
| 3 | **梯度累积** | `config.STAGE2["accum_steps"]` 同上 ⇒ 想用大 batch 又不 OOM 就没辙 |
| 4 | **梯度裁剪** | `config.STAGE2["clip_grad"]` 同上 ⇒ L_de 的梯度穿过声码器，容易炸 |
| 5 | **`--max-hours`** | 云机按小时计费，忘了关就一直烧钱 |
| 6 | **吞吐 / ETA 打印** | 不知道要跑多久，就不敢决定开不开 |

> ★ **"写在 config 里却没人用"是审计里最难发现的一类问题**：
> 不报错、loss 照样降，只是跑得比应该的慢好几倍 / 显存不够。
> 这三个键在 `config.STAGE2` 里躺了很久，谁都没注意。

另外加了一个 **`[L_de 方向自检]`**：训练结束自动比较前 10% 与后 10% 的 L_de，
方向不对直接喊出来 —— A 的 `L_dis` 就是同一种"不报错但方向反了"的坑（§4c.2）。

**实测**（本机 CPU）：

```
--stub --steps 6                → 0.50 it/s，loss 在动 ✅
--stub --steps 9 --resume       → "已从 stage2_last.pt 续训，起始 step = 6" ✅
--stub --vocoder hifigan        → 0.42 it/s（batch 2 / 1 秒片段）✅ 真声码器链路通
```

### 4d.3b ★★★ **全真链路已跑通**（只差权重质量）

用成员 A 的 **3 步冒烟 checkpoint**（当"Stage 1 权重"用），
把 **真 Stage 1 架构 + 真 HiFi-GAN 声码器 + 真微调 E_spk + 真 VCTK** 全接上：

```
$ python algorithms/voiceprint_protection/stage2/train.py \
      --vocoder hifigan --stage1-checkpoint <A 的冒烟权重> \
      --speaker-encoder stage1 --steps 2 --batch-size 2 --max-seconds 1.0

  E_spk      : Stage1SpeakerEncoder      ← 随 Stage 1 微调过的那个（1,423,618 参数）
  Stage 1    : TorchAutoVCStage1         ← 官方 AutoVC Generator（28,422,464 参数）✅ 权重加载成功
  声码器     : HiFiGANVocoder            ← 真 HiFi-GAN 16 kHz ✅
  模型侧 mel : hifigan16k
       步      total      gan       de       per       mel     mask  sim_clone    it/s
       1     9.0417   0.9946   0.6750    0.3115    0.1255   1.8599     0.6750    0.18
       2    16.2494   0.9780   0.6846    0.7899    0.1438   6.4609     0.6846    0.26
```

⇒ **Stage 2 的整条链（真 Stage 1 + 真声码器 + 真数据 + 微调过的 E_spk）已经通了**，
差的只有 Stage 1 权重的**质量**（A 那份 Sim_prot 0.459）。
等 A 的正式权重到位，命令就是：

```bash
python algorithms/voiceprint_protection/stage2/train.py \
    --vocoder hifigan --stage1-checkpoint <A 的 stage1_model.pt> --steps 200000
```

（本机 CPU 3.8 s/step ⇒ 200k 步要跑 8.7 个月 ⇒ 必须上 GPU，见 §4d.5）

### 4d.4 ★★★ 抓到一个"隐藏前提"：**E_spk 必须是 Stage 1 微调过的那个**

论文 §3.1：*"...fine-tuned end-to-end during Stage 1 training"* ——
**Stage 2 用的 E_spk 必须是 Stage 1 微调完的那一个**。

而现在的代码：

| 地方 | 用的是哪个 E_spk |
|---|---|
| `stage2/train.py`（训练） | 上游 GE2E（`TorchSpeakerEncoder`，**没微调过**） |
| `infer.py`（推理） | 随 Stage 1 微调过的那个（`TrainableSpeakerEncoder`） |

⇒ **两个嵌入空间不一样**。后果：G 学的是"怎么扰动**上游** GE2E 的嵌入空间"，
推理时喂进去的却是**微调后**的嵌入 ⇒ **δ 退化成随机方向，防御完全失效，而且不报错**。

**处理**（2026-11-21）：

1. `stage2/models.py` 新增 `Stage1SpeakerEncoder` —— 直接复用 A 的
   `TrainableSpeakerEncoder`（**不重新实现**，保证逐字一致），
   从 Stage 1 checkpoint 的 `speaker_encoder` 段加载微调后的权重；
2. `stage2/train.py` 加 `--speaker-encoder {stage1,upstream}`，
   默认按 checkpoint 内容自动选；
3. ★ **这个选择被写进 Stage 2 的 checkpoint**（`speaker_encoder` / `generator_config`），
   `infer.py` 的 stage2 模式**照着做** ⇒ 训练与推理不可能对不上。

> 这也是为什么 `infer.py` 的 stage2 模式必须先读 checkpoint 的元数据，
> 而不是"猜"用哪个编码器。

### 4d.5 ★ 云机方案：华为云 ECS（含一条**推翻旧结论**的官方规则）

完整指南见 **[`docs/notes/cloud_ecs_stage2.md`](notes/cloud_ecs_stage2.md)**，
脚本 [`scripts/cloud_train_stage2.sh`](../../scripts/cloud_train_stage2.sh)（与 Stage 1 版对称）。

**★ 推翻任务板 §8.2 的一条**：那里写「**优先用竞价实例**（便宜 50–80%）」，
但华为云官方《弹性云服务器 ECS · 计费说明》明确写着：

> 「**代金券不适用于竞价计费型实例。**」

⇒ 我们只有 500 元**代金券**，竞价实例**用不上券**（钱从余额扣，余额里没有）。
⇒ **必须开"按需计费"**，并靠"训完立刻释放实例"来省钱。

其余要点：

| 项 | 结论 |
|---|---|
| 规格 | `p2s.2xlarge.8`（V100 32 GB）正式训练；`pi2.2xlarge.4`（T4 16 GB）先跑通流程 |
| 镜像 | **用干净的 Ubuntu 22.04**，自己装 torch cu121 —— 不要用镜像市场的深度学习镜像（版本对不上） |
| 数据 | **从本机 tar + scp**（1.36 GiB），不要在云机上重下 Edinburgh（国内 30–400 KB/s） |
| 省钱 | 训练完 **释放实例 + 删数据盘 + 释放 EIP**（关机只停算力费，**硬盘和 IP 照收**） |
| 断线 | tmux + `--resume`（优化器状态、随机数状态都在 checkpoint 里） |

### 4d.6 `infer.py` 加了 **stage2 路线**

| 路线 | δ 的来源 | 防御是否生效 |
|---|---|---|
| `clone` | —（用克隆机重新合成） | ❌ 且 Sim_prot 有 0.82 的结构性天花板（§3c） |
| `stage1` | 随机高斯噪声 | ❌（L1 占位） |
| **`stage2`** | **G(h) —— 训练好的生成器** | ✅ **这才是论文的防御** |

`stage1` 与 `stage2` 的**唯一区别就是 δ 从哪来**，其余（内容码、重建、声码器、
攻击者对比）完全一样。CLI 现在可以 `--mode {clone,stage1,stage2,auto}` 了
（之前 `main()` 里根本没暴露 `mode`，只能跑 clone）。

---

## 4e. ★★★ 2026-11-21：M3 的 Sim_prot 到底是多少 —— 一个"空间错配"引发的假象（成员 A）

> **与上面 §4d 第 7 条是同一个坑，两边独立发现、结论一致。**
> 这一节记**排查过程**，因为它同时暴露了
> ① 一个评估口径的坑（**B 的 `evaluate.py` 也会踩**）
> ② 一条"我们其实已经达标了"的结论。

### 4e.1 现象

500,000 步训完后，`scripts/eval_stage1.py` 报：

```
平均 Sim_prot : 0.459     ← 远低于 M3 要求的 0.9（论文 0.95）
平均 L_rec    : 0.0438
重建音 RMS    : 原音的 1/10（0.0028 vs 0.0551）
```

**五层排查全部指向"重建质量差"**：

| 诊断 | 结果 | 当时的结论 |
|---|---|---|
| 纯声码器往返（原 mel → HiFi-GAN → 波形） | 0.9856 | 声码器无损 ✅ |
| 已见说话人（train）vs 未见（test） | 0.387 vs 0.381 | **不是过拟合** |
| 给模型 mel 加 σ=0.02 噪声 | Sim_prot 掉到 0.35 | 误差不是关键，差异是结构性的 |
| 原始 mel std vs 模型 mel std | 0.2065 vs **0.1841** | **过度平滑**：峰值被压平 |
| 过拟合 8 个样本 3000 步 | L_rec 1.306→0.258 仍在降 | **实现没问题，是训练量不够** |

⇒ 一度以为是 "MSE 把 mel 峰值压平了"，并为此准备改 `L_rec`
（加梯度项 / 换 L1 / 换 STFT 损失）。

### 4e.2 ★ 真正的根因：**评测用了「不是训练时那个」的说话人编码器**

Stage 1 训练时：

```python
embedding = speaker_encoder(mel40)     # ★ fine_tune=True，E_spk **正在被微调**
```

而 checkpoint 里**存了微调后的 E_spk**（`"speaker_encoder"` 键）。
但评测脚本当时用的是 `TrainableSpeakerEncoder(fine_tune=False)` ——
**上游预训练**的 E_spk，没装 checkpoint 里的权重。

⇒ **两者的嵌入不在同一个空间里**，于是：
- 解码器收到的是「微调空间的 h」，而我们拿「预训练空间的 E_spk」去评判 ⇒ 全部指标偏低
- 逐帧能量包络看起来被"压平"，其实只是**两个空间的 h 对不上**产生的假象

### 4e.3 装上微调 E_spk 之后（只改评测，**模型一个字没动**）

| E_spk | L_rec | Sim_prot | RMS 比 |
|---|---|---|---|
| 上游预训练（**之前用的**） | 0.04390 | 0.4367 | 0.101 |
| ★ checkpoint 里的微调 | **0.01928** | **0.999996** | **0.707** |

**⇒ Sim_prot = 0.999996，M3 通过。**

### 4e.4 ⚠️ 这个坑的三个要点（**B 的 `evaluate.py` 也要注意**）

1. **微调后的 E_spk 与上游预训练 E_spk 的嵌入空间差异极大** ——
   同一段原音在两者下的嵌入**只相关 0.518**。
   ⇒ **任何比较都必须在同一个 E_spk 下做**。
2. **论文的 E_spk 是"微调后"那个**（§3.1 "initialized from pretrained weights and
   fine-tuned end-to-end during Stage 1 training"）⇒
   **Sim_prot / Sim_clone 应该全线统一用微调后的 E_spk**。
3. **但说话人分类器 W_s 是例外** —— 我们的 W_s 是在**预训练** E_spk 的空间里训的
   （第 1 步 `fine_tune=False`），所以问它"这是谁"时也必须用**预训练** E_spk 的嵌入。
   （`eval_stage1.py` 现在两个编码器各司其职：微调 E_spk 算 Sim_prot，预训练 E_spk 喂 W_s。）

> ⭐ 附带的一个**有意思的推论**：因为微调把 E_spk 的空间挪走了，
> **上游那个"没微调过的"说话人识别器认不出重建出来的语音**
> （预训练 E_spk 下，原音 vs 重建音只有 0.60）⇒
> 对攻击者而言，这本身就是一种"防御"。报告里要如实写清楚这是哪一部分的效果。

### 4e.5 另一个教训：**13.5 小时的训练成果被一次冒烟测试覆盖了**

排查过程中跑了一次 20 步的 `--smoke` 训练，它结束时把
`weights/stage1_model.pt` 覆盖了 —— 500,000 步的权重没了
（`stage1_last_ckpt.pt` 因为只在每 2000 步写，侥幸保住了 6000 步那份，
但也不是 500k 那份）。

⇒ `stage1/train.py` 已加固（2026-11-21）：
- **冒烟/未跑满的步数不再覆盖正式权重**，另存为 `stage1_partial_step{N}.pt`；
- 新增 **`stage1_best.pt`**：每次验证时若 `val total` 更低就更新
  （轻量，不含优化器，115 MB）—— **最优不一定在最后一步**，而且以后不会再被覆盖；
- `stage1_last_ckpt.pt` 仍是含优化器的完整断点（358 MB），用于 `--resume`。

---

## 4f. ★★ 2026-11-23：把「A 交权重之后要干什么」写成手册（成员 B）

**背景**：成员 A 说 Stage 1 还要一段时间（重建质量 Sim_prot 0.459，要改 `L_rec` 重训）。
这一轮不训练，**把成员 B 剩下的全部任务写成一本可照抄的操作手册**
（[`docs/notes/member_b_runbook.md`](notes/member_b_runbook.md)），
等 A 的正式权重一到手就能照着跑。

### 4f.1 ★★ 写手册时抓到的**评测陷阱**（不改的话，手册里的评测步骤会得出错误结论）

`evaluate.py` 原来只有 `--mode {none, random, trained}`，
而 `--mode` **只决定扰动从哪来**，**不决定保护链**：

```
--mode trained  ⇒  δ = G(h)
                  但保护后语音仍然是 "用克隆机重新合成" ⇒ ★ clone 路线
```

⇒ 而 clone 路线有 **0.82 的结构性天花板**（§3c 实测：重新合成本身就改音色/韵律，
与扰动强弱无关）。**拿它的 Sim_prot 去比论文的 0.95，会得出"我们差很远"的错误结论**。

**修法**（2026-11-23）：

| # | 改动 | 说明 |
|---|---|---|
| 1 | 新增 `--protect-mode {clone,stage1,stage2}` | **默认 `clone` ⇒ 现有行为完全不变**（成员 A 的评测数字不受影响）；论文那组数字走 `stage2` |
| 2 | 评测的嵌入统一走 `protector.embed_for_route(wav, route)` | 删掉原来那个进程内缓存的 `_embed()`（固定用上游 SpeakerBridge）。**每条路线用它自己的 E_spk** —— 否则"用 A 的编码器训、用 B 的编码器测"，指标不可比**而且不报错** |
| 3 | `stage2` 路线的 δ 由 Protector 内部算 | 它用的是 **checkpoint 里指定的 E_spk**，与训练时一致。自己另算一份容易对不上 |
| 4 | `make_sample.py` 加 `--mode` | 平台样例要用训练好的 G 出，否则样例展示的还是随机扰动 |
| 5 | 消融能真跑了 | `stage2/losses.py::perceptual_loss` 加 `use_mel_loss` + `stage2/train.py` 加 `--no-mel-loss` ⇒ 论文 Table 2 的 "w/o L_mel" 终于能训（`config.ABLATION` 里写着 `use_mel_loss: False`，**之前根本没实现**） |
| 6 | 云机脚本补参数 | `--outdir` / `--lambda-*` / `--no-mel-loss` / `--log-every` / `--save-every` ⇒ 消融三组用同一条命令（`--outdir` 一定要给，否则覆盖主实验权重） |

### 4f.2 手册的结构

```
1  本机准备（拉代码 / 自检 / 数据对账 / 收 A 的权重 + 验真）
2  开云机（华为云 ECS）
3  装环境 + 自检
4  训练 Stage 2（冒烟 → 正式 → 盯进度 → 续训 → 何时停）
5  评测（三条基线 → 绝对 WER → 消融）
6  收结果 + 释放实例
7  平台样例 + 端到端
8  文档与交付（+ 提交前的全量回归）
9  故障排查速查
附录 A 论文对照表   附录 B 时间与花钱预算
```

每一步都有「**看什么数**」和「**不对怎么办**」—— 这是这个仓库一贯的做法
（任务板 §7 的"三个最容易踩的坑"就是同一个思路）。

### 4f.3 手册里最该记住的三条

1. **评测一定要加 `--protect-mode stage2`**，否则走 clone 路线，数字不能跟论文比；
2. **训完要释放实例**（不是关机）—— 关机只停算力费，云硬盘和 EIP 照收；
3. **消融要重训**，不是改评测参数。

### 4f.4 实测（本机 CPU，2026-11-23）

| 检查 | 结果 |
|---|---|
| `MODE stage1`（论文原版 D(c,h_adv)→HiFi-GAN，δ 随机） | ✅ 11.3 s/条；保护后 (42752,)、攻击者克隆音 (42752,) |
| `MODE stage2`（论文原版 + 训练好的 G） | ✅ 5.9 s/条；**δ 的来源 = `G(h)`** ✅ |
| `MODE stage2` + α=0（**无防御基线，不需要** Stage 2 的 checkpoint） | ✅ sim_prot 0.4773 |
| 回归 | verify_gradients **20/20**、verify_psychoacoustics **20/20**、verify_mel **4/4**、verify_hifigan ✅、verify_rtc_mel ✅、test_stage1 ✅、test_audio_utils ✅ |

> ⚠️ 上面 sim_prot 只有 0.42–0.48，是因为这里用的是成员 A 的 **3 步冒烟** Stage 1 权重
> （重建质量极差）。**换成正式权重后才是有意义的数字** —— 这也再次说明
> 「Stage 1 的 Sim_prot 上不去，Stage 2 训得再好也没用」。

---

## 4g. ★★★ 2026-11-25：`0.999996` 本身是假的 —— E_spk 空间塌缩，以及"开云机前的三连预检"（成员 B）

§4e 把 Sim_prot 从 0.44 修成 0.999996，结论是"评测拿错了 E_spk"。**那一天是对的，但结论只走了一半**：
换上的那个"微调后的 E_spk"自己已经退化到不能当度量用了。
这一节是 B 在准备 Stage 2 正式训练时发现的，直接改变了报数口径和开不开云机的判断。

### 4g.1 触发点：A 的 250k JSON 自己推翻了 §4e

`weights/stage1_fresh_200k/stage1_step250000_wide_test_bn.json`（A 交权重时附的评测）里：

| 打分口径 | 同人间余弦 | 异人间余弦 | **gap** |
|---|---|---|---|
| checkpoint 里**微调过**的 GE2E（§4e 换上的那个） | 0.9453 | 0.8958 | **+0.049** |
| **上游预训练** GE2E | 0.7663 | 0.5670 | **+0.199** |

- B 在本机用 A 的权重**独立复算**过异人间余弦，gap 只有 **+0.013**，比上表还糟 ⇒ 塌缩是真的，不是口径写错；
- 同一份 JSON 的 `upstream_speaker_top1` = clean **0.729** / oracle 0.757 / **rebuilt 0.229** ⇒
  用上游 GE2E 做说话人识别时重建语音 Top-1 掉到 22.9%，而在微调空间里同一批语音**人人 ≈ 0.9**；
- 同一份 JSON 的 `upstream_original_rebuilt_cosine` = **0.7058**，与 B 本机 upstream 口径复算的
  Sim_prot **0.6994** 互相印证 ⇒ 重建本身是**中等水平**，不是"接近完美"。

> ⇒ 在 gap ≈ 0.05 的空间里，`Sim_prot ≈ 1` **任何两条语音都能拿到**。
> 它证明的是"空间退化了"，不是"重建得好"。
> 反过来，论文那个可比的口径是**上游 GE2E**（说话人识别本来就是用通用模型考核的），
> 而它给出的 Sim_prot 只有 **0.70**。
> 这不是"哪个数字更好看"的问题 —— **同一份权重能讲出两个相反的故事**，
> 所以不注明 E_spk 的报数是不可用的。

### 4g.2 处理决定（三条，都是"不这么做就报错数"级别）

1. **评测双口径并报**：`evaluate.py` 加 `--espk-space {auto,upstream,stage1_finetuned}`
   和 `--dual-espk`，报告里同时给两套 Sim_prot / Sim_clone，另存主口径 `espk_space`；
2. **把"L_de 考核在哪个空间"和"δ 在哪个空间生成"解耦**：
   δ 仍然生成于 `--speaker-encoder stage1`（= checkpoint 里微调过的 E2E，这是 §4e.4 立的规矩，**不能动**，
   否则 G 的输入离分布、防御静默失效），但**考核挪到上游空间** —— 新开关
   `--de-space {stage1,upstream}`，默认跟随 `--speaker-encoder`，正式训练用 `--de-space upstream`；
3. **选择写进 checkpoint**：`checkpoint_meta["de_space"]`，`infer.py` 加载时读出并打印
   "L_de 考核于 … 空间"，`evaluate.py` 把 `report["de_space"]` 一起落盘 ⇒
   **权重、推理、评测三处不可能对不上**，事后也不会忘记当时用的哪个口径。

### 4g.3 改了什么（只列事实，不写心得）

| 文件 | 改动 |
|---|---|
| `stage2/pipeline.py` | 新增第二个编码器槽 `de_speaker`（默认 = `speaker`），`describe()` 多打一行"L_de 度量" |
| `stage2/train.py` | `build_pipeline` 按 `--de-space` 装第二个编码器并冻结；`compute_losses` 里 `de_reference` 与 `defense` 走 `de_encoder`，日志多一列 `sim@de`；新增 `--de-space` / `--overfit N` / `--lr`；`checkpoint_meta["de_space"]` |
| `infer.py` | `stage2_de_space` 从 checkpoint 读出、打印 |
| `evaluate.py` | `report["de_space"]`（与已有 `espk_space` / `delta_space` / `secondary_espk` 并存） |
| `scripts/cloud_train_stage2.sh` | `--de-space` 透传（云机默认 `upstream`）+ `export PYTHONUNBUFFERED=1` |
| **新增 `stage2/diagnose_gradient.py`** | 只反传 λ_de·L_de，逐层打印 G 的梯度范数 + δ 的角度预算（§4g.4 / §4g.5 那两个数的出处） |
| **新增 `stage2/diagnose_delta_oracle.py`** | 自由 δ 的 oracle，`--delta-mode unit` 强制 `‖δ‖=1` 与 G 等预算（§4g.7 的出处） |
| **新增 `stage2/diagnose_alpha_sweep.py`** | α–Sim_clone / α–可听度扫描，输出 JSON（§4g.8） |

### 4g.4 ★ 开云机之前的固定动作：本机三连预检（全部实测，非估计）

代金券是按小时烧的，而"训练机制通了"不等于"防御学到了东西"。
本机纯 CPU 慢，但**把变量一个个摘掉**比在云机上跑 200k 步便宜得多：

| 一发 | 配置 | 结果 | 排除了什么 |
|---|---|---|---|
| 150 步预检 | 2 s/条、batch 4、`--de-space upstream`，~12 s/步、28 分钟 | `de` 0.5926→0.5646→0.5757、`sim@de` 0.6796→0.6486→0.6525（每段 16 点均值）；**`mask` 0.70→1.12、`per` 0.077→0.120 同步变差** | 方向轻正但**淹没在噪声里**，150 步定不了；同时说明 λ_per=15 没按住扰动 |
| 过拟合·忠实目标 | `--overfit 4 --lambda-gan 0`，λ_per=15、lr 1e-4，60 步 | `de` 0.549–0.596、`sim@de` 0.629–0.664 来回 ⇒ **平** | 数据换批造成的抖动（去掉之后照样平） |
| 过拟合·纯 L_de | λ_per=0、lr **1e-3**，120 步 | `de` 0.600 →（55–105 步）0.52–0.55 → 120 步回到 0.575；`sim@de` 全程 0.60–0.71 ⇒ **只降约 10% 且回弹** | λ_per 在对着干（摘掉后照样不降）；学习率太小（10 倍仍只降 10%） |
| 梯度实测 | 同一 batch，只反传 λ_de·L_de（`stage2/diagnose_gradient.py`） | **upstream 空间 ‖∂L/∂θ_G‖ = 37.0**，塌缩的 stage1 空间只有 **7.34** | **不是梯度消失**（37 是健康量级，`verify_gradients.py` 20/20）；也说明"换空间"这步确实把信号放大了 5 倍 |
| 过拟合·α 放大 | λ_per=0、lr 1e-4、**α=0.5**，100 步 | `de` 0.435→**0.555（升）**、`sim@de` 0.504→**0.556（升）**；起点比 α=0.1 低 0.15 | **不是"多跑步数就会好"** —— 起点是 α 送的，训练把它练回去了 |

### 4g.5 ★ α 的角度预算是硬的（这是这组实验的物理背景）

`perturb_embedding()` 做的是 `h_adv = normalize(h + α·δ)`。实测 `‖h‖ = 1`、
随机初始化的生成器输出 `‖δ‖ ≈ 1` ⇒ **α=0.10 时 h_adv 只偏离 5.73°**（cos ≥ 0.995）。
换句话说，论文式(12) 的防御**完全依赖"5.7° 的嵌入偏移经 Stage 1 解码 + 声码器之后，
被另一个 GE2E 读出大变化"** —— 放大链是唯一的杠杆，G 学的只是"往哪个 5.7° 偏"。
α 提到 0.5 时数字立刻变好（0.66 → 0.50），这正说明**当前分数是预算决定的，不是学习决定的**。

> ⚠️ 方法论提醒：自由 δ 的 oracle 实验里，如果 `‖δ‖` 从 1 涨到 3.4，
> 实际角度预算就被偷偷放大到 ~19°，**等于换了个方法在算分数**。
> 所以 oracle 必须同时跑 `‖δ‖ = 1` 的等预算版本才可比。

### 4g.6 判读（诚实版，写给三个月后的我们）

1. **`Sim_prot = 0.999996` 从今天的文档起降级为"塌缩空间的副产品"**，
   对外主报数改用**上游口径**，并且必须与同人间/异人间余弦一起给；
2. **`--de-space upstream` 是对的**（梯度 37 vs 7.3 是硬证据），但它**没有让防御学起来**；
3. 现在最可疑的是**放大链容量**：5.7° 的偏移要经过 AutoVC 解码 → mel → HiFi-GAN →
   再被上游 GE2E 分段读出，这条链能不能把 0.5 的余弦推到论文的 0.13，
   是**方法层面**的问题，不是训练时长问题；
4. ⇒ **决定：暂缓 200k 步云机训练**，判据见 §4g.7；改走不花钱的替代路径（§5.1 三条基线 + α–Sim_clone 曲线）。
   **2026-11-25 已确认执行**（成员 B 与发起人二选一里选了"不开云机，走负结果 + 曲线路线"）
   ⇒ 代金券留着，Stage 2 的交付形态变成**"方法层短板的定量定位 + 三条基线 + α 曲线"**，
   而不是"一个训了 200k 步但仍然到不了 0.13 的 G"。

### 4g.7 ★ 最后一刀：自由 δ 的 oracle（绕开 G，直接问"目标本身可达多低"）

`stage2/diagnose_delta_oracle.py`：把每条语音的 δ 当成可优化参数（256 个自由度、逐条独立），
没有任何共享函数的容量限制 —— 如果连它都压不动，就与 G、与步数彻底无关。
★ 关键设置 `--delta-mode unit`：**强制 `‖δ‖ = 1`**，与 G 随机初始化时的工作点**完全等预算**
（α=0.1 ⇒ 5.71°）；不加这个约束时优化器会把 `‖δ‖` 推到 3.4，实际预算变成 19.4°，结论就不可比了。

| 版本（4 条 × 2 s，Adam lr 0.05） | 步 1 | 10 | 20 | 30 | 40 | 50 | 角度预算 |
|---|---|---|---|---|---|---|---|
| 自由 δ，不约束模长 | 0.5582 | 0.4712 | — | — | — | — | `‖δ‖` 1.78→3.39 ⇒ **5.7°→19.4°** |
| **等预算 `‖δ‖=1`**（一发） | 0.5531 | 0.4651 | 0.4944 | 0.4585 | 0.4229 | 0.4415 | **5.71°** |
| **等预算 `‖δ‖=1`**（另一发，独立随机初始化；70 步进入平台后手动停） | 0.5537 | 0.5307 | 0.5148 | 0.4971 | — | 0.4885 → 60 步 **0.4498** → 70 步 **0.4518** | **5.71°** |
| 参照：**α=0（完全无防御）** | 0.5833 | — | — | — | — | — | 0° |

**读法**：

| 量 | 数 | 含义 |
|---|---|---|
| 无防御的上游口径 Sim_clone | **0.583** | 起点 |
| 等预算自由 δ 的地板 | **≈0.42–0.45**（两发分别在 40–70 步进入平台，互相差 ~0.03） | **每条语音 256 个自由度直接优化**能到的下界 |
| G（593k 共享参数）训到 | **0.52** | 已经接近上界，只差 ~0.05 |
| 论文目标 | **0.13** | 需要下降 **0.45**，而这条链总共只给了 **0.10** |

> ⚠️ 附带证据：`‖δ‖=1` 两发的 mel  MSE 从 0.0067 升到 0.0113 ——
> 自由 δ 换来那 0.1 的下降，靠的是**在可听度边缘做牺牲**，
> 而这正是 λ_per=15（论文"不可听"的前提）在训练里禁止做的事。
> 与"摘掉 λ_per 后 G 只降 10% 且回弹"完全自洽。
>
> ★★ **结论（写进答辩也站得住）**：在 **α=0.1 的不可听性约束 + 我们这条 Stage 1 放大链**下，
> 上游口径的 Sim_clone 下界约 **0.45**，**达不到论文量级不是训练时长问题**。
> 200k 步的期望收益 ≤ 0.07，**不值得烧代金券**；
> 要缩小差距必须动**方法**（α、Stage 1 的重建质量 `upstream_speaker_top1 rebuilt = 0.229`、
> 或换放大链更强的扰动落点），而不是加步数。
> 关于 α：把训练里的 α 提到 0.5 那一发，`sim@de` 的**起点**立刻从 ~0.66 降到 ~0.50（batch 8 / 整句口径），
> 但**训练 100 步又把它练回去了**（§4g.4 最后一行）⇒ "加大 α 能买到多少"必须由
> **等步数、等预算**的扫描来回答 —— 已答（§4g.8 正式版）：**能买到一点，但 10 倍预算只换 0.06，
> 而 Sim_prot 要陪葬 0.14**，所以结论仍然是"链路问题，不是 α 问题"。

> ⚠️ **这组 oracle 数字的三个前提**（答辩可能会被追问，先写清楚）：
> ① 只用了 **4 条 × 2 s** 的语音做逐条优化，是"上界"而不是"测试集平均" ——
> 样本少会让地板看起来更低（更容易过拟合到几条数据），所以真实测试集的下界**只会更高**；
> ② 攻击者的说话人嵌入用 `E_stage1(x_prot)` 提取（与 δ 生成空间一致，这是 §4e.4 立的规矩），
> 打分则用 `E_upstream` —— 这已经是**对我们最不利**的组合（放大链走完再换尺子量）；
> ③ 参照语音是 `V(x)`（重建后的原句）而不是 `x` 本身，与 `train.py::compute_losses` 完全一致，
> 所以这几个数之间可以横向比，但**与论文 Table 1 的绝对值比较时仍要注明口径**。


### 4g.8 替代交付路径（都不花钱）

| 路径 | 工具 | 产出 |
|---|---|---|
| §5.1 三条基线（无防御 / 高斯 / 我们的 G）钉同一口径 | `evaluate.py --protect-mode stage2 --espk-space upstream --dual-espk` | 可与论文 Table 1 横向比的三行 + 塌缩口径附列 |
| **α–Sim_clone / α–可听度曲线** ✅ **已实测**（2026-11-25，4 档 × 40 步，见本节末） | `stage2/diagnose_alpha_sweep.py`（每个 α 给随机 δ、自由 δ 上界、Sim_prot、mel  MSE、波形平均差） | 定量回答指南 §11 的"α 敏感吗"，并把上面的结论变成一张图 |
| 判据链本身 | `stage2/diagnose_gradient.py` + `diagnose_delta_oracle.py` | 别人质疑"是不是没训够"时的标准答复流程 |

**★★ α 扫描正式版（2026-11-25，`--alphas 0.1,0.3,0.5,1.0 --steps 40 --count 4 --lr 0.05`，等预算 `‖δ‖=1`，JSON 存 `.tmpB/alpha_sweep_40.json`）**

| α | 角度 | 随机 δ 的 Sim_clone | 优化 40 步后 | 学到 | Sim_prot@upstream ↑ | mel  MSE | 波形平均差 |
|---|---|---|---|---|---|---|---|
| 0.1 | 5.71° | 0.5348 | **0.4978** | 0.037 | 0.6363 | 0.00785 | 0.07861 |
| 0.3 | 16.70° | 0.5754 | **0.4769** | 0.099 | 0.5656 | 0.01054 | 0.07793 |
| 0.5 | 26.57° | 0.5473 | **0.4316** | 0.116 | 0.5250 | 0.01439 | 0.07916 |
| 1.0 | 45.00° | 0.4639 | **0.4346** | 0.029 | 0.4999 | **0.01950** | 0.09068 |

> **读法（这版可以定论了，四条）**：
> ① **等步数之后 α 确实有杠杆**："学到多少"从 α=0.1 的 0.037 涨到 α=0.5 的 0.116
> —— 10 步版看不出来，是因为各档都还没走到自己的平台（见下面那张旧表）；
> ② **但杠杆换不来论文量级**：把预算从 5.71° 一路加到 **45°（α 放大 10 倍）**，
> 逐条自由优化的 Sim_clone 也只从 0.4978 走到 **0.4346**，离论文的 **0.13** 还差 **3.3 倍**
> ⇒ **瓶颈是放大链本身（`upstream_speaker_top1 rebuilt = 0.229`），不是 α、不是步数**；
> ③ **代价是同步且更陡的**：α 0.1→1.0 时 Sim_prot 从 0.6363 掉到 **0.4999**（−0.136）、
> mel  MSE 从 0.00785 涨到 **0.01950**（×2.5）⇒ **每买 0.01 的 Sim_clone 要付 ≈0.02 的 Sim_prot**，
> 而 α=1.0 早已跳出"不可听扰动"的设定（论文用 α=0.1 正是为了守住这一点）
> —— 这一条本身就是对指南 §11"α 敏感吗"的定量回答，**且支持论文的取值**；
> ④ α=1.0 的"学到"反而最小（0.029）：预算大到一定程度时**随机 δ 已经先到 0.4639**，
> 离平台只剩 0.03，40 步就够 ⇒ 说明**大 α 的收益主要来自"扰动本身"而不是"学会往哪偏"**，
> 这跟 G 在 α=0.1 下 300 步只降 0.06 是同一件事的两面。

**α 扫描第一版（2026-11-25，`--alphas 0,0.1,0.2,0.3,0.5,1.0 --steps 10 --count 4`，等预算 `‖δ‖=1`）
—— 保留它不是为了引用数字，而是为了留下"步数不足会把 α 差异盖住"这个教训**

| α | 角度 | 随机 δ 的 Sim_clone | 优化 10 步后 | 学到 | Sim_prot@upstream ↑ | mel  MSE | 波形平均差 |
|---|---|---|---|---|---|---|---|
| 0.0 | 0° | 0.5835 | 0.5835 | — | **0.7157** | 0.00645 | 0.06231 |
| 0.1 | 5.71° | 0.5794 | 0.5332 | 0.046 | 0.7085 | 0.00672 | 0.06195 |
| 0.2 | 11.31° | 0.5427 | 0.5128 | 0.030 | 0.6577 | 0.00734 | 0.06059 |
| 0.3 | 16.70° | 0.5616 | 0.5207 | 0.041 | 0.6892 | 0.00908 | 0.06276 |
| 0.5 | 26.57° | 0.5560 | 0.5106 | 0.046 | **0.6286** | **0.01125** | 0.06052 |
| 1.0 | 45.00° | 0.4983 | 0.4724 | 0.026 | **0.5573** | **0.01448** | 0.06295 |

> **★ 跨发对比的前提（2026-11-25 事后发现的坑，先说清再读表）**：
> 上表的"随机 δ"与 §4g.7 那张表**不在同一条起点上**（α=0.1 的随机 δ 两发分别是
> 0.5794 与 0.5348，差 **0.045**）。根因是 `stage2/dataset.py::_crop()` 用
> **Python 的 `random.randint`** 选裁剪起点，而这三个诊断脚本原来只调
> `torch.manual_seed()` ⇒ 每次启动实际读到的音频片段不同。
> **同一次运行内部**各 α 共用同一个 batch，所以**列与列之间可比**；
> **不同运行之间**的单点数字要按 **±0.05** 的抖动读，别把 0.53 与 0.58 的差异当成结论。
> 已修：三个 `stage2/diagnose_*.py` 改用 `stage2.common.set_seed()`（连 `random`/`np.random` 一起固定）。
>
> **诚实的读法（这一版还不能定论）**：
> ⓪ **口径**：表中 `Sim_clone` 与 `Sim_prot` 都是**上游 GE2E 口径**（脚本里键名
> `sim_prot_upstream` / JSON 里同样），**不是** §4g.5 那种塌缩口径 —— 所以 α=0 行
> 的 Sim_prot **0.7157** 就是"无防御时本来能保住多少音色"，与 §5.1 基线①同源；
> ① 每个 α 只优化 **10 步**，而 §4g.7 显示自由 δ 要到 **40–70 步**才进入平台
> ⇒ 表中"优化 10 步后"这一列**还没走到各自的地板**，α 之间的差异被"Adam 走到哪一步"盖住了，
> **不能**用来回答"加大 α 能买到多少防御"；
> ② 能用的结论只有三条：**随机 δ（不训练）在 α ≤ 0.5 时几乎不随 α 变化**（0.5835→0.5560），
> **但 Sim_prot / mel  MSE 随 α 单调变差**（0.7157→0.6286→**α=1.0 时 0.5573**；
> 0.00645→0.01125→**0.01448**）⇒ 加大 α 的**代价是确定的**，收益还没被证明；
> ③ 一个有用的副观察：**波形平均差 `|Δwav|` 全程基本不变**（0.0606–0.0630），
> 而 mel  MSE 随 α 明显上升 ⇒ 扰动改的是**频谱结构**、不是整体能量/宽带噪声
> —— 这正好是论文"人机感知差异"想利用的方向，说明**方向没错、幅度不够**；
> ④ 所以补了一发**每档 40 步**的扫描 ⇒ **已补，结论见本节最上面的"正式版"表**
> （旧表的 ②⑤ 只能当趋势看，正式版把"α 有没有杠杆"这件事答掉了）；
> ⑤ 但**"随机 δ"那一列是干净的可比量**（同一批 δ、同一预算，不涉及优化进度），
> 从它能看到一条明确的交换比：预算从 5.7° 加到 **45°（α=1.0，8 倍）**，
> Sim_clone 只从 0.5794 换到 0.4983（**−0.08**），Sim_prot 却从 0.7085 掉到 0.5573（**−0.15**）
> ⇒ **在 α=0.1 附近，"加大 α"是用 2 倍于收益的可听度损失买防御强度**，
> 这也解释了论文为什么把 α 定在 0.1 这个量级。

### 4g.9 实测踩到的两个 Windows 坑（都会让"看起来卡死"）

| 坑 | 现象 | 解法 |
|---|---|---|
| **stdout 块缓冲** | `nohup python … > log` 十几分钟零输出，差点 kill 掉一个正在正常跑的任务 | `python -u` / `PYTHONUNBUFFERED=1`；旁证用 `<outdir>/train_log.jsonl`（每行 `with` 写完就关，**永远实时**） |
| **控制台 GBK** | 脚本里打印中文/`‖δ‖` 这类字符直接 `UnicodeEncodeError` | `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`（`.tmpB/*` 与 `evaluate.py` 已加） |
| **"零输出"到底是缓冲还是真卡死** | 光看日志文件分不清（两种都是"没动静"），误 kill 会白扔十几分钟 | 量进程而不是量日志：`powershell -NoProfile -Command '$p=Get-Process -Id <PID>; $p.CPU'` 连查两次，**CPU 秒在涨 = 在算**（本脚本另外会写 JSON，跑完才有文件，中途只能靠这个判活） |

---

## 4h. ★ §5.1 三条基线实测（论文路线 + upstream 口径，成员 B，2026-11-25）

**命令**（三条只差 `--mode`，其余参数完全一致 ⇒ 严格配对）：

```bash
D:/conda/envs/vpp/python.exe -u algorithms/voiceprint_protection/evaluate.py \
    --mode none|random|trained --protect-mode stage2 --alpha 0.1 \
    --stage1-checkpoint algorithms/voiceprint_protection/weights/stage1_fresh_200k/stage1_step250000_wide_test_bn.pt \
    --generator-checkpoint .tmpB/stage2_probe2/stage2_last.pt \
    --split test --limit-per-speaker 5 --espk-space upstream \
    --output results/stage2_eval/base_*.json
```

> **样本与可复现性**：test 清单 20 人 × 5 = **100 条**，`evaluate.py::load_test_items()`
> 按说话人排序后**等间隔**取 5 条 ⇒ 三条跑的是**同一批语音**，可以直接配对相减；
> 这条路径**不做随机裁剪**（整条读入），`--mode random` 的高斯 δ 由
> `infer.random_perturbation(seed=args.seed)` 显式播种 ⇒ **三轮都是确定性的**，
> 不像 §4g 那些诊断脚本有 ±0.05 的裁剪抖动。每轮 CPU **377 s**（3.8 s/条）。
>
> **⚠️ ③ 的权重身份**：`.tmpB/stage2_probe2/stage2_last.pt` 是**本机 150 步短训 G**
> （α=0.1、λ_per=15、λ_de=5、λ_mask=0.1、λ_GAN=1、batch 4、`--de-space upstream`），
> **不是**论文量级的 200k 步权重。这一行只能读作"短训 G 的方向性证据"，
> **不能当最终交付的 Ours 数字**（要正式数字必须回到 §4.2 的开台闸门）。

| 变体 | E_spk 口径 | Sim_prot ↑ | Sim_clone ↓ | DSR(0.3 / 0.4 / 0.5) ↑ |
|---|---|---|---|---|
| ① 无防御（α=0） | upstream | 0.6698 ±0.0812 | 0.6169 ±0.0797 | 0% / 0% / **9.0%** |
| ② 高斯噪声（random） | upstream | 0.6650 | 0.6163 | 0% / 0% / **12.0%** |
| ③ 我们的 G（**150 步短训**） | upstream | **0.7029** ±0.0747 | **0.6775** | 0% / 0% / **1.0%** |
| —— 同一批波形，塌缩口径 `stage1_finetuned` —— | | 0.9243 / 0.9213 / 0.9444 | 0.9048 / 0.9041 / 0.9229 | 全 0% / 0% / 0% |
| 论文 Table 1（可比口径 = 上游 GE2E） | | 0.91 / 0.85 / **0.95** | 0.91 / 0.35 / **0.13** | 0% / 45.2%（τ=0.5）/ **87.2%** |

**三条结论**：

1. **★ DSR 有"链路地板"，无防御也能拿 9%** —— `evaluate.py:452` 的定义是
   `DSR(τ) = mean(Sim_clone < τ)`，而我们的 Stage 1 放大链自己就有 **9% 的克隆音**
   掉到 0.5 以下（与 A 报的 `upstream_speaker_top1 rebuilt = 0.229` 同源）。
   ⇒ **对外报 DSR 必须把"无防御 DSR"并列给出**，否则这 9% 会被误读成防御效果；
   真正可比的量是 **ΔDSR（本方法 − 无防御）**，论文那一行的地板是 0%，不是我们的 9%。
2. **高斯噪声与无防御不可区分**（Sim_clone 0.6163 vs 0.6169，差 **0.0006**）
   —— 随机方向 + 5.71° 预算推不动身份相似度。论文 Table 1 的 Gaussian Noise 行能到
   0.35，是因为它扰动的是**声纹空间本身**，而不是我们这种"嵌入 → 解码 → 声码 → 再被 GE2E 读回"的放大链。
   这条与 §3i.5 的旧结论（clone 路线 0.6995 vs 0.7080）同号同量级
   ⇒ **结论不依赖保护链怎么选**。
3. **★★ 短训 G 的方向是反的** —— ③ 相对 ①：Sim_prot **+0.033**（保真更好）、
   Sim_clone **+0.061**（克隆反而更像受害者）、DSR(0.5) **9% → 1%**。
   也就是 **G 学到的 δ 在帮重建链"守住身份"，而不是破坏它**。
   机理与 §4g 完全自洽：λ_per=15、λ_GAN=1、λ_mask 都在推"改动要小、要像自然语音"，
   而 `L_de` 能借力的只有 5.71°，**等预算自由 δ 的地板也只到 0.42–0.50**
   ⇒ 多目标博弈里 `L_de` 输给了保真项，G 顺手把重建质量抬了一点。
   ⚠️ 严格说不能排除"200k 步翻正"，但翻正要 0.6775 → 0.13（降 **0.55**），
   而逐条 256 维自由优化的上界都到不了 0.42 ⇒ **加步数补不上这 0.3**，
   这正是 §4.2 决定不开台的量化依据。

**怎么用**：M4 要求的"三条基线"到此齐了（含塌缩口径附列与 DSR 地板说明），
写进报告时 ③ 那一行**必须注明"150 步短训 G"**，并把 ① 的 DSR 当作扣除基线。

---

## 5. 已知坑（组内共享，别重复踩）


| 坑 | 现象 | 出处 |
|---|---|---|
| ★★ **E_spk 的「空间错配」** | Stage 1 会**微调** E_spk（论文 §3.1 明写 "fine-tuned end-to-end"），checkpoint 里存了微调后的权重。**评测时若用上游预训练 E_spk**，两者嵌入只相关 **0.518** ⇒ Sim_prot 0.44（装上后是 **0.999996**）、RMS 差 10 倍。**不报错，只是数字偏低** | experiments.md §4e |
| ★★★ **换上微调 E_spk 之后，那个 0.999996 也别信**（2026-11-25 更正） | 微调 250k 步后的 GE2E **空间塌缩**：同人间 0.9453 / 异人间 0.8958（gap **+0.049**，B 本机独立复算只有 **+0.013**）。这种空间里**任何两条语音都能拿到 ≈1 的余弦** ⇒ "Sim_prot 接近 1"只证明度量退化了。**报数必须注明是哪个 E_spk，并和同人间/异人间余弦一起给**；论文可比的口径是**上游 GE2E**（同一份权重的 Sim_prot = **0.70**） | experiments.md §4g.1–4g.2 |
| ★ **训练与考核可以不同空间，但必须写进 checkpoint** | Stage 2 的 δ **必须**生成于微调 E_spk（否则 G 输入离分布、防御静默失效），但 `L_de` 考核在上游空间才有意义（梯度 **37.0 vs 7.34**）。解耦靠 `--de-space`，落盘靠 `checkpoint_meta["de_space"]`，`infer.py` / `evaluate.py` 自动照着读出 ⇒ 忘了当时用哪个也能复查 | stage2/{pipeline,train}.py + infer.py + evaluate.py（2026-11-25） |
| **`nohup … > log` 看不到输出 ≠ 卡死** | Python 的 stdout 重定向到文件是**块缓冲**（8 KB 一刷），训练每 5 步才打一行 ⇒ 十几分钟零输出，容易误 kill 正常任务。用 `python -u` / `PYTHONUNBUFFERED=1`；旁证看 `<outdir>/train_log.jsonl`（每行写完就关，永远实时） | .tmpB 预检（2026-11-25 真实踩到） |
| **Windows 控制台 GBK** | 脚本打印中文或 `‖δ‖` 这类字符 → `UnicodeEncodeError`。加 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`；另外 Git Bash 里 `powershell` 命令串中的 `$_` 会被双引号展开，要么单引号要么换个写法 | .tmpB/*、evaluate.py |
| **上游 GE2E 的分段前向是 4 倍开销** | `TorchSpeakerEncoder` 照抄上游 `embed_utterance`：160 帧一段、50% 重叠，外面再套 Python 逐条循环 ⇒ 一条 3.8 s 语音变成 ~4 个序列。`--de-space upstream` 每步多 2 趟这种前向 ⇒ 本机整句 batch 8 的步长从 ~33 s 涨到**分钟级**。预检的解法是裁短（`--max-seconds 2 --batch-size 4`）而不是砍步数；云机（T4）上不必裁 | stage2/models.py、runbook §4.1.1 |
| **自由 δ 实验会把 α 的预算偷偷放大** | `perturb_embedding` = `normalize(h + α·δ)`，**δ 没归一化** ⇒ 优化器只要把 `‖δ‖` 从 1 涨到 3.4，实际偏移就从 5.73° 变成 ~19°，"目标可达"的结论就失去可比性。等预算对照必须强制 `‖δ‖ = 1`（`stage2/diagnose_delta_oracle.py --delta-mode unit`） | stage2/diagnose_delta_oracle.py（2026-11-25） |
| ★★ **DSR 不是从 0 起算的** | `DSR(τ) = mean(Sim_clone < τ)` 只看克隆音的余弦掉没掉线，**重建链自己的失真也算掉线** ⇒ 本机 upstream 口径下**无防御就有 DSR(0.5)=9.0%**、高斯噪声 12.0%、150 步 G 反而只有 1.0%。**报 DSR 必须并列"无防御 DSR"，可比量是 ΔDSR** | experiments.md §4h（2026-11-25） |
| **诊断脚本只固定 torch 种子 ≠ 实验可复现** | `stage2/dataset.py::_crop()` 的裁剪起点用 **Python 的 `random`**，只写 `torch.manual_seed()` 时每次启动读到的音频片段不同 ⇒ 同一 α 的"随机 δ"两发差 **0.045**，很容易被误读成"α 有效/无效"。凡是要跨发比数字的脚本，统一调 `stage2.common.set_seed()`（`random` + `np.random` + `torch` 三个一起） | stage2/diagnose_{gradient,delta_oracle,alpha_sweep}.py（2026-11-25 修） |
| **冒烟训练覆盖正式权重** | `--smoke` 结束时把 `stage1_model.pt` 覆盖了 ⇒ 13.5 小时成果全丢。已加固：未跑满不覆盖 + 另存 `stage1_best.pt` | experiments.md §4e.5 |
| **训练与推理的 E_spk 必须一致** | 论文说 E_spk 在 Stage 1 里被微调过。训练用**上游** GE2E、推理用**微调后**的那个 ⇒ 两个嵌入空间不同 ⇒ 生成器 G 的输出**离分布** ⇒ **防御静默失效**（不报错，只是指标全错）。现在把选择写进 checkpoint，推理照着做 | stage2/models.py + stage2/train.py + infer.py（2026-11-21） |
| **局部变量别叫 `config`** | 在 `infer.py` 里写 `config = {...}` 会把模块级的 `import config` 遮掉 ⇒ 右边 `config.SPEAKER_DIM` 变成"引用未赋值的局部变量" ⇒ `UnboundLocalError`。而且它**被 try/except 吞掉了**，只打印一句"加载失败" | infer.py::_infer_generator_config（2026-11-21） |
| **Linear 权重的形状方向** | `trunk.<i>.weight` 是 **(hidden, dim)**，`head.weight` 是 **(dim, hidden)** —— 两个不一样。反了会 size mismatch（好）；若用 LayerNorm 的 1 维权重去数层数，会数出 4 而不是 2（**不好，要靠形状区分**） | infer.py::_infer_generator_config（2026-11-21） |
| **L_dis 的符号方向** | `Σ p log p` **本身就是 −H**；实现里再加一个负号 → 最小化它 = **让分类器更确定** ⇒ 解耦**反了**。现象：训练时 `dis` 停在 0.0016（= H≈0，分类器完全确定）。**loss 照样降、不报错**，只能靠看 `dis` 的数值与方向发现 | stage1/losses.py（2026-11-20 修） |
| **mel 口径的近似换算** | AutoVC mel → HiFi-GAN mel 的仿射换算**不可接受**：两套滤波器组不同，仿射补不回来。实测重建音的说话人余弦从 **0.9746 掉到 0.6983** | TASKS.md §0.10.4c |
| **mel 量纲会毁掉论文的 λ** | HiFi-GAN 原始 mel 值域 [-11.5, 0]，直接用会让 `L_rec` 比论文大 **100 倍**，`λ_cd=1.0` 被稀释 40 倍 ⇒ **论文的超参失效**。解法：可逆的固定仿射归一化（送声码器前反变换回去，误差 4.8e-07） | TASKS.md §0.10.4c |
| **下载时 txt 会分掉音频的名额** | `--include wav48_silence_trimmed --include txt` 写在一条命令里 ⇒ 每说话人只拿到 66 个 flac + 34 个 txt ⇒ **两人的清单内容不同** | notes/download_vctk.md §2.2b |
| 上游 GE2E 重构 | 目录变成 `encoder/` 包，权重改从 HuggingFace 下 | ge2e_interface.md §3 |
| 上游 mel 改约定 | **新版不再取 log**，拿旧版 mel 喂新版权重 → 嵌入全错但不报错 | ge2e_interface.md §3 |
| 上游 `load_model` | 设备要传**字符串**、权重路径要传 **Path** | ge2e_interface.md §3.2 |
| librosa 0.10 | `melspectrogram`/`resample` 改成仅关键字，上游位置调用会炸 | ge2e_interface.md §3.3 |
| `np.int` | numpy 1.24+ 已移除；只在**微调 E_spk** 时触发 | ge2e_interface.md §3.1 |
| DataShare 改版 | 老链接 404；老 handle 被重映射 | notes/download_vctk.md |
| VCTK 的第 110 人叫 `s5` | 按 `p+数字` 筛会只剩 109 人 → 划分失败 | TASKS.md §0.7 |
| 忘了 `conda activate vpp` | `No module named 'numpy'`（跑的是系统 Python） | 脚本现在会给出人话提示 |
| f-string 表达式 | 括号多打一个 → SyntaxError（我的检查器 v1 漏报，已升级） | — |
| **FastAPI 的错误格式** | 默认是 `{"detail":...}`，前端读 `detail.error` → 说明全丢 | TASKS.md §0.4b |
| **每次请求重载模型** | `infer.protect()` 每次新建 `Protector()` → 400 MB 重载 → 必超时 | TASKS.md §0.4b |
| **冻结 ≠ 断梯度** | 伪代码里的 `no_grad()` 会让训练**静默失效** | notes/stage2_blockers.md §6.1 |
| **numpy 封装断梯度** | `bridge_autovc` / `bridge_vocoder` 内部 `no_grad()` → Stage 2 用不了 | notes/stage2_blockers.md §1 |
| **掩蔽阈值与 PSD 参考电平不一致** | `L_mask` 恒为 0，约束形同虚设 | notes/stage2_blockers.md §6.4 |
| **"掩蔽不到"≠"极低阈值"** | 整条阈值曲线被压到绝对听阈上 | notes/stage2_blockers.md §6.4 |
| **调性判定方向反了** | 纯音被当成噪声 → 阈值偏高 14 dB | notes/stage2_blockers.md §6.4 |
| **上游 Tacotron 不能喂空文本** | 它自己的裁静音循环会把退化的 mel 裁空 → `index -1 is out of bounds`，而且**间歇性** | experiments.md §3b |

---

## 6. 引用

- 论文：Xue, C., Sun, M., Zhang, Q., Zhang, X., Li, K., Liao, Y., Ge, X., & Yao, K. (2026).
  *Imperceptible Voiceprint Protection via Human-Machine Perception Discrepancy Feature
  Disentanglement.* Interspeech 2026, pp. 5259–5263. DOI: 10.21437/Interspeech.2026-105
- 数据：Veaux, C., Yamagishi, J., & MacDonald, K. (2017). *CSTR VCTK Corpus: English
  Multi-speaker Corpus for CSTR Voice Cloning Toolkit* (version 0.92). CSTR, Edinburgh.
  <https://doi.org/10.7488/ds/2645>
- 组件与许可：见 [`third_party/README.md`](../third_party/README.md)
