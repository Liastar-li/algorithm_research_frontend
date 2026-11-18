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
| 语句数 | 11,004（每人 100 句，**均匀抽样**，不是取前 100 句） |
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
| **梯度通路自检** | ✅ **12/12 通过**（`stage2/verify_gradients.py`） |
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

## 5. 已知坑（组内共享，别重复踩）

| 坑 | 现象 | 出处 |
|---|---|---|
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
