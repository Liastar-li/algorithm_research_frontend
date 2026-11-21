# Stage 2 的阻塞项与待决问题（成员 B · 2026-11-18）

> 这份文档记录**写 Stage 2 时撞到的、必须由组内拍板或由成员 A 补齐**的问题。
> 每一条都写清了：现象 → 影响 → 建议做法 → 谁来做。
>
> Stage 2 的代码骨架已经写完（`algorithms/voiceprint_protection/stage2/`），
> **训练机制已经跑通并通过 12 项梯度通路自测**，
> 剩下的都是"缺组件"而不是"缺代码"。

---

## 一、★ 最要紧的一条：Stage 2 需要一个**可求导**的声码器

### 现象

Stage 2 的梯度必须穿过声码器才能回到生成器 G：

```
G → h_adv → D(c, h_adv) → V(·) → E_spk(·) → L_de
                              ↑
                        这里必须可导
```

而项目里现有的两个声码器封装都是 **numpy + `torch.no_grad()`**：

| 封装 | 位置 | 问题 |
|---|---|---|
| `bridge_autovc.protect()` | `algorithms/voiceprint_protection/bridge_autovc.py` | 内部 `with torch.no_grad()`，返回 numpy |
| `bridge_vocoder.synthesize()` | `algorithms/voiceprint_protection/bridge_vocoder.py` | 同上 |

拿它们训练，**G 收不到任何梯度，loss 一步都不动，而且不报错**。

### 影响

Stage 2 训不起来（静默失败，最坏的一种）。

### 建议做法（三个候选，需要拍板）

| 候选 | 优点 | 缺点 | 建议 |
|---|---|---|---|
| **A. AutoVC 配套的 `wavenet_vocoder`** | mel 约定与 AutoVC **天然一致**，论文的 L_mel / L_mask 才严格成立 | 上游 `vocoder.generate()` 最后一步转 numpy，要搬成 torch；**而且官方权重 404（IBM Box），拿不到** | 暂时搁置 |
| **B. HiFi-GAN**（论文同款，[15]） | 论文用的就是它；torch 原生、可导 | ⚠️ **原来的 `NVIDIA/torch.hifi_gan` 仓库已经不存在了**（实测 404，网页和 API 都是 404） | ★ **推荐，但需要另找权重** |
| **C. 上游 RTC 的 WaveRNN** | 权重已经有了 | 它的输入是**线性频谱**不是 mel，要多加一层转换；而且 `infer_waveform` 也是 numpy | 不推荐 |

### ★★ 2026-11-20 结论：这条**已解决**（成员 B，v0.5）

| 项 | 结果 |
|---|---|
| NVIDIA 官方 `NVIDIA/torch.hifi_gan` | **仓库连同 release 一起被删**（GitHub API / codeload / raw / release 附件全部 404；`api.github.com/repos/pytorch/pytorch` 返回 200 ⇒ 不是墙） |
| 替代品 | HuggingFace **`speechbrain/tts-hifigan-libritts-16kHz`** 的 `generator.ckpt`（55,828,077 B） |
| 与 NVIDIA `hifigan_melbins_16k` 的等价性 | 采样率 16000 / hop 256 / n_fft 1024 / 80 mels、HiFi-GAN v1 架构、参数量同为 **13,936,130**、mel 约定相同，`strict=True` 加载 **234/234 键全匹配** |
| 差别 | 只差训练数据（LibriTTS vs NVIDIA 的 16k 模型） |
| 代码 | **内置**在 `algorithms/voiceprint_protection/hifigan.py`（`bshall/hifigan` 的 HiFi-GAN v1 生成器，MIT，已署名）—— 两台机器都不用再 clone `third_party` |
| 下载 | `python algorithms/voiceprint_protection/scripts/download_hifigan_weights.py`（已下好并校验：sha256 前缀 `2d92a97446467ae5`） |
| 落盘位置 | `<数据根>/models/default/hifigan_16k.ckpt`（即 `D:\vpp\data\models\default\`） |

**mel 约定一并解决**（不用再做近似换算）：

- Stage 1 的模型侧 mel 预设改成 **`hifigan16k`**（`config.STAGE1["mel_preset"]`）；
- `stage2/pipeline.py::autovc_mel_to_hifigan()` **已删除** ——
  实测那个近似换算会把重建音的说话人余弦从 **0.9746 掉到 0.6983**；
- 网络侧 mel 做一次**可逆的固定仿射变换**（`0.08686·s + 0.84`，值域 [-0.11, 0.92]，与 AutoVC 同量纲），
  送声码器前反变换回去（实测误差 **4.8e-07**，零损失）——
  不这么做的话 `L_rec` 会比论文大 100 倍，`λ_cd=1.0` 被稀释 40 倍，**论文的超参就失效了**。

**实测（本机，2026-11-20）**：`scripts/verify_hifigan.py` 全通过 ——
重建音与原音的说话人余弦 **0.9745**、RMS 0.0539 → 0.0585、
网络侧 mel ↔ 原始 mel 一致（最大差 2.56e-06）、**梯度能回传到 mel**（absmax 0.8925）。

---

## 二、成员 A 需要提供：Stage 1 的 checkpoint

### 现象

`stage2/models.py::TorchAutoVCStage1` 需要一个 AutoVC 架构的 checkpoint：

```text
algorithms/voiceprint_protection/weights/stage1_model.pt
```

格式：`{"model": state_dict}` 或裸 `state_dict`（与 AutoVC 官方一致）。

### ★★ 2026-11-21 的状态更新（成员 B）

| 项 | 状态 |
|---|---|
| 成员 A 的 Stage 1 训练 | 已训 500k 步（13.5 h，4060）：rec 0.00177 / cd 0.00007 / dis **−4.38134**（理论极限 −4.3820 的 99.98%）✅ |
| **但 M3 的 Sim_prot 只有 0.459**（要 >0.9） | ❌ 诊断：MSE 把 mel 的峰值压平了，重建音 RMS 只有原音 1/18。A 说要改 `L_rec` 的形式重训 |
| ⇒ **Stage 2 现在还不能开训** | Stage 1 换了权重，Stage 2 就得重训一遍 |
| 本机有没有 A 的权重 | ❌ **没有**（本机只有 `stage1_smoke3steps_NOT_TRAINED.pt`，3 步的冒烟产物，代码会拒绝用它） |
| **训练与推理的 E_spk 必须一致** | ★ 论文说 E_spk 在 Stage 1 里被微调过。Stage 2 训练时要用**同一个** —— 现在用 `--speaker-encoder {stage1,upstream}` 显式选，并写进 checkpoint |

⇒ **给成员 A 的请求**：训完（Sim_prot > 0.9 之后）把 `stage1_model.pt` 给 B 一份，
并说明「Stage 1 训的时候 GE2E 微调了没有」—— 这决定 Stage 2 用哪个 `--speaker-encoder`。

### 影响

没有它，Stage 2 只能用 `StubStage1`（随机权重）跑，**指标全是噪声**，不能写进报告。

### 说明

- AutoVC 官方权重链已 404（任务板 §0.8），所以 Stage 1 必须**自己训**（成员 A 主责）。
- 初始化可以用 `marlenezw/AutoVC_Voice_Conversion` 的 `ckpt_autovc.pth`（171 MB，MakeItTalk 出品），
  里面含 AutoVC 的 Ec + D + GE2E。
- `TorchAutoVCStage1` 与 `bridge_autovc.AutoVCBridge` **模型结构完全一致**，
  区别只是前者留在 torch 里（可导），后者转 numpy（不可导）。
  两者加载同一个 checkpoint 不会冲突。

---

## 三、~~`third_party/AutoVC` 与 `third_party/wavenet_vocoder` 还没克隆~~ ✅ 已解决

`third_party/` 现在已有 `AutoVC`（commit `c1827a6`）、`wavenet_vocoder`（`d4333546`）、
`softvc-hifigan`（MIT 镜像）；`RTC` 在 D 盘，通过 `VPP_RTC_DIR` 指过去。

> ★ HiFi-GAN 的**生成器代码已内置**进 `algorithms/voiceprint_protection/hifigan.py`
> （`bshall/hifigan` 的 HiFi-GAN v1，MIT，已署名），
> 所以连 `softvc-hifigan` 都不是必需的了（保留作参考）。
---

## 四、`Unidecode` + `inflect` 还没装

上游 Tacotron 合成器的 `synthesizers/utils/cleaners.py` 要用它们把文字转音素，
少一个就直接 `ModuleNotFoundError`。

```bash
pip install Unidecode==1.3.2 inflect==5.3.0
```

已写进 `requirements.txt`，但本机还没装。**这条最简单，先装上。**

---

## 五、~~评测要算 WER，还缺 VCTK 的转写文本~~ ✅ 已解决（2026-11-20 夜，成员 B）

### 怎么解决的

源站 `datashare.ed.ac.uk` **恢复了**（此前 3 次尝试全部超时）。
用新增的 `scripts/download_vctk.py`（纯标准库）把 `txt/` 拉下来 ——
测试集 20 个说话人的 **8,000 个转写文件，5.2 分钟，0 失败**。

```bash
python scripts/download_vctk.py --out <数据根>/raw --include txt --per-speaker 100
python algorithms/voiceprint_protection/evaluate.py --mode none --split test --limit-per-speaker 5 --wer --whisper-model <模型> --transcript-dir <数据根>/raw/txt
```

### ★★ 但光把文件下下来还不够 —— 链路上有**三个静默失效**的 bug

不报错，但算出来的 WER 根本没意义：

| # | bug | 后果 |
|---|---|---|
| 1 | `load_transcripts()` 用 `glob("*.txt")` 只扫顶层 | VCTK 的 `txt/` 是**按说话人分子目录**的 ⇒ 一个都找不到 |
| 2 | 拿 `path.stem` 查表 | 我们的音频叫 `p228_001_mic1.wav`（两路麦克风都收了），转写只有一路、叫 `p228_001.txt` ⇒ **永远查不到** |
| 3 | `protect()` 念固定的一句 "The quick brown fox jumps over the lazy dog." | 拿它跟「原句的转写」比 ⇒ **WER 恒等于 ~100%** |

前两个都会退化成「Whisper 转写原音当参考」的**相对 WER**（一个错都不报）；
第三个让 WER 彻底失去意义。

**修法**：`rglob` + `transcript_key()`（去掉 `_mic1`/`_mic2` 后缀）+
**保护时念原句**（有原始转写时）+ 每说话人**均匀抽样**（取前 N 条的话，
每个说话人拿到的都是 `pXXX_001`，而那恰好是同一句 "Please call Stella."）。
另外加了 `transcript_miss` 计数，有缺失就醒目提示。

**验证**：转写查表命中率 **20/20**（修之前 0/20）。

---

## 六、★ 已经解决的坑（记下来免得再踩）

### 6.1 指南伪代码里的 `with torch.no_grad()` 不能照抄

指南第 6 步的伪代码把整段前向包进了 `torch.no_grad()`。
**照抄会让训练完全不工作** —— 梯度被切断，loss 纹丝不动，**而且不报错**。

正确的理解：

- **冻结** = 不更新这些模块的**参数** → `requires_grad_(False)` + `eval()`
- **梯度仍然要流过**它们 → 不要包 `no_grad`，不要 `.detach()`

真正切断梯度的是 `torch.no_grad()` 与 `.detach()`，不是"冻结"。

> 这条已经变成一条**会失败的断言**：`stage2/verify_gradients.py`（**20 项**，全通过；其中【6】段 7 项是成员 A 修的两个 bug 的回归测试）。

### 6.2 torch 版 GE2E 与上游 numpy 版必须逐值一致

Stage 2 用的是自己写的 torch 版 GE2E（因为上游的 numpy 封装断了梯度）。
如果 mel 算得跟上游不一样，训练与推理的说话人嵌入就**不是同一个东西**，
而且**不报错**。

> 已用 `stage2/verify_mel.py` 对照：mel 最大绝对误差 **2.3e-05**，嵌入余弦 **0.99999481** ✓

### 6.3 `L_de` 的方向不能反

`L_de = sim(E_spk(克隆音), E_spk(原音))`，训练时是**最小化**它。
如果写成最大化，训练会**把音频彻底毁掉**。

> 自检方法：训练 100 步后 `L_de` 应该**下降**。`train.py` 每步都打它。

### 6.4 ★ 心理声学掩蔽有三个坑（写的时候全踩了）

`stage2/psychoacoustics.py` 里的每个数字都是手抄标准的，
抄错不会报错，只会让 `L_mask` **静默失去意义**。三个真正致命的地方：

| # | 坑 | 现象 | 解法 |
|---|---|---|---|
| 1 | **参考电平没对齐** | 掩蔽阈值是 dB SPL，扰动 PSD 是数字信号量纲。直接比会差十几个 dB → `L_mask` **恒为 0**，那条约束形同虚设 | 统一到**均方 dB**（满幅正弦 = 94 dB SPL，均方 0.5 → 差 97.01 dB） |
| 2 | **"掩蔽不到"被当成"极低阈值"** | 扩散函数表的低频方向超出 3 带就够不着了。早期用 −200 dB 代替"够不着"，结果一个够不着的掩蔽器算出了 −200 dB 的阈值，**把整条曲线压到听阈上** | `_spreading()` 返回 `None` 表示"够不着"，直接**不参与**取最小/最大 |
| 3 | **安静的频带也能当掩蔽器** | 数字静音的频带（只有数值泄漏，−60 dB）仍高于绝对听阈（−93 dB），于是它算出的极低阈值 dominate 了整条曲线 | 两个条件同时用：① 掩蔽器电平必须**高于**该带的绝对听阈；② 多个掩蔽器**取最强**（物理上可听噪声的地板由最强的掩蔽器决定），不是取最弱 |

还有一条不算坑但值得记：**调性判定方向别搞反** ——
纯音的谱线是"**突出**"的（比邻居高出很多），噪声底是平的。
判据应该是"峰值**高于**邻居"，不是"峰值与邻居**相差不超过** 0.5 dB"。
再加一条"峰值要比频带平均高 6 dB"，否则白噪声里的随机峰会被误判成调性。

> 这三个坑现在都由 `stage2/verify_psychoacoustics.py`（**20/20 通过**）盯着：
> 阈值量级、单调性、零扰动严格为 0、numpy 与 torch 数值一致、梯度非零。

---

## 七、当前状态一览

| 项 | 状态 |
|---|---|
| Stage 2 代码骨架（12 个模块） | ✅ 已写完 |
| 训练机制跑通（`--stub`） | ✅ 已验证（L_de/per/mask/gan 全有值） |
| **训练器（2026-11-21 补齐）** | ✅ **断点续训 / AMP / 梯度累积 / 梯度裁剪 / `--max-hours` / 吞吐 ETA** 全部就位；`[L_de 方向自检]` 自动跑 |
| 梯度通路自检 | ✅ **20/20** 通过（含成员 A 修的两个 bug 的回归断言，见契约 §4.5.3c） |
| torch GE2E 与上游数值对照 | ✅ 误差 2.3e-05，嵌入余弦 0.99999481 |
| 心理声学掩蔽 θ_x | ✅ 已写完（MPEG-1 Model 1），自检 **20/20** 通过 |
| 可求导声码器 | ✅ **已解决**（HiFi-GAN 16 kHz，HF `speechbrain/tts-hifigan-libritts-16kHz`，可导） |
| **E_spk 用哪一个** | ✅ **已解决**（2026-11-21）：`--speaker-encoder {stage1,upstream}`，默认按 checkpoint 自动选，**并写进 checkpoint**，推理照着做 ⇒ 训练与推理不可能对不上 |
| Stage 1 checkpoint | ❌ **仍然阻塞**（第二节）—— 成员 A 已训 500k 步，但 **Sim_prot 0.459 < 0.9**，要改 `L_rec` 重训 |
| **算力** | ❌ **本机没有 NVIDIA GPU**（核显 + torch CPU 版）⇒ 论文量级必须上云机，方案见 [`cloud_ecs_stage2.md`](cloud_ecs_stage2.md) |
| `evaluate.py` | ✅ 已写好（跑起来同样等 `synthesizer.pt`/`vocoder.pt`） |
| `infer.py` 的 stage2 路线 | ✅ **已加**（`δ = G(h)`，训练好的生成器；CLI `--mode {clone,stage1,stage2,auto}`） |
| 绝对 WER | ✅ **已通**（转写已补齐；链路上三个静默失效的 bug 已修，见第五节） |
| Whisper 权重 | ⚠️ 本机到 HF 的 CDN 只有 ~350 KB/s，`large-v3`（3.09 GB）要 ~2.4 h；本轮用 `small` 先跑通（**偏离论文设定，报告里写明**）。云机上 `--setup` 会一并拉下来 |
