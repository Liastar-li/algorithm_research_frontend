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
| **A. AutoVC 配套的 `wavenet_vocoder`** | mel 约定与 AutoVC **天然一致**，论文的 L_mel / L_mask 才严格成立 | 上游 `vocoder.generate()` 最后一步转 numpy，要搬成 torch | ★ **推荐** |
| **B. HiFi-GAN**（论文同款，[15]） | 论文用的就是它；torch.hub 一行装好 | mel 约定不同（`log10` 幅度 + `(x+4.5)/5` vs AutoVC 的 `20log10` + `clip[0,1]`），要做**近似**换算 | 次选 |
| **C. 上游 RTC 的 WaveRNN** | 权重已经有了 | 它的输入是**线性频谱**不是 mel，要多加一层转换；而且 `infer_waveform` 也是 numpy | 不推荐 |

**候选 A 的具体做法**（`stage2/pipeline.py::WaveNetVocoder` 里已经留好了位置和 5 步说明）：

```text
1. git clone --depth 1 https://github.com/r9y9/wavenet_vocoder third_party/wavenet_vocoder
   （固定 commit d4333546 —— 见 config.THIRD_PARTY_COMPONENTS）
2. from wavenet_vocoder import builder → builder.build_model()
3. 加载 checkpoint_step001000000_ema.pth 的 state_dict
4. 把上游 vocoder.py 的 extract_features + model(c, None) 照抄成 torch
5. ★ 只有最后"转 numpy"那一步要留在 torch 里 —— 梯度就断在那里
```

**谁做**：成员 B（实现很快）—— 但**前提是 `third_party/wavenet_vocoder` 先克隆下来**。

---

## 二、成员 A 需要提供：Stage 1 的 checkpoint

### 现象

`stage2/models.py::TorchAutoVCStage1` 需要一个 AutoVC 架构的 checkpoint：

```text
algorithms/voiceprint_protection/weights/stage1.pt
```

格式：`{"model": state_dict}` 或裸 `state_dict`（与 AutoVC 官方一致）。

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

## 三、`third_party/AutoVC` 与 `third_party/wavenet_vocoder` 还没克隆

`third_party/` 目前只有 `README.md`，三个组件一个都没拉下来（`RTC` 在 D 盘，通过 `VPP_RTC_DIR` 指过去）。

```bash
bash scripts/setup_third_party.sh          # Linux / 云机
powershell -ExecutionPolicy Bypass -File scripts/setup_third_party.ps1   # Windows
```

脚本会按 `config.THIRD_PARTY_COMPONENTS` 里钉死的 commit 浅拉取，保证两人版本一致。

---

## 四、`Unidecode` + `inflect` 还没装

上游 Tacotron 合成器的 `synthesizers/utils/cleaners.py` 要用它们把文字转音素，
少一个就直接 `ModuleNotFoundError`。

```bash
pip install Unidecode==1.3.2 inflect==5.3.0
```

已写进 `requirements.txt`，但本机还没装。**这条最简单，先装上。**

---

## 五、评测要算 WER，还缺 VCTK 的转写文本

### 现象

`evaluate.py --wer` 需要每条语音的**标准转写**才能算绝对 WER。
我们下载 VCTK 时**只取了 wav，没取 `txt/`**（转写文本约 2 MB，在同一个 zip 里）。

### 三个办法

| 办法 | 说明 |
|---|---|
| **A. 补下 `txt/`**（推荐） | 用 `scripts/download_vctk.mjs` 再取一次那批条目，然后 `--transcript-dir` 指过去 → 算**绝对 WER**（和论文可比） |
| B. 用 Whisper 转写原始语音当参考 | `evaluate.py` 已经实现了这个回退 → 算的是**相对 WER**（"保护让转写变差多少"），报告里要写清楚 |
| C. 只报 Sim_prot / Sim_clone / DSR | 不算 WER，README 里如实说明 |

---

## 六、★ 已经解决的坑（记下来免得再踩）

### 6.1 指南伪代码里的 `with torch.no_grad()` 不能照抄

指南第 6 步的伪代码把整段前向包进了 `torch.no_grad()`。
**照抄会让训练完全不工作** —— 梯度被切断，loss 纹丝不动，**而且不报错**。

正确的理解：

- **冻结** = 不更新这些模块的**参数** → `requires_grad_(False)` + `eval()`
- **梯度仍然要流过**它们 → 不要包 `no_grad`，不要 `.detach()`

真正切断梯度的是 `torch.no_grad()` 与 `.detach()`，不是"冻结"。

> 这条已经变成一条**会失败的断言**：`stage2/verify_gradients.py`（12 项，全通过）。

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
| 训练机制跑通（`--stub`） | ✅ 已验证（5 步，L_de/per/mask/gan 全有值） |
| 梯度通路自检 | ✅ **12/12** 通过 |
| torch GE2E 与上游数值对照 | ✅ 误差 2.3e-05，嵌入余弦 0.99999481 |
| 心理声学掩蔽 θ_x | ✅ 已写完（MPEG-1 Model 1），自检 **20/20** 通过 |
| 可求导声码器 | ❌ **阻塞**（本文档第一节） |
| Stage 1 checkpoint | ❌ **阻塞**（第二节，等成员 A） |
| 正式训练 | ⛔ 等上面两个 |
| `evaluate.py` | ✅ 已写好（跑起来同样等 `synthesizer.pt`/`vocoder.pt`） |
