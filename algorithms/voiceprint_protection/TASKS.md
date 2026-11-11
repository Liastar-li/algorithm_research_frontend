# 小组任务板 · 语音声纹主动防御复现

> 目标论文：Imperceptible Voiceprint Protection via Human-Machine Perception
> Discrepancy Feature Disentanglement (Interspeech 2026)

---

## 0. 怎么读这份任务板

### 0.1 术语与编号约定

| 记号 | 含义 |
|---|---|
| **W1–W6** | **Week 1–6，即第 1–6 周**（对应指南[第 10 章 排期参考与里程碑](../docs/语音声纹主动防御复现指南_两人分工版.md)） |
| **M0–M5** | 里程碑编号，定义见指南第 10 章 |
| **D1–D5** | 该周内的第 1–5 个工作日 |
| **第 N 步** | 指南[第 5 章](../docs/语音声纹主动防御复现指南_两人分工版.md)的"第 N 步：xxx"小节 |
| **C / A / B 前缀** | C = 共同任务；A = 成员 A 主责；B = 成员 B 主责 |
| **🔗 前置** | 必须先完成的任务，用 ❌ 表示依赖 |
| **📦 产出** | 该任务的交付物，缺了不算完成 |

### 0.2 指南章节 ↔ 任务板 对照

| 指南章节 | 内容 | 对应任务板 |
|---|---|---|
| 第 1–2 步 | 读论文、搭环境 | ✅ 已完成（W1） |
| **第 3 步** | **获取数据与开源组件** | **W2-D1 / W2-D2（见下方 W2 分解）** |
| **第 4 步** | **跑通最小可运行推理链（L1）** | **W2-D2 ~ W2-D5** |
| 第 5 步 | Stage 1 训练 | W4–W5 |
| 第 6 步 | Stage 2 训练 | W4–W5 |
| 第 7 步 | 评测 | W5–W6 |
| 第 8 步 | 服务化 | W3 |
| 第 9 步 | 平台接入 | W3 |
| 第 10 章 | 排期与里程碑 | 本文件 §2 |

### 0.3 当前状态

| 项 | 状态 |
|---|---|
| 环境（conda `vpp` + torch 2.4.1+cu121 + GPU） | ✅ 已就绪 |
| mel 参数契约（`scripts/test_audio_utils.py`，35 项断言） | ✅ 已锁定 |
| 环境指纹 | `36370ba987e72a4b`（两人必须一致） |
| `config.py` / `audio_utils.py` | ✅ 已写好 |
| **当前阶段** | **W2 · 指南第 3 步 + 第 4 步** |
| **下一个里程碑** | **M1：L1 推理链跑通**（产出保护后音频） |
| **待补依赖** | `python-multipart` 升到 0.0.20（起服务前必须） |

---

## 1. W2 详细分解：指南第 3 步 + 第 4 步

> **W2 目标（M1）**：不训练任何东西，用公开预训练组件把论文的推理路径走通，产出 `protected.wav`。
> **分工原则**：按"组件归属"自然拆分，避免两人改同一个文件。

### 1.1 组件归属（避免冲突）

| 组件 | 负责人 | 为什么这样分 |
|---|---|---|
| AutoVC（`Ec`/`D`/PostNet） | **A** | 模型主体，A 后面主责 Stage 1，现在先读懂结构 |
| HiFi-GAN 声码器 | **A** | 与 AutoVC 的 mel 约定要**一起**核对，两者耦合 |
| GE2E 说话人编码器 | **B** | 独立且很小，B 负责验证 `h` 的归一化约定 |
| VCTK 数据管道 | **B** | 已在原分工里 |

> 两人各自只碰自己那一列的目录，**不重叠** → 不会互相覆盖代码。

### 1.2 逐日任务表

#### W2-D1 · 拉组件、建目录、勘察数据

| # | 负责人 | 任务 | 🔗 前置 | 📦 产出 | 验收标准 |
|---|---|---|---|---|---|
| D1-1 | A | 克隆 AutoVC 官方仓库 → `third_party/AutoVC`，记下 **commit hash** | — | `third_party/AutoVC/` | 目录存在，commit 已记录 |
| D1-2 | A | 读它的 `hparams.py`，把 **mel 参数**（`sample_rate / n_fft / hop / win / n_mels / mel_fmin / mel_fmax`）抄进 `docs/notes/autovc_hparams.md` | D1-1 | `docs/notes/autovc_hparams.md` | 六个参数齐全，和 `config.py` 的 `autovc22k` 预设**逐项相同或写明差异** |
| D1-3 | B | 克隆 `CorentinJ/Real-Time-Voice-Cloning` → `third_party/RTC`，记 commit | — | `third_party/RTC/` | 目录存在，commit 已记录 |
| D1-4 | B | 下载 VCTK（Edinburgh DataShare）→ `data/raw/`，解压 | — | `data/raw/` | 解压完成，能看到 `wav48_silence_trimmed/` 或 `wav/` |
| D1-5 | B | 勘察 VCTK 目录：**统计说话人数、每个说话人的句子数、有多少 `p*_eval` 目录** | D1-4 | `docs/notes/vctk_layout.md` | 说话人数 ≈ 110；`p*_eval` 数量已统计 |
| D1-6 | **共同** | 建 `third_party/README.md`：每个组件的 **URL + commit + 许可证 + 用途** | D1-1, D1-3 | `third_party/README.md` | 四个组件（AutoVC / RTC-GE2E / HiFi-GAN / VCTK）都有条目 |

> ⚠️ **D1-5 的坑（指南第 3 步原文）**：VCTK 的说话人文件夹名是 `p225`、`p292` 这种
> （对应论文 t-SNE 图里的 p225/p252/p292/p315/p318）；
> **不要把 `p*_eval` 的噪声版本混进训练集** —— 那是被刻意加噪的评测版本。

#### W2-D2 · ★ mel 数值对照（本周最重要的一天）

> **指南第 4 步的"坑"原文**：*AutoVC 预训练权重的 mel 配置与你用的不一致 → 输出是噪音。
> 必须先读它的 `hparams` / `config` 并对齐。*
> **C2 是 W2 的命门。** 不做，AutoVC 权重加载后输出是噪音，你会误判成"权重坏了"，白查一天。

| # | 负责人 | 任务 | 🔗 前置 | 📦 产出 | 验收标准 |
|---|---|---|---|---|---|
| D2-1 | A | **AutoVC mel 对照**：同一段音频，分别用 AutoVC 仓库的原始 mel 代码 与 我们的 `audio_utils.MelSpec("autovc22k").mel()` 计算 | D1-2 | 对照脚本 + 结论 | **最大绝对差 < 1e-4**；有差异则**改我们的实现**，不是改组件 |
| D2-2 | A | **HiFi-GAN mel 对照**：组件的 mel 处理 与 我们的 `audio_utils.vocoder_mel()` | D1-1 | 对照脚本 + 结论 | 同上；确认 `(x+4.5)/5` 归一化一致 |
| D2-3 | A | **验证 16 kHz 声码器**：`torch.hub.load('NVIDIA/torch.hifi_gan', 'hifigan_melbins_16k')` 能否加载并合成 | — | 合成样例 wav | 能出声；**若可用则 16 kHz 全路线无障碍**（论文本来就是 16k+VCTK） |
| D2-4 | B | **GE2E 输入约定确认**：采样率是多少？内部 mel 是几维、什么参数？输出 `h` 是否已 L2 归一化？ | D1-3 | `docs/notes/ge2e_interface.md` | 四个问题都有明确答案 + 代码位置引用 |
| D2-5 | B | **写 `scripts/prepare_data.py`**（详见下方 §1.3 规格） | D1-5 | `scripts/prepare_data.py` | 能跑出 `data/train.txt` / `val.txt` / `test.txt` / `spk2id.txt` |
| D2-6 | **共同** | 把对照结论写进 `audio_utils.py` 的文件头注释 + `docs/notes/mel_convention.md` | D2-1, D2-2, D2-4 | 两个文件已更新 | 三方 mel 约定在文档里能对照着看 |

#### W2-D3 · 写桥接模块与接口签名

| # | 负责人 | 任务 | 🔗 前置 | 📦 产出 | 验收标准 |
|---|---|---|---|---|---|
| D3-1 | **共同** | **定接口签名**，写进 `docs/interface-contract.md`（指南第 8 章那 10 条 + 下列函数签名） | D2-6 | `docs/interface-contract.md` | 双方确认，后续改动须双方同意 |
| D3-2 | A | `bridge_autovc.py`：加载 AutoVC 权重，提供 `content_encoder(x,h)->c` 与 `decoder(c,h)->mel` | D2-1, D3-1 | `bridge_autovc.py` | 输入 `(T,80)`，输出 `(T',32)` 与 `(T,80)` |
| D3-3 | A | `bridge_vocoder.py`：加载 HiFi-GAN，提供 `mel->wav` | D2-2, D3-1 | `bridge_vocoder.py` | 输出 wav 采样率与预设一致 |
| D3-4 | B | `bridge_speaker.py`：加载 GE2E，提供 `wav->h(256)` | D2-4, D3-1 | `bridge_speaker.py` | 输出形状 `(256,)`，归一化状态明确 |
| D3-5 | B | `scripts/prepare_data.py` 补完 + **核对划分** | D2-5 | `data/*.txt` | **训练说话人不出现在测试集**；`spk2id.txt` 的 K = 80 |

#### W2-D4 · 串起推理主流程

| # | 负责人 | 任务 | 🔗 前置 | 📦 产出 | 验收标准 |
|---|---|---|---|---|---|
| D4-1 | A | 写 `infer.py` 主流程（`wav → x → h → h_adv → c → x_adv → wav'`），**每步打印张量形状** | D3-2, D3-3, D3-4 | `infer.py` | 打印出 `x (T,80)` / `h (256,)` / `c (T',32)` / `x_adv (T,80)` |
| D4-2 | A | **先加随机/高斯扰动**（α=0.1），**不训练 G** —— L1 阶段本来就不训练 | D4-1 | `infer.py` | 扰动可复现（固定 seed） |
| D4-3 | B | 写 `scripts/smoke_test.py`：喂一段真实语音 → 出 `protected.wav` + 打印形状 | D4-1 | `scripts/smoke_test.py` | 一条命令跑完 |
| D4-4 | B | 准备 1 段**公开可用**的测试音频（平台样例输入），确认许可 | D1-4 | `samples/` 下的音频 | 能公开展示 |
| D4-5 | **共同** | 端到端联调 | D4-1, D4-3 | — | **产出一段能播放的 `protected.wav`** |

#### W2-D5 · 验收与固化

| # | 负责人 | 任务 | 🔗 前置 | 📦 产出 | 验收标准 |
|---|---|---|---|---|---|
| D5-1 | **共同** | ★ **每人独立跑一次** `scripts/smoke_test.py`（换环境、不给解释） | D4-5 | 各自的口头/截图记录 | 两人都成功 |
| D5-2 | **共同** | 听一遍输出：确认**不是噪音**、不是静音 | D4-5 | 结论写进 `docs/experiments.md` | 听感正常；若异常，先查 mel 对照 |
| D5-3 | **共同** | 把"每人独立跑通"写进 `docs/experiments.md`，作为 M1 验收证据 | D5-1 | `docs/experiments.md` | 有记录 |
| D5-4 | A | 更新 `requirements.lock.txt` 与 `third_party/README.md`（组件 commit） | D1-6 | 两个文件 | 齐全 |

### 1.3 `prepare_data.py` 的规格（指南第 3 步原文，B 执行）

```text
输入：data/raw/  （VCTK 解压后的目录）
输出：
  data/train.txt   训练集清单（80 说话人）
  data/val.txt     验证集清单（10 说话人）
  data/test.txt    测试集清单（20 说话人）
  data/spk2id.txt  说话人 -> 整数 ID 映射

处理步骤：
  1. 遍历 wav48_silence_trimmed/ 或 wav/ 目录
  2. 重采样到 16 kHz、单声道、16-bit PCM
  3. 按说话人划分 80 / 10 / 20
  4. 每行格式：<音频路径> <说话人ID>
  5. spk2id.txt：训练集说话人 -> 0..79 的整数
```

**三条硬要求：**

| # | 要求 | 原因 |
|---|---|---|
| 1 | **排除 `p*_eval` 目录** | 那是刻意加噪的评测版本，混进训练集会污染实验 |
| 2 | **训练/验证/测试说话人完全不重叠** | 论文用 80 人训练、10 人验证、20 人测试；泄漏会让指标虚高 |
| 3 | **`spk2id.txt` 的 K 必须等于训练集说话人数 = 80** | `L_dis` 的说话人分类器 `Ws` 形状是 **K×256**，K 错了分类器建不起来 |

**验证命令（D3-5 必跑）：**

```bash
# 三个清单的说话人集合必须两两不相交
cut -d' ' -f2 data/train.txt | sort -u > /tmp/tr.spk
cut -d' ' -f2 data/val.txt   | sort -u > /tmp/va.spk
cut -d' ' -f2 data/test.txt  | sort -u > /tmp/te.spk
comm -12 /tmp/tr.spk /tmp/te.spk   # 必须为空
comm -12 /tmp/tr.spk /tmp/va.spk   # 必须为空

# 训练说话人数必须是 80
cut -d' ' -f2 data/train.txt | sort -u | wc -l
```

---

## 2. 里程碑与排期总览（对应指南第 10 章）

| 周 | 里程碑 | 交付物 | 两人共同动作 |
|---|---|---|---|
| **W1** | **M0 对齐** | 论文精读笔记、接口契约 v1、环境就绪 | ✅ 已完成 |
| **W2** | **M1 L1 推理链跑通** | `infer.py` 输出保护后音频；打印所有张量形状 | 一起跑 smoke test，**每人都独立跑一次** |
| **W3** | **M2 L0 平台打通** | 方法 JSON + 服务 + 1 组样例；平台上能上传下载 | 一起过接入自查清单 |
| **W4** | **M3 Stage 1 完成** | Stage1 checkpoint；重建音频听感正常；`Sim_prot(重建)>0.9` | 交叉 review；主责交换 |
| **W5** | **M4 Stage 2 完成** | Stage2 checkpoint；DSR 显著高于高斯噪声基线 | 交叉 review；主讲日各讲一次 |
| **W6** | **M5 评测 + 交付** | 指标表、消融表、实验记录、README、PR | 一起写 README、一起过自查清单、一起发 PR |

---

## 3. W3 详细分解：指南第 8 步 + 第 9 步（平台打通）

> **W3 目标（M2）**：让方法在平台上能被看到、被上传、能出结果、能下载。

| # | 负责人 | 任务 | 🔗 前置 | 📦 产出 | 验收标准 |
|---|---|---|---|---|---|
| P1 | **B 主写 / A review** | 写 `public/methods/group_XX_voiceprint_protection.json` | W2 全部 | 方法配置 JSON | 通过 `npm test` 的契约校验（字段名、MIME、max_mb、参数范围） |
| P2 | 任一人 | 追加 `public/methods/index.json` | P1 | 索引更新 | 方法能在"方法集合"搜到 |
| P3 | **A 主写** | 写 `service.py`（FastAPI + CORS + 错误处理 + 临时文件清理） | P1 | `service.py` | `curl` 能打出 artifacts；错误路径返回 `{"error": "..."}` |
| P4 | **B 主写** | 准备样例：`public/samples/<方法ID>/` 输入 + 输出（**必须是真实跑出来的**） | W2-D5 | 样例目录 | 输入输出确实对应，参数已记录 |
| P5 | **共同** | 平台端到端：上传 → 处理 → 预览 → 下载 | P1–P4 | — | 全部成功 |
| P6 | **共同** | `npm test` && `npm run build` | P5 | — | 全绿 |
| P7 | **共同** | 建 `docs/experiments.md`，记录已跑通的步骤与结果 | P5 | 实验记录 | 有内容，且与代码同 commit |
| P8 | **共同** | 过一遍 `docs/接入自查清单.md` | P6 | 勾选结果 | 逐项确认 |
| P9 | **共同** | 补 `python-multipart==0.0.20`（FastAPI 收文件的必需依赖） | — | 环境更新 | `import python_multipart` 成功 |

> **P3 与 P4 的先后**：P3 需要 P1 里的 `endpoint` 与字段名，所以 **P1 必须先于 P3**；
> P4 的样例要用真实输出，所以 **P4 需要 W2-D5 的产物**。

---

## 4. W4 起的模块主责（W6 交换）

| 阶段 | 成员 A | 成员 B |
|---|---|---|
| W4–W5 | **Stage 1 主责**（指南第 5 步）：`stage1/model.py` 训练循环、`L_rec / L_cd / L_dis`、说话人分类器 `Ws` | **Stage 2 主责**（指南第 6 步）：生成器 `G`、判别器 `D_adv`(LSGAN)、`L_per / L_GAN / L_de`、心理声学掩蔽 θ_x |
| W6 起 | 交换为 **Stage 2 主责** | 交换为 **Stage 1 主责** |

**共同区（任何阶段都不许一人独占）**：`config.py`、`audio_utils.py`、`infer.py`、`evaluate.py`、`service.py`、平台接入、README。

**W4 开始时的分工细节（暂定，W3 末再确认）：**

| 工作 | A | B |
|---|---|---|
| `stage1/train.py` 主循环 | **A** | B 提供 `dataset.py` 并参与 review |
| `stage2/train.py` 主循环 | B | **A** review 防御损失 `L_de` 的方向是否正确 |
| `evaluate.py`（指标脚本） | **共同写** | **共同写** |
| `docs/experiments.md` | **共同维护** | **共同维护** |

---

## 5. 每周节奏

| 频率 | 活动 | 时长 | 产出 |
|---|---|---|---|
| 每天 | 站会：昨天 / 今天 / 卡点 | 15 min | 共享待办 |
| 每周 | **主讲日**：轮流讲自己主责模块，另一人提问并**复述一遍** | 40 min | `docs/notes/主讲记录.md` |
| 每周 | **串行跑通日**：两人一起从头跑一遍，**不许一方代跑** | 2 h | 卡点清单 |
| 每 2 周 | 主责模块交换 | — | `docs/notes/handoff_<阶段>.md` |
| 每个里程碑 | 交叉 code review | 1 h | review 记录 |

---

## 6. 两人能力底线（都要做到，答辩要验）

- [ ] 能画出两阶段数据流，说出每个张量的形状（不看代码）
- [ ] 能独立跑通完整推理脚本
- [ ] 能读懂并讲解**对方主责模块**的代码
- [ ] 能改一个超参并跑出可观察的差异（例：把 `λ_de` 从 5 改成 0，看 DSR 崩掉）
- [ ] 能独立启动服务并在平台完成一次上传→处理→下载
- [ ] 能独立修改方法 JSON 与 README 的对应章节

---

## 7. 三个最容易踩的坑

1. **各写各的 mel 参数** → Stage 2 接入 Stage 1 输出全是噪音。靠 `audio_utils.py` 唯一真源 + **D2-1/D2-2 的对照结果**防住。
2. **只有一个人能跑通** → 换环境就挂。**D5-1「每人独立跑一次」**就是防这个。
3. **文档到最后一个人赶** → README 里全是想象。**实验记录必须和代码同 commit 提交。**

---

## 8. 待决事项（需要组内拍板）

- [ ] **代金券适用产品**：能否开 ECS（NVIDIA 卡）？还是只有昇腾 ModelArts？
      → 决定训练是否需要额外做 NPU 迁移（**越早查越好，直接影响排期**）
- [ ] **采样率方案**：L1 用 `autovc22k`(22.05k) 还是全程 `paper16k`(16k)？
      - 倾向：**L1 用 22.05k（能直接用现成权重，最快看到东西）→ 最终系统用 16k**
      - 若 **D2-3** 验证 `hifigan_melbins_16k` 可用，则 16k 路线全程无障碍
- [ ] **组号与选题名称** → 确定后统一改方法 ID、目录名、README 标题
- [ ] **MOS 听测人数**：论文是 50 人，我们做多少？（如实写明即可）
- [ ] **W4–W5 是否真的交换主责**（W6 起交换是必须的，W4–W5 是否也换需组内定）
