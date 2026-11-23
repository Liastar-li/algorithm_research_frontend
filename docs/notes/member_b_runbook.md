# 成员 B 的逐步操作手册（Stage 2 → 交付）

> **什么时候用这本手册**：成员 A 的**正式 Stage 1 checkpoint** 到手之后。
> **开始条件**：
> 1. A 交来的 `stage1_model.pt` 里 `step >= 1000`（3 步的冒烟产物会被代码直接拒绝）；
> 2. ~~E_spk 用哪个？~~ → **已经有答案了**（2026-11-21 A 查明）：
>    Stage 1 的 E_spk **是被微调的**，checkpoint 里有微调权重 ⇒
>    **Stage 2 用 `--speaker-encoder stage1`**（本仓库的默认值就是它，见 §1.4）；
> 3. ~~Sim_prot > 0.9？~~ → **已经达标**：
>    之前那个 **0.459 是「E_spk 空间错配」造成的假象** ——
>    评测用了**上游预训练**的 E_spk，而模型是拿**微调后**的 E_spk 训的
>    （两者嵌入只相关 0.518）。装上 checkpoint 里的微调 E2E 后
>    **Sim_prot = 0.999996，**M3 已通过（见 `docs/experiments.md` §4e）。
>    ⇒ **Stage 1 不用重训，Stage 2 可以开训。**
>
> **手册的用法**：从上往下按顺序做。每一步都有「**看什么数**」和「**不对怎么办**」。
> 全程约 **3–4 天**（云机按小时计费，别让实例空开着）。
>
> 相关文档：
> · 华为云 ECS 开台细节（**Stage 2 版**）→ [`cloud_ecs_stage2.md`](cloud_ecs_stage2.md)
> · 华为云 ECS 开台细节（**成员 A 写的 Stage 1 版，更详细**）→ [`cloud_ecs_stage1.md`](cloud_ecs_stage1.md)
> · Stage 2 的阻塞项 → [`stage2_blockers.md`](stage2_blockers.md)
> · 实验记录（数字都写在这里）→ [`../../docs/experiments.md`](../../docs/experiments.md)

## TL;DR · A 交权重那一刻，先做这三件事

1. **拿权重**：`stage1_model.pt`（含优化器，358 MB，**能 `--resume`**）
   或 `stage1_best.pt`（验证集最优，115 MB，**不含优化器、更轻**）——
   Stage 2 只要模型权重，**用哪个都行**；要能续训就用前者。
2. **验权重**：`step >= 1000`、有 `model` 段、有 `speaker_encoder` 段
   （§1.4 有一条命令一次验完）。
3. **本机跑 3 步冒烟**：`stage2/train.py --vocoder hifigan --stage1-checkpoint <ckpt> --steps 3`
   ⇒ 通了再开云机。（本机 CPU 约 4 s/step，3 步一分钟。）

---

## 0. 本手册覆盖成员 B 的哪些职责

按 [`algorithms/voiceprint_protection/TASKS.md`](../../algorithms/voiceprint_protection/TASKS.md) §4：

| B 的职责 | 状态 | 本手册对应 |
|---|---|---|
| W2 · GE2E 桥接（`bridge_speaker.py`） | ✅ 已完成 | —— （不再动） |
| W2 · VCTK 数据管道（`prepare_data.py` / `split_speakers.py`） | ✅ 已完成 | §1.3 **对账**（新机器上要重跑一遍） |
| W2 · 冒烟测试、平台样例 | ✅ 已完成 | §7.1 **重生成样例**（用训练好的 G 重新出） |
| **W4–W5 · Stage 2 主责**（G / D_adv / L_per / L_de / θ_x / 训练主循环） | 🟡 **本手册的主线** | §2–§6 |
| W6 起 · **主责交换** → B 改做 Stage 1 主责 | ⬜ 未到 | 用 A 的 `scripts/run_stage1.py` 一条龙；本手册的"步骤 1→4"套路（自检 → 冒烟 → 正式 → 续训）同样适用 |
| 共同区（`config.py` / `audio_utils.py` / `infer.py` / `evaluate.py` / `service.py` / README） | 共同维护 | 改完**必须跑全量回归**（§8.3） |

---

## 0.5 命令速查表

| 目的 | 命令（在仓库根执行） |
|---|---|
| 拉最新代码 | `git pull --ff-only origin group_N005_submission` |
| 环境自检 | `python algorithms/voiceprint_protection/scripts/check_env.py` |
| 数据划分对账 | `python algorithms/voiceprint_protection/scripts/split_speakers.py --check <数据根>/split_speakers.json` |
| Stage 2 冒烟 | `python algorithms/voiceprint_protection/stage2/train.py --vocoder hifigan --stage1-checkpoint <ckpt> --steps 3 --log-every 1` |
| Stage 2 正式 | `bash scripts/cloud_train_stage2.sh --start --steps 200000 --max-hours 24` |
| 训练状态 | `bash scripts/cloud_train_stage2.sh --status` |
| 续训 | `bash scripts/cloud_train_stage2.sh --start --steps 200000 --max-hours 24 --resume` |
| 评测（论文路线） | `python algorithms/voiceprint_protection/evaluate.py --mode trained --protect-mode stage2 --generator-checkpoint <stage2_last.pt> --stage1-checkpoint <ckpt> --split test --limit-per-speaker 5` |
| 评测（绝对 WER） | 上面那条 **加** `--wer --whisper-model <模型> --transcript-dir <数据根>/raw/txt` |
| 重生成平台样例 | `python algorithms/voiceprint_protection/scripts/make_sample.py --mode stage2 --stage1-checkpoint <ckpt> --stage2-checkpoint <stage2_last.pt>` |
| 平台端到端 | `python algorithms/voiceprint_protection/scripts/test_platform_e2e.py` |

> ⚠️ **本机（成员 B 这台）没有 NVIDIA GPU**（Intel Iris Xe 核显，torch 是 `2.4.1+cpu`）。
> 训练必须在云机上跑；本机只做「验证链路 + 评测 + 出样例」。

---

## 1. 本机准备（Windows，约 30 分钟）

### 1.1 拉代码

```bash
cd <仓库根>
git pull --ff-only origin group_N005_submission
git log --oneline -3
```

**看什么**：拉到最新那一条是成员 A 或自己推的 commit。
**不对**：有冲突 → 先 `git stash` 再 pull，再 `git stash pop`。

### 1.2 环境自检

```bash
conda activate vpp
export VPP_DATA_ROOT=D:/vpp/data        # Windows Git Bash 写法
python algorithms/voiceprint_protection/scripts/check_env.py
```

> 💡 **Git Bash 里 `conda activate` 有时不好用**，直接用绝对路径也行：
> ```bash
> export VPP_DATA_ROOT=D:/vpp/data
> PY=/d/conda/envs/vpp/python.exe        # Windows Git Bash 的路径写法
> $PY algorithms/voiceprint_protection/scripts/check_env.py
> ```
> 下面的命令统一写成 `python`（已 `conda activate vpp` 的前提下）。

**看什么**：依赖齐全、上游权重在位、`third_party/` 三个仓库的 commit 对得上。
**不对**：报缺权重 → `python algorithms/voiceprint_protection/scripts/download_encoder_weights.py`
（还有 `download_rtc_weights.py` / `download_hifigan_weights.py`）。

### 1.3 数据对账（★ **两个人必须用同一批数据**）

```bash
python algorithms/voiceprint_protection/scripts/split_speakers.py --check D:/vpp/data/split_speakers.json
```

**看什么**：`110` 个说话人、train/val/test = 80/10/20、与 `docs/notes/vctk_speakers.txt` 一致。
**不对**：对不上 → **不要继续**，数据不对齐的话两个人的指标没法比。

再确认清单条数：

```bash
wc -l D:/vpp/data/train.txt D:/vpp/data/val.txt D:/vpp/data/test.txt
```

**看什么**：**8000 / 1000 / 2000**（合计 11,000）。

### 1.4 收 A 的 Stage 1 权重 + 验真

```bash
# 放到（★ 名字要对，代码按这个名字找）
algorithms/voiceprint_protection/weights/stage1_model.pt
```

> ★ **A 有两个权重，Stage 2 用哪个都行**：
> | 文件 | 大小 | 内容 | Stage 2 用它做什么 |
> |---|---|---|---|
> | `stage1_model.pt` | 358 MB | 模型 + 优化器 + 随机数状态 | 训练 Stage 2 的解码器；**能 `--resume`** |
> | `stage1_best.pt` | 115 MB | 验证集最优的模型（不含优化器） | 同样能用，**传起来更轻** |
>
> ⚠️ **别把 3 步冒烟的 `stage1_partial_step{N}.pt` 当正式权重** ——
> 代码会按 `step` 拦截（< 1000 直接拒绝），但别去动它。

**验真（★ 三条都要过）**：

```bash
python -c "import torch; d=torch.load('algorithms/voiceprint_protection/weights/stage1_model.pt', map_location='cpu', weights_only=False); print('step =', d.get('step')); print('有 speaker_encoder 段:', 'speaker_encoder' in d); print('有 model 段:', 'model' in d)"
```

| 看什么 | 什么算对 | 不对怎么办 |
|---|---|---|
| `step` | **≥ 1000**（A 那个是 500,000） | 是个小数字 → 是冒烟产物，**找 A 重给** |
| `speaker_encoder` 段 | **存在** ⇒ Stage 1 微调过 GE2E ⇒ Stage 2 用 `--speaker-encoder stage1`（**本仓库的默认值就是它**） | 不存在 ⇒ 用 `--speaker-encoder upstream`（并在文档里写明偏离了论文） |
| `model` 段 | 存在 | 不存在 ⇒ 格式不对，找 A |

> ★★ **为什么 E_spk 这件事这么关键（两边独立踩过同一个坑）**：
>
> | | 用**上游预训练** E_spk | 用**checkpoint 里微调**的 E_spk |
> |---|---|---|
> | Stage 1 的 Sim_prot | 0.4367 ❌ | **0.999996** ✅ |
>
> ★ 同一段原音在两个 E_spk 下的嵌入余弦**只有 0.518** —— 两个空间根本不是一回事。
>
> · A 在 **Stage 1 评测**上踩过（2026-11-21，`docs/experiments.md` §4e）；
> · B 在 **Stage 2 训练/推理**上独立发现（§0.14 第 7 条）：
>   训练用上游、推理用微调后的 ⇒ 生成器 G 的输入**离分布** ⇒
>   **δ 退化成随机方向、防御完全失效，而且不报错**。
>
> ⇒ **Stage 2 训练用 `--speaker-encoder stage1`（= checkpoint 里的微调 E2E）**，
> 而且这个选择会被**写进 Stage 2 的 checkpoint**，推理时自动照着做 ——
> 两边不可能对不上。

---

## 2. 开云机（华为云 ECS）

**详细步骤见 [`cloud_ecs_stage2.md`](cloud_ecs_stage2.md)**（Stage 2 版），
或成员 A 写的 [`cloud_ecs_stage1.md`](cloud_ecs_stage1.md)（**更详细**：单价、每步耗时估算、
「出问题就撤」的兜底方案）—— 两份结论一致，可以对着看。

要点复述（别踩这几个）：

| # | 要点 |
|---|---|
| 1 | **计费方式选「按需」，不要选「竞价」** —— 官方《ECS · 计费说明》：「代金券不适用于竞价计费型实例」 |
| 2 | 规格：★ **先用 `pi2.2xlarge.4`（T4 16 GB，约 3–5 元/h）**。Stage 2 要冻结 Stage 1 + HiFi-GAN + GE2E，前向跑 3 次，batch 8 / 3 秒片段 ⇒ **显存约 8–12 GB（fp32）/ 4–6 GB（开 AMP）**，T4 的 16 GB 够用。想更快再换 `p2s.2xlarge.8`（V100 32 GB） |
| 3 | 镜像用**干净的 Ubuntu 22.04**，自己装 torch cu121（脚本里已经写了，还带 NJU 镜像兜底） |
| 4 | 数据**从本机 tar + scp**（1.36 GiB），不要在云机上重下 Edinburgh |
| 5 | 安全组放行 **22**；**绑 EIP**；用**密钥对**登录 |
| 6 | **GPU 配额是按区域分配的** —— 下拉框里看不到 GPU 规格 = 这个区域没配额 ⇒ 换区域 |
| 7 | 训完**释放实例 + 删数据盘 + 释放 EIP** —— 关机只停算力费，硬盘和 IP 照收 |

传东西（本机执行）：

```bash
# 数据（1.36 GiB）
tar czf vpp_data.tgz -C D:/vpp/data audio16k train.txt val.txt test.txt spk2id.txt split_speakers.json
scp vpp_data.tgz root@<EIP>:/root/

# Stage 1 权重（★ Stage 2 训练的前提）
scp algorithms/voiceprint_protection/weights/stage1_model.pt \
    root@<EIP>:<仓库>/algorithms/voiceprint_protection/weights/

# 代码（云机上执行）
git clone git@github.com:Liastar-li/algorithm_research_frontend.git
cd algorithm_research_frontend && git checkout group_N005_submission
```

云机上解包数据：

```bash
mkdir -p /root/vpp/data && tar xzf /root/vpp_data.tgz -C /root/vpp/data
export VPP_DATA_ROOT=/root/vpp/data
```

---

## 3. 装环境 + 自检（云机，约 20 分钟）

```bash
cd algorithm_research_frontend
bash scripts/cloud_train_stage2.sh --setup
source ~/vpp-venv/bin/activate
export VPP_DATA_ROOT=/root/vpp/data

python algorithms/voiceprint_protection/scripts/check_env.py
python algorithms/voiceprint_protection/stage2/verify_gradients.py     # 必须 20/20
```

**看什么**：`nvidia-smi` 能看到卡；`verify_gradients` **20/20 通过**。
**不对**：`verify_gradients` 有失败 ⇒ **别开训**，那是链路坏了（梯度断在某一层，
loss 照样降、指标全错）。

---

## 4. 训练 Stage 2

### 4.1 冒烟（3 步，5 分钟）

```bash
bash scripts/cloud_train_stage2.sh --start --steps 3 --log-every 1 --max-hours 0.2
bash scripts/cloud_train_stage2.sh --status
```

**看什么**（四个数都要是有限的、并且在动）：

| 数 | 期望 | 不对说明 |
|---|---|---|
| `total` | 在动 | 完全不动 ⇒ 梯度断了 |
| `de`（L_de） | 有限（约 0.6–0.8） | NaN ⇒ 梯度炸了，减 `--accum-steps` 或关 AMP |
| `mask` | 有限 | 恒为 0 ⇒ 掩蔽阈值那步没接上 |
| `sim_clone` | 0.6–0.8 | 恒等于 1 ⇒ 扰动没生效 |

### 4.2 正式训练（200k 步）

```bash
bash scripts/cloud_train_stage2.sh --start --steps 200000 --max-hours 24 \
     --batch-size 8 --accum-steps 1
```

> ★ **`--max-hours` 一定要给**：云机按小时计费，到点自动存盘退出，
> 哪怕忘了看，损失也有上限。（实例本身还要手动释放，见 §6。）
>
> ★ **要给够**：按 V100 上 0.2–0.4 s/step 估，200k 步约 **11–22 小时**。
> 给小了会在中途停 —— **没关系，加 `--resume` 接着跑就行**（下一节），
> 但别把 `--max-hours` 设成 8 小时然后以为"训完了"（日志里会有
> `★ 是被 --max-hours 停的`）。

### 4.3 盯进度

```bash
bash scripts/cloud_train_stage2.sh --status
tmux attach -t vpp-stage2      # 实时看（Ctrl-b d 退出来）
```

**看什么**：

| 数 | 期望 | 不对说明 |
|---|---|---|
| `de`（L_de） | **缓慢下降** | 上升 ⇒ 符号反了（`losses.py::defense_loss`） |
| `sim_clone` | **下降**（论文训完 0.13） | 不降 ⇒ 防御没学到东西 |
| `per` | 平稳（不要暴涨） | 暴涨 ⇒ λ_per 相对权重太大 |
| `it/s` | V100 上 0.2–0.5 | 明显更慢 ⇒ batch/显存不对 |

> 训练结束会自动打一行 **`[L_de 方向自检]`** ——
> 方向不对会直接喊出来（A 的 `L_dis` 就是同一种"不报错但方向反了"的坑）。

### 4.4 断了 / 被抢占 ⇒ 续训

```bash
bash scripts/cloud_train_stage2.sh --start --steps 200000 --max-hours 24 --resume
```

**看什么**：打印 `已从 ... 续训，起始 step = N`（N 是断点步数）。
优化器状态与随机数状态都在 checkpoint 里 ⇒ 恢复是**连续**的，不是"接着跑"。

### 4.5 什么时候停

- 论文量级 = **200,000 步**（`config.STAGE2["iterations"]`）。
- 跑完或 `--max-hours` 到点都会自动存盘 `weights/stage2/stage2_last.pt`。

---

## 5. 评测（M4 验收）

### 5.1 ★ 先跑三条基线（**都要走论文路线**）

```bash
CKPT1=algorithms/voiceprint_protection/weights/stage1_model.pt
CKPT2=algorithms/voiceprint_protection/weights/stage2/stage2_last.pt

# ① 无防御（α=0）
python algorithms/voiceprint_protection/evaluate.py --mode none --protect-mode stage2 \
       --stage1-checkpoint $CKPT1 --split test --limit-per-speaker 5

# ② 高斯噪声（论文 Table 1 的 "Gaussian Noise" 那一行）
python algorithms/voiceprint_protection/evaluate.py --mode random --protect-mode stage2 \
       --stage1-checkpoint $CKPT1 --split test --limit-per-speaker 5

# ③ ★ 我们自己（训练好的 G）—— 这一行才是 "Ours"
python algorithms/voiceprint_protection/evaluate.py --mode trained --protect-mode stage2 \
       --generator-checkpoint $CKPT2 --stage1-checkpoint $CKPT1 \
       --split test --limit-per-speaker 5 --output results/stage2_eval/ours.json
```

> ★★ **`--protect-mode stage2` 不能省**：`--mode` 只决定**扰动从哪来**
> （none / random / trained），`--protect-mode` 决定**保护后语音怎么来**
> （clone = 用克隆机重新合成 / stage1 = 论文原版 δ 随机 / stage2 = 论文原版 + 训练好的 G）。
> **不写 `--protect-mode stage2` 就会走 clone 路线**，
> 而 clone 路线的 Sim_prot 有 **0.82 的结构性天花板**（重新合成本身就改音色），
> 拿它跟论文的 0.95 比会得出"我们差很远"的错误结论。

**看什么**（论文 Table 1 的目标）：

| 变体 | Sim_prot ↑ | Sim_clone ↓ | DSR(τ=0.5) ↑ |
|---|---|---|---|
| 无防御 | ≈ 0.91 | ≈ 0.91 | 0% |
| 高斯噪声 | ≈ 0.85 | ≈ 0.35 | 45.2% |
| **论文 · Ours** | **0.95** | **0.13** | **87.2%** |

> ★ **这三条都走论文路线**（`--protect-mode stage2`），而且 Sim_prot / Sim_clone
> 是在**微调后**的 E_spk 空间里算的（`embed_for_route` 自动用这条路线的 E_spk）——
> **不要跟"用上游 E_spk 算出来的数"混着比**（那个口径会把所有指标压低，见 §1.4 的表）。
>
> ★ 好消息：A 的 Stage 1 重建质量是 **Sim_prot 0.999996**（同一口径）⇒
> 论文路线上的 Sim_prot **不该被 Stage 1 拖后腿**了。

**不对怎么办**：

| 症状 | 原因 | 解法 |
|---|---|---|
| 三行的 Sim_prot 都 ≈ 0.8 出头 | 走的是 clone 路线 | 加 `--protect-mode stage2` |
| Sim_prot 明显 < 0.9 | **口径错**（用了上游 E_spk）或 Stage 1 权重不对 | 确认 `--stage1-checkpoint` 指对了、且 `--protect-mode stage2` |
| `sim_clone` 和无防御那行差不多 | δ 没生效 | 检查 `--mode trained` + checkpoint 路径；看训练时 `sim_clone` 有没有降 |
| DSR 全 0 | Sim_clone 还很高（>0.5） | 同上；论文要 0.13 |

### 5.2 绝对 WER

```bash
python algorithms/voiceprint_protection/evaluate.py --mode trained --protect-mode stage2 \
       --generator-checkpoint $CKPT2 --stage1-checkpoint $CKPT1 \
       --split test --limit-per-speaker 5 --wer \
       --whisper-model <Whisper 目录> --transcript-dir <数据根>/raw/txt \
       --output results/stage2_eval/ours_wer.json
```

**看什么**：

| 指标 | 我们已测到的 | 论文 |
|---|---|---|
| WER（保护后） | —— | **5.30%** |
| WER（原音 · ASR 下限） | 3.30%（成员 B 实测） | —— |

**不对怎么办**：
- 打印「有 N 条没找到 VCTK 原始转写」⇒ 那些算的是**相对 WER**，**不能写进报告**；
  修法：`--transcript-dir` 指到 `<数据根>/raw/txt`，并确认 `txt/` 已下载。
- WER 恒等于 ~100% ⇒ `protect()` 念的不是原句（要用 `--transcript-dir`）。

### 5.3 消融（论文 Table 2）

⚠️ **消融要重训，不是改评测参数** —— `config.ABLATION` 里写的是**训练时**要改的
`lambda_de` / `lambda_mask`。正确做法是**用不同的 λ 各训一个 G**，再分别评测。
（评测脚本只负责"读一个 G 算指标"。）

**先在云机上各训一个**（★ 每个都是一次完整训练 ⇒ 12–24 h；
预算紧的话可以训少一点步数，但**三组步数要一样**，否则没法比）：

```bash
# ① w/o L_de（λ_de = 0）—— 论文里 DSR 掉到 12.5%
bash scripts/cloud_train_stage2.sh --start --steps 200000 --max-hours 24 \
     --lambda-de 0 --outdir <仓库>/algorithms/voiceprint_protection/weights/stage2_wo_de

# ② w/o L_mask（λ_mask = 0）
bash scripts/cloud_train_stage2.sh --start --steps 200000 --max-hours 24 \
     --lambda-mask 0 --outdir <仓库>/algorithms/voiceprint_protection/weights/stage2_wo_mask

# ③ w/o L_mel（只留 L_mask）
bash scripts/cloud_train_stage2.sh --start --steps 200000 --max-hours 24 \
     --no-mel-loss --outdir <仓库>/algorithms/voiceprint_protection/weights/stage2_wo_mel
```

**再分别评测**（本机）：

```bash
for ab in wo_de wo_mask wo_mel; do
  python algorithms/voiceprint_protection/evaluate.py --mode trained --protect-mode stage2 \
         --generator-checkpoint algorithms/voiceprint_protection/weights/stage2_$ab/stage2_last.pt \
         --stage1-checkpoint $CKPT1 --split test --limit-per-speaker 5 \
         --output results/stage2_eval/ablation_$ab.json
done
```

**看什么**（论文 Table 2 的目标）：

| 消融 | 论文 DSR | 含义 |
|---|---|---|
| 完整 | 87.2% | — |
| w/o L_de | **12.5%** | ★ 防御损失是主力 —— 去掉它防御基本失效 |
| w/o L_mask | **84.3%** | 掩蔽项主要管**听感**，对 DSR 影响小 |
| w/o L_mel | — | 频谱保持项 |

> ★ **如果预算不够**：消融各训 20k 步（≈2 h/组）也可以，
> 但**必须在报告里写明步数**，并说明它与 200k 的主实验不可直接比较。

---

## 6. 收结果 + 释放实例（★ 别忘了这一步）

```bash
# 本机：把 checkpoint 拉回来
scp root@<EIP>:<仓库>/algorithms/voiceprint_protection/weights/stage2/stage2_last.pt \
    algorithms/voiceprint_protection/weights/stage2/
scp root@<EIP>:<仓库>/logs/stage2.log .
```

然后**去华为云控制台**：

1. **释放实例**（不是只关机）；
2. **删掉数据盘**（竞价实例的数据盘不会自动删，按需实例也要手动勾）；
3. **释放 EIP**、解绑公网带宽。

> 官方原文（按需实例关机）：「ECS 实例本身（vCPU，内存，镜像、GPU）不计费，
> 其他所挂载的资源如云硬盘，公网 IP 或带宽则正常计费。」
> ⇒ 关机只是不烧算力，**硬盘和 IP 还在烧钱**。

---

## 7. 平台样例 + 端到端

### 7.1 重新生成样例（★ 用训练好的 G）

```bash
python algorithms/voiceprint_protection/scripts/make_sample.py --mode stage2 \
       --stage1-checkpoint algorithms/voiceprint_protection/weights/stage1_model.pt \
       --stage2-checkpoint algorithms/voiceprint_protection/weights/stage2/stage2_last.pt
```

产出 `public/samples/group_n005_voiceprint_protection/` 下的
`input.wav` / `output.wav` / `output_clone_baseline.wav` / `output_clone_protected.wav` / `SOURCES.md`。

**看什么 / 怎么验**：
- ★ **必须真的听一遍**（M3-2 的教训）：保护后语音要**听得出来是本人**，
  而"防御后的克隆音"要**明显不像本人**。
- 两个克隆音的差别**就是防御效果本身**（同一个克隆机、同一句话，只差音色卡）。
- 差别很小 ⇒ δ 没生效 ⇒ 回 §5.1 查。

### 7.2 服务 + 平台端到端

```bash
# 终端 1：起服务
conda activate vpp
python algorithms/voiceprint_protection/service.py

# 终端 2：起前端
npm run dev

# 终端 3：端到端自检
python algorithms/voiceprint_protection/scripts/test_platform_e2e.py
```

**看什么**：`test_platform_e2e.py` **29 项全通过**；
`python algorithms/voiceprint_protection/scripts/test_service.py` **41/41**。

---

## 8. 文档与交付

### 8.1 实验记录（`docs/experiments.md`）

新开一节 **§4f**（§4e 是 2026-11-23 写手册那一轮，已占用了），
按 `§4c` / `§4d` 的写法，必须包含：

- 训练实况：步数、耗时、it/s、机器规格、**中断与续训的次数**；
- Stage 1 权重来自哪一版、`--speaker-encoder` 选了哪个；
- **§5.1 三条基线的完整数字**（Sim_prot / Sim_clone / DSR 三个 τ / WER 两个）；
- 消融表（论文 Table 2 对照）；
- **诚实写明偏离论文的地方**（例如 Whisper 用了 small 不是 large-v3、
  数据每人只取 100 句、E_spk 用了上游的）。

### 8.2 README / 任务板

- `README.md` 的状态表：Stage 2 改成 ✅，评测表填上真实数字；
- `TASKS.md` §0.3 状态表 + 新增一节本轮进展（格式照 §0.14 / §0.15）。

### 8.3 提交

```bash
git add -A
git commit -m "feat(stage2): 训练完成 + 评测数字（成员 B）"
git push origin group_N005_submission
```

> ★ **提交前先跑一遍回归**（改了公共代码就必须全跑）：

```bash
for s in stage2/verify_gradients.py stage2/verify_psychoacoustics.py stage2/verify_mel.py \
         scripts/verify_hifigan.py scripts/verify_rtc_mel.py scripts/test_stage1.py \
         scripts/test_audio_utils.py scripts/test_service.py; do
  echo "##### $s"
  python algorithms/voiceprint_protection/$s 2>&1 | tail -3
done
npm test
```

**看什么**：全部"通过"，`npm test` 6/6。

---

## 9. 故障排查速查

| 症状 | 原因 | 解法 |
|---|---|---|
| `找不到 Stage 1 权重` | 名字/位置不对 | 必须叫 `stage1_model.pt` 且在 `weights/` 下 |
| `找到的 Stage 1 权重都只训了很少的步数` | 拿到冒烟产物 | 找 A 要正式的 |
| `x 与 x_adv 的帧数必须相等` | 裁剪长度不是 32 的倍数 | 用 `stage2/dataset.py`（已按 32 对齐），别自己改裁剪 |
| `显存不够 / CUDA out of memory` | batch 太大 | `--batch-size 4 --accum-steps 2`（等效 batch 不变） |
| 训练 loss 是 NaN | 梯度炸了 | `--clip-grad 1.0`；确认 AMP 在起作用 |
| `it/s` 只有 0.05 | 显存不够在反复换页 | 减小 batch；确认 `nvidia-smi` 显示没别的进程占卡 |
| `bash\r: command not found` | Windows 拉下来的 `.sh` 是 CRLF | 仓库已加 `.gitattributes`；实在不行 `sed -i 's/\r$//' scripts/*.sh` |
| 云机脚本跑一半没了 | 竞价实例被抢占 | 用**按需**；或加 `--resume` |
| 训完发现钱在烧 | 实例没释放 | 控制台释放实例 + 删数据盘 + 释放 EIP |
| WER 恒 ~100% | 保护时没念原句 | 加 `--transcript-dir` |
| WER 算的是相对值 | 没找到 VCTK 转写 | 看输出里的「没找到原始转写」计数 |
| 三条基线的 Sim_prot 都一样 | 走错保护链 | 加 `--protect-mode stage2` |
| `sim_clone` 不降 | δ 没生效 | 看训练日志的 `sim_clone`；确认 `--mode trained` + checkpoint 路径 |

---

## 附录 A · 论文对照表（交报告时用）

| 指标 | 论文 | 我们（交报告时填） |
|---|---|---|
| Sim_prot ↑ | 0.95 | |
| Sim_clone ↓ | 0.13 | |
| DSR(τ=0.3) ↑ | 79.8% | |
| DSR(τ=0.4) ↑ | 83.5% | |
| DSR(τ=0.5) ↑ | 87.2% | |
| WER ↓ | 5.30% | |
| 无防御 DSR | 0% | |
| 高斯噪声 DSR | 45.2% | |
| 消融 w/o L_de | DSR 12.5% | |
| 消融 w/o L_mask | DSR 84.3% | |

## 附录 B · 时间与花钱预算

| 阶段 | 时长 | 在哪跑 |
|---|---|---|
| 本机准备 + 权重验收 | 0.5 h | 本机（不花钱） |
| 开台 + 传数据 + 装环境 | 1.5 h | 云机（**按小时计费，这段也在烧钱**） |
| 冒烟 + 自检 | 0.5 h | 云机 |
| Stage 2 正式训练 | **12–24 h**（V100，200k 步） | 云机 |
| 评测（三条基线） | 2–4 h | 本机（CPU，约 27 s/条 × 300 条） |
| 评测（加 WER） | 再 1–2 h | 本机（Whisper 在 CPU 上很慢；`large-v3` 更慢 ⇒ 可先用 `small` 跑通、最后用 `large-v3` 补数） |
| 样例 + 平台 e2e | 1 h | 本机 |
| **合计云机时间** | **约 15–26 h** | ⇒ 按 ¥5/h 估 **¥75–130** |

> ★ **省钱三原则**（任务板 §8.2）：
> ① 只在真正要训练时开机；② 训完立刻**释放**（不是关机）；③ **不要用竞价实例**（用不上代金券）。
