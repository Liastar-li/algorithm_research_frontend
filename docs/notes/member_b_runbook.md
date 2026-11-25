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

1. **拿权重**：A 这次是挂在 **GitHub Release `stage1-250k-bn`** 上的
   （`stage1_step250000_wide_test_bn.pt`，119 MB），仓库里只留同名 JSON ——
   下载完**必须按 `checkpoint_sha256` 对账**（§1.4 有命令）。
   老规矩仍然成立：`stage1_model.pt`（含优化器，358 MB，**能 `--resume`**）
   或 `stage1_best.pt`（验证集最优，115 MB，**不含优化器、更轻**）——
   Stage 2 只要模型权重，**用哪个都行**；要能续训就用前者。
   ★ **A 这次 Release 里的那个文件是"模型 only"**：本机实测 top keys =
   `['bn_calibration', 'config', 'model', 'speaker_encoder', 'step']`，`step = 250000`，
   **没有 optimizer 段** ⇒ Stage 1 不能 `--resume`（Stage 2 不受影响），
   而且文件名**不是** `stage1_model.pt` ⇒ 后面所有命令都要显式传 `--stage1-checkpoint`。
2. **验权重**：`step >= 1000`、有 `model` 段、有 `speaker_encoder` 段
   （§1.4 有一条命令一次验完）。
3. **本机跑 3 步冒烟**：`stage2/train.py --vocoder hifigan --stage1-checkpoint <ckpt> --steps 3`
   ⇒ 通了再开云机。
   ★ **本机 CPU 整句冒烟约 33 s/step**（不是早先估的 4 s/step —— 那个数是 stub 声码器测的），
   3 步冒烟约 2 分钟；**带 `--de-space upstream` 的精简预检约 1 小时**（§4.1.1）。
4. **E_spk 口径**：微调过的那个已经塌缩（§1.4.1），所以
   训练加 `--de-space upstream`、评测加 `--espk-space upstream` 并保留 `--dual-espk`，
   报数时**注明口径**。

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
| Stage 2 冒烟 | `python algorithms/voiceprint_protection/stage2/train.py --vocoder hifigan --stage1-checkpoint <ckpt> --de-space upstream --steps 3 --log-every 1` |
| Stage 2 本机预检 | 同上，把 `--steps` 改大（150 步精简配置 ≈ 半小时，见 §4.1.1） |
| Stage 2 过拟合诊断 | `train.py ... --overfit 4 --steps 200 --lambda-gan 0`（§4.1.1 末，判"路通不通"） |
| Stage 2 正式 | `bash scripts/cloud_train_stage2.sh --start --steps 200000 --max-hours 24`（`--de-space` 已默认 upstream） |
| 训练状态 | `bash scripts/cloud_train_stage2.sh --status` |
| 续训 | `bash scripts/cloud_train_stage2.sh --start --steps 200000 --max-hours 24 --resume` |
| 评测（论文路线） | `python algorithms/voiceprint_protection/evaluate.py --mode trained --protect-mode stage2 --generator-checkpoint <stage2_last.pt> --stage1-checkpoint <ckpt> --split test --limit-per-speaker 5 --espk-space upstream` |
| 评测（绝对 WER） | 上面那条 **加** `--wer --whisper-model <模型> --transcript-dir <数据根>/raw/txt` |
| 重生成平台样例 | `python algorithms/voiceprint_protection/scripts/make_sample.py --mode stage2 --stage1-checkpoint <ckpt> --stage2-checkpoint <stage2_last.pt>` |
| 平台端到端 | `python algorithms/voiceprint_protection/scripts/test_platform_e2e.py` |

> ⚠️ **本机（成员 B 这台）没有 NVIDIA GPU**（Intel Iris Xe 核显，torch 是 `2.4.1+cpu`）。
> 训练必须在云机上跑；本机只做「验证链路 + 评测 + 出样例」。
>
> ⚠️ **Windows 控制台默认 GBK**，脚本里的 `✅`/`★`/`⇒` 会直接把程序炸在**最后一步**
> （评测跑到 2 小时才打印结果那种）。2026-11-25 已给全部入口补了仓库既有片段：
> ```python
> if hasattr(sys.stdout, "reconfigure"):
>     sys.stdout.reconfigure(encoding="utf-8", errors="replace")
> ```
> 新加脚本入口时记得带上；临时兜底用 `export PYTHONIOENCODING=utf-8`。

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

★ **实际交付走的是 GitHub Release，不是仓库里的文件**（2026-11-25 确认）：
A 把 250k 步的权重挂在 Release `stage1-250k-bn` 上（public，不用登录就能下），
仓库里只留了同名 JSON（`weights/stage1_fresh_200k/stage1_step250000_wide_test_bn.json`）。
所以本机拿权重的正确动作是**下载 + 按 sha256 对账**：

```bash
# 1) 看 Release 里挂了什么（digest 字段就是资产文件的 sha256）
curl -s https://api.github.com/repos/<owner>/<repo>/releases | grep -E '"tag_name"|"name"|"size"|"digest"'

# 2) 下载（browser_download_url），放到代码按名字找的位置
#    algorithms/voiceprint_protection/weights/stage1_fresh_200k/stage1_step250000_wide_test_bn.pt

# 3) 本机算 sha256，跟 JSON 里的 checkpoint_sha256 对
python -c "import hashlib;p='algorithms/voiceprint_protection/weights/stage1_fresh_200k/stage1_step250000_wide_test_bn.pt';print(hashlib.sha256(open(p,'rb').read()).hexdigest())"
```

对上的那一刻（`14f7e07b…8832`）才叫"验收通过"——**别只比对文件名和大小**。
（本机实测：HTTPS 到 `api.github.com` 现在是通的，之前"HTTPS 被封"的结论已经过期。）

```bash
# 老的名字也仍然有效，代码按这个名字找（如果 A 以后直接放仓库）
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
# ★ A 这次交付的文件名是 stage1_step250000_wide_test_bn.pt，**不叫 stage1_model.pt**
CKPT1=algorithms/voiceprint_protection/weights/stage1_fresh_200k/stage1_step250000_wide_test_bn.pt
python -c "import torch; d=torch.load('algorithms/voiceprint_protection/weights/stage1_fresh_200k/stage1_step250000_wide_test_bn.pt', map_location='cpu', weights_only=False); print('top keys =', sorted(d)); print('step =', d.get('step')); print('有 speaker_encoder 段:', 'speaker_encoder' in d); print('有 model 段:', 'model' in d)"
```

> ★ **本机实测（2026-11-25）的四个字段**：
> `top keys = ['bn_calibration', 'config', 'model', 'speaker_encoder', 'step']`、
> `step = 250000`、`model` ✅、`speaker_encoder` ✅、**没有 optimizer 段**
> ⇒ Stage 1 **不能 `--resume`**（对 Stage 2 没影响，Stage 2 只要解码器），
> 而且 `bn_calibration` 这一段说明 A 做过 BN 校准 —— **评测时要按它来**。
> 顺带把 sha256 也算了：本机 `14f7e07b…8832` 与 JSON 里的 `checkpoint_sha256`
> （`14f7e07be7ab…dd8832`）**逐字对上** ⇒ 验收通过。

| 看什么 | 什么算对 | 不对怎么办 |
|---|---|---|
| `step` | **≥ 1000**（A 那个是 500,000） | 是个小数字 → 是冒烟产物，**找 A 重给** |
| `speaker_encoder` 段 | **存在** ⇒ Stage 1 微调过 GE2E ⇒ Stage 2 用 `--speaker-encoder stage1`（**本仓库的默认值就是它**） | 不存在 ⇒ 用 `--speaker-encoder upstream`（并在文档里写明偏离了论文） |
| `model` 段 | 存在 | 不存在 ⇒ 格式不对，找 A |

> ★★ **为什么 E_spk 这件事这么关键（两边独立踩过同一个坑）**：
>
> | | 用**上游预训练** E_spk | 用**checkpoint 里微调**的 E_spk |
> |---|---|---|
> | Stage 1 的 Sim_prot | 0.4367 ❌ | **0.999996** ✅（← 2026-11-21 的结论，**已被 §1.4.1 推翻**） |
>
> ★ 同一段原音在两个 E_spk 下的嵌入余弦**只有 0.518** —— 两个空间根本不是一回事。
>
> · A 在 **Stage 1 评测**上踩过（2026-11-21，`docs/experiments.md` §4e）；
> · B 在 **Stage 2 训练/推理**上独立发现（§0.14 第 7 条）：
>   训练用上游、推理用微调后的 ⇒ 生成器 G 的输入**离分布** ⇒
>   **δ 退化成随机方向、防御完全失效，而且不报错**。
>
> ⇒ **Stage 2 的 δ 仍然生成于 `--speaker-encoder stage1`（= checkpoint 里的微调 E2E）**，
> 而且这个选择会被**写进 Stage 2 的 checkpoint**，推理时自动照着做 ——
> 两边不可能对不上。

### 1.4.1 ★ 但 0.999996 是假的 —— 微调后的 GE2E 已经塌缩（2026-11-25 的新结论）

A 那份 250k + BN 校准的 JSON（`stage1_step250000_wide_test_bn.json`）自己就把答案推翻了：

| 口径 | Sim_prot | Sim_clone | 同人间余弦 | 异人间余弦 | gap |
|---|---|---|---|---|---|
| checkpoint 里微调过的 E2E | 0.9467 | 0.9286 | 0.9453 | 0.8958 | **+0.049** |
| 上游预训练 GE2E | 0.6994 | 0.6536 | 0.7663 | 0.5670 | **+0.199** |

（前四列余弦均值取自 A 的 JSON：`stage1_ge2e` / `upstream_ge2e`；
Sim_prot / Sim_clone 取自 B 本机 `evaluate.py --dual-espk` 的复算。）

B 在本机用 A 的权重独立复算过一遍（不靠 A 的脚本），异人间 gap 只有 **+0.013**，
比上表还糟 —— **塌缩是真的**，不是口径写错。
后果是：`Sim_prot = 0.999996` 那种"接近 1"的数字，在塌缩空间里**任何两条语音都能拿到**，
所以它证明的是"空间退化了"，不是"重建得好"。
（旁证：同一份 JSON 里 `upstream_original_rebuilt_cosine = 0.706`，
与 B 本机 upstream 口径的 0.6994 互相印证 —— 重建本身其实是中等水平。）

★ 同一份 JSON 还留了个更能说明问题的数：`upstream_speaker_top1`
= clean 0.729 / oracle 0.757 / **rebuilt 0.229**。
用**上游** GE2E 做说话人识别时，重建语音的 Top-1 直接掉到 22.9%，
而微调后那个空间里同一批语音的余弦仍然人人 ≈ 0.9 ——
⇒ 这不是"重建保真度"的问题，是**度量空间**的问题：
对外报数时**必须说清楚是哪个 E_spk**，否则同一份权重能讲出两个相反的故事。

⇒ 三条处理决定（已落到代码里）：
1. **评测双口径并报**：`evaluate.py --espk-space {auto,upstream,stage1_finetuned}`，
   默认 `--dual-espk` 同时输出两口径 + `secondary_espk` 段（§5.1）。
   对外报数一律注明打分口径，**不许只挑好看的那个**。
2. **L_de 挪到 upstream 空间考核**：`stage2/train.py --de-space upstream`。
   δ 还是生成于 stage1 空间（否则 G 输入离分布），但"攻击者能不能认出这张卡"
   按攻击者真正用的上游 GE2E 算 —— 塌缩空间里几乎**没有梯度信号**。
3. **DSR 只认 upstream 口径**：τ=0.3/0.4/0.5 在塌缩空间里恒为 0%（全员余弦 > 0.89）。

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

# Stage 1 权重（★ Stage 2 训练的前提）—— A 这次交付的是 Release 里那个名字
scp algorithms/voiceprint_protection/weights/stage1_fresh_200k/stage1_step250000_wide_test_bn.pt \
    root@<EIP>:<仓库>/algorithms/voiceprint_protection/weights/stage1_fresh_200k/

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
| `sim@de`（拆开两个空间时才有这列） | 与 `sim_clone` **不同值** | 两个数一样 ⇒ `--de-space` 没生效 |

### 4.1.1 ★ 开云机之前，本机先跑 300 步预检（2026-11-25 加的固定动作）

云机是**代金券 + 按小时烧钱**，而"训练机制通了"并不等于"防御学到了东西"。
本机纯 CPU 虽然慢，但一百来步足够把趋势看出来 —— 等一个小时比开错云机便宜得多：

```bash
nohup python -u algorithms/voiceprint_protection/stage2/train.py \
    --vocoder hifigan --stage1-checkpoint <ckpt> --de-space upstream \
    --steps 150 --log-every 3 --save-every 30 \
    --max-seconds 2 --batch-size 4 --num-workers 2 \
    --outdir .tmpB/stage2_probe > .tmpB/stage2_probe.log 2>&1 &
```

> **为什么是 150 步 × 2 s × batch 4 而不是 300 步 × 整句 × batch 8**：
> 见下面那条 `--de-space` 的代价说明 —— 本机整句配置的步长是**分钟级**，
> 300 步要跑到第二天；裁短之后同一台机器 1 小时就能给出趋势。
> **趋势判据不受影响**（看的是 `sim@de` 有没有在降），
> 但预检产物**不能**当正式权重用，正式训练仍然用整句 + batch 8。

> ⚠️ **`-u` 不能省**：Python 的 stdout 重定向到文件时是**块缓冲**（8 KB 才刷一次），
> 而训练每 5 步才打一行 —— 不加 `-u` 的话十几分钟看不到任何输出，
> 很容易误判"卡死了"然后 kill 掉一个其实在正常跑的任务（2026-11-25 本机真实踩到）。
> 更稳的旁证是 `<outdir>/train_log.jsonl`：它是每行 `with` 写完就关的，**永远是实时的**。
>
> ⚠️ **`--de-space upstream` 是有代价的，纯 CPU 上尤其明显**。
> 两个 GE2E 的算法根本不是一回事（`stage2/models.py` 里都写着）：
> · `Stage1SpeakerEncoder`（微调那个）= 整段 mel **一次过** LSTM ⇒ 每条 1 个序列；
> · `TorchSpeakerEncoder`（上游那个）= 照抄上游 `embed_utterance`，
>   把语音切成 **160 帧一段、50% 重叠** ⇒ 一条 3.8 s 的语音要变成 ~4 个序列过 LSTM，
>   而且外面还套了一层 Python 逐条循环。
> ⇒ 拆空间等于每步**多 2 趟**这种分段前向（其中一趟还要反传），
> 本机整句 + batch 8 时步长从 ~33 s 涨到**分钟级**。
> 预检的解法是**把每条裁短、batch 减小**（上面的 `--max-seconds 2 --batch-size 4`），
> 而不是砍步数 —— 看的是**趋势**，2 s 片段本身就是合理的声纹注册长度。
> 云机（T4）上这件事基本不存在，正式训练不用加这两个参数。

**判据**（只看 `sim@de` 那一列，也就是上游口径的 Sim_clone）：

| 预检结束时 | 结论 |
|---|---|
| `sim@de` 明显低于起点（比如 0.65 → 0.55 以下） | 梯度方向是对的 ⇒ 开云机跑 200k |
| `de`（L_de）在降但 `sim@de` 不动 | G 在拟合塌缩空间的偏移，**换空间没用** ⇒ 先查 α / λ_de |
| 两个都不动 | 梯度断了 ⇒ 跑 `stage2/verify_gradients.py` |

> 300 步的产物**不是交付物**（`step` 太小），只用来判趋势；
> `--save-every 50` 是为了中途崩了能 `--resume`。

**实测记录（2026-11-25，本机 CPU，150 步 / 2 s / batch 4 / `--de-space upstream`，约 12 s/步、28 分钟）**
每段取 ~16 个日志点的均值（`train_log.jsonl`），比看单行可靠得多：

| 指标 | 前 1/3 | 中 1/3 | 后 1/3 | 读法 |
|---|---|---|---|---|
| `de`（上游口径 L_de） | 0.5926 | 0.5646 | 0.5757 | 降了 0.017，**在噪声里** |
| `sim@de` | 0.6796 | 0.6486 | 0.6525 | 同上，方向对但量级不够 |
| `sim_clone`（塌缩口径） | 0.9118 | 0.8992 | 0.9077 | 几乎不动 |
| `mask` | 0.7006 | 0.6129 | **1.1233** | ★ 翻倍 —— G 在拿**可听度**换防御 |
| `per` | 0.0771 | 0.0691 | **0.1203** | 同步变差 |

⇒ **诚实的结论：150 步定不了这件事**。方向是轻正的，但样本噪声（batch 4）就有 ±0.05，
而 `mask` / `per` 在变差说明 λ_per=15 并没有把扰动按住。
所以加了 `--overfit N`（同一天）：**固定前 N 条、不打乱、`--lambda-gan 0` 反复训**，
把"数据换批带来的抖动"这个变量去掉，直接问"**G 到底能不能把上游口径的 L_de 压下去**"：

```bash
python -u algorithms/voiceprint_protection/stage2/train.py \
    --vocoder hifigan --stage1-checkpoint <ckpt> --de-space upstream \
    --overfit 4 --steps 200 --log-every 10 --save-every 0 --lambda-gan 0 \
    --max-seconds 2 --num-workers 0 --outdir .tmpB/stage2_overfit
```

| 过拟合 200 步后 | 结论 |
|---|---|
| `de` / `sim@de` 明显掉下去（比如 0.60 → 0.45） | 梯度链在这个空间**真的能驱动 G** ⇒ 值得开云机跑 200k |
| 同几条数据反复训 200 步都不降 | **不是数据不够，是路不通** ⇒ 先查 α / λ_de / 冻结是否误伤，别开云机 |

**实测记录 2（2026-11-25，`--overfit 4`，本机 CPU）**

| 一发 | 配置 | 结果 |
|---|---|---|
| 忠实目标 | λ_per=15、λ_gan=0、lr=1e-4，60 步（原计划 200，太平所以在 60 步停） | `de` 0.549–0.596 之间来回，`sim@de` 0.629–0.664 —— **平** |
| 纯 L_de | λ_per=0、λ_gan=0、lr=**1e-3**，120 步 | `de` 0.600 →（55–105 步）**0.52–0.55** → 120 步回到 0.575；`sim@de` 全程 0.60–0.71 —— **只降了约 10%，且不稳定** |
| 梯度实测 | 同一条 batch，只反传 λ_de·L_de | **upstream 空间 ‖∂L/∂θ_G‖ = 37.0**，塌缩的 stage1 空间只有 **7.3** |
| ★ α 放大 | λ_per=0、λ_gan=0、lr=1e-4、**α=0.5**（`--overfit 4`，跑到 100 步停） | `de` 0.435 → **0.555（升）**，`sim@de` 0.504 → **0.556（升）**。起点确实比 α=0.1 时低 0.15，**但那是预算送的，不是学出来的** —— 反复训同一批数据 100 步反而把它练回去了 |

三条放在一起能排除两个常见误判：
· **不是梯度消失**（37 是健康量级，`verify_gradients.py` 也 20/20）；
· **不是 λ_per 在对着干**（摘掉之后照样不降）；
· 剩下的解释按顺序查：**α 的角度预算**（见下）、Stage 1 解码链能不能把小扰动放大、
  以及"论文那个 0.13 到底是拿哪个 E_spk 量的"。

> ★ **α 的角度预算是硬的**：`perturb_embedding()` 做的是 `h_adv = normalize(h + α·δ)`，
> 实测 `‖h‖ = 1`、随机初始化的 `‖δ‖ ≈ 1` ⇒ α=0.10 时 h_adv 只偏离 **5.73°**（cos ≥ 0.995）。
> 也就是说防御全靠"5.7° 的嵌入偏移经 Stage 1 + 声码器之后，被**另一个** GE2E 读出大变化"。
>
> **实测分离结果（2026-11-25 补，上面表里"α 放大"那一行）**：把 α 从 0.1 提到 0.5，
> `sim@de` 的**起点**立刻从 ~0.66 掉到 ~0.50，但接着用同 4 条数据反复训 100 步**不降反升**。
> ⇒ 数字的大小是**预算**决定的，G 在这个链路上**没有额外的学习能力可发挥**。
> 这同时否定了"多跑 200k 步自然就好了"这个指望。

**最后一刀：自由 δ 的 oracle（`stage2/diagnose_delta_oracle.py`，绕开 G）**

把"是 G 学不动"还是"是这个目标本来就不可达"彻底分开 —— δ 直接当成可优化参数，
每条 256 个自由度、逐条独立，没有任何共享函数的容量限制：

```bash
D:/conda/envs/vpp/python.exe -u algorithms/voiceprint_protection/stage2/diagnose_delta_oracle.py \
    --alpha 0.1 --space upstream --steps 100 --lr 0.05 --count 4 --max-seconds 2
```

| 自由 δ 的结果 | 结论 | 对云机的决定 |
|---|---|---|
| `L_de` 明显降下去（0.56 → 0.3 以下） | 路是通的，**只是 G 学不动** | 值得开，但要改的是 G 的目标/结构，不是加步数 |
| `L_de` 也压不动 | **α 预算 + 放大链根本到不了论文量级** | **不要开**，200k 步是白烧代金券 |
| 要靠 `‖δ‖` 涨到远大于 1 才降得动 | 角度预算被偷偷放大，等于换了个方法 | 报告口径必须注明，且论文 α=0.1 不可比 |

**实测（2026-11-25，本机 CPU，`--delta-mode unit`，4 条 × 2 s，Adam lr 0.05）**

| 版本 | 步 1 | 10 | 20 | 30 | 40 | 50 | 角度预算 |
|---|---|---|---|---|---|---|---|
| 自由 δ（**不**约束模长） | 0.5582 | 0.4712 | — | — | — | — | `‖δ‖` 1.78→3.39 ⇒ **5.7°→19.4°（偷偷放大）** |
| 等预算 `‖δ‖=1`（一发） | 0.5531 | 0.4651 | 0.4944 | 0.4585 | 0.4229 | 0.4415 | **5.71°（= G 的工作点）** |
| 等预算 `‖δ‖=1`（另一发，独立随机初始化） | 0.5537 | 0.5307 | 0.5148 | 0.4971 | — | 0.4885 → 60 步 0.4498 → 70 步 **0.4518（平台）** | 同上 |
| 参照：**α=0（完全无防御）** | 0.5833 | — | — | — | — | — | 0° |

> ⇒ **地板 ≈ 0.42–0.45**（两发分别在 40–70 步进入平台，互相差 ~0.03），
> 而 G 训练能到 0.52、无防御是 0.583。
> **可动的总空间只有 0.10 左右，论文要的是 0.45** ⇒
> 这不是"G 学不动"或"步数不够"，而是**α=0.1 + 我们这条放大链的上限**（见 §4.2 闸门）。
>
> ★ 另外一个诚实的细节：`‖δ‖=1` 那两发的 mel  MSE 从 0.0067 涨到 0.0113，
> 也就是**自由 δ 是靠在可听度边缘做牺牲才换来 0.1 的下降** ——
> 这正是 λ_per=15 在训练里禁止做的事。摘掉 λ_per 后 G 只降 10% 且回弹，
> 和这个观察一致：**约束与目标在打架，而约束是论文的前提**。
>
> ⚠️ **复现这几张表之前先看这条**（2026-11-25 事后发现的坑）：
> `stage2/dataset.py::_crop()` 的裁剪起点用的是 **Python 的 `random`**，
> 而这三个 `diagnose_*.py` 原来只 `torch.manual_seed()` ⇒ **每次启动读到的音频片段不同**，
> 同一个 α 的"随机 δ"两发实测差 **0.045**（0.5794 vs 0.5348）。
> 同一次运行内部各 α 共用同一个 batch，**列之间可比**；
> **跨运行的单点数字要按 ±0.05 抖动读**。现已改调 `stage2.common.set_seed()`，
> 用当前代码重跑即可逐位复现。

**α 扫描（正式版：每档 40 步、等预算 `‖δ‖=1`，`stage2/diagnose_alpha_sweep.py`）**

| α | 角度 | 优化后 Sim_clone | 相对随机 δ 学到 | Sim_prot@upstream | mel  MSE |
|---|---|---|---|---|---|
| 0.1 | 5.71° | 0.4978 | 0.037 | 0.6363 | 0.00785 |
| 0.3 | 16.70° | 0.4769 | 0.099 | 0.5656 | 0.01054 |
| 0.5 | 26.57° | 0.4316 | 0.116 | 0.5250 | 0.01439 |
| 1.0 | 45.00° | **0.4346** | 0.029 | **0.4999** | **0.01950** |

> ⇒ **α 有杠杆，但买不到论文量级**：预算放大 10 倍只换来 Sim_clone −0.06，
> Sim_prot 要陪葬 −0.14、mel 失真 ×2.5；且 α=1.0 时随机 δ 就已经到 0.4639，
> 说明大 α 的收益来自"扰动本身"而不是"学会往哪偏"。
> **对外报数就用这张表回答"α 敏感吗"**（指南 §11），完整读法在 `docs/experiments.md` §4g.8。

**因此 B 的替代交付路径（都不花钱）**：
① §5.1 三条基线（钉 `--espk-space upstream`）；
② **α–Sim_clone 曲线** —— `stage2/diagnose_alpha_sweep.py`，
每个 α 都测"随机 δ / 自由 δ 优化后 / Sim_prot / mel  MSE / 波形平均差"五个量，
把"我们为什么到不了 0.13"变成一条有数据支撑的定量说明（指南 §11 的"α 敏感吗"顺带答掉）；
③ 把 §4.1.1 这一整套判据写进实验记录（已完成，`docs/experiments.md` §4g）。



### 4.2 正式训练（200k 步）

> ★★ **2026-11-25 的开台闸门（先读这段再决定要不要花钱）**：
> §4.1.1 的三连预检 + 自由 δ oracle 给出的结论是
> **在 α=0.1 的角度预算内，上游口径的 `L_de` 地板 ≈ 0.42–0.45**（无防御 0.583，G 已训到 0.52），
> 而论文目标是 0.13 ⇒ **200k 步最多只能再买 0.07，不值得烧代金券**。
> 开台前必须满足其中一条：
> ① oracle 在等预算（`‖δ‖=1`）下把 `L_de` 压到 **0.3 以下**（说明是 G 的优化问题，不是链路问题）；
> ② 组里决定**偏离论文的 α**（比如 α=0.5，并同时在报告里给 α–Sim_clone 曲线与可听度）；
> ③ 先修 Stage 1 的放大链（`upstream_speaker_top1 rebuilt` 只有 **0.229** ⇒ 重建本身不够干净）。
> 三条都不满足时，**本节的所有命令都不要执行**。
> 替代交付路径见 §5.1（三条基线）+ §4.1.1 末尾（α–Sim_clone 曲线），两条都不花钱。
>
> ✅ **2026-11-25 决定（成员 B 与发起人确认）：不开台。** 上面三条闸门一条都没被满足
> （oracle 等预算地板 0.42–0.45、没改 α、Stage 1 的 `rebuilt` 仍是 0.229），
> 所以本节的 200k 命令**保持"不执行"状态**，交付改走 §5.1 三条基线 + α 曲线 +
> experiments.md §4g 的负结果分析。留着这段是为了说明**为什么不烧这笔钱**，
> 以及将来若有人把闸门①/②/③真做通了，按原命令开台即可。


```bash
bash scripts/cloud_train_stage2.sh --start --steps 200000 --max-hours 24 \
     --batch-size 8 --accum-steps 1
```

> ★ **`--de-space` 现在由云机脚本默认给 `upstream`**（§1.4.1 的结论），
> 要复现论文原口径做消融就显式传 `--de-space stage1`。
> 这个选择会写进 Stage 2 的 checkpoint（`de_space` 字段），
> `evaluate.py` 的报告里会原样带出来（`report["de_space"]`）。

> ★ **`--max-hours` 一定要给**：云机按小时计费，到点自动存盘退出，
> 哪怕忘了看，损失也有上限。（实例本身还要手动释放，见 §6。）
>
> ★ **要给够**：T4 上没实测过，**先跑 §4.1 的 3 步冒烟把 `it/s` 读出来再定 `--steps`**。
> 按 0.2–0.4 s/step 估 200k 步约 **11–22 小时**；如果实际是 0.8 s/step，
> 200k 就要 **44 小时** ⇒ 单张 `--max-hours 24` 一定跑不完，
> 要么分两天 `--resume`，要么把 `--steps` 降到 100k 并在报告里写明偏离论文。
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
| `sim@de` | ★ 这才是**攻击者那侧**的数 —— 以它为准 | 与 `sim_clone` 背离 ⇒ 说明只在塌缩空间里有效 |
| `per` | 平稳（不要暴涨） | 暴涨 ⇒ λ_per 相对权重太大 |
| `it/s` | T4（pi2.2xlarge.4）上 0.5–1.5 | 明显更慢 ⇒ batch/显存不对 |

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
# ★ A 这次交付的实际文件名（Release 下下来就是这个名，**不叫 stage1_model.pt**）
CKPT1=algorithms/voiceprint_protection/weights/stage1_fresh_200k/stage1_step250000_wide_test_bn.pt
CKPT2=algorithms/voiceprint_protection/weights/stage2/stage2_last.pt
# 有些脚本（infer.py / make_sample.py）在不传路径时会去找 weights/stage1_model.pt，
# 那种情况显式传 --stage1-checkpoint $CKPT1，或按 §1.4 说明改名/建链接。
# ★ 三条基线**必须钉同一个打分口径**，否则三行的数没法横向比（见下面的说明）
ESPK=upstream

# ① 无防御（α=0）
python algorithms/voiceprint_protection/evaluate.py --mode none --protect-mode stage2 \
       --stage1-checkpoint $CKPT1 --split test --limit-per-speaker 5 \
       --espk-space $ESPK

# ② 高斯噪声（论文 Table 1 的 "Gaussian Noise" 那一行）
python algorithms/voiceprint_protection/evaluate.py --mode random --protect-mode stage2 \
       --stage1-checkpoint $CKPT1 --split test --limit-per-speaker 5 \
       --espk-space $ESPK

# ③ ★ 我们自己（训练好的 G）—— 这一行才是 "Ours"
python algorithms/voiceprint_protection/evaluate.py --mode trained --protect-mode stage2 \
       --generator-checkpoint $CKPT2 --stage1-checkpoint $CKPT1 \
       --split test --limit-per-speaker 5 --espk-space $ESPK \
       --output results/stage2_eval/ours.json
```

> ★★★ **为什么要 `--espk-space`（2026-11-25 加的）**：
> 打分口径原本是由保护链**顺带决定**的（`route_encoder_space()`），
> 而 `--mode none/random` 根本不加载 G ⇒ 落回 `upstream`，
> `--mode trained` 会落回 `stage1_finetuned` —— **三条基线不在一个空间里，数字当然对不上**。
> 现在用 `--espk-space` 把三条钉在同一个口径，`--dual-espk`（默认开）
> 再顺手把**另一个**口径也算一遍放进 `secondary_espk`，两边都看得到。
> 报告里 `espk_space` = 打分口径、`delta_space` = δ 生成口径、`de_space` = G 训练时的考核口径，
> **三个字段是不同东西**，写报告别混。
> 交叉验证过：两个方向钉法不同但数字一致（upstream 0.6994/0.6536，
> stage1_finetuned 0.9467/0.9286，DSR 全 0%）。

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

> ★ **这三条都走论文路线**（`--protect-mode stage2`），但**打分口径必须钉死**：
> 三条基线统一 `--espk-space upstream`（= 与论文可比的那个通用 GE2E），
> `--dual-espk` 顺带把塌缩口径也算一份存进 `secondary_espk`。
> **不要跟"塌缩空间里的数"混着比** —— 那个口径下人人 ≈ 0.9，横向比较失去意义（§1.4.1）。
>
> ★ ⚠️ 上一版本这里写过"A 的 Stage 1 重建 Sim_prot 0.999996 ⇒ 论文路线不再被 Stage 1 拖后腿"，
> **这句话已经作废**（2026-11-25，§1.4.1）：0.999996 是在**塌缩空间**里量的。
> 换成可比的 upstream 口径，同一份权重的 Sim_prot 只有 **0.6994**，
> 而且 `upstream_speaker_top1` 的 rebuilt 只有 **0.229**（clean 0.729）
> ⇒ **Stage 1 的重建质量仍然是短板**，"Ours"那一行的 Sim_clone 下不去，
> 有一半原因是攻击者那条克隆链本身就不干净，不全是 G 的锅。

**不对怎么办**：

| 症状 | 原因 | 解法 |
|---|---|---|
| 三行的 Sim_prot 都 ≈ 0.8 出头 | 走的是 clone 路线 | 加 `--protect-mode stage2` |
| Sim_prot 明显 < 0.9 | **先分清是哪一列**：`--espk-space upstream` 下 0.70 是**正常的**（§1.4.1：上游口径本来就重建不掉音色），塌缩口径下才有 0.95。真要排查的是 `stage1_finetuned` 那列低 | 确认 `--stage1-checkpoint` 指对了、且 `--protect-mode stage2`；报告里注明用的是哪个口径 |
| `sim_clone` 和无防御那行差不多 | δ 没生效 | 检查 `--mode trained` + checkpoint 路径；看训练时 `sim@de`（上游口径）有没有降 —— 只降 `sim_clone` 说明只在塌缩空间里有用 |
| DSR 全 0 | Sim_clone 还很高（>0.5） | 同上；论文要 0.13。**但注意反过来的坑**：`DSR(τ)=mean(Sim_clone<τ)` 把重建失真也算成功，本机 upstream 口径**无防御就有 9.0%**（§5.1 实测），所以要看的是**相对无防御那一行的 ΔDSR** |

**★★ 2026-11-25 实测结果（100 条 = test 20 人 × 5，同一批语音、确定性、可配对）**

| 变体（`--espk-space upstream`） | Sim_prot ↑ | Sim_clone ↓ | DSR(0.3/0.4/0.5) |
|---|---|---|---|
| ① 无防御 | 0.6698 | 0.6169 | 0% / 0% / **9.0%** |
| ② 高斯噪声 | 0.6650 | 0.6163 | 0% / 0% / 12.0% |
| ③ 我们的 G（**150 步短训**，非 200k） | **0.7029** | **0.6775** | 0% / 0% / **1.0%** |
| 塌缩口径附列（`stage1_finetuned`） | 0.9243 / 0.9213 / 0.9444 | 0.9048 / 0.9041 / 0.9229 | 全 0% |

> 三条都跑通了（各 **377 s** / CPU），但**③ 的方向是反的**：短训 G 让 Sim_prot 升 0.033、
> Sim_clone 升 0.061 —— **δ 在帮重建链守身份，而不是破坏身份**。
> 这与 §4.1.1 的 oracle 地板（0.42–0.50）自洽：**保真项（λ_per/λ_GAN/λ_mask）在多目标里赢了 `L_de`**。
> 完整判读与"为什么这不代表要开云机"写在 `docs/experiments.md` **§4h**。
> ⚠️ 复用这三行时，**③ 必须标注"150 步短训 G"**，且 DSR 要和①的 9.0% 一起给。

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

> ⚠️ **预算先算一遍再决定步数**：三组消融 + 主实验 = 4 次完整训练。
> 按 200k 步 × T4 估，代金券大概率烧不完 ⇒ 建议**四组统一降到同一个步数**
> （比如 50k），并在报告里写明"消融与主实验同预算、仅 λ 不同"。
> **同预算**比**论文步数**重要 —— 差一组步数的消融是没有意义的对照。
> 另外 `--de-space` 四组要保持一致（默认都是 upstream），
> 否则消融对比的是两个变量而不是一个。

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
       --stage1-checkpoint algorithms/voiceprint_protection/weights/stage1_fresh_200k/stage1_step250000_wide_test_bn.pt \
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

> ⚠️ **用 vpp 那个 python 跑**（`D:\conda\envs\vpp\python.exe`）。
> 用系统 python 的话脚本会打印"[环境不对] 依赖缺失 No module named 'torch'"——
> 它给的是提示不是崩溃，容易看漏。
>
> ★ **2026-11-25 B 全量跑过一遍**（A 这一轮改了 B 的主责文件 `evaluate.py`/`infer.py`，
> 所以公共代码的回归是必须的）：8 个 python 脚本 + `npm test` 6/6 全过，
> `verify_gradients.py` **20/20**（加 `--de-space` 之后重跑仍然 20/20）。
> 另外新加了两个专项：
> `split_speakers.py --check <数据根>/split_speakers.json`（划分对账，✅ 完全一致）、
> `--stub --vocoder stub --speaker-encoder stage1 --de-space upstream` 两步冒烟
> （验证两个编码器**真的各自被用到**：日志里 `sim_clone` 与 `sim@de` 必须是两个数）。

---

## 9. 故障排查速查

| 症状 | 原因 | 解法 |
|---|---|---|
| `找不到 Stage 1 权重` | 名字/位置不对 | 要么放到 `weights/stage1_model.pt`，要么**每条命令显式传 `--stage1-checkpoint $CKPT1`**（A 这次 Release 交付的是 `stage1_fresh_200k/stage1_step250000_wide_test_bn.pt`，**不是** `stage1_model.pt`） |
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
| `--de-space` 加了但 `sim_clone` 与 `sim@de` **一模一样** | 第二个编码器没装上（值相同 = 两个槽指向同一个对象） | 确认传的是 `upstream`/`stage1` 且与 `--speaker-encoder` **不同**；看启动横幅有没有 `L_de 空间 : ... ★ 与 δ 生成空间不同` |
| 后台训练日志十几分钟空白 | stdout 重定向到文件是**块缓冲**，不是卡死 | 用 `python -u` / `PYTHONUNBUFFERED=1`；实时进度看 `<outdir>/train_log.jsonl` |
| 三条基线的数字互相矛盾的"感觉" | 打分口径被保护链顺带决定了（none/random 落 upstream、trained 落 stage1_finetuned） | 三条都显式加 `--espk-space`（§5.1） |

---

## 附录 A · 论文对照表（交报告时用）

> ★ **填表前先写一行口径**（§1.4.1）：论文没有说它的 E_spk 有没有被 Stage 1 端到端微调过
> 之后还拿同一个去算 Sim_clone，但我们这边**两个空间差得很远**
> （塌缩口径 Sim_prot 0.947 / 上游口径 0.699）。
> 本表默认填 **`--espk-space upstream`** 那一列（攻击者真正用的空间），
> 并在表注里给出 `stage1_finetuned` 的对照值 —— 只报一个都属于误导。

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
| ★ **本机 Stage 2 预检**（§4.1.1） | **约 0.5 h**（150 步 × 2 s × batch 4，实测 ~11 s/步） | 本机（不花钱） |
| 开台 + 传数据 + 装环境 | 1.5 h | 云机（**按小时计费，这段也在烧钱**） |
| 冒烟 + 自检 | 0.5 h | 云机 |
| Stage 2 正式训练 | **待定** —— 旧表按 V100 写的 12–24 h；我们租的是 **T4（pi2.2xlarge.4）**，200k 步可能到 **40 h+** ⇒ 先用 §4.1 的 3 步冒烟把 `it/s` 读出来再定 `--steps`（§4.2） | 云机 |
| 评测（三条基线） | 2–4 h 是**旧估计**：2026-11-25 本机实测双口径 `--mode none --split val` **10 条 / 33 s ≈ 3.3 s/条** ⇒ §5.1 的 100 条一轮约 **6 分钟**，三条基线 **半小时内**。旧的 27 s/条 是 WaveRNN 时代的数，现在整条链在 HiFi-GAN 上 | 本机（CPU） |
| 评测（加 WER） | 再 1–2 h | 本机（Whisper 在 CPU 上很慢；`large-v3` 更慢 ⇒ 可先用 `small` 跑通、最后用 `large-v3` 补数） |
| 样例 + 平台 e2e | 1 h | 本机 |
| **合计云机时间** | **旧表写约 15–26 h**；T4 的真实值要等冒烟的 `it/s` ⇒ **开台前先算，别按 24 h 一把梭** | 按 ¥5/h 估 **¥75–130**（跑满 24 h 就是 ¥120，代金券 500 元够，但消融四组要另算） |

> ★ **省钱三原则**（任务板 §8.2）：
> ① 只在真正要训练时开机；② 训完立刻**释放**（不是关机）；③ **不要用竞价实例**（用不上代金券）。
