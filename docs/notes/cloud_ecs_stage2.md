# 华为云 ECS 开台指南（Stage 2 训练用）

> 写给**成员 B**（2026-11-21）。目标：在华为云上开一台带 NVIDIA GPU 的 ECS，
> 把 Stage 2 的 20 万步训练跑完，**且不浪费那 500 元代金券**。
>
> 结论先说：**开按需计费的 ECS，不要开竞价实例** ——
> 华为云官方《弹性云服务器 ECS · 计费说明》写明「**代金券不适用于竞价计费型实例**」。
> 任务板 §8.2 里"优先用竞价实例"那条是**错的**，这次改掉。

---

## 0. 先算账：我们需要多少算力

| 训练 | 步数 | 本机 4060 实测 | V100 估计 | 折合按需时长 |
|---|---|---|---|---|
| Stage 1 | 500k | 13.5 h（成员 A 实测，9.7 it/s） | ≈ 4–6 h | — |
| Stage 2 | 200k | ≈ 30 s/step（**本机无 GPU，只能 CPU**） | **≈ 0.2–0.4 s/step ⇒ 11–22 h** | **合计 ~30 h** |

⇒ 只要规格选对、按时释放，**500 元足够跑完两个阶段**。
但前提是：**按需 + 训练完立刻释放**（见 §6）。

---

## 1. 开一台 ECS（控制台一步步）

入口：华为云控制台 → 产品 → 计算 → **弹性云服务器 ECS** → **购买**。

### 1.1 区域与可用区

- 选**离你近**的（华北-北京四 / 华东-上海一 / 华南-广州）—— 主要是为了 SSH 延迟低。
- ⚠️ **GPU 机型在某些区域没有配额**，先在"区域"下拉框里确认能看到 GPU 规格。
  没有就换一个区域（配额是按区域算的）。

### 1.2 实例规格（★ 最关键的一步）

在"实例规格"里按 **GPU 加速型** 筛选。官方文档里可用的几档：

| 规格名 | vCPU / 内存 | GPU | 显存 | 说明 |
|---|---|---|---|---|
| **`p2s.2xlarge.8`** | 8 / 64 GiB | **V100** | **32 GiB** | ★ **推荐**：主售规格，官方软件清单里明确列了 PyTorch |
| `pi2.2xlarge.4` | 8 / 32 GiB | T4 | 16 GiB | 便宜；官方定位"推理加速型"，但 PyTorch 训练也跑得动 |
| `p2v.2xlarge.8` | 8 / 64 GiB | V100 | 16 GiB | 中庸 |
| `g6.xlarge.4` | 4 / 16 GiB | T4 | 16 GiB | 图形加速型，官方没把它列为 PyTorch 训练首选 |

**我们的显存需求**：Stage 2 冻结了 Stage 1（AutoVC 架构，2842 万参数）+ HiFi-GAN
（1394 万参数）+ GE2E，前向要跑 **3 次**（保护后 / 原始 / 攻击者克隆），
batch 8 / 3 秒片段 ⇒ **fp32 约 8–12 GB**。
⇒ **T4 的 16 GB 够用（开 AMP 还能更省），V100 的 32 GB 稳妥**。
**先用 `pi2.2xlarge.4` 跑通流程（便宜），确认无误再换 `p2s.2xlarge.8` 长训。**

### 1.3 操作系统与镜像

- 选 **Ubuntu 22.04 LTS（推荐，HCE 或普通社区镜像都行）**。
- ★ **不要去镜像市场里挑"深度学习镜像"** —— 那些镜像预装的 CUDA / cuDNN / PyTorch
  版本未必跟驱动匹配，而且**升级一次就要重装一遍**。
  我们自己装 torch（cu121），`scripts/cloud_train_stage2.sh --setup` 里已经写了，
  还带了 NJU 镜像兜底 —— **干净、可复现、版本钉死**。

### 1.4 存储

| 盘 | 建议 | 放什么 |
|---|---|---|
| 系统盘 | 100 GB SSD（通用型 gp2） | 代码、python 环境、第三方仓库 |
| 数据盘 | 100–200 GB SSD | VCTK 16 kHz 副本（1.36 GiB）、权重（~500 MB）、Whisper（3 GB）、checkpoint |

> checkpoint 每存一次约 130 MB（生成器 + 判别器 + 两个优化器状态）。
> `--save-every 1000` 跑 200k 步 = 200 次 ≈ 26 GB。**要么调大 `--save-every`，
> 要么定期把旧 checkpoint scp 回来删掉。**

### 1.5 网络

- VPC + 子网：用默认的即可。
- **安全组**：入方向放行 **22（SSH）**。要跑 Jupyter 就再放 8888，要看 TensorBoard 就放 6006。
- **弹性公网 IP（EIP）**：勾一个，否则从外面连不上。
  （训练期间 EIP 只占很小的一笔，但**别忘了在释放时一起删掉**，见 §6。）

### 1.6 登录方式

- 选 **密钥对**（比密码安全，而且免去"重置密码要短信"）。
  没有就新建一个：控制台 → "密钥对" → "创建密钥对" → **下载私钥**（只下这一次！）。

### 1.7 计费方式（★ 钱在这一步）

- 选 **按需计费**。
- **不要选竞价实例** —— 官方原文：「**代金券不适用于竞价计费型实例。**」
  竞价实例还"不支持变更实例规格"、"系统盘随实例释放但**数据盘不会自动删除**"。

---

## 2. 连上去

```bash
# 本机（Windows Git Bash / macOS / Linux 都一样）
chmod 600 vpp_key.pem
ssh -i vpp_key.pem root@<EIP地址>
```

连不上依次检查：
1. 安全组有没有放行 22；
2. EIP 有没有绑定到这台实例；
3. 本机防火墙。

> 备选：**不绑 EIP**，用控制台的"远程登录"（VNC）应急。

---

## 3. 把代码、数据、权重弄上去

### 3.1 代码 —— git clone（最省事）

```bash
# 云机上
git clone git@github.com:Liastar-li/algorithm_research_frontend.git
cd algorithm_research_frontend
git checkout group_N005_submission
```

私有仓库要在云机上加 SSH key：`ssh-keygen` 生成 → 把公钥贴到 GitHub 的 SSH keys。

### 3.2 数据 —— ★ **从本机传，不要在云机上重下**

VCTK 源站在国内实测只有 30–400 KB/s（成员 B 实测），云机也不一定快。
我们只要**每人 100 句 = 1.36 GiB**，直接打包传：

```bash
# 本机
tar czf vpp_data.tgz -C D:/vpp/data audio16k train.txt val.txt test.txt spk2id.txt split_speakers.json
scp vpp_data.tgz root@<EIP>:/root/

# 云机上
mkdir -p /root/vpp/data && tar xzf /root/vpp_data.tgz -C /root/vpp/data
export VPP_DATA_ROOT=/root/vpp/data
```

> 数据是**数据契约**：110 说话人 × 100 句，划分用
> `scripts/split_speakers.py --check` 对账（名单见 `docs/notes/vctk_speakers.txt`）。
> 传错的话，两人训出来的模型对不上，指标没法比。

### 3.3 权重

```bash
# 云机上跑 --setup 会自动下（HF 镜像很快）
bash scripts/cloud_train_stage2.sh --setup
```

### 3.4 ★★ Stage 1 的 checkpoint（**Stage 2 训练的前提**）

Stage 2 的解码器 `D(c, h_adv)` 就是 Stage 1 训出来的那套（AutoVC 架构、键名与官方一致），
**没有它 Stage 2 训不了**（`--stub` 只能验证机制，指标全是噪声）。

```bash
# 本机
scp algorithms/voiceprint_protection/weights/stage1_model.pt root@<EIP>:<仓库>/algorithms/voiceprint_protection/weights/
```

> ⚠️ **必须是正式训练的权重**。冒烟测试（`--iterations 3 --smoke`）也会写
> `stage1_model.pt`，会被静默捡去用、**且不报错** ——
> 2026-11-20 加了防护：训不够 1000 步的 checkpoint 会被拒绝（见 `config.STAGE1["min_steps_for_real_use"]`）。

---

## 4. 装环境

```bash
cd algorithm_research_frontend
bash scripts/cloud_train_stage2.sh --setup
```

它做四件事：建 venv → 装 torch(cu121) 与其余依赖 → 拉 third_party（固定 commit）→
下上游权重 → 准备数据（数据已在就跳过）。

自检：

```bash
source ~/vpp-venv/bin/activate
export VPP_DATA_ROOT=/root/vpp/data
python algorithms/voiceprint_protection/scripts/check_env.py
python algorithms/voiceprint_protection/stage2/verify_gradients.py   # 20 项梯度通路
```

---

## 5. 开训

```bash
# 先冒烟：3 步，确认全链路通（几分钟）
bash scripts/cloud_train_stage2.sh --start --steps 3 --log-every 1

# 正式训练（★ --max-hours 是保命的）
bash scripts/cloud_train_stage2.sh --start --steps 200000 --max-hours 8 --batch-size 8
```

脚本会把训练放进 **tmux**（SSH 断了不影响），日志在 `logs/stage2.log`。

盯进度：

```bash
bash scripts/cloud_train_stage2.sh --status
tmux attach -t vpp-stage2      # 进去看实时输出（Ctrl-b d 退出来）
```

**断了/被抢占后接着跑**（优化器状态、随机数状态都存在 checkpoint 里）：

```bash
bash scripts/cloud_train_stage2.sh --start --steps 200000 --max-hours 8 --resume
```

### ★ 看哪几个数

| 数 | 期望 | 不对说明什么 |
|---|---|---|
| `de`（L_de） | **下降** | 上升 = 符号反了（`losses.py::defense_loss`） |
| `per` | 缓慢上升后平稳 | 暴涨 = λ_per 相对权重太大 |
| `sim_clone` | **下降**（越小防御越有效） | 论文训完是 0.13 |
| `it/s` | 0.2–0.5（V100） | 明显更慢 ⇒ batch/显存不对 |

训练脚本每轮结束会自己打一行 **`[L_de 方向自检]`** ——
它比较前 10% 与后 10% 的 L_de，方向不对会直接喊出来（2026-11-21 新加）。

---

## 6. ★ 收结果 & 省钱（**最容易踩的坑**）

```bash
# 1. 把结果拉回本机
scp root@<EIP>:<仓库>/algorithms/voiceprint_protection/weights/stage2/stage2_last.pt .
scp root@<EIP>:<仓库>/logs/stage2.log .
```

2. **去控制台释放实例**（不是只关机！）：

| 操作 | 还收不收钱 |
|---|---|
| **关机（停止）** | ⚠️ **vCPU/内存/镜像/GPU 不计费了，但云硬盘、EIP、带宽照常计费** |
| **释放（退订）** | 实例本身的费用结束；**但数据盘、EIP、带宽若没删，仍按各自规则计费** |

> 官方原文（按需普通实例关机）：「ECS 实例本身（vCPU，内存，镜像、GPU）不计费，
> 其他所挂载的资源如云硬盘，公网 IP 或带宽则正常计费。」

⇒ **最省的做法**：结果拉回来之后，**释放实例 + 删掉数据盘 + 释放 EIP**，
把控制台里"云硬盘 / EIP / 带宽"也一并清干净。

3. **别忘了这件事**：只要实例还开着，就一直在烧钱。
   我们用 `--max-hours` 给训练本身设了上限，**但实例本身要手动释放**。

---

## 7. 常见坑

| # | 坑 | 怎么防 |
|---|---|---|
| 1 | **代金券买了竞价实例**（用不上券，钱从余额扣） | 结算页确认计费方式是"按需" |
| 2 | 训练跑完了，实例还开着 | 收完结果立刻**释放**（不是关机） |
| 3 | 数据盘没删，按天计费 | 释放时勾"同时删除数据盘" |
| 4 | SSH 断了，训练也停了 | 一定用 tmux（脚本已经包好） |
| 5 | 跑到一半被抢占/超时 | `--resume` 接着跑，checkpoint 里存了优化器状态 |
| 6 | 显存 OOM | 开 AMP（默认开）、减 `--batch-size`、或用 `--accum-steps` 保持等效 batch |
| 7 | 传了冒烟 checkpoint 当正式权重 | `stage1_model.pt` 只训了 3 步 → 代码会拒绝（≥1000 步才认） |
| 8 | 云机数据与本机不是同一批 | `scripts/split_speakers.py --check` 对账 |
| 9 | ★ **`*.sh` 变成 CRLF，Linux 上跑不了** | Windows 的 git 全局是 `core.autocrlf=true` ⇒ Windows 上 clone 的 `.sh` 是 CRLF ⇒ 传到云机上报 `bash\r: No such file or directory`（看不出是换行符的问题）。**仓库已加 `.gitattributes` 把 `*.sh` 钉成 LF**；实在不放心就在云机上 `sed -i 's/\r$//' scripts/*.sh` |

---

## 8. 参考

- 华为云《弹性云服务器 ECS · 计费说明》：<https://support.huaweicloud.com/intl/zh-cn/productdesc-ecs/ecs_01_0065.html>
- 华为云《GPU 加速型 ECS 规格》：<https://support.huaweicloud.com/zh-cn/productdesc-ecs/ecs_01_0045.html>
- 华为云《ECS 规格选型》：<https://support.huaweicloud.com/bestpractice-ecs/ecs_bp_6001.html>
- 本仓库：`scripts/cloud_train_stage1.sh`（Stage 1 版）、`scripts/cloud_train_stage2.sh`（Stage 2 版）
