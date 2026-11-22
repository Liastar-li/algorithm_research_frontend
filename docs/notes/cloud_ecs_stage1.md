# 华为云 ECS 开台指南（**Stage 1 训练**）

> 写给**成员 A**（2026-11-22）。目标：在华为云上开一台带 NVIDIA GPU 的 ECS，
> 把 Stage 1 的 500,000 步训练跑完，**且不浪费那 500 元代金券**。
>
> **结论先说：**
> 1. **开按需计费的 ECS，不要开竞价实例** —— 华为云官方《弹性云服务器 ECS · 计费说明》
>    写明「**代金券不适用于竞价计费型实例**」。
>    （任务板 §8.2 里"优先用竞价实例"那条是**错的**，已改。）
> 2. **Stage 1 的显存需求极小（实测只用 1.66 GB / 8 GB）** ⇒
>    **T4（16 GB）就够**，比 V100 便宜一半。
> 3. **数据从本机传，不要在云机上重下**（源站在国内实测只有 30–420 KB/s）。

---

## 0. 先算账：值不值

| | 本机（RTX 4060 Laptop 8 GB） | 云机 T4 16 GB | 云机 V100 32 GB |
|---|---|---|---|
| Stage 1 单步耗时 | **0.103 s**（实测 9.7 it/s） | 估 ~0.2 s | 估 ~0.1 s |
| 500,000 步 | **13.5 h** | 估 ~10–14 h | 估 ~4–6 h |
| 按需单价（估） | 免费 | 约 3–5 元/小时 | 约 8–13 元/小时 |
| **训练花费** | **0 元** | 约 30–70 元 | 约 35–80 元 |
| 首次配置耗时 | — | **1.5–2 h**（装环境 + 传数据 + 下权重） | 同左 |
| **总耗时** | **13.5 h** | **12–16 h** | **6–8 h** |

⭐ **所以上云机真正买到的是"训练从 13.5 h 压到 6–8 h"**（省 5–7 小时），
代价是几十块 + 1.5–2 小时的配置时间 + 若干踩坑风险。

> 💡 **兜底方案（推荐先看一眼）**：如果开台、配环境、下数据这几步卡住了
> （GPU 配额没有、pip 源慢、云机连不上 GitHub……），
> **直接把那台按需实例释放掉，回本机跑 13.5 小时** ——
> 代码、断点、脚本全都现成，`--resume` 一行就能接上。
> 详见 §9「出问题就撤」。

---

## 1. 开一台 ECS（控制台一步步点）

**入口**：<https://console.huaweicloud.com/> → 右上角「**产品**」→ 左栏
「**计算**」→「**弹性云服务器 ECS**」→「**购买**」

### 1.1 区域与可用区

- 「**区域**」下拉里选离你近的：**华北-北京四** / **华东-上海一** / **华南-广州**（都行，主要为了 SSH 延迟低）
- 「**可用区**」选一个（默认第一个就行）

> ⚠️ ★ **GPU 机型是按区域分配配额的**：如果下拉框里**看不到 GPU 规格**，
> 说明这个区域**没有 GPU 配额** ⇒ **换一个区域**（配额是按区域算的，不是按账号）。
> 这是最常见的"开不了 GPU"的原因。

### 1.2 实例规格（★ 最关键的一步）

「**规格**」下拉 → 筛选「**GPU 加速型**」→ 选一个：

| 规格名 | vCPU / 内存 | GPU | 显存 | 单价（估） | 建议 |
|---|---|---|---|---|---|
| **`pi2.2xlarge.4`** | 8 / 32 GiB | **T4** | **16 GB** | 约 3–5 元/h | ⭐ **推荐**：我们显存只用 1.66 GB，16 GB 绰绰有余，最省钱 |
| **`p2s.2xlarge.8`** | 8 / 64 GiB | **V100** | **32 GB** | 约 8–13 元/h | 更快，预算够就选它 |
| `p2v.2xlarge.8` | 8 / 64 GiB | V100 | 16 GB | 中庸 | 一般不用 |
| `g6.xlarge.4` | 4 / 16 GiB | T4 | 16 GB | 便宜 | 图形加速型，官方没列为 PyTorch 训练首选 |

> **我们的显存需求（实测）**：Stage 1 = AutoVC 架构（2842 万参数）+ GE2E + AMP，
> batch 16 / 3 秒片段 ⇒ **只用 1.66 GB**。**T4 的 16 GB 绰绰有余。**

### 1.3 操作系统与镜像

- 「**镜像**」选 **公共镜像 → Ubuntu → Ubuntu 22.04 LTS server 64bit**
  （HCE 版或普通社区镜像都行）
- ⚠️ ★ **不要去镜像市场挑"深度学习镜像"** —— 那些镜像预装的 CUDA / cuDNN / PyTorch
  版本未必跟驱动匹配，**升级一次就要重装一遍**。
  我们自己在 `cloud_train_stage1.sh --setup` 里装 torch（cu121），带 NJU 镜像兜底 ——
  **干净、可复现、版本钉死**。

### 1.4 存储

| 盘 | 建议 | 放什么 |
|---|---|---|
| 系统盘 | **100 GB SSD**（通用型 gp2） | 代码、python 环境、third_party 仓库 |
| 数据盘 | **100 GB SSD** | VCTK 16 kHz 副本（1.36 GiB）、权重、checkpoint |

> checkpoint 体积：`stage1_last_ckpt.pt` **358 MB**（含优化器状态，每 5000 步覆盖一次，
> 不累积）；`stage1_best.pt` 115 MB（只有权重）。**磁盘压力不大**，
> 不像 Stage 2 那样要存 200 份。

### 1.5 网络

- **VPC + 子网**：用默认的即可
- **安全组**：入方向放行 **22（SSH）**。
  要看 TensorBoard 再放 6006，要跑 Jupyter 再放 8888。
- **弹性公网 IP（EIP）**：**必须勾一个**，否则从外面连不上。
  ⚠️ 训练期间 EIP 只占很小一笔，但**释放实例时记得一起删掉**（见 §7）

### 1.6 登录方式

- 选「**密钥对**」（比密码安全，也免去"重置密码要短信"）
- 没有就新建：控制台 → 左侧「**密钥对**」→「**创建密钥对**」→ **下载私钥**（`vpp_key.pem`）
  ⚠️ **私钥只下这一次**，丢了就删掉重建

### 1.7 计费方式（★ 钱在这一步）

- 选 「**按需计费**」
- ❌ **不要选竞价实例** —— 官方原文：「**代金券不适用于竞价计费型实例。**」
  竞价实例还"不支持变更实例规格"、"系统盘随实例释放但**数据盘不会自动删除**"。

### 1.8 提交订单

点「**立即购买**」→ 弹出「订单」页 → 确认配置 → 「**提交订单**」
→ 支付方式选「**代金券**」（若券只能抵扣部分，差额走余额）

> 💡 **提交后等 1–3 分钟**实例就 Running 了。控制台右上角会有通知。

---

## 2. 连上去

```bash
# 本机（Windows Git Bash / macOS / Linux 都一样）
chmod 600 vpp_key.pem
ssh -i vpp_key.pem root@<EIP地址>
```

连不上依次检查：
1. 安全组有没有放行 **22**
2. EIP 有没有**绑定**到这台实例（控制台 → 弹性公网 IP → 绑定）
3. 本机防火墙

> 备选：不绑 EIP，用控制台的「**远程登录**」（VNC）应急。

---

## 3. 把代码、数据、权重弄上去

### 3.1 代码 —— git clone（最省事）

```bash
# 云机上
git clone git@github.com:Liastar-li/algorithm_research_frontend.git
cd algorithm_research_frontend
git checkout group_N005_submission
```

> ⚠️ **私有仓库**要在云机上加 SSH key：
> ```bash
> ssh-keygen -t ed25519 -C "cloud-vpp"
> cat ~/.ssh/id_ed25519.pub        # 复制输出，贴到 GitHub → Settings → SSH keys
> ```

### 3.2 数据 —— ★ **从本机传，不要在云机上重下**

VCTK 源站（Edinburgh）在国内实测只有 **30–420 KB/s**（成员 B 实测），
云机也不一定快。而我们只要 **11,000 条 = 1.36 GiB**（每人 100 句）。

**本机（Windows Git Bash）打包：**
```bash
cd /d/0github/algorithm_research_frontend/data
tar czf /d/vpp_data.tgz audio16k train.txt val.txt test.txt spk2id.txt split_speakers.json
ls -lh /d/vpp_data.tgz
```

**传到云机：**
```bash
scp /d/vpp_data.tgz root@<EIP>:/root/
```

**云机上解开：**
```bash
mkdir -p /root/vpp/data
tar xzf /root/vpp_data.tgz -C /root/vpp/data
export VPP_DATA_ROOT=/root/vpp/data
ls /root/vpp/data/audio16k | wc -l      # 应该是 110（说话人目录）
wc -l /root/vpp/data/train.txt           # 应该是 8000
```

> ⚠️ **数据是"数据契约"**：110 说话人 × 100 句。
> 传错的话，两人训出来的模型对不上、指标没法比。
> 传完**必须对账**（见 §4 最后一步）。

### 3.3 权重 —— 让脚本下（很快）

```bash
bash scripts/cloud_train_stage1.sh --setup     # 里的 fetch_weights 会下
```

会下三个：`encoder.pt`（17 MB，**GE2E，Stage 1 必需**）、
`hifigan_16k.ckpt`（55 MB，**验收 M3 要用**）、
`synthesizer.pt` + `vocoder.pt`（约 400 MB，**平台服务要用，训练不需要**，但一起下了省事）。

### 3.4 ⭐ 要不要把本机的断点传上去？

本机已经训到 **step 20,000**（`stage1_last_ckpt.pt`，358 MB，含优化器状态）。

| 做法 | 好处 | 代价 |
|---|---|---|
| **A. 传断点续训** | 与本机那条曲线**连续**；省 35 分钟 | 上传 358 MB |
| **B. 云机上从零开始** | 少一次上传 | 曲线另起一条；多跑 35 分钟 |

```bash
# A 的话：
scp /d/0github/algorithm_research_frontend/algorithms/voiceprint_protection/weights/stage1_last_ckpt.pt \
    root@<EIP>:/root/algorithm_research_frontend/algorithms/voiceprint_protection/weights/
```

> 💡 **建议选 A**：358 MB 上传在云服务器上通常几分钟，而它保证**云机上的曲线和本机的是同一条**
> —— 后面报告里画曲线、对比时不会有人说"你俩的图对不上"。
> 脚本会自动检测到断点并 `--resume`（不用手动加参数）。

---

## 4. 装环境（一条命令）

```bash
cd /root/algorithm_research_frontend
bash scripts/cloud_train_stage1.sh --setup
```

它依次做四件事：
1. 建 venv（`~/vpp-venv`）→ 装 torch(cu121) 与其余依赖（**官方索引失败会自动退到 NJU 镜像**）
2. 拉 third_party 官方仓库（**按 commit 钉死**，和本机一致）
3. 下上游权重
4. 准备 VCTK 数据 —— ⚠️ **我们已经从本机传了，所以要加 `--skip-data`**：
   ```bash
   bash scripts/cloud_train_stage1.sh --setup --skip-data
   ```
   （不加的话它会去源站重新下，慢且可能因限流失败）

**自检（必做）：**
```bash
source ~/vpp-venv/bin/activate
export VPP_DATA_ROOT=/root/vpp/data
export HF_ENDPOINT=https://hf-mirror.com

python algorithms/voiceprint_protection/scripts/check_env.py          # 环境
python algorithms/voiceprint_protection/scripts/test_stage1.py       # 19 项形状/损失自检
python algorithms/voiceprint_protection/scripts/split_speakers.py --check \
       /root/vpp/data/split_speakers.json                              # ★ 划分对账
```

⭐ **第三条必须打印「完全一致」** —— 不一致说明数据传错了，
**别开跑**，先回去查。

---

## 5. 开训

```bash
# ① 先冒烟：3 步，确认全链路通（1 分钟）
bash scripts/cloud_train_stage1.sh --start --iterations 3 --stage model --skip-data
tail -20 logs/stage1_model.log

# ② 正式训练（★ --max-hours 是保命的）
bash scripts/cloud_train_stage1.sh --start \
       --stage model --iterations 500000 --max-hours 10 --skip-data
```

> ⚠️ **为什么 `--max-hours 10` 一定要加**：
> 云机按**小时**计费，忘了关就一直烧钱（只有 500 元代金券）。
> `--max-hours` 到点会**自动存盘退出** ⇒ 哪怕忘了看，损失也有上限。
> 断了之后重跑同一条命令会**自动 `--resume`** 接着跑。

---

## 6. 盯进度

```bash
# 方式一：脚本
bash scripts/cloud_train_stage1.sh --status

# 方式二：进 tmux 看实时输出（Ctrl-b 然后按 d 退出来，别按 Ctrl+C！）
tmux attach -t vpp-train

# 方式三：看日志
tail -f logs/stage1_model.log
```

**该看哪个数**（`run_stage1.py` 里也有同样的说明）：

| 数 | 期望 | 含义 |
|---|---|---|
| `rec` | 稳定下降 | 重建在变好 |
| `cd` | 下降 | 内容没丢 |
| **`dis`** | **往 −4.38 靠** | L_dis = −H(p)；越负 = 分类器越猜不出说话人 = **解耦成功** |
| `[验证]` 行 | `dis` 与训练集接近 | 泛化正常（验证集说话人**没见过**） |

> ⭐ **Step 1000 就能判断方向对不对**：如果 `dis` 在往 **正数**走（变接近 0），
> 说明 `L_dis` 的符号反了 ⇒ **立刻停**，别烧钱。
> （我们在本机踩过这个坑，见 `docs/experiments.md §4c`。）

---

## 7. 训完：把权重拉回来 + ★★ 释放实例

### 7.1 拉权重回本机

```bash
# 云机上
ls -lh algorithms/voiceprint_protection/weights/
# 本机
scp root@<EIP>:/root/algorithm_research_frontend/algorithms/voiceprint_protection/weights/stage1_model.pt \
    /d/0github/algorithm_research_frontend/algorithms/voiceprint_protection/weights/
scp root@<EIP>:/root/algorithm_research_frontend/algorithms/voiceprint_protection/weights/stage1_best.pt \
    /d/0github/algorithm_research_frontend/algorithms/voiceprint_protection/weights/
scp root@<EIP>:/root/algorithm_research_frontend/algorithms/voiceprint_protection/weights/stage1_history.json \
    /d/0github/algorithm_research_frontend/algorithms/voiceprint_protection/weights/
```

### 7.2 ★★ 释放实例（**这一步不做，前面全白花**）

> ⚠️⚠️ **「关机」不等于「释放」**：
> - **关机** ⇒ 仍收**存储费**（系统盘 + 数据盘），不收计算费
> - **释放** ⇒ 才彻底停收，**实例和它的 EIP 都会被删掉**（数据也没了！）

**操作路径**：
1. 控制台 → 「**弹性云服务器 ECS**」→ 找到这台实例 → 「**释放**」（不是「停止」！）
2. 弹窗里会问「是否同时删除 EIP / 数据盘」⇒ **勾上删除**（数据我们已经有本机备份了）
3. 「**弹性公网 IP**」页面 → 确认没有残留的 EIP

**还有两张单据**：
- 「**云硬盘 EVS**」/「**OBS 桶**」⇒ 如果创建时单独买过，确认已随实例删除
- 「**流量**」⇒ 云机下载 1.4 GB 数据会走公网流量（一般很少，但看一眼）

---

## 8. 预算（500 元怎么花）

| 阶段 | 是否上云 | 估算 |
|---|---|---|
| **Stage 1（本次）** | ✅ 上云 | **30–80 元**（T4 或 V100，一次 5–14 h） |
| Stage 2 正式训练 | 视情况 | 本机 RTX 4060 也能跑；上云则 11–22 h |
| **Stage 2 消融**（w/o L_de、w/o L_mask） | ★ **上云划算** | 并行开 2–3 台 ⇒ 20–30 h |
| 合计 | | **300–450 元**（在 500 元内，但**别再开别的实例**） |

> 💡 **省钱三原则**（任务板 §8）：
> ① 只在**真正要训练时**开机（开发/调试/评测全在本机）
> ② 训完**立刻释放**（不是关机）
> ③ 优先**竞价实例** —— ❌ **这条作废**：代金券不适用于竞价实例，只能开按需。

---

## 9. 出问题就撤（别硬扛）

云机最常见的三个坑：

| 现象 | 原因 | 处理 |
|---|---|---|
| 规格里**看不到 GPU** | 该区域**没有 GPU 配额** | 换区域 |
| `pip install torch` 卡住 | 官方索引不通 | 脚本已自动退到 NJU 镜像；还不行就手动跑那条命令 |
| 云机上**下不动 VCTK** | 源站限流 | **别下** —— 从本机传（§3.2），或 `--skip-data` |

⭐ **兜底**：上面任何一条卡超过 **30 分钟**，就：

```bash
# 1) 把云机上的断点拉回来（358 MB，别丢）
scp root@<EIP>:/root/algorithm_research_frontend/algorithms/voiceprint_protection/weights/stage1_last_ckpt.pt .
# 2) 去控制台**释放**实例
# 3) 回本机续跑
python algorithms/voiceprint_protection/stage1/train.py --stage model \
       --iterations 500000 --resume --save-every 5000 --val-every 5000 --max-hours 13
```

本机 9.7 it/s，500k 步 13.5 小时 —— **比云机慢，但一定跑得完、且一分钱不花**。

---

## 10. 附录：命令速查

```bash
# ---- 本机（Windows Git Bash）----
tar czf /d/vpp_data.tgz -C /d/0github/algorithm_research_frontend/data \
    audio16k train.txt val.txt test.txt spk2id.txt split_speakers.json
scp /d/vpp_data.tgz root@<EIP>:/root/
scp /d/0github/algorithm_research_frontend/algorithms/voiceprint_protection/weights/stage1_last_ckpt.pt \
    root@<EIP>:/root/algorithm_research_frontend/algorithms/voiceprint_protection/weights/

# ---- 云机 ----
ssh -i vpp_key.pem root@<EIP>
cd /root/algorithm_research_frontend
export VPP_DATA_ROOT=/root/vpp/data
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p /root/vpp/data && tar xzf /root/vpp_data.tgz -C /root/vpp/data

bash scripts/cloud_train_stage1.sh --setup --skip-data
source ~/vpp-venv/bin/activate
python algorithms/voiceprint_protection/scripts/check_env.py
python algorithms/voiceprint_protection/scripts/test_stage1.py
python algorithms/voiceprint_protection/scripts/split_speakers.py --check $VPP_DATA_ROOT/split_speakers.json

bash scripts/cloud_train_stage1.sh --start --stage model --iterations 500000 --max-hours 10 --skip-data
bash scripts/cloud_train_stage1.sh --status
tmux attach -t vpp-train          # Ctrl-b 然后 d 退出

# 训完：拉权重 → 释放实例
scp root@<EIP>:/root/algorithm_research_frontend/algorithms/voiceprint_protection/weights/stage1_{model,best}.pt .
scp root@<EIP>:/root/algorithm_research_frontend/algorithms/voiceprint_protection/weights/stage1_history.json .
# 控制台 → ECS → 「释放」→ 勾上删 EIP → 弹性公网 IP 页面确认无残留
```

---

## 11. 引用与许可

- 云机脚本：`scripts/cloud_train_stage1.sh`（成员 B 起草，成员 A 增量补了 `--max-hours` /
  `--resume` / txt 分开下载 / 划分对账）
- Stage 2 版本：`docs/notes/cloud_ecs_stage2.md` + `scripts/cloud_train_stage2.sh`
- 第三方组件与许可：`third_party/README.md`
- 华为云计费说明（代金券不适用于竞价实例）：见 §1.7
