# 第 N005 组：基于人机感知差异解耦的隐式声纹保护

> **本文件是组内 W6 的工作稿。** 凡是标 `【待填】` 的地方都还没定，
> 填之前**不要**提交 PR —— 任务板 §7 第 3 条踩坑：
> *"文档到最后一个人赶 → README 里全是想象。实验记录必须和代码同 commit 提交。"*
>
> 已经写实的部分（实现内容、接口、环境、命令、来源与许可）都对着代码核过了。

---

## 1. 小组与项目

| 信息 | 本组填写 |
|---|---|
| 组号 | **第 N005 组**【待填：最终组号】 |
| 选题名称 | 基于人机感知差异解耦的隐式声纹保护 |
| 成员 | 【待填：姓名、GitHub 用户名与分工】 |
| 方法 ID | `group_n005_voiceprint_protection`（**必须全小写**，平台契约要求 `/^[a-z][a-z0-9_]{1,63}$/`） |
| 本组仓库与提交分支 | 【待填：Fork 链接】 · 分支 `group_N005_submission` |

**这个方法解决什么问题**

在说话人嵌入（256 维）上加一个人耳听不出来、但语音克隆系统能听出来的扰动，
让针对这段录音的 AI 声音克隆显著失效，同时人耳听感与说话人可识别性基本不变。

原理：先训练解耦-重建网络把语音拆成内容码与说话人嵌入，
再在嵌入上加扰动（`h_adv = h + α·δ`，α = 0.10），最后重新合成音频。
论文的核心洞见是**人机感知差异**：人耳的掩蔽阈值远高于机器特征空间的判别边界，
所以存在一个"人听不出、机器听得出"的扰动窗口。

---

## 2. 论文与代码来源

- **论文**：Xue, C., Sun, M., Zhang, Q., Zhang, X., Li, K., Liao, Y., Ge, X., & Yao, K. (2026).
  *Imperceptible Voiceprint Protection via Human-Machine Perception Discrepancy Feature
  Disentanglement.* Interspeech 2026, pp. 5259–5263. DOI: [10.21437/Interspeech.2026-105](https://doi.org/10.21437/Interspeech.2026-105)
  （本地副本：[`docs/xue26_interspeech.pdf`](docs/xue26_interspeech.pdf)）
- **作者代码**：**没有**。论文未提供官方代码链接，本方法**完全基于公开组件自建**。
- **使用的开源组件**（版本与许可见 [`third_party/README.md`](third_party/README.md)）：

  | 组件 | 仓库 | 固定 commit | 许可 | 用途 |
  |---|---|---|---|---|
  | Real-Time-Voice-Cloning | [CorentinJ/Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) | `890f3a0` | MIT | **GE2E 说话人编码器**（论文脚注 2 直接引用） |
  | AutoVC | [auspicious3000/autovc](https://github.com/auspicious3000/autovc) | `c1827a6` | MIT | Stage 1 的模型架构（Ec / D / PostNet） |
  | wavenet_vocoder | [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder) | `d4333546` | MIT | 与 AutoVC 的 mel 格式配套的声码器 |

  上游权重全部来自 HuggingFace `CorentinJ/SV2TTS`（`encoder.pt` / `synthesizer.pt` / `vocoder.pt`）。
- **数据**：VCTK（Edinburgh DataShare，[DOI 10.7488/ds/2645](https://doi.org/10.7488/ds/2645)），**CC BY 4.0**。
  110 说话人 / 11,000 句，seed=42 划分 80 训练 / 10 验证 / 20 测试。
- **本组算法代码位置**：[`algorithms/voiceprint_protection/`](algorithms/voiceprint_protection/)（本分支）
- **第三方许可证与引用说明**：见 [`third_party/README.md`](third_party/README.md)；
  样例音频的署名见 [`public/samples/group_n005_voiceprint_protection/SOURCES.md`](public/samples/group_n005_voiceprint_protection/SOURCES.md)。

---

## 3. 本组实现内容

### 3.1 功能完成状态

| 功能 | 完成状态 | 实现位置与验证方式 |
|---|---|---|
| GE2E 说话人编码器封装（波形 → 256 维嵌入，L2 归一化） | ✅ 已完成 | [`bridge_speaker.py`](algorithms/voiceprint_protection/bridge_speaker.py)；实测 `‖h‖ = 1.000000` |
| VCTK 数据管道（转 16 kHz + 80/10/20 划分 + 三条硬要求自检） | ✅ 已完成 | [`scripts/prepare_data.py`](algorithms/voiceprint_protection/scripts/prepare_data.py) |
| L1 推理链（体检 → 打疫苗 → 骗子行骗） | ✅ 已完成 | [`infer.py`](algorithms/voiceprint_protection/infer.py)；**已在本机用真权重跑通**（`encoder.pt` / `synthesizer.pt` / `vocoder.pt` 全部就位） |
| 平台算法服务（FastAPI + CORS + 统一错误格式） | ✅ 已完成 | [`service.py`](algorithms/voiceprint_protection/service.py)；契约自测 **41/41**，另有可重复执行的端到端自检 `scripts/test_platform_e2e.py` |
| 平台方法配置 + 索引登记 | ✅ 已完成 | [`public/methods/group_n005_voiceprint_protection.json`](public/methods/group_n005_voiceprint_protection.json) |
| Stage 1（解耦-重建网络训练） | ⬜ 未完成 | 成员 A 主责；架构参考 AutoVC |
| Stage 2（防御性扰动生成，**代码 + 训练器已就绪**） | 🟡 部分完成 | [`stage2/`](algorithms/voiceprint_protection/stage2/)；梯度通路自检 **20/20**；训练器已补齐**断点续训 / AMP / 梯度累积 / 梯度裁剪 / `--max-hours` / 吞吐 ETA**（2026-11-21）；★ 正式训练只缺**成员 A 的 Stage 1 权重**（可求导声码器 HiFi-GAN 16 kHz 已就位）⇒ 云机方案见 [`docs/notes/cloud_ecs_stage2.md`](docs/notes/cloud_ecs_stage2.md) |
| 评测（Sim_prot / Sim_clone / DSR / WER） | 🟡 部分完成 | [`evaluate.py`](algorithms/voiceprint_protection/evaluate.py)；**基线与绝对 WER 已出**（见 [`docs/experiments.md`](docs/experiments.md) §3i）—— ⏳ 但「Ours」那一行仍缺训练好的 Stage 2 |
| 心理声学掩蔽 θ_x（MPEG-1 模型） | ✅ 已完成 | [`stage2/psychoacoustics.py`](algorithms/voiceprint_protection/stage2/psychoacoustics.py) |

### 3.2 相对 baseline 做了什么

- **把论文的两阶段拆成可独立验证的两步**：Stage 1 的解耦-重建（保内容、换音色）、
  Stage 2 的扰动生成（换音色、破克隆）。
- **写了一套可求导的冻结组件**（`stage2/models.py`）：
  现有的 `bridge_autovc` / `bridge_vocoder` 是 numpy 封装（内部 `no_grad()`），
  梯度到不了生成器 G，训练会**静默失效**。
- **把两个"会静默失败"的坑变成会失败的断言**：
  梯度通路自检（12 项）+ torch/numpy GE2E 数值对照（4 项）。
- **服务层把模型做成单例**：平台的请求是同步等待的，
  每次请求重载 400 MB 权重必然超时。

---

## 4. 输入、输出与参数

### 输入

| 输入字段 | 素材类型与格式 | 必填 | 限制与说明 |
|---|---|---|---|
| `input_file` | 音频（`audio/wav`） | 是 | ≤ 50 MB，≤ 60 秒，16 kHz 单声道（服务内部自动重采样） |

### 参数

| 参数字段 | 类型 | 默认值与范围 | 作用 |
|---|---|---|---|
| `alpha` | number | **0.1**，范围 [0, 1]，步长 0.01 | 扰动强度 α（论文取值 0.10）。值越大克隆越难，但音质可能下降 |
| `attack_mode` | select | **clone**（`clone` / `none`） | 是否同时输出攻击者的克隆音，便于直观对比 |
| `report` | boolean | **true** | 是否返回评测报告 JSON |

### 输出

| 输出 | 文件类型 | 含义 |
|---|---|---|
| 保护后语音 | WAV (16 kHz) | 加了扰动后重新合成的音频，人耳听感应与原音频基本一致 |
| 未防御时攻击者的克隆音 | WAV (16 kHz) | 用**原始**嵌入克隆出来的声音（应很像本人） |
| 防御后攻击者的克隆音 | WAV (16 kHz) | 攻击者从保护后语音重新提卡再克隆（应明显不像本人） |
| 评测报告 | JSON | Sim_prot / Sim_clone / 各步张量形状、耗时 |

---

## 5. 环境与安装

- **已验证**：Windows 11，Python 3.9（conda 环境 `vpp`），torch 2.4.1（CPU 版即可推理），
  numpy 1.26.4 / librosa 0.10.2.post1 / soundfile 0.12.1 / scikit-learn 1.5.2 /
  fastapi 0.110.3 / uvicorn 0.30.1 / python-multipart 0.0.20
- **GPU**：训练用；本地 RTX 4060（8 GB）可跑缩减版，论文量级（Stage 1 500k + Stage 2 200k）建议上云
- **依赖文件**：[`algorithms/voiceprint_protection/requirements.txt`](algorithms/voiceprint_protection/requirements.txt)、
  [`requirements.lock.txt`](algorithms/voiceprint_protection/requirements.lock.txt)
- **权重**（不提交到仓库，`.gitignore` 已忽略）：

  | 权重 | 大小 | 来源 | 放置位置 |
  |---|---|---|---|
  | `encoder.pt` | 17 MB | HF `CorentinJ/SV2TTS` | `<数据根>/models/default/` |
  | `synthesizer.pt` | 371 MB | 同上 | 同上 |
  | `vocoder.pt` | 54 MB | 同上 | 同上 |

安装命令：

```bash
conda activate vpp
pip install torch==2.4.1 torchaudio==2.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r algorithms/voiceprint_protection/requirements.txt
pip install Unidecode==1.3.2 inflect==5.3.0     # 上游合成器的文字处理，克隆机必需

# 第三方组件（按固定 commit 拉取，保证两人版本一致）
bash scripts/setup_third_party.sh                # Windows 用 .ps1

# 权重
python algorithms/voiceprint_protection/scripts/download_encoder_weights.py
python algorithms/voiceprint_protection/scripts/download_rtc_weights.py
```

> **国内网络**：`HF_ENDPOINT=https://hf-mirror.com` 可显著加快权重下载。

---

## 6. 启动与运行

### 6.1 启动算法服务

```bash
conda activate vpp
uvicorn algorithms.voiceprint_protection.service:app --host 127.0.0.1 --port 8000
# 或者：python algorithms/voiceprint_protection/service.py
```

接口地址：`http://127.0.0.1:8000/process`（与
[`public/methods/group_n005_voiceprint_protection.json`](public/methods/group_n005_voiceprint_protection.json)
里的 `adapter.endpoint` 一致）。

自检（**不需要权重**就能验协议）：

```bash
python algorithms/voiceprint_protection/scripts/test_service.py --mock   # 40 项
```

### 6.2 启动前端

```bash
npm install
npm run dev
```

打开页面 → 「方法集合」里找 **「基于人机感知差异解耦的隐式声纹保护」**。

### 6.3 第一次运行

1. 用 `public/samples/group_n005_voiceprint_protection/input.wav`（测试集说话人 p228，CC BY 4.0）
2. 参数保持默认：`alpha = 0.1`、`attack_mode = clone`、`report = true`
3. 点「运行方法」

预期得到 3 个音频（保护后语音 / 未防御克隆音 / 防御后克隆音）+ 1 个 JSON 报告。

### 6.4 结果在哪里 / 怎么清理

- 服务把结果写在 `algorithms/voiceprint_protection/results/`，
  通过 `http://127.0.0.1:8000/files/<文件名>` 下载。
- 服务**启动时会清空**上一次的残留（结果目录只保留最近 50 个文件）；
  上传的临时文件**用完即删**。
- 完全清空：停掉服务后删掉 `algorithms/voiceprint_protection/results/` 即可。

### 6.5 离线跑通整条链

```bash
python algorithms/voiceprint_protection/scripts/smoke_test.py
```

---

## 7. 样例与实际结果

| 样例 | 输入链接 | 输出链接 | 参数 | 观察重点 |
|---|---|---|---|---|
| 保护前后与攻击者克隆对比 | [`input.wav`](public/samples/group_n005_voiceprint_protection/input.wav) | [`output.wav`](public/samples/group_n005_voiceprint_protection/output.wav)<br>[`output_clone_baseline.wav`](public/samples/group_n005_voiceprint_protection/output_clone_baseline.wav)<br>[`output_clone_protected.wav`](public/samples/group_n005_voiceprint_protection/output_clone_protected.wav) | `alpha=0.1, attack_mode=clone, report=true` | 三个音频一起听：`output.wav` 跟 `input.wav` 差不多；但未防御的克隆音像本人，防御后的克隆音明显不像 |

> ⚠️ **L1 阶段**：扰动 δ 还是**随机高斯方向**（Stage 2 的生成器 G 尚未训练），
> 所以两个克隆音的差别**还不够大**（当前 Sim_clone ≈ 0.90，论文是 0.13）。
> Stage 2 训练完成后需要**重新生成**这组样例。

---

## 8. 验证与已知限制

### 实际执行过的测试

| 测试 | 结果 |
|---|---|
| `npm test` / `npm run build` | ✅ 6/6 通过 / 构建通过 |
| `scripts/test_service.py --mock` | ✅ **41/41** 通过（协议全项，不需要权重） |
| `scripts/test_service.py`（真权重） | ✅ **41/41** 通过（含真实推理 + 文件下载 + 空文本兜底） |
| `stage2/verify_gradients.py` | ✅ **20/20** 通过（梯度确实穿过三层冻结组件回到 G；含 Stage 1 ↔ Stage 2 接口的 7 项回归断言） |
| `stage2/verify_mel.py` | ✅ **4/4** 通过（torch 版 GE2E 与上游 mel 误差 2.3e-05，嵌入余弦 0.99999481） |
| `stage2/train.py --stub --steps 6` | ✅ 训练机制跑通（8000 条 / 80 说话人） |
| `stage2/train.py --stub --steps 9 --resume` | ✅ **断点续训**（"已从 stage2_last.pt 续训，起始 step = 6"，优化器 + 随机数状态都恢复） |
| `stage2/train.py --stub --vocoder hifigan` | ✅ **真声码器**链路跑通（本机 CPU 0.42 it/s） |
| `infer.py --mode stage2` | ✅ **Stage 2 路线**（δ = G(h)，训练好的生成器；CLI 另有 `--mode {clone,stage1,stage2,auto}`） |
| `scripts/prepare_data.py` | ✅ 三条硬要求自检全通过 |

### 接口当前状态

- 服务代码已完成并通过协议自测；**真实推理需要上游三个权重**。
- 权重缺失时接口返回 **503 + 中文说明**（明确指出缺哪个文件、去哪儿下），不是 500 traceback。

### 已知限制

1. **L1 阶段扰动是随机高斯噪声**（尚未训练 Stage 2 的生成器 G），
   因此当前指标反映的是**链路是否正确**，**不代表论文水平**。
   实测基线（40 条测试集样本）：

   | 变体 | Sim_prot ↑ | Sim_clone ↓ | DSR(τ=0.5) | 论文对应 |
   |---|---|---|---|---|
   | 无防御 | 0.8089 | 0.6967 | 0.0% | Original 0.91 / 0.91 / 0% |
   | 高斯噪声扰动（α=0.1） | 0.8171 | 0.6994 | 2.5% | Gaussian Noise 0.85 / 0.35 / 45.2% |
   | 论文 · Ours | 0.95 | 0.13 | 87.2% | — |

2. ★ **Sim_prot 有结构性天花板 ≈ 0.82**（论文 0.95）：
   我们的保护后语音是**用克隆机重新合成**的，不是"原音频 + 扰动"，
   重新合成本身就会改变音色/韵律 —— **与扰动强弱无关**。
   论文的 0.95 需要 Stage 1 的解码器**重建原句**（内容码 c 保证"说了什么"不变，只换音色）。
   **→ 做到论文那组数字绕不开 Stage 1。**
3. ~~**Stage 2 缺一个可求导的声码器**~~ → ✅ **已解决**（2026-11-19）：
   `bridge_vocoder` 已换成 torch 版 HiFi-GAN（HF `speechbrain/tts-hifigan-libritts-16kHz`），
   本身就可求导。剩下的阻塞只有**成员 A 的 Stage 1 权重**。
   详见 [`docs/notes/stage2_blockers.md`](docs/notes/stage2_blockers.md)。
4. **本机没有 NVIDIA GPU**（核显 + torch CPU 版）⇒ 论文量级必须上云机，
   流程见 [`docs/notes/cloud_ecs_stage2.md`](docs/notes/cloud_ecs_stage2.md)
   与 [`scripts/cloud_train_stage2.sh`](scripts/cloud_train_stage2.sh)。
   ★ 华为云**代金券不适用于竞价实例** ⇒ 只能开**按需**，训完**立刻释放**。
5. **MOS 听测未做**（论文 50 人）——【待填：实际人数】。
6. **黑盒迁移未做**（论文的 Figure 2 跨系统实验，最难，放最后）。
7. VCTK 每人只取了 **100 句**（均匀抽样），不是全部 ~400–800 句 —— 为把 10.94 GiB 的下载压到 1.36 GiB。

### 耗时与资源

- **单条请求（CPU，实测）**：`attack_mode=none` **11.4 s**；`attack_mode=clone` **21.6 s**。
  平台是**同步等待**的，两者都在 60 秒预算内。
  （优化前 clone 模式要 201.8 s —— 已通过调整上游声码器的折叠块长解决，
  实测与音质影响见 [`docs/experiments.md`](docs/experiments.md) §3d。）
- 服务**只加载一次模型**（约 15 s），后续请求复用，不会每次重载 400 MB 权重。

---

## 9. 附加文档

- [实验记录（与代码同 commit）](docs/experiments.md)
- [两人接口契约（形状与签名）](docs/interface-contract.md)
- [任务板与分工](algorithms/voiceprint_protection/TASKS.md)
- [★ 成员 B 的逐步操作手册（Stage 2 → 交付）](docs/notes/member_b_runbook.md)
- [★ 华为云 ECS 开台指南](docs/notes/cloud_ecs_stage2.md)
- [Stage 2 的阻塞项与待决问题](docs/notes/stage2_blockers.md)
- [GE2E 接口核实记录](docs/notes/ge2e_interface.md)
- [VCTK 目录勘察结果](docs/notes/vctk_layout.md)
- [VCTK 下载方案](docs/notes/download_vctk.md)
- [数据与第三方组件的存放位置（C 盘不够用怎么办）](docs/notes/data-location.md)
- [第三方组件来源与许可登记](third_party/README.md)
- [方法配置](public/methods/group_n005_voiceprint_protection.json)
- [样例素材与署名](public/samples/group_n005_voiceprint_protection/SOURCES.md)
