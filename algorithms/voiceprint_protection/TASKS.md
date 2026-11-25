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
| 环境指纹 | ⬜ **待重新生成**（新增了 `scikit-learn==1.5.2`，指纹算法已跟着改）—— 两人装完后重跑 `check_env.py` 比对 |
| `config.py` / `audio_utils.py` | ✅ 已写好（★ 已按 AutoVC 官方实测对齐） |
| `third_party/AutoVC` | ✅ 已克隆，commit `c1827a6` |
| `third_party/RTC`（GE2E） | ✅ 已克隆，commit `890f3a0`；权重 `encoder.pt` 已下载并校验 |
| VCTK 数据 | ✅ **两边已下好并逐条对齐**：110 说话人 × 100 句 = **11,000 条**（A 侧实测与 B 侧 `EXACT MATCH: True`，差集为空）；划分 80/10/20 已自检通过，`scripts/split_speakers.py --check` **对账完全一致** ✅ |
| Stage 1 代码 | ✅ 已实现，19 项自检全过（`scripts/test_stage1.py`） |
| Stage 1 训练 | ⏳ **2026-11-21 重训中**（500k 步 / 13.5 h；上次那批权重被一次冒烟测试覆盖，已加固脚本防再犯） |
| **M3 验收** | ✅ **Sim_prot = 0.999996**（要求 > 0.9，论文 0.95）、`dis` = −4.38134（理论极限的 99.98%）、内容码泄露率 0.0000 —— 详见 `docs/experiments.md §4d` |
| HiFi-GAN 16 kHz 声码器 | ✅ **权重就位**（HF `speechbrain/tts-hifigan-libritts-16kHz`，234/234 键 strict 通过，可求导） |
| 接口契约 | ✅ **v0.5 双方已签字**（`docs/interface-contract.md` §9） |
| Stage 2 回归 | ✅ `verify_gradients` **20/20**（含成员 A 修的两个 bug 的回归断言） |
| 平台端到端 | ✅ `scripts/test_platform_e2e.py` **29/29**（成员 B 的 P5） |
| **训练链路** | ✅ **2026-11-18 审计后全部打通**：AMP / 梯度累积 / 断点续训 / 验证回路 / 两种启动方式 / 帧数约定 —— 详见 **§0.10** |
| **当前阶段** | **W4：Stage 1 训练中**（W3 平台已通） |
| **下一个里程碑** | **M3 Stage 1 完成**（三条验收：checkpoint / 听感正常 / Sim_prot > 0.9） |
| **仍待组内拍板** | **云端代金券的适用产品**（ECS/NVIDIA 还是只有昇腾 ModelArts）→ 决定 50 万步长训在哪跑 |
| **待补依赖** | ✅ 依赖已装齐（实测 2026-11-20：`python-multipart` **0.0.20**（起服务必须）/ `scikit-learn` **1.5.2**（上游 GE2E 必需）/ `faster-whisper` 1.1.0）；⚠️ **`Unidecode` + `inflect`（上游 Tacotron 必需）B 那台还没装** |

### ★ 本周已取得的关键结论（2026-11-13 实测，详见 [docs/notes/autovc_alignment.md](docs/notes/autovc_alignment.md)）

| # | 结论 | 影响 |
|---|---|---|
| 1 | AutoVC 官方仓库是 **`auspicious3000/autovc`**（原 `cyqxiaozhen/AutoVC` 已 404） | D1-1 已完成 |
| 2 | **AutoVC 官方权重是 16 kHz 的**（不是 22.05 kHz） | **全程 16 kHz**，原"22.05k 预设"取消 |
| 3 | **AutoVC 的 mel 不是 librosa**（reflect 填充 + 幅度 + dB + clip[0,1] + fmin=90 + 30Hz 高通） | 已实现 `autovc_mel()`，必须跑对照脚本验证 |
| 4 | content code 实际 **64 维**（dim_neck 32 × 双向），下采样 `freq=32` | 契约已更正 |
| 5 | AutoVC 的说话人嵌入用 `D_VECTOR`（**输入 80 维** mel）；论文脚注的 GE2E 用 **40 维** mel | **两套权重不可混用**，L1 用前者、最终系统用后者 |
| 6 | AutoVC 官方权重在 **IBM Box**（不是 Google Drive）；Google Drive 直连不通 | 需代理或找镜像 |

### 0.4 本轮进展（成员 B · 2026-11-17）

**已产出（代码/文档写完并已实测跑通）**

| 任务 | 产出 | 状态 |
|---|---|---|
| D1-3 | 克隆 RTC（commit `890f3a0`）+ 从 HuggingFace 下 `encoder.pt`（17,090,379 B，已校验） | ✅ |
| D1-4 | VCTK 下载：**1.36 GiB / 11,000 条 / 110 说话人**（用 `scripts/download_vctk.py`，云机没 Node 也能跑） | ✅ |
| D1-5 | `scripts/survey_vctk.py` → `docs/notes/vctk_layout.md`（自动生成） | ✅ |
| D2-4 | `docs/notes/ge2e_interface.md`（四个问题按上游源码逐条核实 + 实测记录） | ✅ **已收口** |
| D2-5 / D3-5 | `scripts/prepare_data.py`：转换 + 80/10/20 划分 + **三条硬要求自检** | ✅ **全通过** |
| D3-4 | `bridge_speaker.py`（上游 GE2E → `h (256,)`，实测 `\|h\|=1.000000`） | ✅ **已跑通** |
| D4-4 | 平台样例**输入** `public/samples/.../input.wav`（测试集说话人 p228）+ `SOURCES.md`（CC BY 4.0 署名） | ✅ |
| D3-1 | `docs/interface-contract.md`（**草案 v0.1，待成员 A 签字**） | ⬜ 待 A 确认 |
| D1-6 | `third_party/README.md`（四个组件的 URL/commit/许可登记表） | ✅ 已合并双方信息 |
| W3-P1 | `public/methods/group_n005_voiceprint_protection.json` + `npm run check:method` 校验器 | ⬜ 待 A review |

### 0.4b ★ 本轮进展（成员 B · 2026-11-18：平台服务化）

**背景**：任务板 §3 的 P3（`service.py`）原派给成员 A，但 A 还没写，
而它是 M2 平台打通的**唯一卡点**（平台 POST 不过去）。
B 先补上这块，两边都不耽误。

| 任务 | 产出 | 状态 |
|---|---|---|
| W3-P3 | **`algorithms/voiceprint_protection/service.py`**（FastAPI + CORS + 统一错误格式 + 临时文件清理 + 模型单例缓存） | ✅ **40 项契约自测全通过** |
| W3-P3 验收 | `algorithms/voiceprint_protection/scripts/test_service.py`（`--mock` 模式不用权重也能验协议） | ✅ |
| W3-P2 | `public/methods/index.json` 追加本方法 | ✅ |
| W3-P6 | `npm test`（6/6）+ `npm run build` | ✅ **全绿** |
| W3-P4 | 样例**输出** `output.wav` | ✅ 已产出（B 的 pipeline 跑出来的真实样例） |

#### 服务层踩到的三个坑（写下来免得 A 再踩一遍）

| # | 坑 | 后果 | 怎么解 |
|---|---|---|---|
| 1 | **FastAPI 默认返回 `{"detail": ...}`** | 前端读的是 `detail.error`（`src/services/algorithms.js:104`），读不到就只显示"算法服务返回 HTTP 422。"，中文说明全丢 | 覆盖全局异常处理器，**所有**错误一律 `{"error": "..."}` |
| 2 | **`infer.protect()` 每次调用都新建 `Protector()`** | 每次请求重载 400 MB 权重（encoder+synthesizer+vocoder），平台同步等待必然超时 | 服务层缓存**一个** `Protector` 实例，懒加载只建一次 |
| 3 | 用户上传的**文件名不能拼进路径** | 可能含 `..`、中文、斜杠 | 临时文件一律 uuid 命名，结果文件也用 uuid 前缀 |

#### 服务怎么起

```bash
conda activate vpp
uvicorn algorithms.voiceprint_protection.service:app --host 127.0.0.1 --port 8000
# 或者：python algorithms/voiceprint_protection/service.py
```

自测（不用下权重）：

```bash
python algorithms/voiceprint_protection/scripts/test_service.py --mock    # 40 项
python algorithms/voiceprint_protection/scripts/test_service.py           # 真权重
```

### 0.5 ★ 上游 RTC 在 2026-03 做过重构（成员 B 核实，三处影响）

| 项 | 变化 | 我们的处理 |
|---|---|---|
| 目录结构 | 平铺 `speaker_encoder.py` → 包结构 `encoder/` | `bridge_speaker.py` **两种布局都支持** |
| **权重来源** | 仓库内 `pretrained.pt` → **HuggingFace `CorentinJ/SV2TTS` 的 `encoder.pt`** | 新增 `scripts/download_encoder_weights.py`；clone 完目录里那个 `pretrained.pt` 是**HTML 错误页**，要删 |
| **GE2E 内部 mel** | **不再取 log**（改成线性 mel） | ⚠️ 拿旧版 mel 喂新版权重 → **嵌入全错但不报错**。桥接层直接调上游的 mel 函数，不混用 |
| 依赖 | 新增 `scikit-learn` | 已加进 `requirements.txt`，**两人都要装** |

已核实 commit：`890f3a03187195b9829db2079b75c2ba2ab0405c`（2026-03-09，MIT）。
详见 [`docs/notes/ge2e_interface.md`](../docs/notes/ge2e_interface.md)（含上游三处 API 漂移的适配）。

### 0.6 ★ DataShare 改版 + 下载方案（成员 B 实测）

| 发现 | 内容 |
|---|---|
| 老链接失效 | `.../bitstream/handle/10283/3191/VCTK-2.tar.gz` → **404**；老 handle 在新站点被重新映射到别的数据集 |
| 新直链 | `https://datashare.ed.ac.uk/server/api/core/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/content`（`VCTK-Corpus-0.92.zip`，11,747,302,977 字节） |
| 包内结构 | 只有 `wav48_silence_trimmed/`（88,329 个 flac / 10.92 GiB / 111 个目录）+ `txt/`（2 MB）。**没有 `wav/` 目录** |
| 速度实测 | 单连接 **31.8 KB/s**；16 连接 ~280 KB/s；**48 连接 423 KB/s** |
| **最终方案** | 新增 `scripts/download_vctk.mjs`：**按中央目录定向取条目 + 48 并发 Range + 直接解压落盘**。每人只取 100 句 → **1.36 GiB（12%），约 55 分钟** |

### 0.7 ★ VCTK 的第 110 个说话人叫 `s5`（成员 B 踩到）

数据里 `wav48_silence_trimmed/` 下是 **109 个 `pXXX` + 1 个 `s5`**，
`s5` 是 VCTK 真实的第 110 个说话人（`speaker-info.txt` 第 111 行：

```text
s5    22  F    British (new speaker, more data would be released for training a speaker-dependent model)
```

**它不遵循 `p+数字` 的命名规律**，按 `p` 前缀筛说话人就会只剩 109 人 → 80/10/20 划分直接失败。
已修：`prepare_data.py::_is_speaker_dir()` 显式把 `s5` 认作说话人。
**组内任何"筛 VCTK 说话人"的代码都要注意这一条。**

### 0.8 ★ 路线取舍已定（成员 A · 2026-11-17，依据实测）

**结论：L1 走路线 B（论文脚注的 GE2E + 上游合成器/声码器）；AutoVC 降级为「Stage 1 的架构参考」。**

#### 决定性依据：AutoVC 官方权重链实测已失效

```
curl -I https://ibm.box.com/s/upmfbstk8n8ded0xd3aaic4ayclh90ee
→ HTTP/1.1 404 Not Found
```

AutoVC 官方 README 给的三件套权重（`autovc.ckpt` / `3000000-BL.ckpt` / WaveNet 声码器）
**全部托管在这个 IBM Box 链接上，现已 404** —— 路线 A 不是"难下"，是**下不到**。

#### 对比

| | 路线 A（AutoVC 全家桶） | **路线 B（上游 RTC 全套）← 选定** |
|---|---|---|
| 说话人编码器 | AutoVC `D_VECTOR`（吃 80 维 mel） | **GE2E**（论文脚注 2，吃 40 维线性 mel） |
| 权重来源 | IBM Box → **404 失效** | HF `CorentinJ/SV2TTS` → **实测 200 OK** |
| 需要的权重 | 3 个（含配套声码器） | 3 个（encoder 17MB / synthesizer 353MB / vocoder 51MB） |
| mel 约定 | 要自己复刻 AutoVC 的自定义链 | **合成器与声码器自带配套，不用猜** |
| 与论文的一致性 | 架构一致，但权重链断了 | **脚注 2 直接引用的就是它** |
| 当前状态 | 阻塞 | **encoder.pt 已跑通**（成员 B） |

#### 路线 B 的推理链（全部有可用权重）

```
[保护]
  wav ──GE2E──→ h (256) ──加扰动 a·d──→ h_adv
  h_adv ──上游合成器(text, h_adv)──→ mel ──上游 WaveRNN──→ 保护后语音 x_adv
[攻击]
  x_adv ──GE2E──→ h_stolen ──上游合成器(text, h_stolen)──→ 攻击者克隆音
[基线]
  h ──上游合成器(text, h)──→ 未防御克隆音（像原说话人）
```

★ **上游合成器就是论文 Figure 1 红框里的那个"攻击者"**：
用同一个克隆机跑两次（原始嵌入 vs 被扰动嵌入），两者的差别**就是防御效果本身**。

#### AutoVC 没有被抛弃，改任新职

| AutoVC 的价值 | 用在哪 |
|---|---|
| **架构**（Ec / D / PostNet、dim_neck=32、双向 LSTM、freq=32） | **Stage 1 我们自己实现并训练**（论文本来就要训） |
| **mel 预处理链**（reflect 填充 + 幅度 + dB + clip[0,1] + fmin=90 + 30Hz 高通） | 已复刻在 `audio_utils.autovc_mel()`，**自训 Stage 1 时沿用** |
| 官方权重 `3000000-BL.ckpt`（= GE2E） | 已被上游 `encoder.pt` 取代（同一个 Wan et al. [2018] GE2E 家族） |
| **备选初始化权重**：`marlenezw/AutoVC_Voice_Conversion` 的 `ckpt_autovc.pth`（171 MB，MakeItTalk 出品，含 Ec + D + GE2E） | **Stage 1 训练的初始化权重来源**（HF 可下载） |

> `marlenezw` 那个 ckpt 的 README 明确写着：
> *"The speaker encoder is a pre-trained model provided Wan et al. [2018]"* —— 正是论文用的 GE2E。

#### 因此分工微调

| | 原分工 | 调整后 |
|---|---|---|
| 成员 A | AutoVC 权重链 | **克隆机（合成器+声码器）桥接 + 上游权重下载 + infer.py 整链** |
| 成员 B | GE2E 桥接 + 数据 + 平台 | 不变（**B 的路线 B 是主线，A 转为补齐 B 缺的"攻击者"那一半**） |

#### A 的新行动项（已开工）

- [x] `bridge_cloner.py`：上游合成器 + 声码器封装（= 攻击者）
- [x] `scripts/download_rtc_weights.py`：定点下载 `synthesizer.pt` + `vocoder.pt`（带大小校验）
- [x] `infer.py` 改写为路线 B，并加**模块级 `protect(wav_path, alpha, output_path)`**（成员 B 的 `make_sample.py` 约定一依赖它）
- [x] `scripts/smoke_test.py` 步骤 2/3 改为「克隆机分支」与「端到端 + 攻击者对比」
- [x] 下载 `synthesizer.pt` / `vocoder.pt`，跑通端到端
      （**M1 已达成**：三个音频都产出，`stolen_embedding_cosine = 0.8985`）
- [x] Stage 1 代码：模型 / 损失 / 数据集 / 可微 GE2E / 训练主循环 + 自检脚本
- [ ] Stage 1 训练：等 `data/train.txt` 就绪后跑
      （第 1 步训 W_s → 第 2 步训解耦网络，**顺序不能反**）
- [ ] Stage 1 权重初始化：试 `marlenezw/AutoVC_Voice_Conversion` 的 `ckpt_autovc.pth`

---

### 0.9 ★ Stage 1 已实现（成员 A · 2026-11-18）

`algorithms/voiceprint_protection/stage1/` 全部写完，`scripts/test_stage1.py` **19 项自检全过**。

| 文件 | 职责 |
|---|---|
| `model.py` | `Ec` / `D` / `Postnet`（照搬 AutoVC 官方 `model_vc.py`），**28,422,464** 参数 |
| `losses.py` | `L_rec` / `L_cd` / `L_dis` + 说话人分类器 `W_s` |
| `dataset.py` | VCTK 清单 → **两套 mel** + 说话人 ID，片段裁成 `freq` 的整数倍 |
| `speaker_encoder.py` | **可微的 GE2E**（论文说它端到端微调） |
| `train.py` | 两阶段训练主循环（第 1 步训 `W_s`，第 2 步训解耦网络） |

**★ 过程中纠正的两个"论文 vs 代码"矛盾（都以代码为准）：**

| # | 论文写的 | 官方代码实测 | 我们的处理 |
|---|---|---|---|
| 1 | `c ∈ R^{T'×32}`，`T' = T//2` | **`(T//32, 64)`**（`freq=32`、`dim_neck=32`×双向） | 以代码为准，已更正契约 §2 |
| 2 | `W_s ∈ R^{K×256}`，`c̃ ∈ R^256`，但 content code 是 32 维 —— 对不上 | 论文说 c̃ 是 "**concatenated** ... **along the time dimension**" | **沿时间维拼接 4 个 content code = 4×64 = R^256** ✅ 与论文的 `R^{K×256}` 严丝合缝 |

**第二点是这轮最有价值的发现**：把 c̃ 理解成"沿时间维拼接 4 个 content code"，
论文的每一个数字就都对上了，而且 `W_s` 能同时吃说话人嵌入 `h`(R^256) 和内容表示 `c̃`(R^256)
—— 两个空间维度相同，分类器才有意义。参数见 `config.STAGE1["content_window"] = 4`。

**第三套 mel 约定**（Stage 1 每条音频要算两次）：
`audio_utils.autovc_mel()` → 80 维给 Ec/D；`audio_utils.ge2e_mel()` → 40 维给 GE2E。

契约已更新为 **v0.3**（成员 A 已签字，新增 §7 Stage 1 的约定）。

---

## 0.10 ★ 训练链路审计与修补（成员 A · 2026-11-18）

> 起因：W3 结束时准备开训，做了一次**"训练到底能不能跑起来"的全面审计**。
> 结果：代码写完了，但**一条命令都跑不通**。下面是一次性补齐的清单。

### 0.10.1 审计发现的 6 个真问题（都已修）

| # | 问题 | 现象 | 修法 |
|---|---|---|---|
| 1 | **训练脚本两种启动方式都是坏的** | `python -m algorithms.voiceprint_protection.stage1.train` → `No module named 'stage1'`；`python stage1/train.py` → `No module named 'config'` | `stage1/__init__.py` 里做 sys.path 引导 + **相对导入**；`train.py` 用 try/except 包住相对/绝对导入。**两种方式现在都实测通过** |
| 2 | **配置里的 `amp` / `accum_steps` 根本没被用** | `config.STAGE1["amp"]=True`、`accum_steps=2`，但训练循环里一个字都没引用 | 真正实现混合精度（`torch.amp.GradScaler`）与梯度累积，8 GB 显存下这是必需品不是可选项 |
| 3 | **没有断点续训** | 论文量级 50 万步要跑几天，而云上是**竞价实例**（随时被抢占），断了就白跑 | 每 `--save-every` 步存完整 checkpoint（模型+优化器+步数），`--resume` 续训；另有 `--max-hours` 时间上限 |
| 4 | **没有验证集回路** | 论文说超参是在验证集上网格搜索选出来的，我们却连验证都不跑 | 每 `--val-every` 步跑一次验证集并记曲线 |
| 5 | **帧数会差 1 帧 → `L_rec` 直接崩** | 两套 mel 链都给 `1 + N//hop` 帧，而采样点数取的是 `crop_frames*hop` ⇒ 多出 1 帧；`freq=32` 摊不开 ⇒ 解码输出与输入不等长 | 采样点数改成 `N = (crop_frames-1)*hop`，并在 Dataset 里加断言。见契约 §7.1b |
| 6 | **短片段时 `content_windows` 维度会错** | 不足 4 个 content code 时退化成 `cat(codes)`，维度变成 `len(codes)*64 ≠ 256`，分类器直接形状报错 | 改成**重复最后一个 code 凑满窗口**，维度恒为 256 |

### 0.10.2 ★ 一处架构性偏离：声码器与 mel 约定

**问题**：论文 §2.3 写的是 `x_adv = D(c, h_adv)` 之后"**用 HiFi-GAN 声码器 V(·) 转成波形**"。

**实测**：**NVIDIA 官方的 `NVIDIA/torch.hifi_gan` 仓库已经下架了**（2026-11-18 验证）：

```
https://api.github.com/repos/NVIDIA/torch.hifi_gan        -> 404
https://codeload.github.com/NVIDIA/torch.hifi_gan/...    -> 404
https://raw.githubusercontent.com/NVIDIA/torch.hifi_gan/..-> 404
https://api.ngc.nvidia.com/.../hifigan_melbins_16k.pth  -> 404
```

（`api.github.com/repos/pytorch/pytorch` 是 200，所以不是网络被墙，是仓库真没了。）

**我们唯一能用的声码器**是上游 Real-Time-Voice-Cloning 自带的 **WaveRNN**（`vocoder.pt`，51 MB，已下载校验）。
但它**只吃自己那套 mel 约定**，喂别的进去**出来是噪音而且不报错**。

**⇒ 决定：模型侧 mel 从 `model16k`（AutoVC 官方链）切换到 `rtc16k`（WaveRNN 那套）。**

| | `model16k`（AutoVC 官方） | **`rtc16k`（WaveRNN）← 当前默认** |
|---|---|---|
| n_fft / hop | 1024 / 256 | 800 / 200 |
| fmin / fmax | 90 / 7600 | 55 / 7600 |
| 前处理 | 30 Hz 高通 | preemphasis 0.97 |
| 取值 | [0, 1] | **[-1, 1]** |
| 有配套声码器？ | ❌（WaveNet 权重 404，HiFi-GAN 下架） | ✅ 上游 WaveRNN |
| 与论文的关系 | 论文用 AutoVC 架构，未规定 mel | **AutoVC 架构完全照搬**，mel 是预处理约定 |

**依据不是猜的**，有三条硬证据：

1. `vocoder/vocoder_dataset.py:27-28` —— WaveRNN 训练时的 mel 被**明确除以 4** 归到 `[-1,1]`；
2. `vocoder/inference.py::infer_waveform` 内部 `mel / 4` ⇒ 喂进去必须是 `[-4,4]`；
3. 上游合成器**真实输出**的 mel 实测范围 `[-4.19, +0.66]` ⇒ 对应
   `synthesizer/audio.py::melspectrogram`（**带 preemphasis、fmax=7600、对称归一化**）。

> ⚠️ **踩过的坑**：一开始照抄了 `vocoder/audio.py::melspectrogram`
> （没有 preemphasis、fmax 用默认 8000、归一化成 [0,1]），
> **数值对照能过（1.6e-07）但送进声码器出来是噪音**。
> ⇒ 所以 `scripts/verify_rtc_mel.py` 必须有**两层对照**：数值 + 端到端。
> 现在两层都过：数值 `2.98e-08`，mel 往返 20.8%（错误量纲是 73%）。

### 0.10.3 本轮新增/改动的文件

| 文件 | 状态 | 说明 |
|---|---|---|
| `stage1/__init__.py` | 🆕 重写 | sys.path 引导 + 相对导入 |
| `stage1/train.py` | 🆕 重写 | AMP / 梯度累积 / 断点续训 / 验证回路 / TensorBoard |
| `stage1/losses.py` | ✏️ 修 | `content_windows` 短片段维度错误 |
| `stage1/dataset.py` | ✏️ 修 | mel 预设化、hop 硬编码、帧数公式、加断言、加 `deterministic`、返回原波形 |
| `audio_utils.py` | ✆️ 增 | `rtc_mel()` / `rtc_mel_raw()` / `rtc_mel_filterbank()` / **`model_mel()`（模型侧统一入口）** |
| `config.py` | ✆️ 增 | `MEL_PRESETS["rtc16k"]`、`STAGE1["mel_preset"]` |
| `paths.py` | ✏️ 修 | `models_dir()` 不再跟随 `VPP_DATA_ROOT`（否则冒烟测试会找不到 encoder.pt） |
| `bridge_vocoder.py` | 🆕 重写 | `mel_to_wav` / `wav_to_mel`，内部用上游 WaveRNN |
| `scripts/verify_rtc_mel.py` | 🆕 新 | 两层对照（数值 + 端到端 mel 往返） |
| `scripts/eval_stage1.py` | 🆕 新 | **M3 验收脚本**：L_rec / Sim_prot / 内容码泄露率 / 重建音说话人准确率 |
| `scripts/make_dummy_data.py` | 🆕 新 | 生成假数据，**不用等 55 分钟的 VCTK 就能冒烟测试训练** |
| `docs/interface-contract.md` | ✏️ v0.4 | §3.3/§3.4/§7.1 重写，新增 §7.1b 帧数约定 |

### 0.10.4 训练链路现在的状态（**假数据实测通过**）

```
$ python algorithms/voiceprint_protection/scripts/make_dummy_data.py
$ $env:VPP_DATA_ROOT = "D:/vpp_dummy_data"
$ python algorithms/voiceprint_protection/stage1/train.py --stage classifier --iterations 3 --smoke
  → step 1..3  loss 4.3658 → 4.3283（在降）✅
$ python algorithms/voiceprint_protection/stage1/train.py --stage model --iterations 6 --smoke --val-every 3 --save-every 3
  → 训练 / 断点 / 验证 三条回路全部跑通 ✅
$ python algorithms/voiceprint_protection/stage1/train.py --stage model --iterations 9 --resume
  → "已从 stage1_last_ckpt.pt 续训，起始 step = 6" ✅
$ python algorithms/voiceprint_protection/scripts/eval_stage1.py --split val --num 4 --save-audio
  → 端到端跑通（指标是垃圾，因为只训了 9 步）✅
```

**⇒ 代码层面已经没有任何阻塞开训的东西了。剩下的唯一阻塞是数据。**

### 0.10.4b ★ 训练速度实测（2026-11-18，RTX 4060 Laptop）

一开始测出来 **5.3 秒/步**（50 万步 = 74 小时，根本没法训）。
profile 之后发现**三个原因叠加**，修完变成 **0.3 秒/步**（提速 ~18 倍）：

| # | 原因 | 修法 |
|---|---|---|
| 1 | **DataLoader 每个 epoch 重新 spawn worker**（Windows 上 spawn 要重新 import torch，十几秒）。假数据只有 32 条 → 一个 epoch 才 2 步 → **每 2 步就重启一次 worker**，平均下来 5 秒/步 | `persistent_workers=True`（`stage1/dataset.py`） |
| 2 | **LSTM 输入非连续张量** → cuDNN 走不了快路径，退回纯 Python 慢路径 | `transpose(...).contiguous()`（`stage1/model.py` 三处） |
| 3 | **batch 8 时 GPU 严重欠利用** | 实测 batch 8 只用 949 MiB / 8 GB，吞吐 69 条/秒；batch 16 用 1.01 GB、吞吐 **149 条/秒**。⇒ `config` 改成 `batch_size=16, accum_steps=1`（有效 batch 不变，但少跑一次前向） |

**profile 明细（batch 16）：**

```
取一个 batch（从 loader）   : 0.057 s
GE2E 编码器单独             : 0.017 s
GE2E + Stage1 前向          : 0.036 s
完整一步（不含取数据）      : 0.116 s
batch= 16: 0.107 s/step -> 149.0 条/秒   显存 1.01 GB
batch= 32: 0.143 s/step -> 223.2 条/秒   显存 1.34 GB
```

**⇒ 时间预算（本机 4060，按 0.3 s/步保守估）**

| 迭代数 | 耗时 | 说明 |
|---|---|---|
| 5k | ~25 分钟 | 第一次跑，先看 loss 形状 |
| 20k | ~1.7 小时 | 应该有像样的重建效果 |
| 100k | ~8 小时 | 建议目标 |
| **500k（论文量级）** | **~42 小时** | ⚠️ 建议还是上云，本机跑要一天半 |

> ⚠️ **本机只有 15.7 GB 内存（可用 4.3 GB）** ——
> `num_workers` **最多 2**，开 4 个会被系统直接杀掉（报 "worker exited unexpectedly"）。
> 云机上内存多，可以开到 8。

### 0.10.4c ★★ 找到了 16 kHz 的 HiFi-GAN 权重（成员 A · 2026-11-19）—— **解掉了 B 的 Stage 2 阻塞点**

**背景**：B 在远端提交了 Stage 2 全套代码（5053 行），但 `docs/notes/stage2_blockers.md`
把第一件事列为**公开阻塞点**：

> ### 一、★ 最要紧的一条：Stage 2 需要一个**可求导**的声码器
> | 候选 | 建议 |
> |---|---|
> | A. AutoVC 配套 wavenet_vocoder | 暂时搁置（权重 404） |
> | B. HiFi-GAN（论文同款） | ★ **推荐，但需要另找权重** |
> | C. 上游 RTC 的 WaveRNN | 不推荐 |
>
> 「还差的部分（需要能连 GitHub 的人补）」：
> ```
> curl -L -o hifigan_melbins_16k.pt https://github.com/NVIDIA/torch.hifi_gan/releases/download/v0.1/hifigan_melbins_16k.pt
> ```

**A 的实测结论**：那个 release 附件**也是 404** —— NVIDIA 把仓库连同 release 一起删了。
（`api.github.com/repos/pytorch/pytorch` 返回 200，所以不是墙。）

#### ★ 找到的替代品

HuggingFace **`speechbrain/tts-hifigan-libritts-16kHz`** 的 `generator.ckpt`：

| | NVIDIA `hifigan_melbins_16k` | speechbrain 这个 | |
|---|---|---|---|
| 采样率 / hop / n_fft | 16000 / 256 / 1024 | **同** | ✅ |
| n_mels | 80 | **同** | ✅ |
| 架构 | HiFi-GAN v1（4 upsample + 12 resblock） | **同** | ✅ |
| 参数量 | 13,936,130 | **13,936,130** | ✅ |
| mel 约定 | slaney mel → `log(clamp(1e-5))` | **同** | ✅ |
| 权重格式 | `weight_g`/`weight_v` | **同**（原生） | ✅ |
| 加载 | — | **strict=True，234/234 键全匹配** | ✅ |
| 可求导 | ✅ | ✅ | ✅ |

只差训练数据（LibriTTS 16 kHz vs NVIDIA 的 16 kHz 模型）—— **架构与约定逐项相同**。

已下载：`data/models/default/hifigan_16k.ckpt`（55,828,077 字节，sha256 `2d92a974...`）
下载脚本：`scripts/download_hifigan_weights.py`
生成器**内置**在 `algorithms/voiceprint_protection/hifigan.py`（bshall/hifigan，MIT，署名在文件头）
→ **两台机器、云机都零配置可用，不用再 clone third_party**。

#### ★ 由此确定的 mel 约定（实测，不是猜的）

```
torchaudio Spectrogram(hop=256, win=1024, n_fft=1024, power=1, normalized=False)
  -> MelScale(sr=16000, n_mels=80, f_min=0, f_max=8000, norm='slaney', mel_scale='slaney')
  -> log(clamp(mel, min=1e-5))          ← 自然对数，不是 log10
```

来源：该模型的 README 给的确切参数 + speechbrain `lobes/models/FastSpeech2.py::mel_spectrogram`。

**实测判据（三层）**：

| 判据 | 我们的实现 | NVIDIA 那条 log10 链 |
|---|---|---|
| 重建音 vs 原音的**说话人余弦** | **0.9746** ✅ | 0.70（近似换算） |
| 通用 mel 相对误差 | 39.9% | **2512%**（纯噪音） |
| RMS | 0.0539 → 0.0585 | 0.7662 |

#### ★★ 一个关键的量化结论：**仿射换算不可接受**

B 的 `stage2/pipeline.py::autovc_mel_to_hifigan()` 想在 AutoVC mel 与 HiFi-GAN mel 之间换算。
A 实测了这个想法：

| 方案 | 说话人余弦 |
|---|---|
| **精确的 HiFi-GAN mel** | **0.9746** ✅ |
| AutoVC mel → 仿射换算 | 0.6983 ❌ |
| 再夹到实际取值范围 | 0.6967 ❌ |
| 再做逐帧 min-max | 0.3640 ❌ |

**原因**：两套 mel 的**滤波器组不同**（AutoVC: fmin 90 / fmax 7600 / 带 30 Hz 高通；
HiFi-GAN: fmin 0 / fmax 8000 / slaney），底层能量分布不同，**仿射补不回来**。

⇒ **结论：Stage 1 的模型侧 mel 必须直接用 HiFi-GAN 的约定**（`config.STAGE1["mel_preset"] = "hifigan16k"`），
⇒ **B 的 `autovc_mel_to_hifigan()` 可以删掉**。

#### ★★ 顺带解决的一个隐患：论文的 λ 会被稀释

HiFi-GAN 的原始 mel 值域是 **[-11.5, 0]**，而 AutoVC 的是 **[0, 1]**。
直接用会让 `L_rec` 比论文大 **~100 倍**（实测 165 vs 0.86），
`λ_cd = 1.0` 的相对权重被稀释 **~40 倍** ⇒ **论文的超参失效**。

**解法**：给网络侧的 mel 加一个**固定仿射变换**（可逆、零损失）：
```
网络侧 mel  a = 0.08686 · s + 0.84      （值域 [-0.11, 0.92]，与 AutoVC 同量纲）
送声码器前  s = (a - 0.84) / 0.08686    （反变换，误差 4.8e-07）
```
实测 `L_rec` 从 165 回到 **0.66**，与论文同一量级 ⇒ **λ_cd=1.0、λ_dis=0.01 直接可用**。

### 0.10.4d ★★ 顺手修了 B 的 Stage 2 两个**致命 bug**（成员 A · 2026-11-19，合并后 review 出来的）

在合并 B 的 Stage 2 代码后，A 做了交叉 review，发现 `stage2/models.py::TorchAutoVCStage1`
有两个 bug —— **症状是"训练能跑、loss 在降、但防御机制根本没生效"**，最坏的一类：

| # | bug | 后果 | 修法 |
|---|---|---|---|
| 1 | `self.model(mel, embedding, embedding)` —— 官方 `Generator.forward(x, c_org, c_trg)` 的**第三个参数才是目标嵌入** | **`h_adv` 根本没进解码器** ⇒ 扰动嵌入对 `x_adv` 毫无影响 ⇒ **整个防御机制失效**，且**不报错** | `decode()` 自己实现摊开 + 解码，c_trg = 给定的目标嵌入 |
| 2 | `decode()` 的 `content` 参数**被完全忽略** | 论文的 **"arbitrary content"**（`c_arb`）没起作用，`x_clone` 用的还是原句内容 ⇒ `L_de` 测的不是论文那个场景 | `decode(..., content=c)` 支持注入内容码 |

**修复后实测**（A 的 Stage 1 checkpoint + B 的封装）：

```
content_code -> (2, 7, 64)                  ✅ 与契约 §4.5.1 一致（原来是 list）
D(c,h) 与 D(c,h_adv) 的平均差 = 0.525248    ✅ 修 bug 前是 0
梯度能回到 h_adv: ✅ (absmax = 2165)        ✅ 修 bug 前是 0
content 被真正使用: ✅                       ✅ 修 bug 前是 0
```

> ★ **验收 Stage 2 时请先确认这三行** —— 写法错了 loss 照样在降，但指标是假的。
> 已写进契约 §4.5.3b。

> ★ 顺带把 `ConvNorm` 的卷积属性名从 `conv_layer` 改成官方的 `conv`
> ⇒ 我们的 checkpoint 的键名与**官方 AutoVC 的 `Generator` 完全一致**（实测 107 = 107）
> ⇒ ① B 的 `TorchAutoVCStage1`（内部用官方 `Generator`）能直接加载我们的权重；
> ② 将来拿到 AutoVC 官方权重也能直接加载进我们的模型。

### 0.10.5 开训前还缺什么（只剩数据）

| # | 缺什么 | 谁 | 怎么办 |
|---|---|---|---|
| 1 | **VCTK 数据** | **你（成员 A）** | ★ **源站已恢复**（2026-11-20 夜实测 HTTP 200，11.75 GiB 可下）。用 B 新写的 **Python 版**定向下载器：`python scripts/download_vctk.py --out <数据根>/raw`（只用标准库，**云机没装 Node 也能跑**）。约 55 分钟 / 1.36 GiB |
| 2 | `data/train.txt` / `val.txt` / `test.txt` / `spk2id.txt` | 你（B 的 `prepare_data.py` 已就绪） | `python algorithms/voiceprint_protection/scripts/prepare_data.py` |
| 3 | 云端 GPU（论文量级训练） | 你 | ⚠️ **代金券能开 ECS(NVIDIA) 还是只有昇腾 ModelArts 仍未确认**（见 §8） |

---

### 0.11 ★ 本轮进展（成员 B · 2026-11-20：review 成员 A 的 v0.5 + 修三个自坑）

**背景**：成员 A 的 v0.5（契约）已合并（5 个提交），带来三处改动：
(1) 16 kHz 的 HiFi-GAN 权重；(2) 模型侧 mel 改成 `hifigan16k`；
(3) 修 Stage 1 <-> Stage 2 的两个致命 bug。B 按 A 的要求做了复核 + 补断言。

| # | 事项 | 结论 | 产出 |
|---|---|---|---|
| 1 | 复核 A 的三处改动 | ✅ **全部同意**，详见 `docs/experiments.md` §3f.1 | 契约 §9 已签字（v0.5） |
| 2 | ★ 把 A 修的两个 bug **加成会失败的断言** | ✅ `stage2/verify_gradients.py`【6】段 7 项（**不依赖训练好的权重**，`random_init=True` 即可验） | 现在共 **20 项**全通过 |
| 3 | ★ 修 `paths.models_dir()` 指向不存在目录的 bug | ✅ 这是一个**真阻塞**：GE2E 权重找不到 ⇒ **Stage 1/2 训练 + L1 推理链 + 平台服务全部起不来** | `paths.models_dirs()` 候选目录列表 |
| 4 | 修 `Stage2Pipeline.vocode()` 写死 mel 预设 | ✅ 换预设就会送出错的 mel（噪音且不报错） | 跟着 `config.STAGE1['mel_preset']` 走 |
| 5 | 修 `audio_utils.MelSpec` 写死 `autovc_mel()` | ✅ `MelSpec('hifigan16k')` 也返回 AutoVC 口径 | 改走 `model_mel(wav, self.name)` |
| 6 | 声码器实现去重 | ✅ 删掉 `stage2/models.py::TorchHiFiGANVocoder`（无人 import 的第二份拷贝），只留 `stage2/pipeline.py::HiFiGANVocoder`，内部包 `bridge_vocoder.VocoderBridge` | 契约 §4.5.3 收口 |
| 7 | ★ **数据划分对账件**（A 的行动项 (6)） | ✅ 名单 + 种子就能复现划分 ⇒ **A 不用再下一份 1.36 GiB VCTK** | `docs/notes/vctk_speakers.txt`（110 人）+ `scripts/split_speakers.py` |
| 8 | 契约签字 | ✅ `docs/interface-contract.md` §9 成员 B 已签 v0.5，新增 §4.5.3c | — |
| 9 | 回归 | ✅ `verify_gradients` 19 / `verify_psychoacoustics` 20 / `test_service --mock` 41 / `verify_hifigan` / `verify_rtc_mel` / `test_stage1` / `test_audio_utils` / `verify_mel` 全绿 | `docs/experiments.md` §3f.4 |

> ★ **本机环境提醒**：`vpp` 环境是 **torch 2.4.1+cpu**（无 GPU），
> 上面所有自检都在 CPU 上跑（够用）；正式训练要另开 CUDA 环境或云机。


### 0.12 ★ 本轮进展（成员 B · 2026-11-20 晚：Stage 1/2 在真数据上跑通 + 抓到一个致命 bug）

**背景**：A 的 Stage 1 之前只在**假数据**上验过（`test_stage1.py` 19 项），
Stage 2 也只在 `--stub`（替身 Stage 1 + 替身声码器）上跑过。
这一轮把它们接到**真 VCTK + 真 HiFi-GAN + 真 GE2E** 上跑 —— 结果**抓到一个接上就崩的 bug**。

| # | 事项 | 结论 |
|---|---|---|
| 1 | ★ 把 `stage2/train.py` 的 `--stub` 与 `--vocoder` **解耦** | 原来 `--stub` 强制用替身声码器 ⇒ **真实声码器从来没在 Stage 2 里上过场**，一直没人验过。解耦后 `--stub --vocoder hifigan` = 替身 Stage 1 + 真声码器 |
| 2 | Stage 1 训练循环在**真 VCTK** 上跑通 | ✅ 3 步，loss 递减，模型侧 mel = `hifigan16k`（A 的 v0.5 生效），**14 s/step**（CPU, batch 4） |
| 3 | Stage 2 在**真数据 + 真声码器** 上跑通 | ✅ L_de / L_mask / L_mel / gan 全部有限且在动，**45.5 s/step**（CPU, batch 8） |
| 4 | ★★ Stage 2 接**真实 Stage 1 checkpoint** | ❌ **崩**：`x 与 x_adv 的帧数必须相等，实际 188 vs 192` |
| 5 | ★ 修帧数不变式（两处） | `decode()` 输出裁回原始帧数（不变式**由构造保证**）+ `stage2/dataset.py` 按 Stage 1 的同一套约定裁（帧数是 32 的整数倍、长度恒定） |
| 6 | ★ 修「冒烟 checkpoint 被当正式权重」的坑 | `stage1/train.py` 存出的 `stage1_model.pt` 会被 `_guess_checkpoint()` **优先捡走** ⇒ 3 步的垃圾权重会被静默使用。已改名 `stage1_smoke3steps_NOT_TRAINED.pt`；**建议 A 把 `_guess_checkpoint()` 改成只认显式路径** |
| 7 | 加断言 | `verify_gradients.py`【6】段 (e)：`T=90`（非 32 倍数）时 `decode()` 输出仍是 90 帧 ⇒ 现在 **20 项**全通过 |

> 详见 `docs/experiments.md` §3g。**云机预算的实测依据**也在那里：
> Stage 2 用真 Stage 1 + 真 HiFi-GAN 是 **30~35 s/step**（CPU, batch 8）；
> 按 A 的 4060 实测（0.3 s/step）推算，200k 步约 **17 小时**。


### 0.13 ★ 本轮进展（成员 B · 2026-11-20 夜：P2 基线 / P3 绝对 WER / P5 云机脚本 + 修两个**静默失效**的 bug）

**背景**：A 的三个行动项里 (1)(2)(3) 已完成（§0.11）。剩下 B 能独立推进的是
**P2（测试集基线指标）**、**P3（绝对 WER）**、**P5（云机一键脚本）**。
这一轮三件事都推到底了，**顺带抓出两个"看起来正常、其实结果错了"的 bug**。

| # | 事项 | 结论 |
|---|---|---|
| 1 | ★★ `datashare.ed.ac.uk` **恢复了** | ✅ HTTP 200，11.75 GiB 源可下（此前 3 次尝试全部超时）。**A 现在可以直接自取 VCTK，不必等 B 打包** |
| 2 | ★ 新增 `scripts/download_vctk.py`（Python 版定向下载器） | ✅ 云机常常没装 Node，`.mjs` 用不了；这个版本**只用标准库**，支持 `--speakers` / `--per-speaker` / `--include`，且**断点续传**（已落盘且大小对就跳过） |
| 3 | ★★ 修「下载选择**不可复现**」 | ✅ 旧 `.mjs` 用 `localeCompare` 排序（**依赖运行机器的 locale 与 ICU 版本**），且把 `wav48_silence_trimmed/log.txt` 也算进了每说话人配额。两处都改成**码点序 + 只按音频文件计配额**；`.py` 与 `.mjs` 现在选出**完全相同**的 11,000 个文件 |
| 4 | ★★ B 的数据集与规则**重新对齐** | ✅ B 机器上多出 4 个"野文件"（p303/p304 各 2 个，早期分批下载的残留）⇒ 清单 11,004 条 ≠ 规则的 11,000 条。删掉这 4 个并重跑 `prepare_data.py`（**0 条重转**）⇒ 现在**两边逐条一致**（`EXACT MATCH: True`），清单 train 8,000 / val 1,000 / test 2,000 |
| 5 | ★★ 抓到一个**静默失效**：`load_transcripts()` 只扫顶层 | VCTK 的 `txt/` 是**按说话人分子目录**的（`txt/p225/p225_001.txt`），`glob("*.txt")` 一个都找不到 ⇒ **绝对 WER 会无声退化成"Whisper 转写原音"的相对 WER，而且不报错**。改 `rglob` |
| 6 | ★★ 抓到一个**静默失效**：转写键对不上 | 我们把两路麦克风都收了，转 16 kHz 时改名成 `p228_001_mic1.wav`，而 VCTK 的转写只有一路、叫 `p228_001.txt`。拿 `path.stem` 去查**永远查不到** ⇒ 同上，退化成相对 WER。新增 `transcript_key()` 去掉 `_mic1`/`_mic2` 后缀 |
| 7 | ★ 让退化**可见** | 统计"没找到原始转写"的条数，有就用醒目提示打印出来，不再默默算 |
| 8 | ★ P2 基线（用 v0.5 的新代码复测） | ✅ **100 条分层抽样**（测试集 20 个说话人 x 5 句，均匀抽样）：无防御 Sim_prot **0.8043** / Sim_clone **0.7080** / DSR **0%·0%·0%**；高斯噪声 Sim_prot **0.8053** / Sim_clone **0.6995** / DSR **0%·0%·0%** ⇒ **高斯噪声与无防御无法区分**（与 §3c 的旧结论一致）。**27 s/条**（CPU） |
| 9 | ★ P3 绝对 WER | ✅ 补齐了 VCTK 的 `txt/`（8,000 个文件 / 5.2 分钟下完）；链路上**三个静默失效**已修（转写查不到 0/20 → **20/20**、WER 恒 ~100%、样本全是同一句话）。**结果**：WER(保护后) **5.81%** / WER(原音·ASR 下限) **3.30%** ⇒ 防御引入 **2.51 个百分点**（含「TTS 往返」损失）。详见 `docs/experiments.md` §3i.5 |
| 10 | ★ P5 云机一键脚本 | `scripts/cloud_train_stage1.sh`：`--setup` / `--start` / `--status` / `--stop`。**"配环境"和"烧钱训练"分成两条命令** —— 云机按小时计费，别一开机就跑满 |
| 11 | 回归 | ✅ 7 个自检脚本全绿（`test_audio_utils` / `verify_mel` 4 / `verify_psychoacoustics` 20 / `verify_gradients` 20 / `verify_rtc_mel` / `verify_hifigan` / `test_stage1`） |
| 12 | ★ 新增 `scripts/parallel_download.py` | ✅ HuggingFace 的 CDN 从国内**单连接只有 ~350 KB/s**，而 Whisper `large-v3` 的 `model.bin` 有 **3.09 GB**（单连接要 2.4 小时）。这个脚本并发拉取；实测 `large-v3` 用 24 路并发约 20 分钟下完。**三个坑**（都写在脚本注释里）：urllib 默认 UA 被 **403**；CDN 对 **HEAD** 不响应（会挂住）；hf-mirror 对**小文件**会忽略 `Range` ⇒ 必须先探，不支持就退回单流。**验证**：8 路并发下 HiFi-GAN 权重与本机已下的那份**逐字节一致**（sha256 `2d92a97446467ae5…`） |

> ★ **给成员 A 的数据复现配方**（源站已恢复，可直接照做）：
>
> ```bash
> python scripts/download_vctk.py --out <数据根>/raw --include wav48_silence_trimmed --include txt --per-speaker 100
> python algorithms/voiceprint_protection/scripts/prepare_data.py
> ```
>
> 应当得到 **11,000 条**（train 8,000 / val 1,000 / test 2,000），说话人划分用
> `scripts/split_speakers.py --check` 对账（名单见 `docs/notes/vctk_speakers.txt`）。
> 逐条对账：B 侧 `EXACT MATCH: True`（11,000 == 11,000，差集为空）。

### 0.14 ★ 本轮进展（成员 B · 2026-11-21：Stage 2 开训前的准备 + 云机方案）

**背景**：A 的 Stage 1 训完 500k 步（rec 0.00177 / cd 0.00007 / dis −4.38134），
最初报 Sim_prot 只有 0.459（要 >0.9），A 一度准备改 `L_rec` 的形式重训。
⇒ **Stage 2 不能现在开训**（Stage 1 换了要重训一遍）。
这一轮做的是：**把 Stage 2 的训练器做成"能真的跑完 20 万步"的样子**，
并把云机这条路走通。

| # | 事项 | 结论 |
|---|---|---|
| 1 | 拉 A 的 7 个 commit | ✅ `git pull --ff-only` 干净快进（0 ahead / 7 behind，代码零冲突） |
| 2 | 合并后的回归 | ✅ **7 个自检脚本全绿**（verify_gradients 20/20、verify_psychoacoustics 20/20、verify_mel 4/4、verify_hifigan、verify_rtc_mel、test_stage1、test_audio_utils） |
| 3 | ★★ **B 这台机器跑不了论文量级** | **只有 Intel Iris Xe 核显，没有 NVIDIA GPU**（torch 2.4.1+cpu），内存空闲仅 2.2 GB ⇒ Stage 2 ≈ 30 s/step，200k 步要**两个月** ⇒ 必须上 GPU（**A 那台有 RTX 4060**） |
| 4 | ★★ **Stage 2 训练器补齐六件事** | 断点续训 `--resume` / AMP / 梯度累积 / 梯度裁剪 / `--max-hours` / 吞吐+ETA |
| 5 | ★★ **`config.STAGE2` 三个键写了没人用** | `amp` / `accum_steps` / `clip_grad` —— 与 A 在 Stage 1 审计里发现的**同一类问题**（不报错、loss 照样降，只是慢/显存不够）。现已全部接上 |
| 6 | ★ 加 **`[L_de 方向自检]`** | 训练结束自动比较前 10% / 后 10% 的 L_de，方向反了直接喊出来（A 的 `L_dis` 就是同一种坑，§0.10.4c） |
| 7 | ★★ 抓到一个**隐藏前提** | **E_spk 必须是 Stage 1 微调过的那个**（论文 §3.1）⇒ 训练与推理必须用**同一个** E_spk，否则两个嵌入空间不一样、**δ 退化成随机方向、防御静默失效**。已加 `--speaker-encoder {stage1,upstream}`，并把选择**写进 checkpoint**，推理照着做<br>⭐ **A 这边确认：Stage 1 训练时 E_spk 就是被微调的**（`fine_tune=True`，权重存进了 checkpoint 的 `"speaker_encoder"` 键）⇒ **Stage 2 必须用 `--speaker-encoder stage1`**。详见 §0.15 |
| 8 | ★ `stage2/dataset.py` 加 `persistent_workers` | Windows 上 DataLoader 的 worker 每个 epoch 重启一次（十几秒）—— A 在 Stage 1 上踩过同一个坑（§0.10.4b 第 1 条） |
| 9 | ★ `infer.py` 加 **stage2 路线** | `δ = G(h)`（训练好的生成器）—— 这才是论文的防御。CLI 现在有 `--mode {clone,stage1,stage2,auto}`（之前根本没暴露 `mode`） |
| 10 | ★★ **云机方案：华为云 ECS** | 完整指南 [`docs/notes/cloud_ecs_stage2.md`](../../docs/notes/cloud_ecs_stage2.md) + 脚本 [`scripts/cloud_train_stage2.sh`](../../scripts/cloud_train_stage2.sh) |
| 11 | ★★ **推翻 §8.2 的一条旧结论** | 华为云官方《ECS · 计费说明》：**「代金券不适用于竞价计费型实例」** ⇒ 我们只有代金券 ⇒ **必须开按需，不能开竞价**（§8.2 已改） |
| 12 | 本机实测 | `--stub` 0.50 it/s；`--stub --vocoder hifigan` **0.42 it/s**（batch 2 / 1 秒）；`--resume` 正常（"起始 step = 6" ✅） |

> ★ **给成员 A 的两件事**：
> ① **Stage 1 的正式 checkpoint 训完请给 B 一份**（`weights/stage1_model.pt`，约 358 MB）——
>    Stage 2 训练**必须**有它（冻结的解码器 `D(c, h_adv)`）；
>    也请把 `--speaker-encoder` 的选择对齐：**A 的 Stage 1 训的是「微调后的 E_spk」**
>    ⇒ Stage 2 必须用 `--speaker-encoder stage1`。
> ② ~~Stage 1 改成新的 `L_rec` 之后，**Sim_prot 要 > 0.9** 再开 Stage 2~~
>    → ✅ **2026-11-21 已满足：Sim_prot = 0.999996**（见 §0.15）。
>    **不需要改 `L_rec`**，Stage 2 可以开训。

---

**背景**：A 的 Stage 1 训完 500k 步（rec 0.00177 / cd 0.00007 / dis −4.38134），
但 **Sim_prot 只有 0.459**（要 >0.9），A 说要改 `L_rec` 的形式重训。
⇒ **Stage 2 不能现在开训**（Stage 1 换了要重训一遍）。
这一轮做的是：**把 Stage 2 的训练器做成"能真的跑完 20 万步"的样子**，
并把云机这条路走通。

| # | 事项 | 结论 |
|---|---|---|
| 1 | 拉 A 的 7 个 commit | ✅ `git pull --ff-only` 干净快进（0 ahead / 7 behind，代码零冲突） |
| 2 | 合并后的回归 | ✅ **7 个自检脚本全绿**（verify_gradients 20/20、verify_psychoacoustics 20/20、verify_mel 4/4、verify_hifigan、verify_rtc_mel、test_stage1、test_audio_utils） |
| 3 | ★★ **本机跑不了论文量级** | **本机只有 Intel Iris Xe 核显，没有 NVIDIA GPU**（torch 2.4.1+cpu），内存空闲仅 2.2 GB ⇒ Stage 2 ≈ 30 s/step，200k 步要**两个月** ⇒ 必须上 GPU |
| 4 | ★★ **Stage 2 训练器补齐六件事** | 断点续训 `--resume` / AMP / 梯度累积 / 梯度裁剪 / `--max-hours` / 吞吐+ETA |
| 5 | ★★ **`config.STAGE2` 三个键写了没人用** | `amp` / `accum_steps` / `clip_grad` —— 与 A 在 Stage 1 审计里发现的**同一类问题**（不报错、loss 照样降，只是慢/显存不够）。现已全部接上 |
| 6 | ★ 加 **`[L_de 方向自检]`** | 训练结束自动比较前 10% / 后 10% 的 L_de，方向反了直接喊出来（A 的 `L_dis` 就是同一种坑，§0.10.4c） |
| 7 | ★★ 抓到一个**隐藏前提** | **E_spk 必须是 Stage 1 微调过的那个**（论文 §3.1）。现在训练用上游 GE2E、推理用微调后的 ⇒ **两个嵌入空间不一样 ⇒ δ 退化成随机方向、防御静默失效**。已加 `--speaker-encoder {stage1,upstream}`，并把选择**写进 checkpoint**，推理照着做 |
| 8 | ★ `stage2/dataset.py` 加 `persistent_workers` | Windows 上 DataLoader 的 worker 每个 epoch 重启一次（十几秒）—— A 在 Stage 1 上踩过同一个坑（§0.10.4b 第 1 条） |
| 9 | ★ `infer.py` 加 **stage2 路线** | `δ = G(h)`（训练好的生成器）—— 这才是论文的防御。CLI 现在有 `--mode {clone,stage1,stage2,auto}`（之前根本没暴露 `mode`） |
| 10 | ★★ **云机方案：华为云 ECS** | 完整指南 [`docs/notes/cloud_ecs_stage2.md`](../../docs/notes/cloud_ecs_stage2.md) + 脚本 [`scripts/cloud_train_stage2.sh`](../../scripts/cloud_train_stage2.sh) |
| 11 | ★★ **推翻 §8.2 的一条旧结论** | 华为云官方《ECS · 计费说明》：**「代金券不适用于竞价计费型实例」** ⇒ 我们只有代金券 ⇒ **必须开按需，不能开竞价**（§8.2 已改） |
| 12 | 本机实测 | `--stub` 0.50 it/s；`--stub --vocoder hifigan` **0.42 it/s**（batch 2 / 1 秒）；`--resume` 正常（"起始 step = 6" ✅） |

> ★ **给成员 A 的两件事**：
> ① **Stage 1 的正式 checkpoint 训完请给 B 一份**（`weights/stage1_model.pt`，约 358 MB）——
>    Stage 2 训练**必须**有它（冻结的解码器 `D(c, h_adv)`）；
>    也请把 `--speaker-encoder` 的选择对齐：若 Stage 1 训的是"上游 GE2E 不微调"，
>    就显式用 `--speaker-encoder upstream`，两边必须一致。
> ② Stage 1 改成新的 `L_rec` 之后，**Sim_prot 要 > 0.9** 再开 Stage 2
>    —— 否则 Stage 2 学的是"怎么扰动一个坏掉的空间"。

### 0.15 ★★ 2026-11-21：M3 通过 —— 一个「E_spk 空间错配」引发的假象（成员 A）

> ⭐ **与下面 §0.14 第 7 条是同一个坑，两边独立发现 —— 结论一致。**

**现象**：500k 步训完后 `eval_stage1.py` 报 Sim_prot = 0.459（要求 > 0.9），
且重建音 RMS 只有原音的 1/10。五层排查全部指向"重建质量差"（声码器无损、不是过拟合、
误差不是关键、mel 过度平滑、实现没问题）。

**真因**：Stage 1 训练时 **E_spk 是被微调的**（论文 §3.1 明写 "fine-tuned end-to-end"），
checkpoint 里存了微调后的权重；而评测用的是**上游预训练**的 E_spk
⇒ **两者的嵌入不在同一个空间**（同一段原音在两者下只相关 **0.518**）⇒ 指标全偏低。

| E_spk | L_rec | Sim_prot | RMS 比 |
|---|---|---|---|
| 上游预训练（之前用的） | 0.04390 | 0.4367 | 0.101 |
| ★ checkpoint 里的微调 | **0.01928** | **0.999996** | **0.707** |

⇒ **M3 通过：Sim_prot = 0.999996**，`dis` = −4.38134，内容码泄露率 0.0000。

> ⚠️ **成员 B 的 `evaluate.py` 也要注意这个口径**（他那边是共同区）：
> 论文的 E_spk 是**微调后**那个 ⇒ Sim_prot / Sim_clone 应全线统一用它。
> 但**说话人分类器 W_s 是例外**（我们的 W_s 在**预训练** E_spk 的空间里训的），
> 问它"这是谁"时必须用**预训练** E_spk 的嵌入。
> `scripts/eval_stage1.py` 现在两个编码器各司其职。

**另一个教训**：排查时跑了一次 20 步的 `--smoke`，它结束时**覆盖了 `stage1_model.pt`**，
500k 步的权重没了。已加固 `stage1/train.py`：
- 未跑满步数**不覆盖**正式权重（另存 `stage1_partial_step{N}.pt`）；
- 新增 **`stage1_best.pt`**：验证集最优单独留一份（轻量 115 MB，不含优化器）。

---

---

### 0.16 ★ 本轮进展（成员 B · 2026-11-23：把「A 交权重之后要干什么」写成手册）

**背景**：A 说 Stage 1 还要一段时间。这一轮不训练，**把成员 B 剩下的全部任务
写成一本可照抄的操作手册**，等 A 的正式权重一到手就能照着跑，不用临场想。

> ★ **写完手册当天，A 就查清了那个 Sim_prot 0.459**：
> 那是「用**上游预训练**的 E_spk 去评测**微调过**的模型」造成的**空间错配**，
> 换成 checkpoint 里的微调 E2E 后 **Sim_prot = 0.999996，**M3 已通过（见 §0.15）。
> ⇒ **Stage 1 不用重训，Stage 2 可以开训了**；
> ⇒ E_spk 的问题也有了答案：**A 的 checkpoint 里有微调 E2E ⇒
>    Stage 2 用 `--speaker-encoder stage1`**（B 侧的默认值就是它，见 §0.14 第 7 条）。

| # | 事项 | 结论 |
|---|---|---|
| 1 | ★★ **新增 [`docs/notes/member_b_runbook.md`](../../docs/notes/member_b_runbook.md)** | 9 步操作手册：准备 → 云机 → 训练 → 评测 → 消融 → 收结果释放实例 → 样例 → 平台 → 交付。每步都有「**看什么数**」与「**不对怎么办**」，另有故障排查速查表 + 论文对照表 + 花钱预算 |
| 2 | ★★ 补上 `evaluate.py` 的 **保护链开关** | 原来 `--mode trained` **只换 δ，保护链仍是 clone 路线** ⇒ Sim_prot 有 0.82 的结构性天花板 ⇒ 拿它跟论文的 0.95 比会得出错误结论。新增 `--protect-mode {clone,stage1,stage2}`（默认 clone ⇒ **现有行为完全不变**） |
| 3 | ★ 评测的 E_spk 统一 | 删掉 `evaluate.py` 里那个进程内缓存的 `_embed()`（上游 SpeakerBridge），改成统一走 `protector.embed_for_route(wav, route)` —— **每条路线用它自己的 E_spk**。否则会「用 A 的编码器训、用 B 的编码器测」，指标不可比而且不报错 |
| 4 | ★ `make_sample.py` 加 `--mode` | 平台样例要用**训练好的 G** 出（`--mode stage2`），否则样例展示的还是随机扰动、没有防御效果 |
| 5 | ★ 消融能真跑了 | `stage2/losses.py::perceptual_loss` 加 `use_mel_loss`，`stage2/train.py` 加 `--no-mel-loss` ⇒ 论文 Table 2 的 "w/o L_mel" 这一行终于能训（之前 `config.ABLATION` 里写着 `use_mel_loss: False` 但**根本没实现**） |
| 6 | ★ 云机脚本补参数 | `cloud_train_stage2.sh` 加 `--outdir` / `--lambda-de` / `--lambda-per` / `--lambda-mask` / `--lambda-gan` / `--no-mel-loss` / `--log-every` / `--save-every` ⇒ **消融的三组实验可以用同一条命令跑**（`--outdir` 一定要给，否则会覆盖主实验的权重） |
| 7 | 生成器结构的反推逻辑收敛到一处 | `stage2/generator.py::config_from_state` 是唯一实现，`infer.py` 与 `evaluate.py` 都用它（之前两边各写一份，正是"慢慢跑偏"的典型） |

> ★ **手册里最该记住的三条**：
> ① 评测一定要加 `--protect-mode stage2`，否则走 clone 路线，数字不能跟论文比；
> ② 训完要**释放实例**（不是关机）—— 关机只停算力费，硬盘和 EIP 照收；
> ③ 消融要**重训**，不是改评测参数。

### 0.17 ★★ 本轮进展（成员 B · 2026-11-25：权重到手 + **两条旧结论被推翻** + 开云机前的三连预检）

**背景**：A 的 Stage 1 权重通过 **GitHub Release `stage1-250k-bn`** 交付（仓库里只留同名 JSON），
本机下载后按 **sha256 `14f7e07b…8832`** 与 JSON 双向对账 + 三段验真（`step` /
`speaker_encoder` / `model`）通过 ⇒ M3 收口、Stage 2 具备开训条件。
**但"能开训"不等于"值得开"** —— 代金券按小时烧，所以先在本机把机制性问题问完。

> ★★★ **推翻之一：`Sim_prot = 0.999996` 是塌缩空间的副产品，不是重建质量。**
> A 那份 250k JSON 自带的两组余弦把 §0.15 的结论顶回去了：
>
> | 打分口径 | 同人间 | 异人间 | **gap** |
> |---|---|---|---|
> | checkpoint 里**微调过**的 GE2E | 0.9453 | 0.8958 | **+0.049** |
> | **上游预训练** GE2E | 0.7663 | 0.5670 | **+0.199** |
>
> B 本机独立复算异人间 gap 只有 **+0.013**；同一份 JSON 的
> `upstream_speaker_top1` = clean **0.729** / oracle 0.757 / **rebuilt 0.229**，
> `upstream_original_rebuilt_cosine` = **0.7058**（与 B 复算的 Sim_prot **0.6994** 互相印证）。
> ⇒ 在 gap ≈ 0.05 的空间里**任何两条语音都能拿到 ≈1 的余弦**；
> 与论文可比的是**上游口径**，那里 A 的 Stage 1 只是**中等水平**，**仍然是短板**。
> 详见 [`docs/experiments.md`](../../docs/experiments.md) §4g、
> [`docs/notes/member_b_runbook.md`](../../docs/notes/member_b_runbook.md) §1.4.1。

> ★★★ **推翻之二：本机预检的结论是"路基本不通"，不是"还没训够"。**
> 五发实验（产物在本机 scratch `.tmpB/`（已 .gitignore），**三个探针脚本已固化为
> `stage2/diagnose_gradient.py` / `diagnose_delta_oracle.py` / `diagnose_alpha_sweep.py`**，
> 全部本机 CPU 实测，非估计）：
>
> | 一发 | 配置 | 结果 |
> |---|---|---|
> | 150 步预检 | 2 s/条、batch 4、`--de-space upstream` | `sim@de` 0.680→0.649→0.653（每段 16 点均值）——**方向轻正、淹没在噪声里**；`mask` 0.70→1.12、`per` 0.077→0.120 **同步变差** |
> | 过拟合·忠实目标 | `--overfit 4`、λ_per=15、lr 1e-4，60 步 | `de` 0.549–0.596 **平** ⇒ 排除"换批抖动" |
> | 过拟合·纯 L_de | λ_per=0、lr **1e-3**，120 步 | 只降约 **10%** 且回弹 ⇒ 排除"λ_per 对着干"与"学习率太小" |
> | 梯度实测 | 同一 batch 只反传 λ_de·L_de | **upstream ‖∂L/∂θ_G‖ = 37.0** vs 塌缩空间 **7.34** ⇒ **不是梯度消失**，也证明"换考核空间"确实把信号放大 5 倍 |
> | 过拟合·**α=0.5** | λ_per=0、lr 1e-4，100 步 | 起点从 ~0.66 掉到 ~0.50（**α 送的**），但训练 100 步**练回去了**（0.504→0.556） ⇒ **"多跑步数就会好"这个指望不成立** |
>
> **最后一刀（自由 δ 的 oracle，绕开 G）**：把每条的 δ 直接当参数、
> 并强制 `‖δ‖ = 1`（= G 随机初始化时的工作点，角度预算与 α=0.1 **完全等量**），
> 两发独立随机初始化分别在 40 / 60–70 步进入平台，
> `L_de` = **0.4229** 与 **0.4498/0.4518**（起点 0.553，无防御 0.583）
> ⇒ **地板 ≈ 0.42–0.45**，比 G 训到的 0.52 只好一点，离论文 0.13 差 3 倍以上。
> 物理背景：`perturb_embedding()` = `normalize(h + α·δ)`，实测 `‖h‖=1`、`‖δ‖≈1`
> ⇒ α=0.10 时 h_adv 只偏离 **5.73°**（cos ≥ 0.995），**防御完全押在放大链上**。
>
> ⇒ **结论：α=0.1 + 我们这个 Stage 1 放大链，在 upstream 口径下达不到论文量级；
> 200k 步云机训练（≈代金券大头）在拿到新证据之前不应开。**
> 改成先做**不用花大钱**的两件事：① §5.1 三条基线（钉 `--espk-space upstream`）；
> ② **α–Sim_clone 曲线**（论文"α 敏感吗"那一问，我们反而能给出定量回答）。
>
> ✅ **2026-11-25 决定已落地（成员 B 确认）：不开云机，走 ① + ② 的负结果路线。**
> 三条开台闸门（oracle 压到 0.3 以下 / 偏离论文 α / 先修 Stage 1 放大链）当时**一条都没满足**，
> 因此 runbook §4.2 的 200k 命令保持"不执行"，代金券不花；
> M4 的交付口径相应变成**三条基线的完整数字 + α 曲线 + §4g 的方法层短板定位**。

**本轮改动（代码 5 处 + 文档 4 处）**：

| # | 改动 | 说明 |
|---|---|---|
| 1 | ★★ `stage2/pipeline.py` 新增 **`de_speaker` 第二编码器槽** | `L_de` 的**度量空间**与 δ 的**生成空间**解耦（默认同一个 ⇒ 行为不变） |
| 2 | ★★ `stage2/train.py` 新增 **`--de-space {stage1,upstream}`** | 装配第二个编码器并冻结；`de_reference` 与 `defense` 走 `de_encoder`；日志多一列 **`sim@de`**；`checkpoint_meta["de_space"]` **落盘** |
| 3 | ★ `stage2/train.py` 新增 **`--overfit N`** 与 **`--lr`** | 固定前 N 条、batch=N、不打乱，专门用来判"路通不通"；`--lr` 覆盖 `config.STAGE2["lr"]`（原来三处硬引用 config，改不动） |
| 4 | ★ `infer.py` / `evaluate.py` 读出口径 | `stage2_de_space` 加载时打印"L_de 考核于 … 空间"；`report["de_space"]` 与 `espk_space` / `delta_space` / `secondary_espk` 一起落盘 ⇒ **事后不会忘记当时用的哪个口径** |
| 5 | ★ `scripts/cloud_train_stage2.sh` | `--de-space` 透传（云机默认 `upstream`）+ **`export PYTHONUNBUFFERED=1`**（重定向时 stdout 块缓冲，十几分钟零输出会误杀正常任务） |
| 6 | 文档 | `experiments.md` 新增 **§4g**（含 §5 坑表 +6 行、修掉一个错位的 `4e.4` 编号）；`member_b_runbook.md` §1.4 改写成 Release 下载 + 双向对账、新增 **§1.4.1** 与 **§4.1.1**（预检判读顺序）、§5.1 三条基线统一口径；`README.md` 状态表 + 已知限制第 8 条；本节后插入本节 |

**给 A 的两条**：
① Stage 1 的 Sim_prot 如果只在微调空间里报，**答辩会被追问**（同一份权重的 upstream 口径是 0.70）
⇒ 建议两个口径一起报，并把 `upstream_speaker_top1 = 0.229` 当作"还需要继续训/换声码器"的依据；
② checkpoint 里有 `speaker_encoder` 段 ⇒ Stage 2 的 δ 生成空间照旧用 `stage1`，
**这个规矩别动**（用了会让 G 的输入离分布、防御静默失效，§0.14 第 7 条）。

---

### 0.18 ★★ 本轮进展（成员 B · 2026-11-25 下半场：**不开台落地 + §5.1 三条基线实测 + α 曲线补齐**）

**1) 决定已落地**：200k 步云机**不开**（§0.17 的三条闸门一条没满足），
runbook §4.2 保留命令与闸门作为"为什么不花这笔钱"的证据链，§8 预算注明不动。

**2) α–Sim_clone 曲线补成正式版**（每档 **40 步**、等预算 `‖δ‖=1`，
`stage2/diagnose_alpha_sweep.py`，JSON 在 `.tmpB/alpha_sweep_40.json`）：

| α | 角度 | 优化后 Sim_clone | 学到 | Sim_prot@upstream | mel  MSE |
|---|---|---|---|---|---|
| 0.1 | 5.71° | 0.4978 | 0.037 | 0.6363 | 0.00785 |
| 0.3 | 16.70° | 0.4769 | 0.099 | 0.5656 | 0.01054 |
| 0.5 | 26.57° | 0.4316 | 0.116 | 0.5250 | 0.01439 |
| 1.0 | 45.00° | **0.4346** | 0.029 | **0.4999** | **0.01950** |

⇒ 指南 §11 的"α 敏感吗"有了定量答案：**敏感但极不划算**（预算 ×10 只换 Sim_clone −0.06，
Sim_prot 陪葬 −0.14、mel 失真 ×2.5），且 α=1.0 时随机 δ 已经先到 0.4639
⇒ 大 α 的收益来自"扰动本身"，不是"学会往哪偏"。**这也反过来支持论文取 α=0.1**。

**3) §5.1 三条基线跑通并出数**（100 条、同一批语音、确定性、可配对，各 377 s）：

| 变体 | Sim_prot | Sim_clone | DSR(0.3/0.4/0.5) |
|---|---|---|---|
| ① 无防御 | 0.6698 | 0.6169 | 0% / 0% / **9.0%** |
| ② 高斯噪声 | 0.6650 | 0.6163 | 0% / 0% / 12.0% |
| ③ 150 步短训 G | **0.7029** | **0.6775** | 0% / 0% / **1.0%** |

→ **两个新发现**（都写进 `docs/experiments.md` §4h 与 §5 坑表）：
① **DSR 有链路地板**：`DSR(τ)=mean(Sim_clone<τ)` 把重建失真也算成功 ⇒ 无防御就有 9.0%，
**报数必须并列①、可比量是 ΔDSR**；
② **短训 G 方向是反的**：Sim_prot +0.033 但 Sim_clone +0.061 ⇒ δ 在**帮**重建链守身份，
说明保真项（λ_per=15 / λ_GAN=1 / λ_mask）在多目标里压过了 `L_de`（λ_de=5）。

**4) 修掉一个可复现性 bug**：三个 `stage2/diagnose_*.py` 原来只 `torch.manual_seed()`，
而 `stage2/dataset.py::_crop()` 用的是 **Python 的 `random`** ⇒ 跨运行的单点数字差 **0.045**
（同一 α 的"随机 δ"两发 0.5794 vs 0.5348）。已统一改用 `stage2.common.set_seed()`。
`evaluate.py` 不受影响（整条读取、不裁剪，随机 δ 显式播种）。

**5) 本轮提交**（本地 4 个 commit，**尚未 push**，push 前需组长确认权限）：
解耦 `--de-space` 与口径落盘 / 不开台决定与判据 / `set_seed` 修复 / α 正式版 + §4h 三条基线。

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
| D1-1 | A | 克隆 AutoVC 官方仓库 → `third_party/AutoVC`，记下 **commit hash** | — | `third_party/AutoVC/` | ✅ **已完成** `c1827a6`，仓库见 `third_party/README.md` |
| D1-2 | A | 读它的 `hparams.py`，把 **mel 参数**抄进 `docs/notes/` | D1-1 | `docs/notes/autovc_alignment.md` | ✅ **已完成**，实测参数已记录 |
| D1-3 | B | 克隆 `CorentinJ/Real-Time-Voice-Cloning` → `third_party/RTC`，记 commit | — | `third_party/RTC/` | ✅ **已完成** commit `890f3a0`；⚠️ 权重改从 HF 下 `encoder.pt` |
| D1-4 | B | 下载 VCTK（Edinburgh DataShare）→ `data/raw/`，解压 | — | `data/raw/` | ✅ **已完成** 1.36 GiB / 11,000 条 / 110 说话人 |
| D1-5 | B | 勘察 VCTK 目录：**统计说话人数、每个说话人的句子数、有多少 `p*_eval` 目录** | D1-4 | `docs/notes/vctk_layout.md` | ✅ **已完成** 110 人（含 `s5`）；本包无 `p*_eval` |
| D1-6 | **共同** | 建 `third_party/README.md`：每个组件的 **URL + commit + 许可证 + 用途** | D1-1, D1-3 | `third_party/README.md` | ✅ **已完成**（待 D1-3 后补 RTC 条目） |
| **D1-7** | **A** | **下载 AutoVC 官方权重**（IBM Box，主模型 + WaveNet 声码器）；`3000000-BL.ckpt` 也要 | D1-1 | `algorithms/voiceprint_protection/weights/` | 权重文件就位，**不提交到 git**（`.gitignore` 已忽略 `*.pt`） |

> ⚠️ **D1-5 的坑（指南第 3 步原文）**：VCTK 的说话人文件夹名是 `p225`、`p292` 这种
> （对应论文 t-SNE 图里的 p225/p252/p292/p315/p318）；
> **不要把 `p*_eval` 的噪声版本混进训练集** —— 那是被刻意加噪的评测版本。

#### W2-D2 · ★ mel 数值对照（本周最重要的一天）

> **指南第 4 步的"坑"原文**：*AutoVC 预训练权重的 mel 配置与你用的不一致 → 输出是噪音。
> 必须先读它的 `hparams` / `config` 并对齐。*
> **C2 是 W2 的命门。** 不做，AutoVC 权重加载后输出是噪音，你会误判成"权重坏了"，白查一天。

| # | 负责人 | 任务 | 🔗 前置 | 📦 产出 | 验收标准 |
|---|---|---|---|---|---|
| D2-1 | A | **AutoVC mel 对照**：用官方 `make_spect.py` 原始代码 与 我们的 `audio_utils.autovc_mel()` 比较 | D1-2 | ✅ `scripts/verify_autovc_mel.py` | **最大绝对差 < 1e-5**；脚本已就绪，**待运行** |
| D2-2 | A | **HiFi-GAN mel 对照**：组件 mel 处理 与 我们的 `vocoder_mel()` | D1-1 | 对照脚本 + 结论 | 同上；确认 `(x+4.5)/5` 归一化一致 |
| D2-3 | A | **验证 16 kHz 声码器**：`torch.hub.load('NVIDIA/torch.hifi_gan', 'hifigan_melbins_16k')` 能否加载并合成 | — | 合成样例 wav | 能出声；**若可用则 16 kHz 全路线无障碍** |
| D2-4 | B | **GE2E 输入约定确认**：采样率？内部 mel 几维、什么参数？输出 `h` 是否已 L2 归一化？ | D1-3 | `docs/notes/ge2e_interface.md` | 四个问题都有明确答案 + 代码位置引用✅ **已完成并收口**（16 kHz / 40 维线性 mel / 已 L2 归一化，见 ge2e_interface.md） |
| D2-5 | B | **写 `scripts/prepare_data.py`**（详见下方 §1.3 规格） | D1-5 | `scripts/prepare_data.py` | ✅ **已完成** 三份清单 + `spk2id.txt`（K=80）已产出 |
| D2-6 | **共同** | 把对照结论写进 `audio_utils.py` 注释 + `docs/notes/autovc_alignment.md` | D2-1, D2-2, D2-4 | 两个文件 | ✅ 主体已完成，D2-1/D2-2 通过后补最终结论 |

#### W2-D3 · 写桥接模块与接口签名

| # | 负责人 | 任务 | 🔗 前置 | 📦 产出 | 验收标准 |
|---|---|---|---|---|---|
| D3-1 | **共同** | **定接口签名**，写进 `docs/interface-contract.md`（指南第 8 章那 10 条 + 下列函数签名） | D2-6 | `docs/interface-contract.md` | 双方确认，后续改动须双方同意 |
| D3-2 | A | `bridge_autovc.py`：加载 AutoVC 权重，提供 `content_encoder(x,h)->c` 与 `decoder(c,h)->mel` | D2-1, D3-1 | `bridge_autovc.py` | 输入 `(T,80)`，输出 `(T',32)` 与 `(T,80)` |
| D3-3 | A | `bridge_vocoder.py`：加载 HiFi-GAN，提供 `mel->wav` | D2-2, D3-1 | `bridge_vocoder.py` | 输出 wav 采样率与预设一致 |
| D3-4 | B | `bridge_speaker.py`：加载 GE2E，提供 `wav->h(256)` | D2-4, D3-1 | `bridge_speaker.py` | ✅ **已跑通** `(256,)`，`|h|=1.000000` |
| D3-5 | B | `scripts/prepare_data.py` 补完 + **核对划分** | D2-5 | `data/*.txt` | ✅ **三条硬要求自检全通过**（不相交 / K=80 / 无 p*_eval） |

#### W2-D4 · 串起推理主流程

| # | 负责人 | 任务 | 🔗 前置 | 📦 产出 | 验收标准 |
|---|---|---|---|---|---|
| D4-1 | A | 写 `infer.py` 主流程（`wav → x → h → h_adv → c → x_adv → wav'`），**每步打印张量形状** | D3-2, D3-3, D3-4 | `infer.py` | 打印出 `x (T,80)` / `h (256,)` / `c (T',32)` / `x_adv (T,80)` |
| D4-2 | A | **先加随机/高斯扰动**（α=0.1），**不训练 G** —— L1 阶段本来就不训练 | D4-1 | `infer.py` | 扰动可复现（固定 seed） |
| D4-3 | B | 写 `scripts/smoke_test.py`：喂一段真实语音 → 出 `protected.wav` + 打印形状 | D4-1 | `scripts/smoke_test.py` | ✅ 已写；⭐ 与 A 的 smoke_test 合并为两版（重建/保护分两步测）；待 A 的权重就位后跑通 |
| D4-4 | B | 准备 1 段**公开可用**的测试音频（平台样例输入），确认许可 | D1-4 | `samples/` 下的音频 | ✅ **输入已生成**（测试集说话人 p228 + CC BY 4.0 署名）；输出待 `infer.py` |
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
| P4 | **B 主写** | 准备样例：`public/samples/<方法ID>/` 输入 + 输出（**必须是真实跑出来的**） | W2-D5 | 样例目录 | ✅ **已完成**（2026-11-18 真实产出：`input.wav` + `output.wav` + 两个克隆对比音） | **B 主写** | 准备样例：`public/samples/<方法ID>/` 输入 + 输出（**必须是真实跑出来的**） | W2-D5 | 样例目录 | 输入输出确实对应，参数已记录 |
| P5 | **共同** | 平台端到端：上传 → 处理 → 预览 → 下载 | P1–P4 | — | ✅ **已完成**（2026-11-20，成员 B；`scripts/test_platform_e2e.py` **29 项全通过**，见 experiments.md §3h） | **共同** | 平台端到端：上传 → 处理 → 预览 → 下载 | P1–P4 | — | 全部成功 |
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

### 4b. ★ W4–W5 Stage 2 的产出清单（成员 B 主责，2026-11-18 起）

| 文件 | 内容 | 状态 |
|---|---|---|
| `stage2/generator.py` | 生成器 G：h (256) → δ (256)，**单位长度** | ✅ |
| `stage2/discriminator.py` | 判别器 D_adv：mel → 真假分数（**LSGAN，无 sigmoid**） | ✅ |
| `stage2/losses.py` | L_mel / L_mask / L_per / L_GAN / L_D / L_de / L_G（论文式 7–13） | ✅ |
| `stage2/psychoacoustics.py` | 心理声学掩蔽阈值 θ_x（MPEG-1 模型） | ✅ |
| `stage2/models.py` | **可求导**的冻结组件（torch 版 GE2E / AutoVC Stage 1 / 替身） | ✅ |
| `stage2/pipeline.py` | mel ↔ 波形、阈值、对齐、声码器适配 | ✅ |
| `stage2/dataset.py` | VCTK 数据集（按清单读、随机裁） | ✅ |
| `stage2/train.py` | 训练主循环 | ✅ |
| `stage2/verify_gradients.py` | **梯度通路自检**（12 项，全通过） | ✅ |
| `stage2/verify_mel.py` | torch GE2E ↔ 上游 numpy 数值对照（4 项，全通过） | ✅ |
| `stage2/verify_psychoacoustics.py` | 心理声学掩蔽自检（20 项，全通过） | ✅ |
| `algorithms/voiceprint_protection/evaluate.py` | Sim_prot / Sim_clone / DSR / WER + 高斯噪声基线 + 消融入口 | ✅ |
| `docs/notes/stage2_blockers.md` | **阻塞项清单**（可求导声码器 / Stage 1 checkpoint 等） | ✅ |

> **正式训练还差两样**（都不是代码问题）：
> ① 一个**可求导**的声码器；② 成员 A 的 Stage 1 checkpoint。
> 详见 [`docs/notes/stage2_blockers.md`](../../docs/notes/stage2_blockers.md)。

> ⚠️ **A 需要 review 的一点**：`stage2/losses.py::defense_loss` 是**最小化**余弦相似度
> （论文式 12 的 "Minimizing L_de"）。方向反了训练会把音频毁掉 ——
> `stage2/verify_gradients.py` 会打印 `L_de`，**它应该下降**。

> ★ **2026-11-19 更新（成员 A）**：阻塞点 ① **已解掉** ——
> 找到了 16 kHz 的 HiFi-GAN 权重（`speechbrain/tts-hifigan-libritts-16kHz`），
> `bridge_vocoder.VocoderBridge` 换成 torch 版 HiFi-GAN 后**本身就可求导**，
> `stage2/pipeline.py::HiFiGANVocoder` 可以删掉。
> 详见 §0.10.4c。阻塞点 ②（Stage 1 checkpoint）**仍差数据**，见 §0.10.5。

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

## 8. 云端算力预算（华为云体验代金券 500 元）

> **已知**：华为云体验代金券 500 元。
> **未知（需在控制台确认）**：① 适用产品是否包含 ECS（NVIDIA 卡）还是只有 ModelArts（昇腾）
> ② 有效期多长 ③ 能否用于按需/竞价实例。
>
> ✅ **2026-11-25 成员 B 的决定：Stage 2 的 200k 步**不开台**，本预算暂时不动。**
> 依据是等预算自由 δ oracle 把 `L_de` 的地板钉在 **0.42–0.50**（论文 0.13），
> 加上 §4h 三条基线显示短训 G 的**方向是反的** ⇒ 加步数补不上这 0.3（判据见 §0.17 与 runbook §4.2）。
> 如果将来要重新开台，**优先把钱花在"修放大链"而不是"多跑步数"**
> （例如重训 Stage 1 把 `upstream_speaker_top1 rebuilt` 从 **0.229** 抬上去）。

### 8.1 500 元够不够？够，但必须精打细算

本地 RTX 4060（8 GB）能跑**缩减版**训练，但要跑论文量级（Stage 1 500k + Stage 2 200k 迭代）
必须有云上 GPU。500 元大约能覆盖 **1–2 次完整训练**，前提是：

| 规则 | 原因 |
|---|---|
| **只在真正要训练时才开机** | 开发、调试、评测全在本地（推理单条 <1s，CPU 也够） |
| **训练完立刻释放实例** | 按需实例"关机"仍收存储费，"释放"才彻底停收 |
| ~~**优先用竞价实例**~~ → **★ 改：不要用竞价** | **2026-11-21 实测（华为云官方《ECS · 计费说明》）：「代金券不适用于竞价计费型实例。」** 我们只有代金券，竞价实例用不上券（钱从余额扣）⇒ 只能开**按需** |
| **先在小卡上跑通流程** | 别一上来就开 V100 烧钱（`pi2.2xlarge.4` T4 先跑通，`p2s.2xlarge.8` V100 再长训） |
| **VCTK 下载在本地做** | 数据 11 GB，别在云机上慢慢下（**实测从本机 tar + scp 更稳**，见 `docs/notes/cloud_ecs_stage2.md`） |

### 8.2 花钱顺序（按这个来，别乱开）

| 阶段 | 是否需要云 | 说明 |
|---|---|---|
| W2 L1 推理链 | ❌ | 本地 CPU/GPU 都够 |
| W3 平台打通 | ❌ | 推理服务本地跑 |
| W4 Stage 1 训练 | ✅ **第一次开云** | 先用 5k 迭代验证能收敛（几分钟），确认无误再放长训 |
| W5 Stage 2 训练 | ✅ | Stage 1 冻结后显存压力小，可与 Stage 1 用同一台 |
| W6 评测 | ❌ | 本地跑 |
| **云上常驻推理服务** | ❌ **不要** | 本地起服务即可（除非演示需要外网访问） |

### 8.3 建议的云上配置

- **实例**：先开中等规格（能放下 AutoVC + 8 GB 显存即可），跑通再换大的
- **镜像**：**选 PyTorch 预置镜像**（CUDA 版 torch 开箱即用，**零下载**，省掉 2GB 的安装折腾）
- **代码同步**：`git push` 到 GitHub → 云机 `git clone`（顺便保证两人同一份代码）
- **数据**：VCTK 在云机上重新下载，或先在本地下载后传 OBS
- **断线保护**：**必须用 `tmux`**（SSH 断了训练不能停）
- ★ **一键脚本**：`scripts/cloud_train_stage1.sh` —— `--setup`（配环境 + 下数据 + 下权重）/ `--start`（tmux 里开跑）/ `--status` / `--stop`。
  它把**「配环境」和「烧钱训练」分成两条命令**：云机按小时计费，别一开机就跑满；先 `--setup` + `check_env.py` 自检，确认无误再 `--start`
- **结果回收**：训练完把 checkpoint + 指标 `scp` 回本地，再释放实例

### 8.4 如果券只能开昇腾（ModelArts）

⚠️ 那就要额外评估迁移成本：**本项目是纯 CUDA 栈**（torch cu121 / torch.hub 拉 HiFi-GAN /
AutoVC 权重），迁到昇腾要装 CANN + `torch_npu`，并把 `.cuda()` 改成 `.npu`，
而 AutoVC / GE2E / HiFi-GAN **都没有人在昇腾上验证过**，出问题难定位。

**对策**：这时优先考虑用券的**其他用途**（如 ECS 的 NVIDIA 卡、或折算成别的服务），
实在不行就走"本地 4060 缩减规模训练 + 诚实写明配置差距"的路线 ——
**平台能展示 + 有一组真实指标 > 训练规模做全**。

---

## 9. 待决事项（需要组内拍板）

- [x] ~~**采样率方案**~~ → **已定：全程 16 kHz**（AutoVC 官方权重就是 16k）
- [x] ~~**L1 走哪条路线**~~ → **已定：路线 B**（上游 RTC 全套），依据见 §0.8
      —— AutoVC 官方权重链实测 404 失效，路线 A 走不通
- [ ] **代金券适用产品**（控制台确认）：能否开 ECS（NVIDIA 卡）？还是只有昇腾 ModelArts？
      → 决定 W4–W5 的训练路线，**W3 结束前必须有答案**
- [ ] **代金券有效期** → 训练排期必须卡在有效期内
- [ ] **采样率方案**：~~L1 用 22.05k~~ → **已定：全程 16 kHz**（AutoVC 官方权重就是 16k）
- [ ] **说话人编码器选型**：
      - L1 用 AutoVC 自带 `D_VECTOR`（80 维输入，与官方权重配套）
      - 最终系统用论文脚注的 **GE2E / RTC**（40 维输入）
      → **建议就按此执行，待组内确认**
- [ ] **组号与选题名称** → 确定后统一改方法 ID、目录名、README 标题
- [ ] **MOS 听测人数**：论文是 50 人，我们做多少？（如实写明即可）
- [ ] **W4–W5 是否也交换主责**（W6 起交换是必须的）
