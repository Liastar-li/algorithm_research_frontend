# 第三方组件登记（任务板 W2-D1-6 · 共同任务）

> **规矩**：本项目**不使用论文作者的代码**（论文未提供官方代码链接），
> 而是基于公开组件自行搭建。因此**每一个外部组件都必须在这里登记**：
> 来源、版本（commit）、许可证、用途。
>
> README 的"baseline 来源与许可证"一节直接引用本文件。

---

## 1. 组件清单

| # | 组件 | 用途 | 仓库 / 地址 | 负责人 | commit | 许可证 |
|---|---|---|---|---|---|---|
| 1 | **AutoVC** | Stage 1 的内容编码器 `Ec` / 解码器 `D` / PostNet | <https://github.com/cyqxiaozhen/AutoVC> | 成员 A | ⬜ 待填 | ⬜ 待核实 |
| 2 | **Real-Time-Voice-Cloning (GE2E)** | 说话人编码器 `E_spk`，产出 256 维嵌入 `h` | <https://github.com/CorentinJ/Real-Time-Voice-Cloning> | 成员 B | ⬜ 待填 | MIT（待核实） |
| 3 | **HiFi-GAN** | 声码器 `V(·)`：mel → 波形 | <https://github.com/jaywalnut310/hifi-gan> ／ <https://github.com/NVIDIA/torch.hgan> | 成员 A | ⬜ 待填 | BSD-3-Clause（待核实） |
| 4 | **VCTK Corpus** | 训练 / 验证 / 测试语音（80 / 10 / 20 人） | <https://datashare.ed.ac.uk/handle/10283/3191> | 成员 B | 2020 版（Zenodo 3537857） | CC BY 4.0 / ODC-By（需同意条款） |

---

## 2. 组件详情

### 2.1 AutoVC（成员 A）

| 项 | 内容 |
|---|---|
| 仓库 | <https://github.com/cyqxiaozhen/AutoVC> |
| 论文引用 | Qian et al., "Towards Voice Conversion with ...", ICML 2019（论文参考文献 [14]） |
| 我们用它做什么 | Stage 1 的网络结构参考 + **预训练权重**（内容编码器 `Ec` / 解码器 `D` / PostNet） |
| 关键文件 | `hparams.py`（mel 参数真源）、`wav_encoder.py`、`decoder.py`、`models.py` |
| 注意 | 其原生采样率是 **22050 Hz / 80 mel / hop 256 / n_fft 1024 / fmin 40 / fmax 7600**，与我们的 `autovc22k` 预设一致 |
| 记录 | commit：`⬜ 待填` |

```bash
git clone https://github.com/cyqxiaozhen/AutoVC.git third_party/AutoVC
git -C third_party/AutoVC rev-parse HEAD
```

### 2.2 Real-Time-Voice-Cloning / GE2E（成员 B）

| 项 | 内容 |
|---|---|
| 仓库 | <https://github.com/CorentinJ/Real-Time-Voice-Cloning>（国内可用 gitee 镜像 <https://gitee.com/mirrors/real-time-voice-cloning>） |
| 论文引用 | **论文脚注 2 直接引用的仓库**（GE2E，Wan et al., ICASSP 2018） |
| 我们用它做什么 | 说话人编码器 `E_spk`：wav → `h` (256,) |
| **commit** | **`890f3a03187195b9829db2079b75c2ba2ab0405c`**（2026-03-09，MIT）✅ 已核实 |
| 目录结构 | ⚠️ **2026-03 起重构过**：现在是包结构 `encoder/`（`model.py` / `inference.py` / `audio.py` / `params_data.py` / `params_model.py`），不再是平铺的 `speaker_encoder.py` |
| **权重来源** | ⚠️ **不再随仓库分发**。从 HuggingFace 下载：<https://huggingface.co/CorentinJ/SV2TTS> 的 `encoder.pt`（**17,090,379 字节**） |
| 权重下载 | `python algorithms/voiceprint_protection/scripts/download_encoder_weights.py` |
| 必需依赖 | `scikit-learn==1.5.2`（上游 `encoder/model.py` import 时就要） |
| 可选依赖 | `webrtcvad==2.0.10`（裁长静音；缺失不影响运行） |
| 记录 | 国内镜像 clone 的 commit 与官方一致，可直接用上面这个 hash |

```bash
git clone --depth 1 https://gitee.com/mirrors/real-time-voice-cloning.git third_party/RTC
git -C third_party/RTC rev-parse HEAD        # 应为 890f3a03187195b9829db2079b75c2ba2ab0405c
```

> ⚠️ **clone 完目录里那个 `pretrained.pt` 是假的** —— 它是**下载失败留下的 HTML 错误页**
> （上游新版不再分发权重，有些脚本会尝试下载并把错误页存成 .pt）。
> **直接删掉它**，用上面的脚本下真正的 `encoder.pt`。
>
> 接口细节（含新旧版差异）见 [`docs/notes/ge2e_interface.md`](../docs/notes/ge2e_interface.md)。

### 2.3 HiFi-GAN（成员 A）

| 项 | 内容 |
|---|---|
| 仓库 | <https://github.com/jaywalnut310/hifi-gan>（原版）／ <https://github.com/NVIDIA/torch.hgan>（torch.hub 版） |
| 论文引用 | Kim et al., ICASSP 2020（论文参考文献 [15]） |
| 我们用它做什么 | 声码器 `V(·)`：mel → 波形 |
| 关键文件 | `mel_processing.py`（**mel 约定的真源**：`fmin=0, fmax=8000, power=1, log10(·, 1e-5), (x+4.5)/5`）、`models.py`、`config.json` |
| 注意 | 官方主推 **22.05 kHz** 权重；要用 16 kHz 必须换成 16 k 权重，**不能拿 22.05 k 的权重合成 16 k 的 mel** |
| 记录 | commit：`⬜ 待填` |

```bash
git clone https://github.com/jaywalnut310/hifi-gan.git third_party/hifi-gan
git -C third_party/hifi-gan rev-parse HEAD
```

或直接用 `torch.hub.load('NVIDIA/torch.hgan', 'hifigan_melbins_16k')`（任务板 D2-3 要验证 16 k 版本能否加载）。

### 2.4 VCTK Corpus（成员 B）

| 项 | 内容 |
|---|---|
| 数据主页 | <https://datashare.ed.ac.uk/handle/10283/3191> |
| 论文引用 | 论文 §3.1 的数据设置：80 训练 / 10 验证 / 20 测试 |
| 我们用它做什么 | 全部语音数据，统一重采样到 16 kHz 单声道 16-bit PCM |
| 目录 | `wav48_silence_trimmed/`（推荐，已裁静音） / `wav/` |
| **必须排除** | `p*_eval` 目录（刻意加噪的评测版本） |
| 引用 | Veaux, C., Yamagishi, J., & MacDonald, K. (2017). *CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit* (version 0.92). The Centre for Speech Technology Research (CSTR), Edinburgh. **DOI: <https://doi.org/10.7488/ds/2645>**（2026-11-17 实测核实，指向我们下载的同一个 item，handle `10283/3443`） |
| 许可 | **需在 DataShare 页面同意条款后下载**；语音部分基于 CC BY 4.0 / ODC-By，**对外发布样例时必须署名来源** |

下载命令见 [`docs/notes/download_vctk.md`](../docs/notes/download_vctk.md)。

---

## 3. 许可证核实

克隆后逐个确认，把上表的"待核实"改成实际许可证：

```bash
for d in third_party/*/; do
  echo "=== $d ==="
  ls "$d" | grep -iE "^(licen[sc]e|copying|notice)" || echo "  (无许可证文件)"
done
```

| 组件 | 许可证 | 已核实 |
|---|---|---|
| AutoVC | ⬜ | ⬜ |
| Real-Time-Voice-Cloning | MIT | ⬜ |
| HiFi-GAN | BSD-3-Clause | ⬜ |
| VCTK | CC BY 4.0 / ODC-By | ⬜ |

> **注意**：`third_party/` 目录**不提交到 git**（体积大 + 第三方代码不宜塞进主仓库）。
> 上面记录的是 **URL + commit hash**，任何人按这两个信息都能还原出完全一样的组件。
> 这就是"记录 commit"的意义 —— 让实验可复现。

---

## 4. 使用第三方代码的规矩

1. **只 import，不修改**。要改行为就在**我们这一侧**包一层（`bridge_*.py` 就是这么设计的）。
2. **不把 `third_party/` 提交进仓库**。
3. **不在 `third_party/` 目录里执行命令**（避免误跑上游脚本）。
4. 运行本项目脚本时建议加 `python -I`，减少被污染的解释器环境。
5. 论文未提供官方代码这一事实，要写进小组 README 的"实现说明"一节。

---

## 5. 待办

- [ ] 成员 A：克隆 AutoVC + HiFi-GAN，回填 commit hash 与许可证
- [ ] 成员 B：克隆 RTC + 下载 VCTK，回填 commit hash 与许可证
- [ ] 共同：跑 `third_party/README.md` §3 的许可证核实命令
- [ ] 共同：在小组 README 里加上"未使用作者代码，基于公开组件自建"的说明
