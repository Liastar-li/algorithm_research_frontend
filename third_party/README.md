# 第三方组件来源记录

> 用于满足 `docs/算法提交指南.md` 与 `docs/接入自查清单.md` 对"第三方项目的来源与许可证应保留"的要求。
>
> **规矩**：本项目**不使用论文作者的代码**（论文未提供官方代码链接），
> 而是基于公开组件自行搭建。因此**每一个外部组件都必须在这里登记**：
> 来源、版本（commit）、许可证、用途。

---

## 一键搭建（两人 / 每台机器都跑这一条）

`third_party/` **不提交进仓库**（那里是别人的官方代码和许可证，混进我们的成果仓库不合适）。
代价是每个人要自己拉一次 —— 所以用脚本把版本钉死，保证两人拿到**完全一致**的代码。

**Windows（PowerShell）：**
```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_third_party.ps1
```

**Linux / macOS（云端机器用这个）：**
```bash
bash scripts/setup_third_party.sh
```

脚本会：按固定 commit 浅拉取三个仓库 → 自动生成 `wavenet_vocoder/version.py` → 打印状态。
**幂等**：已是目标版本会跳过；版本不符会删除重拉。

> 组件清单的唯一真源在 `algorithms/voiceprint_protection/config.py` 的
> `THIRD_PARTY_COMPONENTS`，脚本里的表与之保持一致。

---

## 组件清单

| 组件 | 来源 | 固定 commit | 许可证 | 用途 |
|---|---|---|---|---|
| **AutoVC** | <https://github.com/auspicious3000/autovc> | `c1827a60e83e6d26030b8953843b2f7cf2bd36db`（2024-10-23） | MIT | Stage 1 的模型架构（`Ec`/`D`/PostNet）+ 官方预训练权重 + 配套声码器封装 |
| **wavenet_vocoder** | <https://github.com/r9y9/wavenet_vocoder> | `d4333546eccd3b681b7092e4f52160ad7e62dcc9`（2019-08-25） | MIT | WaveNet 声码器（与 AutoVC 的 mel 格式配套） |
| **Real-Time-Voice-Cloning** | <https://github.com/CorentinJ/Real-Time-Voice-Cloning>（国内镜像 <https://gitee.com/mirrors/real-time-voice-cloning>） | **`890f3a03187195b9829db2079b75c2ba2ab0405c`**（2026-03-09）✅ 已核实 | MIT | GE2E 说话人编码器 `E_spk`（**论文脚注 2 直接引用**），最终系统使用 |
| **VCTK Corpus**（数据集，非代码） | <https://datashare.ed.ac.uk/handle/10283/3443> | version 0.92 | **CC BY 4.0** | 训练 / 验证 / 测试语音（80 / 10 / 20 人） |

### 三条必须知道的版本约束

1. **AutoVC 的仓库地址变了**
   论文引用的是 "AutoVC [14] Qian et al., ICML 2019"。
   原地址 `cyqxiaozhen/AutoVC` **已失效（404）**，官方现址是 `auspicious3000/autovc`。

2. **wavenet_vocoder 必须固定在 `d4333546`**
   这是**最后一个还含 `builder.py` 的提交**。AutoVC 的 `synthesis.py` 依赖
   `from wavenet_vocoder import builder`，而 2019-08-26 之后的新版删掉了它
   且模型结构有变 → 权重加载失败。
   另外该 commit 的 `wavenet_vocoder/version.py` 在 git 里**不存在**
   （那个年代由 `setup.py` 在打包时自动生成），脚本会按官方
   `build_py.create_version_file()` 的方式补上。

3. **⚠️ RTC 在 2026-03 做过一次重构（成员 B 于 2026-11-17 核实）**
   | | 旧版（2026-03 之前） | 新版（**当前上游 master**） |
   |---|---|---|
   | 目录 | 平铺 `speaker_encoder.py` / `voice_encoder.py` / `preprocess.py` | 包结构 `encoder/`（`model.py` / `inference.py` / `audio.py` / `params_data.py`） |
   | **权重** | 仓库内 `pretrained.pt`（~30 MB） | **不再随仓库分发** → HuggingFace `CorentinJ/SV2TTS` 的 **`encoder.pt`（17,090,379 字节）** |
   | **内部 mel** | log + 均值方差归一化 | **线性 mel，不取 log** ← 拿旧版 mel 喂新版权重会**嵌入全错但不报错** |
   | 依赖 | librosa | 额外需要 **scikit-learn** |

   下载新版权重：
   ```bash
   python algorithms/voiceprint_protection/scripts/download_encoder_weights.py
   ```
   详细接口约定与踩过的坑见 [`docs/notes/ge2e_interface.md`](../docs/notes/ge2e_interface.md)。

---

## 权重（不提交到 git）

| 权重 | 用途 | 下载地址 |
|---|---|---|
| `autovc.ckpt` | 主模型（Generator + 说话人嵌入） | `https://ibm.box.com/s/upmfbstk8n8ded0xd3aaic4ayclh90ee` |
| （另两个 AutoVC 权重） | 备用 | `https://ibm.box.com/s/k4nyvzkz26r9ux1o8rcq9tov36h69o2e`<br>`https://ibm.box.com/s/pdr8b0z5oim9q0p4be496mt3ue5lb004` |
| `3000000-BL.ckpt` | 说话人编码器 `D_VECTOR`（**80 维 mel** 输入，AutoVC 自带） | AutoVC 官方 README 的 IBM Box 链接 |
| `checkpoint_step001000000_ema.pth` | WaveNet 声码器 | `https://ibm.box.com/s/asvv554v0zd09yipl2qadz49i7jpdhta` |
| **`encoder.pt`** | **GE2E 说话人编码器（40 维 mel 输入，论文脚注引用）** | HuggingFace <https://huggingface.co/CorentinJ/SV2TTS> → `encoder.pt`（17,090,379 字节）<br>国内：`set HF_ENDPOINT=https://hf-mirror.com` |

> ⚠️ **两套说话人编码器权重不可混用**：输入 mel 维度不同（80 vs 40），结构也不同。
> AutoVC 的 `D_VECTOR` 吃 80 维 mel；论文的 GE2E 吃 40 维线性 mel。

**放到：**
- AutoVC 三件套 → `algorithms/voiceprint_protection/weights/`
- `encoder.pt` → `<数据根>/models/default/`（由 `scripts/download_encoder_weights.py` 自动放好；
  也可用 `VPP_MODELS_DIR` 指定到别的盘）

（`.gitignore` 已忽略 `*.pt` / `*.pth`，不会被误提交）

**网络实测（2026-11-13 / 11-17）**：`github.com` ✅、`gitee.com` 镜像 ✅、`hf-mirror.com` ✅、
`modelscope.cn` ✅、`huggingface.co` ❌（慢/超时）、`drive.google.com` ❌（需代理或镜像）、
`datashare.ed.ac.uk` ✅（但**单连接只有 ~30 KB/s**，见 `docs/notes/download_vctk.md`）。

---

## 数据集

| 项 | 值 |
|---|---|
| 数据集 | CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit（version 0.92） |
| **DOI（权威标识）** | **<https://doi.org/10.7488/ds/2645>** → <https://datashare.ed.ac.uk/handle/10283/3443> |
| 直接下载 | <https://datashare.ed.ac.uk/server/api/core/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/content>（`VCTK-Corpus-0.92.zip`，11,747,302,977 字节） |
| 许可 | **CC BY 4.0**（数据自带 `VCTK.txt`）→ **对外发布样例时必须署名来源** |
| 说话人数 | **110**（109 个 `pXXX` + 1 个 `s5`），与论文一致 |
| 下载方式 | `node scripts/download_vctk.mjs --out <数据根>/raw --per-speaker 100`（见 `docs/notes/download_vctk.md`） |

> ⚠️ **引用勘误**：指南里写的 `Zenodo. https://doi.org/10.5281/zenodo.3235654` **实测 404**
> —— Zenodo 上没有 VCTK 的主记录（而且那条其实是 AutoVC 的论文）。
> **要引用 VCTK 请用 DataShare 的 DOI `10.7488/ds/2645`。**

---

## 许可证核实

克隆后逐个确认（脚本）：

```bash
for d in third_party/*/; do
  echo "=== $d ==="
  ls "$d" | grep -iE "^(licen[sc]e|copying|notice)" || echo "  (无许可证文件)"
done
```

| 组件 | 许可证 | 已核实 |
|---|---|---|
| AutoVC | MIT | ⬜ |
| wavenet_vocoder | MIT | ⬜ |
| Real-Time-Voice-Cloning | MIT | ✅（仓库内 `LICENSE` 头部已确认） |
| VCTK | CC BY 4.0 | ✅（数据自带 `VCTK.txt` 已确认） |

---

## 使用约定

1. **不提交权重文件**到仓库（`.gitignore` 已忽略 `*.pt`/`*.pth`/`*.safetensors`）。
2. README 中必须写明**每个组件的仓库地址 + 版本/commit + 许可证**。
3. 若最终使用某个第三方项目的代码，需保留其原始版权声明与许可证。
4. 权重下载方式（链接 / 命令）写进 README，**不写具体凭据**。
5. **成员 B 不需要 A 的 third_party**：B 自己跑一遍上面的脚本即可，
   且 B 需要的全部"信息"（mel 约定、张量形状、实测记录）都在本仓库：
   - `algorithms/voiceprint_protection/config.py` —— 超参与契约的唯一真源
   - `algorithms/voiceprint_protection/audio_utils.py` —— mel 实现
   - `algorithms/voiceprint_protection/docs/notes/autovc_alignment.md` —— AutoVC 侧的实测记录（成员 A）
   - `algorithms/voiceprint_protection/docs/notes/ge2e_interface.md` —— GE2E 侧的实测记录（成员 B）
