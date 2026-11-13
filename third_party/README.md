# 第三方组件来源记录

> 用于满足 `docs/算法提交指南.md` 与 `docs/接入自查清单.md` 对"第三方项目的来源与许可证应保留"的要求。

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
| **Real-Time-Voice-Cloning** | <https://github.com/CorentinJ/Real-Time-Voice-Cloning> | 默认分支 `master` | MIT | GE2E 说话人编码器（论文脚注引用），最终系统使用 |

### 两条必须知道的版本约束

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

---

## 权重（不提交到 git）

| 权重 | 用途 | 下载地址 |
|---|---|---|
| `autovc.ckpt` | 主模型（Generator + 说话人嵌入） | `https://ibm.box.com/s/upmfbstk8n8ded0xd3aaic4ayclh90ee` |
| （另两个 AutoVC 权重） | 备用 | `https://ibm.box.com/s/k4nyvzkz26r9ux1o8rcq9tov36h69o2e`<br>`https://ibm.box.com/s/pdr8b0z5oim9q0p4be496mt3ue5lb004` |
| `3000000-BL.ckpt` | 说话人编码器 `D_VECTOR`（80 维 mel 输入） | AutoVC 官方 README 的 IBM Box 链接 |
| `checkpoint_step001000000_ema.pth` | WaveNet 声码器 | `https://ibm.box.com/s/asvv554v0zd09yipl2qadz49i7jpdhta` |
| `pretrained.pt` | GE2E 说话人编码器（40 维 mel 输入） | RTC 官方 README（Google Drive / Dropbox） |

**放到：** `algorithms/voiceprint_protection/weights/`
（`.gitignore` 已忽略 `*.pt` / `*.pth`，不会被误提交）

**网络实测（2026-11-13）**：`github.com` ✅、`hf-mirror.com` ✅、`modelscope.cn` ✅、
`huggingface.co` ❌、`drive.google.com` ❌（需代理或镜像）。
若走 HF 镜像，先设 `HF_ENDPOINT=https://hf-mirror.com`。

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
   - `algorithms/voiceprint_protection/docs/notes/autovc_alignment.md` —— 实测记录
