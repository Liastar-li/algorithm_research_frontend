# 数据放在哪个盘（成员 B · 实操说明）

> 起因：C 盘空间不够（95% 已用，剩 21 GB），而 VCTK 压缩包 + 解压 + 16 kHz 副本
> 合计要 **约 27 GB**，C 盘放不下。

---

## 0. 一句话结论

> **仓库留在 C 盘，数据放 D 盘。**
>
> 仓库本身只有 **77 MB**（`.git` 1.2 MB + `node_modules` 69 MB + 其余 ~2 MB），
> 搬它省不了多少空间；而真正占地方的 VCTK 有 27 GB，必须挪走。
>
> 代码已经改好：**设一个环境变量 `VPP_DATA_ROOT` 就行**，脚本会自动跟着走。

---

## 1. 方案 A：环境变量（推荐）

### 1.1 在 D 盘建目录

```powershell
mkdir D:\vpp
mkdir D:\vpp\data      # 数据根：raw/ + audio16k/ + train.txt ...
mkdir D:\vpp\third_party
```

### 1.2 设置环境变量

```powershell
# 永久生效（新开的终端都能用）
setx VPP_DATA_ROOT "D:\vpp\data"

# 只在当前窗口生效（调试用）
$env:VPP_DATA_ROOT = "D:\vpp\data"
```

> Git Bash 里写法一样：
> ```bash
> export VPP_DATA_ROOT="D:/vpp/data"
> ```

### 1.3 克隆第三方组件到 D 盘

```bash
git clone https://github.com/CorentinJ/Real-Time-Voice-Cloning.git /d/vpp/third_party/RTC
export VPP_RTC_DIR="D:/vpp/third_party/RTC"     # bridge_speaker.py 认这个变量
```

### 1.4 下载 VCTK 到 D 盘

```bash
mkdir -p /d/vpp/data/raw
# 正确直链与解压方式见 download_vctk.md（老链接已 404，现在是 zip 不是 tar.gz）
aria2c -x 16 -s 16 -k 1M -c \
  -d /d/vpp/data/raw -o VCTK-Corpus-0.92.zip \
  "https://datashare.ed.ac.uk/server/api/core/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/content"
unzip -q /d/vpp/data/raw/VCTK-Corpus-0.92.zip -d /d/vpp/data/raw
```

### 1.5 跑脚本（命令一个字都不用改）

```bash
conda activate vpp
python algorithms/voiceprint_protection/scripts/survey_vctk.py
python algorithms/voiceprint_protection/scripts/prepare_data.py
```

脚本启动时会打印当前解析到的路径，先核对一下：

```text
  路径解析：
    仓库根 : C:\Users\z\Desktop\26秋\语音识别\语音识别code
    数据根 : D:\vpp\data（来自 VPP_DATA_ROOT）
    原始数据: D:\vpp\data\raw
    权重   : C:\Users\z\Desktop\26秋\语音识别\语音识别code\algorithms\voiceprint_protection\weights
```

---

## 2. 方案 B：目录联接（junction，零代码改动）

如果你不想设环境变量，可以让 `data/` 这个路径"指"到 D 盘：

```powershell
# 先把仓库里已有的 data 目录挪走或删掉（它现在是空的）
cmd //c mklink /J "C:\Users\z\Desktop\26秋\语音识别\语音识别code\data" "D:\vpp\data"
```

效果：脚本里写 `data/train.txt` 依然成立，实际落在 D 盘。
Python、git、VSCode 都会透明跟随，**不用改任何代码**。

> ⚠️ 注意事项
> - junction 只在**本地 NTFS 磁盘之间**有效；D 盘若不是 NTFS（FAT32/exFAT）请用方案 A。
> - 创建 junction 不需要管理员权限（**符号链接 `mklink /D` 才需要**）。
> - `data/` 已在 `.gitignore` 里，git 不会跟着联接把 27 GB 提交进去。
> - 想撤销：`cmd //c rmdir "C:\...\data"`（只删链接，不删 D 盘的数据）。

---

## 3. 各脚本怎么用

| 脚本 | 默认数据位置 | 怎么改 |
|---|---|---|
| `scripts/survey_vctk.py` | `<数据根>/raw` | 设 `VPP_DATA_ROOT` |
| `scripts/prepare_data.py` | `--raw-root <数据根>/raw`、`--out-root <数据根>` | 设 `VPP_DATA_ROOT` 或直接传参 |
| `scripts/smoke_test.py` | `<数据根>/test.txt`、`<仓库>/public/samples/...` | 设 `VPP_DATA_ROOT` |
| `scripts/make_sample.py` | `--data-root <数据根>` | 设 `VPP_DATA_ROOT` 或传参 |
| `bridge_speaker.py` | 找 `third_party/RTC` | 设 `VPP_RTC_DIR` 或传 `--rtc-dir` |

**两个变量的作用：**

| 变量 | 作用 | 默认 |
|---|---|---|
| `VPP_DATA_ROOT` | 数据根目录（VCTK、清单、转换后的音频） | `<仓库根>/data` |
| `VPP_RTC_DIR` | Real-Time-Voice-Cloning 仓库目录 | `<仓库根>/third_party/RTC` |

---

## 4. 还有哪些东西可能占地方

| 东西 | 大小 | 放哪 |
|---|---|---|
| VCTK 压缩包 | 11 GB | **D 盘**，`data/raw/`，解压后可删 |
| VCTK 解压目录 | 11 GB | **D 盘**，`data/raw/VCTK_2/` |
| 16 kHz 转换副本 | ~5 GB | **D 盘**，`data/audio16k/` |
| `third_party/RTC` | ~100 MB | D 盘也行（`VPP_RTC_DIR`） |
| AutoVC / HiFi-GAN 权重 | ~500 MB | 仓库的 `weights/`（已 gitignore） |
| **Whisper-large-v3**（算 WER 用） | **~3 GB** | ⚠️ 建议也放 D 盘，见下 |
| `node_modules` | 69 MB | 留着，无所谓 |

### Whisper 模型放 D 盘

`faster-whisper` 默认把模型缓存在 `~/.cache/huggingface`，也就是 C 盘。
先设缓存目录再跑评测：

```powershell
setx HF_HOME "D:\vpp\hf_cache"
```

（Windows 下 `faster-whisper` 用 `HF_HOME`；老版本可能认 `HUGGINGFACE_HUB_CACHE`。）

---

## 5. 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| 脚本还是读 C 盘的数据 | 环境变量没生效（终端是设之前开的） | 重开终端，或 `echo $VPP_DATA_ROOT` 确认 |
| `mklink` 报 "拒绝访问" | 用了 `/D`（符号链接）而不是 `/J`（联接） | 改用 `/J`；或干脆用方案 A |
| junction 建了但 `data/` 变成空 | 目标目录不存在 | 先 `mkdir D:\vpp\data` |
| git 提示 data 有大量变更 | 联接没被 `.gitignore` 覆盖 | 确认 `.gitignore` 里有 `data/` |
| 转换特别慢 | 数据在 D 盘是机械硬盘/网络盘 | `--workers 8` 试试；或把 `data/audio16k` 放 SSD |
| 两个人环境不一致 | 一个人设了 `VPP_DATA_ROOT` 另一个没设 | **约定统一**：都用 D 盘 + 同一个变量名，或都用 junction |

> 最后一条很重要：**两个人必须用同一种方案**，否则一个跑出来的清单在另一个人机器上找不到文件。
> 建议：组内统一用**方案 A**，并在群里贴一次 `paths.describe()` 的输出作为凭证。
