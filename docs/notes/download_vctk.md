# 下载数据与第三方组件（成员 B 主责 · 任务板 W2-D1-3 / W2-D1-4）

> 本文件是**可复制的命令清单**。成员 B 在自己的机器上执行一遍，
> 成功后把实际输出（commit hash / 文件大小 / 说话人数）回填到
> [`third_party/README.md`](../../third_party/README.md) 与
> [`vctk_layout.md`](vctk_layout.md)。
>
> 两个人必须**下载到同一份数据**，否则划分和指标就没法比。
>
> ⚠️ **C 盘空间不够？** VCTK 压缩包 + 解压 + 16 kHz 副本合计要 **约 27 GB**。
> 本文件里的命令默认写在 `data/raw/`；如果你要把数据放到 D 盘，
> 先看 [`data-location.md`](data-location.md)（设 `VPP_DATA_ROOT` 或用目录联接），
> 然后把下面的路径换成 `/d/vpp/data/raw/` 即可，**其余命令一字不改**。

---

## 1. 克隆 Real-Time-Voice-Cloning（含 GE2E 预训练权重）

任务板 D1-3。**论文脚注 2 直接引用的仓库**，不是我们自选的实现。

```bash
cd <仓库根目录>
mkdir -p third_party
git clone https://github.com/CorentinJ/Real-Time-Voice-Cloning.git third_party/RTC
git -C third_party/RTC rev-parse HEAD        # ← 记下这串 commit hash
```

**国内网络慢就用 gitee 镜像（内容与官方一致）：**

```bash
git clone --depth 1 https://gitee.com/mirrors/real-time-voice-cloning.git third_party/RTC
git -C third_party/RTC rev-parse HEAD        # 应为 890f3a03187195b9829db2079b75c2ba2ab0405c
```

### 1.1 ★ 权重不在仓库里，要单独下载（新版上游的改动）

⚠️ **2026-03 起上游重构，不再把预训练权重放在仓库中**，
改从 Hugging Face 下载：`CorentinJ/SV2TTS` 的 `encoder.pt`（**17,090,379 字节**）。

```bash
# 国内网络建议先设镜像
set HF_ENDPOINT=https://hf-mirror.com

conda activate vpp
python algorithms/voiceprint_protection/scripts/download_encoder_weights.py
```

脚本会**核对文件大小必须精确等于 17,090,379 字节**，
不符就报错（防止把错误页面当权重存下来）。

> ⚠️ **clone 完目录里那个 `pretrained.pt` 是假的** —— 它是下载失败留下的
> **HTML 错误页**，不是权重。直接删掉：
> ```bash
> rm third_party/RTC/pretrained.pt
> ```

> ⚠️ 第三方仓库是**外部代码**。本项目只 `import` 它的模型类，不改它、
> 不运行它的脚本（尤其别跑它的 `demo_cli.py`，它会把 synthesizer 也下下来），
> 也不要把工作目录切进 `third_party/RTC` 里执行命令。
> 运行本项目的脚本时建议加 `python -I`。
>
> ⚠️ 新版上游的 `encoder/model.py` 需要 **scikit-learn**，两人都要装：
> ```bash
> pip install scikit-learn==1.5.2
> ```

---

## 2. 下载 VCTK

任务板 D1-4。**现在不需要在网页上同意条款了** —— 直链实测可直接下载（见 2.1）。

> 历史上需要在 <https://datashare.ed.ac.uk/handle/10283/3191> 页面点 "Agree"，
> 但该 handle 在新站点上已被重新映射，**下面的 2.1 才是当前有效的地址**。

### 2.1 ★ 正确下载地址（2026-11-17 实测）

⚠️ **老链接已经失效了**。DataShare 官网已重建成 DSpace 7（Angular），
`https://datashare.ed.ac.uk/bitstream/handle/10283/3191/VCTK-2.tar.gz` 现在返回 **404**，
而且老 handle `10283/3191` 在新站点上已经被**重新映射到别的数据集**（手语数据集），不再是 VCTK。

**现在正确的地址（成员 B 于 2026-11-17 经 REST API 核实）：**

```text
数据集：CSTR VCTK Corpus: English Multi-speaker Corpus for
        CSTR Voice Cloning Toolkit (version 0.92)
文件：  VCTK-Corpus-0.92.zip
大小：  11,747,302,977 字节（约 10.9 GiB）
直链：  https://datashare.ed.ac.uk/server/api/core/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/content
```

实测这个直链**不需要登录/不需要同意条款**，并且 **`Accept-Ranges: bytes`（支持断点续传与多线程）**。

> 顺便记一下查询方法（以后 DataShare 再改版可以照着用）：
> ```bash
> # 1) 用 REST API 找 item
> curl -s -H "Accept: application/json" \
>   "https://datashare.ed.ac.uk/server/api/discover/search/objects?query=VCTK&size=10"
> # 2) 取 item 的 bundles → ORIGINAL → bitstreams → content 链接
> ```

**下载命令（先装 aria2c，多线程）：**

```cmd
:: 检查有没有装
where aria2c
:: 没装就一条命令（Windows 10/11 自带 winget）
winget install aria2.aria2
:: 装完重开终端确认
aria2c --version
```

```cmd
aria2c -x 16 -s 16 -k 1M -c ^
  -d D:\vpp\data\raw -o VCTK-Corpus-0.92.zip ^
  "https://datashare.ed.ac.uk/server/api/core/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/content"
```

**没有 aria2c 时用 curl（单线程约 5.4 小时，但支持断点续传）：**

```cmd
curl -L --retry 10 --retry-delay 5 -C - ^
  -o D:\vpp\data\raw\VCTK-Corpus-0.92.zip ^
  "https://datashare.ed.ac.uk/server/api/core/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/content"
```

> 也可以用 IDM / Motrix / 浏览器下载器手动拉，存成
> `D:\vpp\data\raw\VCTK-Corpus-0.92.zip`（**文件名和大小对上就行**，不较真）。

### 2.2 ★ 推荐：用定向下载器（比整包下载快 8 倍）

**先说实测数据**（2026-11-17，成员 B 测的）：

| 项 | 实测值 |
|---|---|
| 单连接速度 | **31.8 KB/s** |
| 16 连接（aria2c） | ~280 KB/s |
| 48 连接 | **423 KB/s** |
| 整包 10.94 GiB 按 48 连接 | **约 7.4 小时** |
| **只取每人 100 句（1.36 GiB）+ 48 连接** | **约 55 分钟** ✅ |

同时查明这个 zip 的**内部结构**（用中央目录解析得到）：

```text
VCTK-Corpus-0.92.zip  (11,747,302,977 字节)
├── wav48_silence_trimmed/   88,329 个 .flac   10.917 GiB   ← 111 个说话人目录
├── txt/                     44,583 个 .txt       2 MiB      ← 文本标注，用不上
├── update.txt
└── speaker-info.txt
```

> 注意：**包里没有 `wav/` 目录**（未裁静音那版不在这个包里），
> 所以"只要 flac 就能省一半"的想法不成立 —— 真正省时间的是**按说话人限量**。

**论文的测试集本身就是每个说话人 100 句**，所以每人取 100 句是有依据的：

```cmd
:: ① 先停掉正在跑的 aria2c（Ctrl+C），它在抢带宽
:: ② 定向下载：111 个说话人，每人 100 句，48 并发
node scripts/download_vctk.mjs --out D:\vpp\data\raw --per-speaker 100 --concurrency 48
```

脚本做的事：

1. 用 HTTP Range 只读 zip 的**中央目录**（12.3 MB，**不下载数据本体**）
2. 按说话人**均匀取样** N 句（不是只取前 N 句，避免句表偏差）
3. 48 个并发 Range 请求，**直接解压落盘**，不需要先下整个 zip 再解压
4. **断点续传**：已落盘且大小正确的文件会跳过，中断后重跑同一条命令即可
5. 中央目录会缓存到 `<输出目录>/.vctk_entries.json`，重跑不用再拉一次

其他用法：

```cmd
node scripts/download_vctk.mjs --out D:\vpp\data\raw --dry-run          # 只看要下多少
node scripts/download_vctk.mjs --out D:\vpp\data\raw --limit 5           # 只下 5 个试跑
node scripts/download_vctk.mjs --out D:\vpp\data\raw --speakers p225,p292 --per-speaker 200
node scripts/download_vctk.mjs --out D:\vpp\data\raw --per-speaker 0    # 全要（10.92 GiB）
```

> ⚠️ 节点内网/直连慢时，把 `--concurrency` 调到 64 或 96 还能再快一些。

### 2.3 备选：整包下载（要全部 10.94 GiB 才用这个）

```bash
cd /d/vpp/data/raw
unzip -q VCTK-Corpus-0.92.zip          # Git Bash 自带 unzip
rm VCTK-Corpus-0.92.zip                 # 解压后删掉压缩包，省 11 GB
```

> ⚠️ **GNU tar 不支持 zip**（`tar -xzf` 会失败），必须用 `unzip` 或 7-Zip。
> PowerShell 的 `Expand-Archive` 也能用，但 11 GB / 4 万多个文件会非常慢，不推荐。

解压后大致是：

```text
data/raw/
├── wav48_silence_trimmed/     ← 我们用这个（已裁掉首尾静音，.flac）
├── txt/                       ← 文本标注（我们用不到）
├── update.txt
└── speaker-info.txt
```

**磁盘预算**：定向下载 1.36 GiB + 16 kHz 副本约 1.3 GiB ≈ **3 GB**（整包方案要 22 GB）。

### 2.4 什么时候用哪个目录

| 目录 | 内容 | 本项目用哪个 |
|---|---|---|
| `wav48_silence_trimmed/` | 48 kHz FLAC，已裁掉首尾静音 | ✅ **默认用这个** |
| `wav/` | 48 kHz WAV，未裁静音 | 备选（需加 `--trim-silence`） |

`prepare_data.py` 会自动找 `wav48_silence_trimmed`，找不到再退到 `wav`。

---

## 3. 下载完后的三步

```bash
conda activate vpp

# ① 勘察目录 → 自动生成 docs/notes/vctk_layout.md
python algorithms/voiceprint_protection/scripts/survey_vctk.py

# ② 预处理 + 划分（80/10/20），产物在 data/
python algorithms/voiceprint_protection/scripts/prepare_data.py

# ③ 提取平台样例
python algorithms/voiceprint_protection/scripts/make_sample.py
```

---

## 4. 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `aria2c` 报 `Resource not found` / 404 | 用了老链接（DataShare 已改版） | 用 2.1 里的**新直链** |
| `curl` 返回 403 / 需要登录 | 同上 | 同上（新直链实测不需要登录） |
| 下载到一半断 | 网络抖动 | 用 `-C -` 或 aria2c 续传（服务器支持 Range） |
| 解压报 "unexpected EOF" | 包没下全 | 对照文件大小重下（下面命令查） |
| `tar -xzf` 报错 | **它现在是 zip，不是 tar.gz** | 用 `unzip`（GNU tar 不支持 zip） |
| 找不到 `p225` 目录 | 解压层级多了一层 | `ls data/raw/VCTK-Corpus-0.92/`，脚本会自动向下找 |
| `soundfile` 读不了 `.flac` | 缺 libsndfile 1.0+ | `pip install soundfile==0.12.1`（会带上 libsndfile） |
| 转换很慢 | 单线程太慢 | `--workers 8`（默认 4） |

```bash
# 查压缩包实际大小，应该正好是 11,747,302,977 字节
ls -l data/raw/VCTK-Corpus-0.92.zip
```

---

## 5. 引用与许可（写 README 时要用）

> Yamagishi, J., et al. (2020). *The VCTK Corpus: A multi-speaker English dataset
> for speaker adaptation and multi-speaker TTS.* Zenodo.
> <https://doi.org/10.5281/zenodo.3235654>
>
> 数据集主页：<https://datashare.ed.ac.uk/handle/10283/3191>
> 语音部分基于 CC BY 4.0 / ODC-By，**对外发布样例时必须署名来源**。
