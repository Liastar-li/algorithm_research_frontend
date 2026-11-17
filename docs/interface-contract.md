# 双人接口契约（任务板 W2-D3-1）

> **状态**：草案 v0.1 —— 成员 B 起草，**待成员 A 确认签字**。
> 依据：指南第 8 章 §8.5 的 10 条 + 本项目三个桥接模块的实际签名。
>
> **规矩**：本文件里的任何一条，改动都要**两人同意**并更新本文件。
> 这是两人协作里性价比最高的一件事 —— 没有它，Stage 2 拿到 Stage 1 的输出时
> 形状/归一化对不上，会浪费一整天。

---

## 1. 音频与 mel（不可协商的硬约定）

| # | 约定 | 值 | 真源 |
|---|---|---|---|
| 1 | 采样率 | **16000 Hz**，单声道，16-bit PCM | `config.MEL_PRESETS` |
| 2 | mel 参数 | `n_fft=1024, hop=256, win=1024, n_mels=80, fmin=40, fmax=7600`，`power=2`，`log10` 下限 `1e-10` | `audio_utils.MelSpec` |
| 3 | 预设 | `paper16k`（16k，最终系统）/ `autovc22k`（22.05k，L1 阶段用官方权重） | `config.MEL_PRESETS` |

> **规则**：**任何地方都不许再手写 mel 参数**，一律调 `audio_utils.MelSpec`。
> 两个预设除采样率外完全一致，切换只改一行。
>
> **特例（唯一允许的例外）**：GE2E 内部的 40 维 mel（`n_fft=400, hop=160, fmin=0, fmax=8000`）
> 是**上游自己的约定，我们不碰**，只通过 `bridge_speaker.py` 使用。
> 详见 [`notes/ge2e_interface.md`](notes/ge2e_interface.md)。

---

## 2. 张量形状（指南 §8.5 第 3–9 条）

| 符号 | 名称 | 形状 | dtype | 归一化 | 说明 |
|---|---|---|---|---|---|
| `x` | 输入 mel | `(T, 80)` | float32 | 见 `audio_utils` | `T` = 帧数 |
| `c` | 内容码 | `(T', 32)` | float32 | — | `T' = T // 2`（AutoVC 瓶颈降采样） |
| `h` | 说话人嵌入 | `(256,)` | float32 | **L2 归一化** | 由 `bridge_speaker.embed()` 出口保证 |
| `δ` | 扰动 | `(256,)` | float32 | — | Stage 2 生成器 `G(h)` 的输出 |
| `h_adv` | 被污染的嵌入 | `(256,)` | float32 | 继承 h | `h_adv = h + α·δ` |
| `x_adv` | 保护后 mel | `(T, 80)` | float32 | — | `x_adv = D(c, h_adv)` |
| `wav` / `wav_adv` | 波形 | `(samples,)` | float32 | — | `wav_adv = V(x_adv)`，16 kHz |
| `δ_wav` | 波形扰动 | `(samples,)` | float32 | — | `δ_wav = wav_adv − wav`，供 `L_mask` 用 |

**形状不变量**：`T' = T // 2`；`x` 与 `x_adv` 的帧数必须**完全相等**。
> 不相等时 `L_mel` 会退化成恒定大常数，**而且不报错** —— 这是最难查的一类 bug。

---

## 3. 三个桥接模块的签名

> A 负责 `bridge_autovc.py` / `bridge_vocoder.py`，B 负责 `bridge_speaker.py`（已完成）。
> 三个模块的**调用方式必须一致**：构造时加载权重，方法名统一。

### 3.1 `bridge_speaker.py`（成员 B）

```python
from bridge_speaker import SpeakerBridge

bridge = SpeakerBridge(rtc_dir=None, device=None, weights_path=None)

h        = bridge.embed(wav: np.ndarray, sample_rate: int) -> torch.Tensor  # (256,) 已归一化
h        = bridge.embed_path(path: str | Path)               -> torch.Tensor  # (256,) 已归一化
H        = bridge.embed_batch(waves: list, srs: list)       -> torch.Tensor  # (N, 256)
info     = bridge.describe()                                 -> dict           # 接口自述
```

- `sample_rate` 是**输入音频的原生采样率**；不是 16 kHz 时桥接层内部自动重采样。
- 输出**必定已 L2 归一化**，调用方不要再归一化。

### 3.2 `bridge_autovc.py`（成员 A）

```python
from bridge_autovc import AutoVCBridge

bridge = AutoVCBridge(preset: str = "autovc22k", weights_path: str | None = None)

c     = bridge.content_encoder(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor  # (T', 32)
x_out = bridge.decoder(c: torch.Tensor, h: torch.Tensor)       -> torch.Tensor  # (T, 80)
```

- `x` 是 `(T, 80)` 的 mel（**不是** batch 维），`h` 是 `(256,)`。
- `c` 的帧数必须是 `T // 2`。
- 成员 A 需在文件头写明：权重来源、是否含 PostNet、mel 约定用的是哪个预设。

### 3.3 `bridge_vocoder.py`（成员 A）

```python
from bridge_vocoder import VocoderBridge

bridge = VocoderBridge(preset: str = "autovc22k", weights_path: str | None = None)

wav = bridge.mel_to_wav(x: torch.Tensor) -> torch.Tensor  # (samples,) float32
```

- 输入是 `(T, 80)` 的 **AutoVC 侧 mel**；声码器若用 HiFi-GAN 官方权重，
  内部负责转成 HiFi-GAN 的 mel 约定（`(x+4.5)/5` 归一化那套）。
- 输出采样率必须与预设一致（`autovc22k` → 22050，`paper16k` → 16000）。

---

## 4. 推理主流程 `infer.py`（成员 A 主责，B 写冒烟测试要用）

```python
def protect(
    wav_path: str,
    alpha: float = 0.10,
    output_path: str | None = None,
) -> dict:
    """一段语音 → 保护后语音。

    返回
    ----
    {
      "protected_path": str,          # 保护后 wav 的路径
      "alpha": float,
      "shapes": {
        "x": (T, 80),
        "h": (256,),
        "c": (T', 32),
        "x_adv": (T, 80),
      },
    }
    """
```

- **L1 阶段**：`δ` 用**随机方向**（固定 seed）代替训练好的 `G`，不训练任何东西。
- `output_path` 为 `None` 时写到 `algorithms/voiceprint_protection/results/protected.wav`。
- `scripts/smoke_test.py` 会直接调用它；**签名一变就要同步改冒烟测试**。

---

## 5. 数据清单格式（成员 B）

```text
data/train.txt   每行：<相对 data/ 的音频路径> <说话人名>
data/val.txt     同上
data/test.txt    同上
data/spk2id.txt  每行：<说话人名> <整数 ID>，只含训练集说话人（K = 80）
```

| 硬要求 | 检查方式 |
|---|---|
| 排除 `p*_eval` | `grep -c "_eval" data/*.txt` 必须为 0 |
| 三集合说话人不相交 | `comm -12 <(cut -d' ' -f2 data/train.txt\|sort -u) <(cut -d' ' -f2 data/test.txt\|sort -u)` 必须为空 |
| `spk2id.txt` 的 K = 80 | `cut -d' ' -f2 data/train.txt \| sort -u \| wc -l` → 80，且 ID 覆盖 0..79 |

> `prepare_data.py` 跑完会自动执行这三条检查，不过则返回码非 0。

---

## 6. 随机性与复现

| 项 | 约定 | 出处 |
|---|---|---|
| 全局 seed | `config.SEED = 42` | `config.py` |
| 说话人划分 | `prepare_data.py --seed`，默认取 `config.SEED` | 划分名单记录在 `data/split_speakers.json` |
| L1 随机扰动 | `numpy.random.default_rng(config.SEED)` | `scripts/smoke_test.py` |
| 训练随机性 | Stage 1/2 的 DataLoader、初始化**全部**显式用 `config.SEED` | 待 A/B 各自实现 |

---

## 7. 变更记录

| 日期 | 版本 | 改动 | 提出 | 同意 |
|---|---|---|---|---|
| 2026-11-17 | v0.1 | 起草：补齐三个桥接模块签名、`infer.protect()` 签名、清单格式 | 成员 B | ⬜ 待成员 A 确认 |

---

## 8. 签字

| 角色 | 姓名 | 确认日期 | 备注 |
|---|---|---|---|
| 成员 A | | | |
| 成员 B | | | |

> **签字前请重点核对：**
> - [ ] A 的 `bridge_autovc.py` 签名与 §3.2 一致
> - [ ] A 的 `bridge_vocoder.py` 签名与 §3.3 一致
> - [ ] A 的 `infer.protect()` 签名与 §4 一致（`scripts/smoke_test.py` 依赖它）
> - [ ] `c` 的帧数确实是 `T // 2`
> - [ ] 双方都同意"只有 GE2E 内部的 40 维 mel 是例外"这一条
