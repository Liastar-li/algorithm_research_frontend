# 样例说明 · group_n005_voiceprint_protection

## 文件

| 文件 | 说明 |
|---|---|
| `input.wav` | 原始语音（VCTK 测试集说话人 p228） |
| `output.wav` | **保护后语音**，由本项目推理链实际生成（α = 0.1） |
| `output_clone_baseline.wav` | **未防御时**攻击者克隆出来的声音（用原始身份卡）—— 应与本人很像 |
| `output_clone_protected.wav` | **防御后**攻击者克隆出来的声音（攻击者从保护后语音重新提卡）—— 应明显不像本人 |

> ★ 三个音频一起听，差别就是**防御效果本身**：
> `output.wav` 听起来跟 `input.wav` 差不多（人耳听不出扰动），
> 但 `output_clone_baseline.wav` 像本人而 `output_clone_protected.wav` 不像。
> 这是论文 Figure 1 的直观演示。

## 参数

```json
{ "alpha": 0.1, "attack_mode": "clone", "report": true }
```

生成命令：

```bash
conda activate vpp
python algorithms/voiceprint_protection/scripts/make_sample.py
```

> ⚠️ **L1 阶段**：扰动 δ 是**随机高斯方向**（Stage 2 的生成器 G 还没训练），
> 所以 `output_clone_protected.wav` 与 `output_clone_baseline.wav` 的差别
> 还**不够大**（当前 Sim_clone ≈ 0.90，论文是 0.13）。
> 等 Stage 2 训练完成后需要**重新生成**这组样例。

## 样例来源与许可

- **数据集**：CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning
  Toolkit（version 0.92），The Centre for Speech Technology Research (CSTR), Edinburgh
- **DOI（权威标识）**：<https://doi.org/10.7488/ds/2645>
  → <https://datashare.ed.ac.uk/handle/10283/3443>
- **许可**：CC BY 4.0（见数据自带的 `VCTK.txt`：“licensed under the Creative Commons
  License: Attribution 4.0 International”）
  → **对外展示本样例时必须保留上述署名**。
- **引用格式**：
  Veaux, C., Yamagishi, J., & MacDonald, K. (2017). *CSTR VCTK Corpus: English
  Multi-speaker Corpus for CSTR Voice Cloning Toolkit* (version 0.92). The Centre
  for Speech Technology Research (CSTR), Edinburgh.
  <https://doi.org/10.7488/ds/2645>
- **处理过程**：原始 48 kHz FLAC → 重采样到 16 kHz / 单声道 / 16-bit PCM；
  保护后音频由本项目的推理链实际生成。

## 具体条目

- 来源：VCTK 测试集（说话人 p228）
- 原始文件：`audio16k/p228/p228_001_mic1.wav`
- 说话人：`p228`（测试集，未参与训练）
