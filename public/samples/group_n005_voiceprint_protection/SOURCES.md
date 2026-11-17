# 样例说明 · group_n005_voiceprint_protection

## 文件

| 文件 | 说明 |
|---|---|
| `input.wav` | 原始语音（VCTK 测试集说话人） |
| `output.wav` | **待生成** —— 需要成员 A 的 `infer.py` 就位后重跑本脚本 |

## 参数

```json
{ "alpha": 0.1 }
```

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
