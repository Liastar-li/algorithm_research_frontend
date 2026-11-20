"""Stage 1 评测 —— 里程碑 M3 的验收脚本。

【M3 的三条验收标准（任务板 §2）】
  1. 产出 Stage 1 checkpoint
  2. **重建音频听感正常**（不是噪音、不是静音）
  3. **Sim_prot(重建) > 0.9**

【这个脚本在做什么】
    原波形 x ──GE2E──→ h
                └──Ec──→ c ──D──→ x̂ (mel) ──V(WaveRNN)──→ x̂ (波形)
    原波形 x ──GE2E──→ h ──(与重建波形的嵌入比)──→ Sim_prot

【★ 为什么必须过声码器】
Stage 1 的解码器只输出 **mel 频谱**，不是声音。
"听感正不正常"这件事**只能靠耳朵**，也就是必须把 mel 变成波形。
而"Sim_prot"要比较的是**语音的说话人相似度**，也必须在波形域做
（说话人编码器 GE2E 吃的是波形算出来的 40 维 mel）。

【★ 三个指标怎么读】
| 指标 | 期望 | 含义 |
|---|---|---|
| `L_rec` | 小 | mel 重建误差。太大 = 解码器没训好 |
| `Sim_prot` | > 0.9 | 重建音与原音的说话人像不像。论文 0.95 |
| `speaker_acc` | **越低越好** | 说话人分类器 W_s 从**重建音**里猜说话人的准确率 |
| `content_acc` | 越低越好 | 从**内容码**里猜说话人的准确率（解耦得越好越低） |

  `content_acc` 低 = "内容码里已经没有说话人信息了" = 解耦成功。
  `speaker_acc` 高 = "重建出来的声音还是本人" = 重建保真。

【用法】
    conda activate vpp
    # 先训（W3 之后）
    python algorithms/voiceprint_protection/stage1/train.py --stage classifier --iterations 2000
    python algorithms/voiceprint_protection/stage1/train.py --stage model --iterations 20000
    # 再评测
    python algorithms/voiceprint_protection/scripts/eval_stage1.py
    python algorithms/voiceprint_protection/scripts/eval_stage1.py --split test --num 20 --save-audio
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 控制台默认 GBK，中文会变乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import audio_utils  # noqa: E402
import config  # noqa: E402
import paths  # noqa: E402

paths.ensure_import_paths()

from stage1.dataset import build_dataloader  # noqa: E402
from stage1.losses import SpeakerClassifier, content_windows  # noqa: E402
from stage1.model import Stage1Model, count_parameters  # noqa: E402
from stage1.speaker_encoder import TrainableSpeakerEncoder  # noqa: E402

WEIGHTS_DIR = paths.WEIGHTS_DIR
RESULTS_DIR = paths.MODULE_DIR / "results"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


def main() -> int:
    print("=" * 70)
    print("  Stage 1 评测（M3 验收）")
    print("=" * 70)

    parser = argparse.ArgumentParser(description="Stage 1 评测")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default=None,
                        help="模型权重（默认 weights/stage1_model.pt）")
    parser.add_argument("--num", type=int, default=20, help="评测多少条")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-audio", action="store_true",
                        help="把原音与重建音写到 results/stage1_eval/ 供试听")
    parser.add_argument("--no-vocoder", action="store_true",
                        help="跳过声码器（只算 mel 域指标；没有声码器权重时用）")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # -- 加载模型 --------------------------------------------------------
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (WEIGHTS_DIR / "stage1_model.pt")
    if not checkpoint_path.exists():
        print(f"\n[失败] 找不到 checkpoint：{checkpoint_path}")
        print("  先训练：")
        print("    python algorithms/voiceprint_protection/stage1/train.py --stage classifier --iterations 2000")
        print("    python algorithms/voiceprint_protection/stage1/train.py --stage model --iterations 20000")
        return 1

    model = Stage1Model().to(device)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    model.eval()
    print(f"  checkpoint : {checkpoint_path}（step {saved.get('step', '?')}）")
    print(f"  模型参数量 : {count_parameters(model):,}")
    print(f"  设备       : {device}")

    # -- 说话人编码器（推理用，冻结）--------------------------------------
    speaker_encoder = TrainableSpeakerEncoder(device=str(device), fine_tune=False)
    speaker_encoder.model.eval()

    # -- 说话人分类器 W_s（用来测"内容码里还剩多少说话人信息"）------------
    classifier = None
    classifier_path = WEIGHTS_DIR / "stage1_speaker_classifier.pt"
    if classifier_path.exists():
        classifier = SpeakerClassifier(
            num_speakers=config.STAGE1["num_speakers"],
            embedding_dim=config.SPEAKER_DIM).to(device)
        state = torch.load(classifier_path, map_location="cpu", weights_only=False)
        classifier.load_state_dict(state["model"])
        classifier.freeze()
        print(f"  分类器     : {classifier_path}")
    else:
        print(f"  [警告] 没有 {classifier_path}，跳过「内容码泄露多少说话人」的测量")

    # -- 声码器 ------------------------------------------------------------
    vocoder = None
    if not args.no_vocoder:
        try:
            from bridge_vocoder import VocoderBridge

            vocoder = VocoderBridge()
            print("  声码器     : HiFi-GAN 16k（论文 §2.3 的 V(·)，可求导）")
        except Exception as error:  # noqa: BLE001
            print(f"  [警告] 声码器加载失败（{error}），只算 mel 域指标。")

    # -- 数据 --------------------------------------------------------------
    loader = build_dataloader(args.split, batch_size=args.batch_size,
                              num_workers=0, shuffle=False, deterministic=True)

    total = 0
    similarity_sum = 0.0
    rec_loss_sum = 0.0
    speaker_correct = 0
    content_correct = 0
    classified = 0
    records: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            if total >= args.num:
                break
            mel80 = batch["mel80"].to(device)
            mel40 = batch["mel40"].to(device)

            embedding = speaker_encoder(mel40)                      # (B, 256)
            mel_out, mel_postnet, codes = model(mel80, embedding, embedding)

            rec_loss = float(torch.nn.functional.mse_loss(mel_postnet, mel80))
            # ★ rec_loss 是**batch 均值**，累加时要乘 batch size ——
            #   否则最后除以样本数，得到的数会小 batch_size 倍（看起来"重建很好"，其实没有）。
            rec_loss_sum += rec_loss * mel80.shape[0]

            # 重建波形 -> 说话人嵌入
            if vocoder is not None:
                for index in range(mel_postnet.shape[0]):
                    if total >= args.num:
                        break
                    mel_numpy = mel_postnet[index].detach().cpu().numpy()
                    rebuilt = vocoder.mel_to_wav(mel_numpy)
                    original = batch["wav"][index].numpy()

                    rebuilt_mel40 = audio_utils.ge2e_mel(rebuilt)
                    rebuilt_embedding = speaker_encoder(
                        torch.from_numpy(rebuilt_mel40).unsqueeze(0).to(device))[0]

                    similarity = cosine(rebuilt_embedding.cpu().numpy(),
                                        embedding[index].cpu().numpy())
                    similarity_sum += similarity

                    # W_s 从重建音里猜说话人 —— 猜得越准，说明音色保真越好
                    if classifier is not None:
                        logits = classifier(rebuilt_embedding.unsqueeze(0))
                        predicted = int(logits.argmax(-1))
                        truth = int(batch["speaker_id"][index])
                        speaker_correct += int(predicted == truth)

                    records.append({
                        "path": batch["path"][index],
                        "speaker": batch["speaker"][index],
                        "sim_prot": similarity,
                        "l_rec": rec_loss,
                        "rms_original": float(audio_utils.rms_energy(original)),
                        "rms_rebuilt": float(audio_utils.rms_energy(rebuilt)),
                    })

                    if args.save_audio:
                        out_dir = RESULTS_DIR / "stage1_eval"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        name = Path(batch["path"][index]).stem
                        audio_utils.save_audio(out_dir / f"{name}_original.wav",
                                               original, 16000)
                        audio_utils.save_audio(out_dir / f"{name}_rebuilt.wav",
                                               rebuilt, 16000)

                    total += 1

            # W_s 从**内容码**里猜说话人 —— 猜得越准，说明解耦越失败
            if classifier is not None:
                windows = content_windows(codes)
                aggregated = torch.cat(windows, dim=0)              # (B*W, 256)
                logits = classifier(aggregated)
                predicted = logits.argmax(-1)
                truth = batch["speaker_id"].to(device).repeat(len(windows))
                content_correct += int((predicted == truth).sum())
                classified += truth.numel()

    # -- 汇总 --------------------------------------------------------------
    print("\n" + "-" * 70)
    if total == 0:
        print("  没有评测到任何样本（数据清单为空？）")
        return 1

    mean_similarity = similarity_sum / total
    print(f"  样本数           : {total}")
    print(f"  平均 L_rec       : {rec_loss_sum / total:.5f}")
    print(f"  平均 Sim_prot    : {mean_similarity:.4f}   "
          f"（论文 0.95，M3 要求 > 0.9）")
    print(f"  ★ M3 验收        : {'通过' if mean_similarity > 0.9 else '未通过'}"
          f"（Sim_prot > 0.9）")
    if classified:
        print(f"  内容码泄露率     : {content_correct / classified:.4f}   "
              f"（W_s 从内容码猜说话人的准确率，**越低越好** = 解耦成功）")
    if classifier is not None:
        print(f"  重建音说话人准确率: {speaker_correct / total:.4f}   "
              f"（越高说明音色保真越好）")

    print("\n  前 5 条明细：")
    for record in records[:5]:
        print(f"    {record['speaker']:<6} Sim_prot={record['sim_prot']:.4f}  "
              f"L_rec={record['l_rec']:.4f}  "
              f"RMS 原 {record['rms_original']:.4f} -> 重建 {record['rms_rebuilt']:.4f}")

    print("\n  ★ 一定要**试听**一遍 results/stage1_eval/ 里的 *_original.wav 与 "
          "*_rebuilt.wav：")
    print("    指标好看不代表听起来正常（Postnet 权重没训好时 mel 看着正常、声音是糊的）。")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "num_samples": total,
        "mean_l_rec": rec_loss_sum / total,
        "mean_sim_prot": mean_similarity,
        "content_leak_accuracy": (content_correct / classified) if classified else None,
        "rebuilt_speaker_accuracy": (speaker_correct / total) if classifier is not None else None,
        "records": records,
    }
    report_path = RESULTS_DIR / f"stage1_eval_{args.split}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  报告已写：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
