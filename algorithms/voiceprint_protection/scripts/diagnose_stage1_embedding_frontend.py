"""用上游正式 GE2E 波形入口复核 Stage 1 的 clean/oracle/rebuilt Top-1。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge_speaker import SpeakerBridge
from bridge_vocoder import VocoderBridge
from stage1.dataset import build_dataloader, collate_stage1
from stage1.losses import SpeakerClassifier, content_windows, disentanglement_loss
from stage1.model import Stage1Model, remap_to_official_keys
from stage1.speaker_encoder import GE2EMel, TrainableSpeakerEncoder


def top1(queries, clean, speaker_count):
    reference = torch.nn.functional.normalize(clean[:speaker_count])
    queries = torch.nn.functional.normalize(queries[speaker_count:])
    labels = torch.arange(len(queries)) % speaker_count
    return float(((queries @ reference.T).argmax(1) == labels).float().mean())


def identity_gap(embeddings, speaker_count):
    embeddings = torch.nn.functional.normalize(embeddings)
    reference, queries = embeddings[:speaker_count], embeddings[speaker_count:]
    scores = queries @ reference.T
    same = torch.arange(len(queries))[:, None] % speaker_count == torch.arange(speaker_count)
    same_mean = float(scores[same].mean())
    different_mean = float(scores[~same].mean())
    return {"same": same_mean, "different": different_mean, "gap": same_mean - different_mean}


def sample_key(path):
    return "/".join(str(path).replace("\\", "/").split("/")[-2:])


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="algorithms/voiceprint_protection/weights/"
                        "stage1_fresh_200k/stage1_step250000_wide_test_bn.pt")
    parser.add_argument("--calibration-report", default="algorithms/voiceprint_protection/weights/"
                        "stage1_fresh_200k/stage1_step250000_wide_test_bn.json")
    parser.add_argument("--output", default="results/stage1_embedding_frontend.json")
    parser.add_argument("--speaker-count", type=int, default=4,
                        help="先抽前 N 个说话人做入口复核，默认 4，设 20 跑全量")
    parser.add_argument("--whisper-model", help="本地 Whisper 模型目录；用原音转写计算相对 WER")
    parser.add_argument("--wer-reference-report", type=Path,
                        help="复用同一固定样本和 Whisper 配置的原音转写及 oracle WER")
    parser.add_argument("--classifier-checkpoint", type=Path, help="记录 W_s 对内容窗口的预测熵")
    args = parser.parse_args()
    if args.wer_reference_report and not args.whisper_model:
        parser.error("--wer-reference-report requires --whisper-model")

    torch.set_num_threads(2)
    calibration = json.loads(Path(args.calibration_report).read_text(encoding="utf-8"))
    all_speakers = calibration["held_out_speakers"]
    speaker_count = min(args.speaker_count, all_speakers)
    utterances = calibration["held_out_utterances_per_speaker"]
    if speaker_count < 2 or utterances < 2:
        parser.error("identity evaluation requires at least two speakers and two utterances per speaker")
    ordered = np.asarray(calibration["held_out_paths"]).reshape(all_speakers, utterances)
    selected_paths = ordered[:speaker_count].T.reshape(-1).tolist()
    dataset = build_dataloader(calibration["held_out_split"], num_workers=0,
                               deterministic=True).dataset
    by_path = {sample_key(relative): index for index, (relative, _) in enumerate(dataset.samples)}
    indices = [by_path[sample_key(path)] for path in selected_paths]
    batch = collate_stage1([dataset[index] for index in indices])

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Stage1Model().eval()
    model.load_state_dict(remap_to_official_keys(saved["model"]))
    encoder = TrainableSpeakerEncoder(device="cpu", fine_tune=False).eval()
    encoder.load_checkpoint_state(saved["speaker_encoder"])
    vocoder = VocoderBridge(device="cpu")

    source = batch["mel80"]
    h = encoder(batch["mel40"])
    _, rebuilt_mel, codes = model(source, h, h)
    content_features = torch.stack(content_windows(codes)).mean(0)
    waves = {"clean": [wave.numpy() for wave in batch["wav"]],
             "oracle": [], "rebuilt": []}
    for original, mel, rebuilt in zip(batch["wav"], source, rebuilt_mel):
        waves["oracle"].append(vocoder.mel_to_wav(mel.numpy()))
        waves["rebuilt"].append(vocoder.mel_to_wav(rebuilt.numpy()))

    bridge = SpeakerBridge(device="cpu")
    upstream_whole_clip = TrainableSpeakerEncoder(device="cpu", fine_tune=False).eval()
    utterances_per_speaker = calibration["held_out_utterances_per_speaker"]
    formal = {}
    for name, items in waves.items():
        rows = []
        for index, wave in enumerate(items):
            rows.append(bridge.embed(wave, 16000))
            if (index + 1) % 16 == 0:
                print(f"  formal {name}: {index + 1}/{len(items)}", flush=True)
        formal[name] = torch.from_numpy(np.stack(rows))
    ge2e_mel = GE2EMel().eval()
    with torch.no_grad():
        whole_clip = {
            name: upstream_whole_clip(ge2e_mel(torch.from_numpy(np.stack(items))))
            for name, items in waves.items()
        }

    report = {
        "checkpoint": args.checkpoint,
        "step": saved.get("step"),
        "note": "同一批固定3秒音频；formal=SpeakerBridge.embed（音量归一化+分段平均，"
                "本机缺少 webrtcvad 所以跳过 VAD），whole_clip=GE2EMel + "
                "上游 GE2E 整段入口",
        "paths": selected_paths,
        "speaker_count": speaker_count,
        "utterances_per_speaker": utterances_per_speaker,
        "content_nearest_reference_top1": top1(content_features, content_features, speaker_count),
        "top1": {
            frontend: {name: top1(embeddings, variants["clean"], speaker_count)
                       for name, embeddings in variants.items()}
            for frontend, variants in (("formal_speaker_bridge", formal),
                                       ("whole_clip_ge2e", whole_clip))
        },
        "identity_gap": {
            frontend: {name: identity_gap(embeddings, speaker_count)
                       for name, embeddings in variants.items()}
            for frontend, variants in (("formal_speaker_bridge", formal),
                                       ("whole_clip_ge2e", whole_clip))
        },
        "original_rebuilt_cosine": {
            frontend: float(torch.nn.functional.cosine_similarity(
                variants["clean"], variants["rebuilt"]).mean())
            for frontend, variants in (("formal_speaker_bridge", formal),
                                       ("whole_clip_ge2e", whole_clip))
        },
        "original_oracle_cosine": {
            frontend: float(torch.nn.functional.cosine_similarity(
                variants["clean"], variants["oracle"]).mean())
            for frontend, variants in (("formal_speaker_bridge", formal),
                                       ("whole_clip_ge2e", whole_clip))
        },
    }
    if args.classifier_checkpoint:
        classifier_state = torch.load(args.classifier_checkpoint, map_location="cpu", weights_only=True)
        classifier = SpeakerClassifier(classifier_state["num_speakers"], classifier_state["embedding_dim"]).eval()
        classifier.load_state_dict(classifier_state["model"])
        report["content_classifier_entropy_nats"] = -float(disentanglement_loss(codes, classifier))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.whisper_model:
        from evaluate import WhisperScorer, word_error_rate

        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        scorer = WhisperScorer(args.whisper_model)
        decode_options = {"temperature": 0.0, "max_new_tokens": 48}
        names = ("oracle", "rebuilt")
        oracle_wer = None
        if args.wer_reference_report:
            previous = json.loads(args.wer_reference_report.read_text(encoding="utf-8"))
            previous_wer = previous["relative_wer"]
            if (previous["paths"] != selected_paths or previous_wer["model"] != args.whisper_model
                    or previous_wer["decode_options"] != decode_options):
                parser.error("WER reference report must use the same samples, model and decoding options")
            references = previous_wer["reference_transcripts"]
            oracle_wer = previous_wer["oracle"]
            names = ("rebuilt",)
        else:
            print(f"  Whisper clean: {len(waves['clean'])} utterances", flush=True)
            references = [scorer.transcribe(wav, 16000, **decode_options) for wav in waves["clean"]]
        reference_words = sum(len(reference.split()) for reference in references)
        report["relative_wer"] = {
            "model": args.whisper_model, "decode_options": decode_options,
            "reference_transcripts": references, "reference_words": reference_words,
            "empty_references": sum(not ref for ref in references), "oracle": oracle_wer,
        }
        for name in names:
            print(f"  Whisper {name}: {len(waves[name])} utterances", flush=True)
            errors = sum(word_error_rate(reference, scorer.transcribe(wav, 16000, **decode_options))
                         * len(reference.split())
                         for reference, wav in zip(references, waves[name]) if reference)
            report["relative_wer"][name] = errors / reference_words if reference_words else None
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "paths"},
                     ensure_ascii=False, indent=2))
    print(f"写出 {output}")


if __name__ == "__main__":
    main()
