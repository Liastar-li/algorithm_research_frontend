"""Verify a saved W_s on a fixed, balanced sample of VCTK training speakers."""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1.dataset import build_dataloader, collate_stage1
from stage1.losses import SpeakerClassifier
from stage1.speaker_encoder import TrainableSpeakerEncoder


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--per-speaker", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.per_speaker < 1:
        parser.error("--per-speaker must be positive")

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    classifier = SpeakerClassifier(saved["num_speakers"], saved["embedding_dim"]).eval()
    classifier.load_state_dict(saved["model"])
    encoder = TrainableSpeakerEncoder(device="cpu", fine_tune=False).eval()
    dataset = build_dataloader("train", num_workers=0, deterministic=True).dataset
    by_speaker = defaultdict(list)
    for index, (_, speaker) in enumerate(dataset.samples):
        by_speaker[speaker].append(index)
    if len(by_speaker) != saved["num_speakers"] or min(map(len, by_speaker.values())) < args.per_speaker:
        parser.error("the training manifest does not provide the expected balanced speaker sample")
    indices = [rows[i * len(rows) // args.per_speaker]
               for _, rows in sorted(by_speaker.items()) for i in range(args.per_speaker)]
    loader = DataLoader(Subset(dataset, indices), batch_size=16, collate_fn=collate_stage1)
    correct = 0
    entropy_sum = 0.0
    for batch in loader:
        logits = classifier(encoder(batch["mel40"]))
        log_probs = logits.log_softmax(-1)
        entropy_sum -= float((log_probs.exp() * log_probs).sum())
        correct += int((logits.argmax(-1) == batch["speaker_id"]).sum())

    report = {
        "step": saved["step"],
        "classifier_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "encoder_sha256": hashlib.sha256(encoder.weights_path.read_bytes()).hexdigest(),
        "split": "train",
        "speaker_count": len(by_speaker),
        "examples_per_speaker": args.per_speaker,
        "correct": correct,
        "examples": len(indices),
        "accuracy": correct / len(indices),
        "mean_entropy_nats": entropy_sum / len(indices),
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
