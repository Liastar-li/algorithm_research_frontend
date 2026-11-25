"""Calibrate Stage 1 BatchNorm on train data and verify on held-out speakers."""

import argparse
import hashlib
import json
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audio_utils
import config
from bridge_vocoder import VocoderBridge
from stage1.dataset import build_dataloader, collate_stage1
from stage1.model import Stage1Model
from stage1.speaker_encoder import GE2EMel, TrainableSpeakerEncoder


def held_out_batches(split, speaker_count, utterances, spread=False):
    dataset = build_dataloader(split, num_workers=0, deterministic=True).dataset
    groups = {}
    for index, (_, speaker) in enumerate(dataset.samples):
        groups.setdefault(speaker, []).append(index)
    groups = [indices for indices in groups.values() if len(indices) >= utterances]
    if len(groups) < speaker_count:
        raise ValueError("Insufficient held-out speakers with enough utterances")
    groups = groups[:speaker_count]
    if spread:
        groups = [[indices[j] for j in np.linspace(0, len(indices) - 1, utterances, dtype=int)]
                  for indices in groups]
    return [collate_stage1([dataset[indices[i]] for indices in groups])
            for i in range(utterances)]


def score(model, encoder, batches, device):
    errors, embeddings, speakers = [], [], []
    for batch in batches:
        source = batch["mel80"].to(device)
        h = encoder(batch["mel40"].to(device))
        pre, post, _ = model(source, h, h)
        pre_mse, post_mse = [(out - source).square().mean((1, 2)) for out in (pre, post)]
        delta_mae = (post.diff(dim=1) - source.diff(dim=1)).abs().mean((1, 2))
        errors.append(torch.stack((pre_mse, post_mse, delta_mae), dim=1).cpu())
        embeddings.append(h.cpu())
        speakers.extend(batch["speaker"])
    pre_mse, post_mse, delta_mae = torch.cat(errors).mean(0).tolist()
    metrics = {"pre_mse": pre_mse, "post_mse": post_mse, "post_delta_mae": delta_mae}
    return metrics, torch.cat(embeddings), speakers


def cosine_summary(embeddings, speakers):
    embeddings = torch.nn.functional.normalize(embeddings)
    row, column = torch.triu_indices(len(speakers), len(speakers), offset=1)
    same = torch.tensor([speakers[i] == speakers[j] for i, j in zip(row, column)])
    cosine = (embeddings[row] * embeddings[column]).sum(1)
    return {
        name: {"count": len(values), "min": float(values.min()),
               "mean": float(values.mean()), "max": float(values.max())}
        for name, values in (("same_speaker", cosine[same]), ("different_speaker", cosine[~same]))
    }


def reference_accuracy(queries, clean, speaker_count):
    """Use the first utterance per speaker as enrollment."""
    reference = torch.nn.functional.normalize(clean[:speaker_count])
    queries = torch.nn.functional.normalize(queries[speaker_count:])
    labels = torch.arange(len(queries)) % speaker_count
    return float(((queries @ reference.T).argmax(1) == labels).float().mean())


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calibration-batches", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--held-out-speakers", type=int, default=8)
    parser.add_argument("--held-out-utterances", type=int, default=4)
    parser.add_argument("--spread-utterances", action="store_true")
    args = parser.parse_args()
    if min(args.held_out_speakers, args.held_out_utterances) < 2:
        parser.error("Evaluation requires at least two speakers and two utterances each")
    if min(args.calibration_batches, args.batch_size) < 1:
        parser.error("Calibration batch count and batch size must be positive")
    report_path = args.output.with_suffix(".json")
    if args.output.exists() or report_path.exists():
        parser.error("Choose a new output path")

    torch.set_num_threads(1)
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Stage1Model().to(args.device).eval()
    model.load_state_dict(saved["model"])
    upstream = TrainableSpeakerEncoder(device=args.device, fine_tune=False).eval()
    encoder = TrainableSpeakerEncoder(device=args.device, fine_tune=False).eval()
    encoder.load_checkpoint_state(saved["speaker_encoder"])
    held_out = held_out_batches(args.split, args.held_out_speakers, args.held_out_utterances,
                                spread=args.spread_utterances)
    baseline, embeddings, speakers = score(model, encoder, held_out, args.device)
    upstream_embeddings = torch.cat([upstream(b["mel40"].to(args.device)).cpu()
                                     for b in held_out])

    batch_norms = [module for module in model.modules()
                   if isinstance(module, torch.nn.BatchNorm1d)]
    for module in batch_norms:
        module.reset_running_stats()
        module.momentum = None
        module.train()
    train = build_dataloader("train", batch_size=args.batch_size, num_workers=0, shuffle=True)
    count = 0
    for batch in islice(train, args.calibration_batches):
        h = encoder(batch["mel40"].to(args.device))
        model(batch["mel80"].to(args.device), h, h)
        count += 1
    if count != args.calibration_batches:
        raise ValueError(f"Expected {args.calibration_batches} training batches, got {count}")
    model.eval()
    calibrated, _, _ = score(model, encoder, held_out, args.device)
    if not np.isfinite(list(calibrated.values())).all():
        raise ValueError("Calibrated reconstruction has non-finite error")

    audio_dir = args.output.with_suffix("") / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    vocoder = VocoderBridge(device=args.device)
    reconstructed, oracle_mels = [], []
    swap_errors, swapped_embeddings = [], []
    ge2e_frontend = GE2EMel().to(args.device)
    for batch_index, batch in enumerate(held_out):
        source = batch["mel80"].to(args.device)
        h = encoder(batch["mel40"].to(args.device))
        _, post, _ = model(source, h, h)
        _, swapped, _ = model(source, h, h.roll(1, dims=0))
        swap_errors.append(float((swapped - post).square().mean()))
        swapped_waveform = torch.cat([vocoder.mel_to_wav(part, batch=True)
                                     for part in swapped.split(4)])[:, :len(batch["wav"][0])]
        swapped_embeddings.append(upstream(ge2e_frontend(swapped_waveform)).cpu())
        for index, (original, mel, prediction) in enumerate(zip(batch["wav"], source, post)):
            rebuilt = vocoder.mel_to_wav(prediction.cpu().numpy())
            oracle = vocoder.mel_to_wav(mel.cpu().numpy())
            reconstructed.append(audio_utils.ge2e_mel(rebuilt))
            oracle_mels.append(audio_utils.ge2e_mel(oracle))
            if batch_index == 0:
                name = Path(batch["path"][index]).stem
                for label, waveform in (("original", original.numpy()),
                                        ("oracle", oracle), ("rebuilt", rebuilt)):
                    audio_utils.save_audio(audio_dir / f"{name}_{label}.wav", waveform, 16000)
    rebuilt_mels = torch.from_numpy(np.stack(reconstructed)).to(args.device)
    oracle_mels = torch.from_numpy(np.stack(oracle_mels)).to(args.device)
    rebuilt_upstream = upstream(rebuilt_mels).cpu()
    oracle_upstream = upstream(oracle_mels).cpu()
    independent_cos = torch.nn.functional.cosine_similarity(
        rebuilt_upstream, upstream_embeddings)
    converted = torch.cat(swapped_embeddings)
    target_embeddings = torch.cat([chunk.roll(1, dims=0) for chunk in
                                   upstream_embeddings.split(len(held_out[0]["speaker"]))])
    converted_source_cos = torch.nn.functional.cosine_similarity(converted, upstream_embeddings)
    converted_target_cos = torch.nn.functional.cosine_similarity(converted, target_embeddings)
    report = {
        "source": str(args.checkpoint),
        "step": saved["step"],
        "calibration_batches": count,
        "calibration_batch_size": args.batch_size,
        "calibration_seed": config.SEED,
        "device": args.device,
        "held_out_split": args.split,
        "held_out_speakers": args.held_out_speakers,
        "held_out_utterances_per_speaker": args.held_out_utterances,
        "spread_utterances": args.spread_utterances,
        "held_out_paths": [path for batch in held_out for path in batch["path"]],
        "original_eval": baseline,
        "calibrated_eval": calibrated,
        "stage1_ge2e": cosine_summary(embeddings, speakers),
        "upstream_ge2e": cosine_summary(upstream_embeddings, speakers),
        "stage1_original_rebuilt_cosine_mean": float(torch.nn.functional.cosine_similarity(
            encoder(rebuilt_mels).cpu(), embeddings).mean()),
        "upstream_original_oracle_cosine_mean": float(torch.nn.functional.cosine_similarity(
            oracle_upstream, upstream_embeddings).mean()),
        "upstream_speaker_top1": {
            name: reference_accuracy(queries, upstream_embeddings, len(held_out[0]["speaker"]))
            for name, queries in (("clean", upstream_embeddings),
                                  ("oracle", oracle_upstream), ("rebuilt", rebuilt_upstream))
        },
        "different_target_embedding_mel_mse": float(np.mean(swap_errors)),
        "swapped_speaker": {
            "source_cosine": float(converted_source_cos.mean()),
            "target_cosine": float(converted_target_cos.mean()),
            "target_beats_source_fraction": float(
                (converted_target_cos > converted_source_cos).float().mean()),
        },
        "upstream_original_rebuilt_cosine": {
            "min": float(independent_cos.min()), "mean": float(independent_cos.mean()),
            "max": float(independent_cos.max()),
        },
    }
    payload = {key: saved[key] for key in ("speaker_encoder", "step", "config")}
    if "speaker_adaptation" in saved:
        payload["speaker_adaptation"] = saved["speaker_adaptation"]
        report["speaker_adaptation"] = payload["speaker_adaptation"]
    payload["model"] = model.state_dict()
    payload["bn_calibration"] = {
        "train_batches": count, "batch_size": args.batch_size, "seed": config.SEED,
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.rename(args.output)
    digest = hashlib.sha256()
    with args.output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    report["checkpoint_sha256"] = digest.hexdigest()
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "held_out_paths"},
                     indent=2))


if __name__ == "__main__":
    main()
