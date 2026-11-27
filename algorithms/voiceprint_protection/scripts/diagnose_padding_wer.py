"""对照补零与真实静音 mel 对短音频重建 WER 的影响。"""

from __future__ import annotations

import argparse
import math
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path("algorithms/voiceprint_protection").resolve()))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

import audio_utils
import config
from evaluate import WhisperScorer, load_test_items, load_transcripts, transcript_key, word_error_rate
from stage2.common import set_seed
from stage2.models import Stage1SpeakerEncoder, TorchAutoVCStage1, freeze
from stage2.pipeline import HiFiGANVocoder, Stage2Pipeline

CKPT = "algorithms/voiceprint_protection/weights/stage1_fresh_200k/stage1_step250000_wide_test_bn.pt"


def zero_pad(self: TorchAutoVCStage1, mel: torch.Tensor) -> torch.Tensor:
    target = int(self.pad_base * math.ceil(mel.shape[1] / self.pad_base))
    if target > mel.shape[1]:
        mel = torch.nn.functional.pad(mel, (0, 0, 0, target - mel.shape[1]))
    return mel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=1.0)
    parser.add_argument("--whisper-model", default="small")
    args = parser.parse_args()

    set_seed(config.SEED)
    device = "cpu"
    speaker = Stage1SpeakerEncoder(CKPT, device=device)
    stage1 = TorchAutoVCStage1(checkpoint=CKPT, device=device)
    vocoder = HiFiGANVocoder(device=device)
    pipeline = Stage2Pipeline(speaker_encoder=speaker, stage1=stage1,
                              vocoder=vocoder, device=device)
    for model in (speaker, stage1, vocoder):
        freeze(model)

    items = load_test_items("test", limit_per_speaker=1)[:args.count]
    transcripts = load_transcripts(Path("data/VCTK-Corpus/txt"))
    scorer = WhisperScorer(model_name=args.whisper_model, device=device)

    silence_pad = TorchAutoVCStage1._pad
    results = []
    print(f"{'文件':<28} {'帧':>4} {'原音':>8} {'补零':>8} {'静音':>8}")
    for path, _ in items:
        wav = audio_utils.load_audio(path, 16000)
        start = min(len(wav) // 4, int(0.5 * 16000))
        samples = int(args.max_seconds * 16000)
        wav = wav[start:start + samples]
        waveform = torch.from_numpy(wav).unsqueeze(0)
        with torch.no_grad():
            mel = pipeline.mel(waveform)
            embedding = pipeline.speaker(waveform, torch.tensor([len(wav)]))
            content = stage1.content_code(mel, embedding)
            hypotheses = {"original": scorer.transcribe(wav, 16000)}
            for name, padding in (("zero", zero_pad), ("silence", silence_pad)):
                stage1._pad = types.MethodType(padding, stage1)
                rebuilt_mel = stage1.decode(mel, embedding, content=content)
                rebuilt = pipeline.vocode(rebuilt_mel).reshape(-1).numpy()
                hypotheses[name] = scorer.transcribe(rebuilt, 16000)
        stage1._pad = types.MethodType(silence_pad, stage1)

        reference = transcripts.get(transcript_key(path)) or hypotheses["original"]
        wers = {name: word_error_rate(reference, text) for name, text in hypotheses.items()}
        results.append({"path": str(path), "frames": mel.shape[1],
                        "reference": reference, "hypotheses": hypotheses, "wer": wers})
        print(f"{path.name:<28} {mel.shape[1]:>4} {wers['original']:>8.3f} "
              f"{wers['zero']:>8.3f} {wers['silence']:>8.3f}")

    print("\n逐条转写")
    for row in results:
        print(f"\n{Path(row['path']).name}  参考：{row['reference']}")
        for name, text in row["hypotheses"].items():
            print(f"  {name:<8} {text}")


if __name__ == "__main__":
    main()
