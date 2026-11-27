"""自由 δ 的 oracle 测试：绕开生成器 G，直接把每条的 δ 当参数优化。

回答的问题和 `--overfit` 不一样：
  --overfit  测的是「G 这个 593k 的共享函数，学不学得动」；
  本脚本     测的是「在 α 的角度预算内，L_de 这个**目标本身**可达多低」。
如果连自由 δ（每条 256 个自由度、逐条独立）都压不下去，那不是 G 的问题，
是 α=0.1 的预算 / 放大链根本到不了论文那个量级。

用法（仓库根，用 vpp 的 python）：
    python -u algorithms/voiceprint_protection/stage2/diagnose_delta_oracle.py \
        --alpha 0.1 --space upstream --delta-mode unit --steps 100
★ `--delta-mode unit` 才是与 G 等预算（‖δ‖=1、α=0.1 ⇒ 5.73°）的可比版本；
   `free` 版本里优化器会把 ‖δ‖ 推到 3 倍以上，等于偷偷放大角度预算。
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path("algorithms/voiceprint_protection").resolve()))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

import config
from stage2.common import cosine_similarity, perturb_embedding, set_seed
from stage2.dataset import build_dataloader
from stage2.models import Stage1SpeakerEncoder, TorchAutoVCStage1, TorchSpeakerEncoder, freeze
from stage2.pipeline import HiFiGANVocoder, Stage2Pipeline

CKPT = "algorithms/voiceprint_protection/weights/stage1_fresh_200k/stage1_step250000_wide_test_bn.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--space", choices=("upstream", "stage1"), default="upstream")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=2.0)
    parser.add_argument("--delta-mode", choices=("free", "unit"), default="free",
                        help="free=δ 模长自由增长（会偷偷放大 α 的角度预算）；"
                             "unit=每条强制 ‖δ‖=1，与 G 随机初始化时的工作点完全同预算")
    parser.add_argument("--output", default=None, help="可选，把本次 oracle 结果写成 JSON")
    args = parser.parse_args()

    set_seed(config.SEED)  # 连 Python 的 random 一起固定，数据集裁剪才可复现
    speaker = Stage1SpeakerEncoder(CKPT, device="cpu")
    freeze(speaker)
    stage1 = TorchAutoVCStage1(checkpoint=CKPT, device="cpu")
    freeze(stage1)
    vocoder = HiFiGANVocoder(device="cpu")
    de_encoder = TorchSpeakerEncoder(device="cpu") if args.space == "upstream" else speaker
    freeze(de_encoder)
    pipeline = Stage2Pipeline(speaker_encoder=speaker, stage1=stage1, vocoder=vocoder,
                              device="cpu", de_speaker=de_encoder)

    loader = build_dataloader("train", batch_size=args.count, max_seconds=args.max_seconds,
                              num_workers=0, shuffle=False)
    waveforms, lengths, _ = next(iter(loader))

    with torch.no_grad():
        original_mel = pipeline.mel(waveforms)
        h = pipeline.speaker(waveforms.float(), lengths)
        original_waveform = pipeline.vocode(original_mel).float()
        reference = de_encoder(original_waveform, lengths)
        content_codes = pipeline.content(original_mel, h)
        baseline_waveform = pipeline.vocode(
            pipeline.stage1.decode(original_mel, h, content=content_codes)).float()
        baseline_waveform, _ = pipeline.align(baseline_waveform, original_waveform)
        baseline_de = float(cosine_similarity(
            de_encoder(baseline_waveform, lengths), reference).mean())
    permutation = torch.tensor([(index + 1) % args.count for index in range(args.count)])

    delta = torch.nn.Parameter(torch.randn_like(h) * 0.1)
    optimizer = torch.optim.Adam([delta], lr=args.lr)

    print("=" * 70)
    print(f"  自由 δ oracle · α={args.alpha} · L_de 在 {args.space} 空间 · {args.count} 条"
          f" · δ={args.delta_mode} · 参照=E_spk(V(x))")
    print("=" * 70)
    print(f"{'步':>4} {'L_de':>8} {'sim_prot':>9} {'prot@de':>9} {'‖δ‖':>7} {'角度°':>7} "
          f"{'mel':>8} {'|Δwav|':>8}")

    history = []
    best = (float("inf"), None)

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        raw_norm = delta.detach().norm(dim=1, keepdim=True).clamp_min(1e-12)
        delta_used = delta / raw_norm if args.delta_mode == "unit" else delta
        h_adv = perturb_embedding(h, delta_used, args.alpha)
        protected_mel = pipeline.stage1.decode(original_mel, h_adv, content=content_codes)
        protected_waveform = pipeline.vocode(protected_mel).float()
        stop = min(protected_waveform.shape[-1], original_waveform.shape[-1])
        protected_waveform, trimmed = protected_waveform[..., :stop], original_waveform[..., :stop]
        stolen = pipeline.speaker(protected_waveform, lengths)
        cloned_mel = pipeline.stage1.decode(original_mel, stolen,
                                            content=content_codes[permutation])
        cloned_waveform = pipeline.vocode(cloned_mel).float()
        loss = cosine_similarity(de_encoder(cloned_waveform, lengths), reference).mean()
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best[0]:
            best = (value, step)

        if step % 10 == 0 or step == 1:
            with torch.no_grad():
                eff = float(delta.detach().norm(dim=1).mean())
                if args.delta_mode == "unit":
                    eff = 1.0
                angle = math.degrees(math.atan2(args.alpha * eff, 1.0))
                sim_prot = float(cosine_similarity(stolen, h).mean())
                sim_prot_de = float(cosine_similarity(
                    de_encoder(protected_waveform, lengths), reference).mean())
                mel_mse = float((protected_mel - original_mel).square().mean())
                wave_mae = float((protected_waveform - trimmed).abs().mean())
                history.append({"step": step, "l_de": value, "sim_prot": sim_prot,
                                "sim_prot_de_space": sim_prot_de, "delta_norm": eff,
                                "angle_degrees": angle, "mel_mse": mel_mse,
                                "waveform_mae": wave_mae})
                print(f"{step:>4} {value:>8.4f} "
                      f"{sim_prot:>9.4f} {sim_prot_de:>9.4f} {eff:>7.3f} "
                      f"{angle:>7.2f} {mel_mse:>8.5f} {wave_mae:>8.5f}")

    print("-" * 70)
    print(f"  最低 L_de = {best[0]:.4f}（第 {best[1]} 步）  ‖δ‖ 约束 = {args.delta_mode}")
    if args.output:
        report = {
            "checkpoint": CKPT,
            "de_reference": "E_spk(V(x))",
            "h_adv": "h + alpha * delta",
            "args": vars(args),
            "baseline_l_de": baseline_de,
            "initial_l_de": history[0]["l_de"],
            "best_l_de": best[0],
            "best_step": best[1],
            "history": history,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  写出 {output}")


if __name__ == "__main__":
    main()
