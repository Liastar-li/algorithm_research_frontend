"""α–Sim_clone 扫描（等预算 ‖δ‖=1，自由 δ 作为**上界**）。

回答指南 §11 的"α 敏感吗"，同时给"放大链到底能吃下多少预算"定量答案：
每个 α 都先把 δ 自由优化 `--steps` 步，记录能到的最低 L_de（= 上游口径 Sim_clone 的下界），
再记录随机 δ（第 1 步）的数 —— 两者之差就是"学"能贡献的部分，
不同 α 之间的差就是"预算"能贡献的部分。

用法（仓库根，用 vpp 的 python）：
    python -u algorithms/voiceprint_protection/stage2/diagnose_alpha_sweep.py --steps 20 --count 4
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
    parser.add_argument("--alphas", default="0.0,0.1,0.2,0.3,0.5,1.0")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=2.0)
    parser.add_argument("--out", default="results/alpha_sweep.json",
                        help="扫描结果 JSON（目录不存在会自动建；results/ 已在 .gitignore 里）")
    args = parser.parse_args()

    # set_seed 而不是 torch.manual_seed：dataset 的随机裁剪用的是 Python 的 random，
    # 只固定 torch 会让每次启动拿到不同的音频片段，整表数字跟着漂 ~0.05（2026-11-25 实测）。
    set_seed(config.SEED)
    speaker = Stage1SpeakerEncoder(CKPT, device="cpu")
    stage1 = TorchAutoVCStage1(checkpoint=CKPT, device="cpu")
    vocoder = HiFiGANVocoder(device="cpu")
    de_encoder = TorchSpeakerEncoder(device="cpu")
    for module in (speaker, stage1, vocoder, de_encoder):
        freeze(module)
    pipeline = Stage2Pipeline(speaker_encoder=speaker, stage1=stage1, vocoder=vocoder,
                              device="cpu", de_speaker=de_encoder)

    loader = build_dataloader("train", batch_size=args.count, max_seconds=args.max_seconds,
                              num_workers=0, shuffle=False)
    waveforms, lengths, _ = next(iter(loader))
    with torch.no_grad():
        original_mel = pipeline.mel(waveforms)
        h = speaker(waveforms.float(), lengths)
        reference = de_encoder(waveforms.float(), lengths)
        original_waveform = pipeline.vocode(original_mel).float()
        content_codes = pipeline.content(original_mel, h)
    permutation = torch.tensor([(i + 1) % args.count for i in range(args.count)])

    def evaluate(delta_used, alpha):
        """给定 δ，跑完整条放大链，返回 (L_de, Sim_prot, mel 误差, 音频差)."""
        h_adv = perturb_embedding(h, delta_used, alpha)
        protected_mel = pipeline.stage1.decode(original_mel, h_adv, content=content_codes)
        protected_waveform = pipeline.vocode(protected_mel).float()
        stop = min(protected_waveform.shape[-1], original_waveform.shape[-1])
        pw, ow = protected_waveform[..., :stop], original_waveform[..., :stop]
        stolen = speaker(pw, lengths)
        cloned_mel = pipeline.stage1.decode(original_mel, stolen,
                                           content=content_codes[permutation])
        cloned_waveform = pipeline.vocode(cloned_mel).float()
        loss = cosine_similarity(de_encoder(cloned_waveform, lengths), reference).mean()
        return (float(loss.detach()),
                float(cosine_similarity(de_encoder(pw, lengths), reference).mean()),
                float((protected_mel - original_mel).square().mean()),
                float((pw - ow).abs().mean()),
                loss)

    print("=" * 82)
    print(f"  α–Sim_clone 扫描 · 等预算 ‖δ‖=1 · 自由 δ 优化 {args.steps} 步 · {args.count} 条")
    print("=" * 82)
    print(f"{'α':>5} {'角度°':>7} {'随机δ':>8} {'优化后δ':>9} {'学到多少':>9} "
          f"{'Sim_prot':>9} {'mel误差':>9} {'|Δwav|':>8}")

    rows = []
    for alpha in [float(x) for x in args.alphas.split(",")]:
        torch.manual_seed(config.SEED)
        raw = torch.randn_like(h)
        with torch.no_grad():
            unit = raw / raw.norm(dim=1, keepdim=True)
            sim_prot0 = evaluate(unit, alpha)[:4]
            random_de = sim_prot0[0]
        if alpha > 0 and args.steps:
            delta = torch.nn.Parameter(raw.clone())
            opt = torch.optim.Adam([delta], lr=args.lr)
            best = float("inf")
            for step in range(args.steps):
                opt.zero_grad(set_to_none=True)
                used = delta / delta.detach().norm(dim=1, keepdim=True).clamp_min(1e-12)
                _, _, _, _, loss = evaluate(used, alpha)
                loss.backward()
                opt.step()
                best = min(best, float(loss.detach()))
        else:
            best = random_de
        with torch.no_grad():
            used = delta / delta.norm(dim=1, keepdim=True) if (alpha > 0 and args.steps) else unit
            de, sim_prot, mel_err, dwav = evaluate(used, alpha)[:4]
            de = best if (alpha > 0 and args.steps) else de
        angle = math.degrees(math.atan2(alpha, 1.0))
        print(f"{alpha:>5.2f} {angle:>7.2f} {random_de:>8.4f} {de:>9.4f} "
              f"{random_de - de:>9.4f} {sim_prot:>9.4f} {mel_err:>9.5f} {dwav:>8.5f}")
        rows.append({"alpha": alpha, "angle_deg": angle, "de_random_delta": random_de,
                     "de_best_free_delta": de, "learned_gain": random_de - de,
                     "sim_prot_upstream": sim_prot, "mel_mse": mel_err,
                     "abs_waveform_diff": dwav})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "note": "等预算 ‖δ‖=1 的自由 δ 上界；L_de = 上游 GE2E 口径的 Sim_clone（要最小化）",
        "count": args.count, "max_seconds": args.max_seconds, "steps": args.steps,
        "lr": args.lr, "delta_mode": "unit", "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
