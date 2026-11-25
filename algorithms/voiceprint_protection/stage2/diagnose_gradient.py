"""诊断：上游口径的 L_de 到底能不能把梯度送到 G（以及有多大）。

用法（仓库根执行）：
    D:/conda/envs/vpp/python.exe algorithms/voiceprint_protection/stage2/diagnose_gradient.py upstream
    D:/conda/envs/vpp/python.exe algorithms/voiceprint_protection/stage2/diagnose_gradient.py stage1
第二个参数可以覆盖 Stage 1 权重路径。
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path("algorithms/voiceprint_protection").resolve()))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

import config
from stage2.common import set_seed
from stage2.dataset import build_dataloader
from stage2.discriminator import MelDiscriminator
from stage2.generator import PerturbationGenerator
from stage2.models import Stage1SpeakerEncoder, TorchAutoVCStage1, TorchSpeakerEncoder, freeze
from stage2.pipeline import HiFiGANVocoder, Stage2Pipeline
from stage2.train import compute_losses

CKPT = "algorithms/voiceprint_protection/weights/stage1_fresh_200k/stage1_step250000_wide_test_bn.pt"


def grad_norm(module):
    total = 0.0
    per_layer = {}
    for name, param in module.named_parameters():
        if param.grad is None:
            per_layer[name] = 0.0
            continue
        value = float(param.grad.detach().norm())
        per_layer[name] = value
        total += value * value
    return total ** 0.5, per_layer


def main():
    de_space = sys.argv[1] if len(sys.argv) > 1 else "upstream"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else CKPT
    device = torch.device("cpu")
    set_seed(config.SEED)  # 含 Python 的 random，否则数据集随机裁剪每次不同

    args = SimpleNamespace(alpha=0.10, lambda_de=5.0, lambda_per=0.0, lambda_mask=0.0,
                           lambda_gan=0.0, no_mel_loss=True, de_space=de_space,
                           speaker_encoder="stage1", stage1_checkpoint=ckpt,
                           stub=False, vocoder="hifigan", max_seconds=2.0)

    speaker = Stage1SpeakerEncoder(ckpt, device="cpu")
    freeze(speaker)
    de_encoder = TorchSpeakerEncoder(device="cpu") if de_space == "upstream" else None
    if de_encoder is not None:
        freeze(de_encoder)
    stage1 = TorchAutoVCStage1(checkpoint=ckpt, device="cpu")
    freeze(stage1)
    pipeline = Stage2Pipeline(speaker_encoder=speaker, stage1=stage1,
                              vocoder=HiFiGANVocoder(device="cpu"), device=device,
                              de_speaker=de_encoder)

    generator = PerturbationGenerator(dim=config.SPEAKER_DIM, hidden=512, layers=2)
    discriminator = MelDiscriminator()

    loader = build_dataloader("train", batch_size=4, max_seconds=2.0, num_workers=0,
                              shuffle=False)
    batch = next(iter(loader))

    out = compute_losses(batch, pipeline, generator, discriminator, args, device,
                         autocast_enabled=False, measure_dual=True)
    out["total"].backward()
    total_norm, per_layer = grad_norm(generator)

    dual = out["sim_clone_de_space"]
    print("=" * 62)
    print(f"  L_de 在 **{de_space}** 空间  ⇒  ‖∂(λ_de·L_de)/∂θ_G‖ = {total_norm:.4e}")
    print("=" * 62)
    print(f"  de = {float(out['de'].detach()):.4f}   "
          f"sim_clone = {float(out['sim_clone'].detach()):.4f}   "
          f"sim@de = {'—（两个空间同一个，不另算）' if dual is None else f'{float(dual.detach()):.4f}'}")
    for name, value in per_layer.items():
        print(f"    {name:<22} {value:.4e}")

    # δ 的量级：α·‖δ‖ 相对单位球面上的 h 有多大（决定防御上限）
    generator.zero_grad(set_to_none=True)
    waveforms, lengths, _ = batch
    with torch.no_grad():
        h = pipeline.speaker(waveforms.float(), lengths)
        delta = generator(h)
    print(f"\n  ‖h‖ = {float(h.norm(dim=1).mean()):.4f}   "
          f"‖δ‖ = {float(delta.norm(dim=1).mean()):.4f}   "
          f"α·‖δ‖/‖h‖ = {0.10 * float(delta.norm(dim=1).mean()) / float(h.norm(dim=1).mean()):.4f}  "
          f"⇒ 角度预算约 {57.3 * 0.10 * float(delta.norm(dim=1).mean()) / float(h.norm(dim=1).mean()):.2f}°")


if __name__ == "__main__":
    main()
