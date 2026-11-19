"""Stage 2 训练主循环（成员 B 主责，论文 §2.3 / 指南第 6 步）。

【论文的 Stage 2 在干什么】
冻结：Stage 1 的 Ec/D/PostNet、E_spk、声码器 V
可训：生成器 G、判别器 D_adv

    h      = E_spk(x)                            冻结
    δ      = G(h)                                ★ 训练
    h_adv  = h + α·δ                             ★ 梯度从这里往回传
    x_adv  = D(c, h_adv)                         冻结（但梯度要穿过它）
    wav_adv= V(x_adv)                            冻结（但梯度要穿过它）
    h_stolen = E_spk(wav_adv)                    冻结（但梯度要穿过它）← 攻击者重新提卡
    x_clone  = D(c_arb, h_stolen)                冻结（但梯度要穿过它）← 攻击者拿去克隆
    wav_clone= V(x_clone)                        冻结（但梯度要穿过它）

    L_per  = L_mel + λ_mask·L_mask               听感约束
    L_de   = sim(E_spk(wav_clone), E_spk(wav))   ★ 要**最小化**
    L_GAN  = E[(D_adv(x_adv) − 1)²]
    L_D    = E[(D_adv(x)−1)²] + E[D_adv(x_adv)²]
    L_G    = L_GAN + λ_de·L_de + λ_per·L_per

【★★ 一个必须讲清楚的点：指南伪代码里的 `with torch.no_grad()`】
指南第 6 步的伪代码把整段前向都包进了 `torch.no_grad()`。
**照抄会让训练完全不工作** —— 梯度被切断，G 收不到任何梯度，
loss 一步都不动，而且**不报错**（这是最难查的一类 bug）。

正确的理解是：
  * **冻结** = 不更新这些模块的**参数** → `requires_grad_(False)` + `eval()`
  * **梯度仍然要流过**它们 → 不要包 `no_grad`，不要 `.detach()`
真正切断梯度的是 `torch.no_grad()` 与 `.detach()`，不是"冻结"。
本文件全程**没有**用 `no_grad` 包住前向（只有原始语音那一段用，那里确实不需要梯度）。

【怎么跑】
    # 离线冒烟测试（用随机权重的替身，验证训练机制跑得通，不看指标）
    python algorithms/voiceprint_protection/stage2/train.py --stub --steps 20

    # 正式训练（需要成员 A 的 Stage 1 checkpoint）
    python algorithms/voiceprint_protection/stage2/train.py --steps 200000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR.parent))

try:
    import torch  # noqa: E402

    import config  # noqa: E402
    import paths  # noqa: E402
except ImportError as error:  # 依赖缺失（多半是忘了 conda activate vpp）
    import paths as _paths  # noqa: E402

    _paths.print_env_hint(error)
    raise SystemExit(1) from error

from stage2 import losses as loss_module  # noqa: E402
from stage2.common import cosine_similarity, perturb_embedding, set_seed  # noqa: E402
from stage2.dataset import build_dataloader, describe_dataset  # noqa: E402
from stage2.discriminator import MelDiscriminator  # noqa: E402
from stage2.generator import PerturbationGenerator, RandomPerturbationGenerator  # noqa: E402
from stage2.models import (  # noqa: E402
    StubStage1,
    TorchAutoVCStage1,
    TorchSpeakerEncoder,
    describe,
    freeze,
)
from stage2.pipeline import HiFiGANVocoder, Stage2Pipeline  # noqa: E402


# ---------------------------------------------------------------------------
# 组件装配
# ---------------------------------------------------------------------------
def build_pipeline(args, device) -> Stage2Pipeline:
    """把 Stage 2 需要的东西都装好。

    ★ 三个冻结组件都必须**保持梯度可流过**，只是参数不更新。
    """
    # -- E_spk（上游 GE2E，权重现在就有）--------------------------------
    print("加载说话人编码器 E_spk（上游 GE2E）…")
    speaker = TorchSpeakerEncoder(device=str(device))
    freeze(speaker)
    print(f"  {describe(speaker)}")

    # -- Stage 1（Ec + D）------------------------------------------------
    if args.stub:
        print("使用 StubStage1（随机权重替身）—— 只验证训练机制，指标无意义")
        stage1 = StubStage1(device=str(device)).to(device)
    else:
        print("加载 Stage 1（AutoVC 架构）…")
        stage1 = TorchAutoVCStage1(checkpoint=args.stage1_checkpoint, device=str(device))
    freeze(stage1)
    print(f"  {describe(stage1)}")

    # -- 声码器 V --------------------------------------------------------
    vocoder = None
    convention = "hifigan"
    if args.stub:
        vocoder = stage1.vocoder          # StubStage1 自带一个可导的替身
        convention = "hifigan"
        print("声码器 V：StubVocoder（随机权重，指标无意义）")
    elif args.vocoder == "hifigan":
        vocoder = HiFiGANVocoder(device=str(device))
        convention = "hifigan"
        print("声码器 V：HiFi-GAN（论文同款；★ 不再做 mel 口径近似换算）")
    else:
        raise SystemExit(
            "正式训练需要一个**可求导**的声码器。\n"
            "  --vocoder hifigan  论文同款（★ NVIDIA 官方仓库已下架，用 HF 上等价的\n"
            "                     speechbrain/tts-hifigan-libritts-16kHz）\n"
            "  --vocoder stub     随机权重，只用来验证训练机制\n"
            "详见 docs/notes/stage2_blockers.md"
        )

    return Stage2Pipeline(
        speaker_encoder=speaker,
        stage1=stage1,
        vocoder=vocoder,
        vocoder_convention=convention,
        device=device,
    )


# ---------------------------------------------------------------------------
# 一步训练
# ---------------------------------------------------------------------------
def train_step(batch, pipeline: Stage2Pipeline, generator, discriminator,
               optimizer_g, optimizer_d, args, device) -> dict[str, float]:
    """跑一步，返回各损失（用于日志）。

    batch : (waveforms (B,L), lengths (B,), speakers (B,))
    """
    waveforms, lengths, _ = batch
    waveforms = waveforms.to(device)
    lengths = lengths.to(device)

    batch_size = waveforms.shape[0]
    alpha = args.alpha

    # -- 1. 原始语音的 mel 与说话人嵌入（**不需要梯度**）------------------
    with torch.no_grad():
        original_mel = pipeline.mel(waveforms)             # (B, T, 80)
        original_embedding = pipeline.speaker(waveforms, lengths)   # (B, 256)

    # -- 2. 生成器出扰动（★ 从这里开始有梯度）---------------------------
    delta = generator(original_embedding)                   # (B, 256)
    advanced_embedding = perturb_embedding(original_embedding, delta, alpha)

    # -- 3. 用被污染的嵌入重建保护后的 mel（梯度穿过冻结的 Stage 1）------
    protected_mel = pipeline.stage1.decode(original_mel, advanced_embedding)
    if protected_mel.shape[1] != original_mel.shape[1]:
        raise RuntimeError(
            f"x 与 x_adv 的帧数必须相等（契约 §2 的形状不变量），"
            f"实际 {original_mel.shape[1]} vs {protected_mel.shape[1]}"
        )

    # -- 4. 过声码器得到波形（L_mask 要用波形域的扰动）--------------------
    protected_waveform = pipeline.vocode(protected_mel)             # (B, N)
    original_waveform = pipeline.vocode(original_mel)              # (B, N0)
    protected_waveform, original_waveform = pipeline.align(
        protected_waveform, original_waveform
    )
    delta_waveform = protected_waveform - original_waveform

    # -- 5. 模拟攻击：攻击者从保护后语音重新提卡，再拿去克隆 ----------------
    stolen_embedding = pipeline.speaker(protected_waveform, lengths)  # ★ 梯度穿过声码器
    permutation = torch.randperm(batch_size, device=device)
    content_codes = pipeline.content(original_mel, original_embedding)
    arbitrary_content = content_codes[permutation]                    # 论文的 "arbitrary content"
    cloned_mel = pipeline.stage1.decode(original_mel, stolen_embedding,
                                        content=arbitrary_content)
    cloned_waveform = pipeline.vocode(cloned_mel)

    # -- 6. 算损失 --------------------------------------------------------
    thresholds = pipeline.thresholds(original_waveform)
    perceptual = loss_module.perceptual_loss(
        original_mel, protected_mel, delta_waveform, thresholds,
        lambda_mask=args.lambda_mask,
    )
    defense = loss_module.defense_loss(
        pipeline.speaker(cloned_waveform, lengths), original_embedding
    )
    gan = loss_module.generator_gan_loss(discriminator(protected_mel))

    total = loss_module.generator_loss(
        gan, defense, perceptual,
        lambda_de=args.lambda_de, lambda_per=args.lambda_per, lambda_gan=args.lambda_gan,
    )

    # -- 7. 更新生成器 -----------------------------------------------------
    optimizer_g.zero_grad(set_to_none=True)
    if args.lambda_gan > 0:
        total.backward()
    else:
        # 关掉 GAN 时，判别器那部分不参与（但 L_per / L_de 还要）
        (args.lambda_de * defense + args.lambda_per * perceptual).backward()
    optimizer_g.step()

    # -- 8. 更新判别器（★ x_adv 要 detach，否则会同时改生成器）-----------
    discriminator_loss_value = torch.zeros((), device=device)
    if args.lambda_gan > 0:
        optimizer_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            real_scores = discriminator(original_mel)
        fake_scores_detached = discriminator(protected_mel.detach())
        discriminator_loss_value = loss_module.discriminator_gan_loss(
            real_scores, fake_scores_detached
        )
        discriminator_loss_value.backward()
        optimizer_d.step()

    return {
        "total": float(total.detach()),
        "gan": float(gan.detach()),
        "de": float(defense.detach()),
        "per": float(perceptual.detach()),
        "mel": float(loss_module.mel_loss(original_mel, protected_mel).detach()),
        "mask": float(loss_module.mask_loss(delta_waveform, thresholds).detach()),
        "d": float(discriminator_loss_value.detach()),
        "sim_clone": float(cosine_similarity(stolen_embedding, original_embedding)
                           .mean().detach()),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 训练（防御性扰动生成）")
    parser.add_argument("--stub", action="store_true",
                        help="用随机权重的替身组件（离线冒烟测试，不产出可用模型）")
    parser.add_argument("--vocoder", choices=("stub", "hifigan"), default="stub",
                        help="声码器；stub=随机权重替身，hifigan=论文同款（需联网）")
    parser.add_argument("--steps", type=int, default=100,
                        help="训练步数（正式训练用 config.STAGE2['iterations'] = 200000）")
    parser.add_argument("--alpha", type=float, default=config.STAGE2["alpha"])
    parser.add_argument("--lambda-per", type=float, default=config.STAGE2["lambda_per"])
    parser.add_argument("--lambda-de", type=float, default=config.STAGE2["lambda_de"])
    parser.add_argument("--lambda-mask", type=float, default=config.STAGE2["lambda_mask"])
    parser.add_argument("--lambda-gan", type=float, default=config.STAGE2["lambda_gan"],
                        help="设 0 可先关掉 GAN（指南建议：DSR 有了再打开）")
    parser.add_argument("--perturb", choices=("train", "random"), default="train",
                        help="'random' = 用高斯噪声基线（不训练 G）")
    parser.add_argument("--hidden", type=int, default=512, help="生成器隐层宽度")
    parser.add_argument("--layers", type=int, default=2, help="生成器隐层个数")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--outdir", default=None, help="checkpoint 与日志的输出目录")
    parser.add_argument("--stage1-checkpoint", default=None, help="成员 A 的 Stage 1 checkpoint")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    outdir = Path(args.outdir) if args.outdir else paths.WEIGHTS_DIR / "stage2"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  Stage 2 训练 · 防御性扰动生成")
    print("=" * 68)
    print(f"  设备        : {device}")
    print(f"  α           : {args.alpha}")
    print(f"  λ_per/λ_de  : {args.lambda_per} / {args.lambda_de}")
    print(f"  λ_mask      : {args.lambda_mask}")
    print(f"  λ_GAN       : {args.lambda_gan}"
          f"{'（已关闭：先只训 L_per + L_de）' if args.lambda_gan == 0 else ''}")
    print(f"  步数        : {args.steps}")
    print()

    pipeline = build_pipeline(args, device)
    print()
    print(pipeline.describe())
    print()

    # -- 可训练组件 --------------------------------------------------------
    if args.perturb == "random":
        generator = RandomPerturbationGenerator(seed=args.seed)
        print("\n生成器 G：随机方向（高斯噪声基线，不训练）")
    else:
        generator = PerturbationGenerator(
            dim=config.SPEAKER_DIM, hidden=args.hidden, layers=args.layers
        ).to(device)
        print(f"\n生成器 G：{describe(generator)}")
    discriminator = MelDiscriminator().to(device)
    print(f"判别器 D_adv：{describe(discriminator)}")

    # -- 数据 --------------------------------------------------------------
    loader = build_dataloader("train", batch_size=config.STAGE2["batch_size"])
    print(f"\n数据：{describe_dataset(loader.dataset)}\n")

    # -- 优化器 ------------------------------------------------------------
    optimizer_g = torch.optim.AdamW(
        [p for p in generator.parameters() if p.requires_grad],
        lr=config.STAGE2["lr"],
    )
    optimizer_d = torch.optim.AdamW(
        [p for p in discriminator.parameters() if p.requires_grad],
        lr=config.STAGE2["lr"], betas=(0.5, 0.9),
    )

    # -- 训练循环 ----------------------------------------------------------
    log_path = outdir / "train_log.jsonl"
    step = 0
    started = time.time()
    print(f"{'步':>8}  {'total':>9} {'gan':>8} {'de':>8} {'per':>9} "
          f"{'mel':>9} {'mask':>8} {'sim_clone':>10}")
    print("-" * 68)

    generator.train()
    discriminator.train()

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            metrics = train_step(batch, pipeline, generator, discriminator,
                                 optimizer_g, optimizer_d, args, device)

            if step % args.log_every == 0 or step == args.steps - 1:
                elapsed = time.time() - started
                print(f"{step:>8}  {metrics['total']:>9.4f} {metrics['gan']:>8.4f} "
                      f"{metrics['de']:>8.4f} {metrics['per']:>9.4f} {metrics['mel']:>9.4f} "
                      f"{metrics['mask']:>8.4f} {metrics['sim_clone']:>10.4f}"
                      f"   [{elapsed:.0f}s]")
                metrics["step"] = step
                metrics["elapsed"] = round(elapsed, 1)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")

            if (step + 1) % args.save_every == 0:
                save_checkpoint(generator, discriminator, optimizer_g, optimizer_d,
                                step, outdir)
            step += 1

    save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, step, outdir)

    print("-" * 68)
    print(f"训练结束，共 {step} 步，用时 {time.time() - started:.0f} 秒")
    print(f"checkpoint: {outdir / 'stage2_last.pt'}")
    print(f"日志      : {log_path}")
    print()
    print("★ 下一步：跑 `evaluate.py` 算 Sim_prot / Sim_clone / DSR")
    print("  ★ L_de 应该在下降。如果它在**上升**，说明符号反了 —— 见 losses.py 的警告。")
    return 0


def save_checkpoint(generator, discriminator, optimizer_g, optimizer_d,
                    step: int, outdir: Path) -> Path:
    path = outdir / "stage2_last.pt"
    torch.save({
        "step": step,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
    }, path)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
