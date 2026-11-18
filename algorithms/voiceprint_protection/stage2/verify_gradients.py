"""Stage 2 自检：梯度到底流不流得回生成器 G？

【为什么必须有这个脚本】
指南第 6 步的伪代码把整段前向包进了 `with torch.no_grad()`。
**照抄的后果**：梯度被切断 → G 收不到任何梯度 → loss 一步都不动，
而且**不报错**。这是最难查的一类 bug（不报错，只是"没效果"）。

本脚本把这件事变成**一条会失败的断言**：
    · G 的参数梯度必须非零
    · 单独对 L_de 反传，梯度也必须非零
      （这验证了梯度确实穿过了声码器、Stage 1 解码器、E_spk 三层冻结组件）
    · 关掉 λ_GAN 时，梯度依然非零

用法：
    conda activate vpp
    python algorithms/voiceprint_protection/stage2/verify_gradients.py
"""

from __future__ import annotations

import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR.parent))

try:
    import torch  # noqa: E402

    import config  # noqa: E402
except ImportError as error:  # 依赖缺失（多半是忘了 conda activate vpp）
    from paths import print_env_hint  # noqa: E402

    print_env_hint(error)
    raise SystemExit(1) from error
from stage2 import losses as loss_module  # noqa: E402
from stage2.common import perturb_embedding, set_seed  # noqa: E402
from stage2.discriminator import MelDiscriminator  # noqa: E402
from stage2.generator import PerturbationGenerator  # noqa: E402
from stage2.models import StubStage1, TorchSpeakerEncoder, freeze  # noqa: E402
from stage2.pipeline import Stage2Pipeline  # noqa: E402

PASSED = 0
FAILED = 0


def check(ok: bool, name: str, extra: str = "") -> bool:
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  [通过] {name}")
    else:
        FAILED += 1
        print(f"  [失败] {name}" + (f"  —— {extra}" if extra else ""))
    return ok


def grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return total ** 0.5


def forward_pass(pipeline, generator, waveforms, lengths):
    """跑一遍完整前向，返回各中间量。

    ★ 每次检查都**重新前向**一遍 —— `backward()` 会把图释放掉，
      同一张图不能反传两次（那会报 "backward through the graph a second time"）。
      真实的训练循环本来也是每步重新前向。
    """
    with torch.no_grad():
        original_mel = pipeline.mel(waveforms)
        original_embedding = pipeline.speaker(waveforms, lengths)

    delta = generator(original_embedding)
    advanced = perturb_embedding(original_embedding, delta, 0.1)
    protected_mel = pipeline.stage1.decode(original_mel, advanced)
    protected_waveform = pipeline.vocode(protected_mel)
    with torch.no_grad():
        original_waveform = pipeline.vocode(original_mel)
    protected_waveform, original_waveform = pipeline.align(protected_waveform, original_waveform)
    delta_waveform = protected_waveform - original_waveform

    stolen = pipeline.speaker(protected_waveform, lengths)
    content = pipeline.content(original_mel, original_embedding)
    permutation = torch.randperm(waveforms.shape[0])
    cloned_mel = pipeline.stage1.decode(original_mel, stolen, content=content[permutation])
    cloned_waveform = pipeline.vocode(cloned_mel)

    defense = loss_module.defense_loss(pipeline.speaker(cloned_waveform, lengths),
                                       original_embedding)
    perceptual = loss_module.perceptual_loss(original_mel, protected_mel,
                                              delta_waveform, None)
    return {
        "original_mel": original_mel,
        "original_embedding": original_embedding,
        "delta": delta,
        "protected_mel": protected_mel,
        "protected_waveform": protected_waveform,
        "stolen": stolen,
        "defense": defense,
        "perceptual": perceptual,
    }


def main() -> int:
    set_seed(config.SEED)
    device = torch.device("cpu")   # CPU 就够验梯度通路

    print("=" * 66)
    print("  Stage 2 梯度通路自检")
    print("=" * 66)

    speaker = TorchSpeakerEncoder(device="cpu")
    freeze(speaker)
    stage1 = StubStage1(device="cpu").to(device)
    freeze(stage1)
    pipeline = Stage2Pipeline(speaker_encoder=speaker, stage1=stage1,
                              vocoder=stage1.vocoder, device=device)
    generator = PerturbationGenerator().to(device)

    # -- 造一批假数据（不必真读 VCTK，验的是通路）------------------------
    torch.manual_seed(0)
    waveforms = (0.2 * torch.randn(2, 16000 * 2)).to(device)
    lengths = torch.tensor([16000 * 2, 16000 * 2])
    print(f"\n测试样本：{tuple(waveforms.shape)} 波形 / {tuple(lengths)} 长度")

    # -- 1. 前向 ----------------------------------------------------------
    print("\n【1】前向：形状与梯度图")
    state = forward_pass(pipeline, generator, waveforms, lengths)
    original_mel = state["original_mel"]
    original_embedding = state["original_embedding"]
    check(original_mel.dim() == 3 and original_mel.shape[2] == config.FRAME_DIM,
          f"原始 mel 形状 (B, T, 80) —— 实际 {tuple(original_mel.shape)}")
    check(original_embedding.shape == (2, config.SPEAKER_DIM),
          f"说话人嵌入形状 (B, 256) —— 实际 {tuple(original_embedding.shape)}")

    delta = state["delta"]
    check(delta.shape == (2, config.SPEAKER_DIM), "δ 形状 (B, 256)")
    check(abs(float(torch.linalg.vector_norm(delta[0])) - 1.0) < 1e-4,
          "δ 是单位长度（α 是唯一的强度旋钮）")

    protected_mel = state["protected_mel"]
    check(protected_mel.shape == original_mel.shape,
          f"x_adv 与 x 帧数相等（契约 §2）—— {tuple(protected_mel.shape)} vs "
          f"{tuple(original_mel.shape)}")

    check(bool(state["protected_waveform"].requires_grad),
          "wav_adv 带梯度（说明梯度穿过了声码器）")
    check(bool(state["stolen"].requires_grad),
          "h_stolen 带梯度（说明梯度穿过了 E_spk）")

    # -- 2. L_de 的梯度 ----------------------------------------------------
    print("\n【2】L_de 单独反传：梯度必须能穿过三层冻结组件")
    state = forward_pass(pipeline, generator, waveforms, lengths)
    generator.zero_grad(set_to_none=True)
    state["defense"].backward()
    norm = grad_norm(generator)
    check(bool(state["defense"].requires_grad), "L_de 带梯度")
    check(norm > 0, f"梯度确实到达 G（‖grad‖ = {norm:.3e}）",
          "梯度是 0 —— 某处的 no_grad()/detach() 把通路切断了")
    print(f"         L_de = {float(state['defense']):.4f}"
          f"（越小越好：攻击者克隆音越不像本人）")

    # -- 3. 总损失的梯度 ----------------------------------------------------
    print("\n【3】总损失 L_G 反传")
    state = forward_pass(pipeline, generator, waveforms, lengths)
    generator.zero_grad(set_to_none=True)
    total = loss_module.generator_loss(torch.zeros(()), state["defense"], state["perceptual"])
    total.backward()
    norm_total = grad_norm(generator)
    check(norm_total > 0, f"总损失的梯度到达 G（‖grad‖ = {norm_total:.3e}）")

    # -- 4. 关掉 GAN 也要有梯度 ---------------------------------------------
    print("\n【4】λ_GAN = 0 时梯度依然存在")
    state = forward_pass(pipeline, generator, waveforms, lengths)
    generator.zero_grad(set_to_none=True)
    only_per_de = (config.STAGE2["lambda_de"] * state["defense"]
                   + config.STAGE2["lambda_per"] * state["perceptual"])
    only_per_de.backward()
    norm_no_gan = grad_norm(generator)
    check(norm_no_gan > 0, f"仅 L_per + L_de 的梯度到达 G（‖grad‖ = {norm_no_gan:.3e}）")

    # -- 5. 判别器的梯度不会污染生成器 ---------------------------------------
    print("\n【5】判别器更新时生成器不受影响")
    discriminator = MelDiscriminator().to(device)
    state = forward_pass(pipeline, generator, waveforms, lengths)
    real_scores = discriminator(state["original_mel"])
    fake_scores = discriminator(state["protected_mel"].detach())
    discriminator.zero_grad(set_to_none=True)
    generator.zero_grad(set_to_none=True)
    loss_module.discriminator_gan_loss(real_scores, fake_scores).backward()
    check(grad_norm(generator) == 0.0,
          "判别器反传时 G 的梯度为 0（x_adv 已 detach）")

    print("\n" + "=" * 66)
    print(f"  通过 {PASSED} 项，失败 {FAILED} 项")
    print("=" * 66)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
