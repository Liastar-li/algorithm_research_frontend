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
    · ★ Stage 1 <-> Stage 2 的**接口**（成员 A 修的两个致命 bug 的回归测试）：\n      content_code 的形状、h_adv 有没有真的进解码器、content 有没有被用上、\n      梯度能不能回到 h_adv、decode() 与官方 AutoVC Generator 是否逐值一致\n
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
from stage2.models import (  # noqa: E402
    StubStage1,
    TorchAutoVCStage1,
    TorchSpeakerEncoder,
    freeze,
)
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
    # ★ 2026-11-19（成员 A）：内容码要先算（用原说话人的嵌入编码原句），
    #   再拿它去重建 —— 否则 c_adv 里会混进扰动。
    content = pipeline.content(original_mel, original_embedding)
    protected_mel = pipeline.stage1.decode(original_mel, advanced, content=content)
    protected_waveform = pipeline.vocode(protected_mel)
    with torch.no_grad():
        original_waveform = pipeline.vocode(original_mel)
    protected_waveform, original_waveform = pipeline.align(protected_waveform, original_waveform)
    delta_waveform = protected_waveform - original_waveform

    stolen = pipeline.speaker(protected_waveform, lengths)
    permutation = torch.randperm(waveforms.shape[0])
    cloned_mel = pipeline.stage1.decode(original_mel, stolen, content=content[permutation])
    cloned_waveform = pipeline.vocode(cloned_mel)

    defense = loss_module.defense_loss(pipeline.speaker(cloned_waveform, lengths),
                                       pipeline.speaker(original_waveform, lengths))
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

    # -- 6. Stage 1 <-> Stage 2 的接口（★ 成员 A 修的两个致命 bug 的回归测试）----
    #
    # 【为什么必须有这一段】
    # A 修的两个 bug 都在 `stage2/models.py::TorchAutoVCStage1`：
    #   (1) `self.model(mel, embedding, embedding)` —— 官方 forward 的第三个参数
    #       才是目标嵌入，两个都传 embedding，于是 h_adv 根本没进解码器；
    #   (2) `decode()` 把 content 参数完全忽略 —— 论文的 "arbitrary content" 失效。
    # 这两个 bug 的症状是"训练能跑、loss 在降，但防御机制根本没生效"，而且**不报错**。
    #
    # ★ 它们之所以能溜过去，正是因为本文件原来**只验 StubStage1** ——
    #   而 StubStage1 的 decode 逻辑跟官方 Generator 根本不是一回事。
    #   所以这里直接拿**官方 AutoVC 的 Generator** 来验接口。
    #
    # ★ 用 `random_init=True`：权重是随机的，**指标毫无意义**，
    #   但这几条断言验的是**接口**（形状 / 有没有传对 / 梯度通不通），
    #   它们**不依赖训练好的权重** —— Stage 1 还没训完也能验。
    print("\n【6】★ Stage 1 <-> Stage 2 的接口（h_adv 进解码器 / content 被使用）")
    stage1_real = TorchAutoVCStage1(device="cpu", random_init=True)
    freeze(stage1_real)

    torch.manual_seed(1)
    test_mel = torch.randn(2, 96, config.FRAME_DIM)          # T = 96 = 3 x freq(32)
    test_emb = torch.randn(2, config.SPEAKER_DIM)
    test_emb_adv = torch.nn.functional.normalize(
        test_emb + 0.1 * torch.randn(2, config.SPEAKER_DIM), dim=-1)

    codes = stage1_real.content_code(test_mel, test_emb)
    check(isinstance(codes, torch.Tensor),
          "content_code 返回**张量**（不是 list）—— train.py 要写 content_codes[permutation]")
    check(codes.dim() == 3 and codes.shape[0] == 2
          and codes.shape[2] == config.CONTENT_ACTUAL_DIM,
          f"content_code 形状 (B, T', 64) —— 实际 {tuple(codes.shape)}")
    check(codes.shape[1] == test_mel.shape[1] // config.CONTENT_DOWNSAMPLE,
          f"内容码帧数 T' = T // {config.CONTENT_DOWNSAMPLE} —— 实际 {codes.shape[1]}"
          f"（T = {test_mel.shape[1]}）")

    # (a) h_adv 真的进了解码器
    out_h = stage1_real.decode(test_mel, test_emb, content=codes)
    out_adv = stage1_real.decode(test_mel, test_emb_adv, content=codes)
    embedding_difference = float((out_h - out_adv).abs().mean())
    check(embedding_difference > 1e-6,
          f"h_adv 真的进了解码器（D(c,h) 与 D(c,h_adv) 的平均差 = {embedding_difference:.6f}）",
          "差为 0 —— 目标嵌入没传进解码器，整个防御机制等于没生效")

    # (b) 梯度能回到 h_adv
    test_emb_grad = test_emb_adv.clone().requires_grad_(True)
    stage1_real.decode(test_mel, test_emb_grad, content=codes).abs().mean().backward()
    gradient = test_emb_grad.grad
    check(gradient is not None and float(gradient.abs().sum()) > 0,
          "梯度能回到 h_adv",
          "梯度是 0 —— h_adv 到解码器之间被 detach()/no_grad() 切断了")

    # (c) content 真的被用上
    out_other = stage1_real.decode(test_mel, test_emb, content=codes.flip(0))
    content_difference = float((out_h - out_other).abs().mean())
    check(content_difference > 1e-6,
          f"content 参数真的被用上了（换内容后的平均差 = {content_difference:.6f}）",
          "差为 0 —— content 被忽略了，论文的 'arbitrary content' 没生效")

    # (d) decode() 与官方 AutoVC Generator 的语义**逐值一致**
    with torch.no_grad():
        official = stage1_real.model(test_mel, test_emb, test_emb_adv)[1].squeeze(1)
    ours = stage1_real.decode(test_mel, test_emb_adv, content=codes)
    official_difference = float((official - ours).abs().max())
    # (e) ★ 帧数不变式：T **不是** freq 的整数倍时，decode() 也要返回同样帧数
    #     （2026-11-20 实测踩到：Stage 2 接上**真实 Stage 1 checkpoint** 时直接崩
    #      —— "x 与 x_adv 的帧数必须相等，实际 188 vs 192"。
    #      原因：AutoVC 的解码器要求 T 是 freq 的整数倍（内部 pad 到 32 的倍数），
    #      但输出没裁回来；而 Stage 2 的数据管道当时也没保证帧数是 32 的倍数。）
    odd_mel = torch.randn(2, 90, config.FRAME_DIM)      # 90 不是 32 的倍数
    odd_codes = stage1_real.content_code(odd_mel, test_emb)
    odd_out = stage1_real.decode(odd_mel, test_emb_adv, content=odd_codes)
    check(odd_out.shape[1] == odd_mel.shape[1],
          f"decode() 在 T 不是 freq 整数倍时仍返回同样帧数"
          f"（实际 {odd_out.shape[1]} vs {odd_mel.shape[1]}）",
          "不相等 —— 解码器内部 pad 了但没裁回来，x 与 x_adv 的帧数会不等")

    check(official_difference < 1e-5,
          f"decode() 与官方 Generator.forward(x, c_org, c_trg) 逐值一致"
          f"（最大差 = {official_difference:.2e}）",
          "不一致 —— decode() 手工复刻的摊开/解码逻辑与官方对不上了")
    print("\n" + "=" * 66)
    print(f"  通过 {PASSED} 项，失败 {FAILED} 项")
    print("=" * 66)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
