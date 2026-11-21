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

【★★★ 2026-11-21 补齐的六件事（成员 B）】
A 在 Stage 1 上做过一次"训练到底能不能跑起来"的审计（任务板 §0.10.1），
发现并修掉了 6 个问题。**Stage 2 当时一件都没有**，这次照着补齐：

| # | 缺什么 | 不补的后果 |
|---|--------|-----------|
| 1 | **断点续训** `--resume` | 200k 步在竞价实例上被抢占 = 白跑（云机随时抢占，任务板 §8.2） |
| 2 | **AMP**（`config.STAGE2["amp"]` 写了但代码里一个字都没用） | 8 GB 显存跑 batch 8 会 OOM，白花钱 |
| 3 | **梯度累积**（`config.STAGE2["accum_steps"]` 同上） | 想用大 batch 又不 OOM 就没辙 |
| 4 | **梯度裁剪**（`config.STAGE2["clip_grad"]` 同上） | L_de 的梯度要穿过声码器，很容易炸 |
| 5 | **`--max-hours`** | 云机按小时计费，忘了关就一直烧钱 |
| 6 | **吞吐 / ETA 打印** | 不知道要跑多久，就不敢决定开不开 |

★ 这三个 config 键**写在配置文件里却没人用**，是审计里最难发现的一类问题：
不报错、loss 照样降，只是"跑得比应该的慢好几倍 / 显存不够"。

【怎么跑】
    # 离线冒烟测试（用随机权重的替身，验证训练机制跑得通，不看指标）
    python algorithms/voiceprint_protection/stage2/train.py --stub --steps 20

    # 正式训练（需要成员 A 的 Stage 1 checkpoint）
    python algorithms/voiceprint_protection/stage2/train.py --vocoder hifigan --steps 200000

    # 断了接着跑（**云机被抢占后的标准动作**）
    python algorithms/voiceprint_protection/stage2/train.py --vocoder hifigan --steps 200000 --resume
"""

from __future__ import annotations

import argparse
import json
import random
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
    Stage1SpeakerEncoder,
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
def resolve_speaker_encoder(args) -> str:
    """★ 决定 Stage 2 用**哪一个** E_spk（说话人编码器）。

    论文 §3.1：E_spk 在 Stage 1 里被**端到端微调**过 ⇒ Stage 2 必须用那一个。
    但上游 GE2E（`TorchSpeakerEncoder`）也能用 —— **只要训练与推理用同一个**。

    所以规则是：
      · `--speaker-encoder stage1`  → 用 Stage 1 checkpoint 里微调过的那个（★ 论文原意）
      · `--speaker-encoder upstream` → 用上游那个（回退方案）
      · 不给（默认）                → checkpoint 里有 `speaker_encoder` 就用 stage1，
                                      没有就退回 upstream

    ★ 这个选择会被**写进 Stage 2 的 checkpoint**，推理时照着做 ——
      两边不可能对不上（对不上是最难查的一类 bug：不报错，只是指标全错）。
    """
    if args.speaker_encoder:
        return args.speaker_encoder
    if args.stage1_checkpoint and Path(args.stage1_checkpoint).exists():
        try:
            state = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            state = None
        if isinstance(state, dict) and "speaker_encoder" in state:
            return "stage1"
    return "upstream"


def build_pipeline(args, device) -> Stage2Pipeline:
    """把 Stage 2 需要的东西都装好。

    ★ 三个冻结组件都必须**保持梯度可流过**，只是参数不更新。
    """
    # -- E_spk（说话人编码器）---------------------------------------------
    print(f"加载说话人编码器 E_spk（{args.speaker_encoder}）…")
    if args.speaker_encoder == "stage1":
        speaker = Stage1SpeakerEncoder(args.stage1_checkpoint, device=str(device))
    else:
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
    # ★ `--stub` 与 `--vocoder` 现在是**正交**的两个开关（2026-11-20，B）：
    #   `--stub` 只决定 **Stage 1 用不用替身**；`--vocoder` 决定声码器用哪个。
    #   为什么要拆开：`--stub --vocoder hifigan` 才是**最有用的冒烟测试** ——
    #   Stage 1 用随机权重（指标无意义），但**数据管道 / 梯度通路 / 真实声码器 /
    #   心理声学掩蔽全部是真的**。A 的 Stage 1 checkpoint 还没出来时，
    #   用它就能先把整条 Stage 2 训练链跑通并测出每步耗时，不用干等。
    if args.vocoder == "hifigan":
        vocoder = HiFiGANVocoder(device=str(device))
        print("声码器 V：HiFi-GAN（论文同款；★ 不再做 mel 口径近似换算）")
    elif args.stub:
        vocoder = stage1.vocoder          # StubStage1 自带一个可导的替身
        print("声码器 V：StubVocoder（随机权重，指标无意义）")
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
def compute_losses(batch, pipeline: Stage2Pipeline, generator, discriminator,
                   args, device, autocast_enabled: bool = False) -> dict:
    """跑一遍前向，返回各项损失（**未 backward**）。

    拆成"算损失"与"更新参数"两步，是为了支持**梯度累积**：
    累积时前向要跑 N 次，但优化器只更新一次。

    返回的 dict 里：
      * 带张量的键 —— 用来 backward
      * `protected_mel` / `original_mel` —— 判别器更新时要用（已 detach）
    """
    waveforms, lengths, _ = batch
    waveforms = waveforms.to(device, non_blocking=True)
    lengths = lengths.to(device, non_blocking=True)

    batch_size = waveforms.shape[0]
    alpha = args.alpha
    device_type = device.type

    def _autocast():
        # CPU 上 autocast 基本没用（很多算子没有 fp16 kernel），还可能更慢
        return torch.autocast(device_type=device_type,
                              dtype=torch.float16,
                              enabled=autocast_enabled and device_type == "cuda")

    def _speak_fp32(waveform: torch.Tensor) -> torch.Tensor:
        """★ E_spk 一律用 **fp32** 算。

        为什么单独拎出来：autocast 只覆盖卷积/矩阵乘这类算子，
        **LSTM 不在列表里** —— 它按**输入的数据类型**跑。
        要是让 fp16 的波形进 GE2E，它的 LSTM 就在 fp16 里跑了，
        嵌入算出来就有可见误差，而 L_de 正是拿嵌入算的。
        所以这里显式关掉 autocast 并把输入转回 fp32。
        """
        with torch.autocast(device_type=device_type, enabled=False):
            return pipeline.speaker(waveform.float(), lengths)

    # -- 1. 原始语音的 mel 与说话人嵌入（**不需要梯度**）------------------
    with torch.no_grad():
        original_mel = pipeline.mel(waveforms)             # (B, T, 80)
        original_embedding = _speak_fp32(waveforms)         # (B, 256)

    # -- 2. 生成器出扰动（★ 从这里开始有梯度）---------------------------
    with _autocast():
        delta = generator(original_embedding)                   # (B, 256)
        advanced_embedding = perturb_embedding(original_embedding, delta, alpha)

        # -- 2b. 内容码 c = E_c(x, h)（**用原说话人的嵌入编码原句**）--------
        #   ★ 2026-11-19（成员 A）修：内容码必须**先算出来**再重建 ——
        #     否则 `decode()` 会拿 h_adv 去编码原句，内容里就混进了扰动，
        #     论文的 "x_adv = D(c, h_adv)" 就不成立了。
        content_codes = pipeline.content(original_mel, original_embedding)   # (B, T', 64)

        # -- 3. 用被污染的嵌入重建保护后的 mel（梯度穿过冻结的 Stage 1）----
        protected_mel = pipeline.stage1.decode(original_mel, advanced_embedding,
                                               content=content_codes)
        if protected_mel.shape[1] != original_mel.shape[1]:
            raise RuntimeError(
                f"x 与 x_adv 的帧数必须相等（契约 §2 的形状不变量），"
                f"实际 {original_mel.shape[1]} vs {protected_mel.shape[1]}"
            )

        # -- 4. 过声码器得到波形（L_mask 要用波形域的扰动）----------------
        protected_waveform = pipeline.vocode(protected_mel)             # (B, N)
        with torch.no_grad():
            # ★ 原始语音那一路**不需要梯度** —— 包 no_grad 省显存也省时间。
            #   （以前没包，梯度图白白建了一整条。）
            original_waveform = pipeline.vocode(original_mel)            # (B, N0)

    # -- 5. 一律转回 fp32 再算损失（余弦、掩蔽阈值在 fp32 更稳）-----------
    protected_mel = protected_mel.float()
    protected_waveform = protected_waveform.float()
    original_waveform = original_waveform.float()

    protected_waveform, original_waveform = pipeline.align(
        protected_waveform, original_waveform
    )
    delta_waveform = protected_waveform - original_waveform

    # -- 6. 模拟攻击：攻击者从保护后语音重新提卡，再拿去克隆 ----------------
    stolen_embedding = _speak_fp32(protected_waveform)      # ★ 梯度穿过声码器
    permutation = torch.randperm(batch_size, device=device)
    arbitrary_content = content_codes[permutation]            # 论文的 "arbitrary content"
    with _autocast():
        cloned_mel = pipeline.stage1.decode(original_mel, stolen_embedding,
                                            content=arbitrary_content)
        cloned_waveform = pipeline.vocode(cloned_mel)
    cloned_waveform = cloned_waveform.float()

    # -- 7. 算损失 --------------------------------------------------------
    thresholds = pipeline.thresholds(original_waveform)
    perceptual = loss_module.perceptual_loss(
        original_mel, protected_mel, delta_waveform, thresholds,
        lambda_mask=args.lambda_mask,
    )
    defense = loss_module.defense_loss(
        _speak_fp32(cloned_waveform), original_embedding
    )
    gan = loss_module.generator_gan_loss(discriminator(protected_mel))

    total = loss_module.generator_loss(
        gan, defense, perceptual,
        lambda_de=args.lambda_de, lambda_per=args.lambda_per, lambda_gan=args.lambda_gan,
    )

    return {
        "total": total,
        "gan": gan,
        "de": defense,
        "per": perceptual,
        "mel": loss_module.mel_loss(original_mel, protected_mel),
        "mask": loss_module.mask_loss(delta_waveform, thresholds),
        # 判别器更新要用（已 detach）
        "protected_mel": protected_mel.detach(),
        "original_mel": original_mel.detach(),
        # 指标
        "sim_clone": cosine_similarity(stolen_embedding, original_embedding).mean(),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
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
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--outdir", default=None, help="checkpoint 与日志的输出目录")
    parser.add_argument("--stage1-checkpoint", default=None, help="成员 A 的 Stage 1 checkpoint")
    parser.add_argument("--speaker-encoder", choices=("stage1", "upstream"), default=None,
                        help="Stage 2 用哪个 E_spk：stage1=随 Stage 1 微调过的那个（论文原意）；"
                             "upstream=上游 GE2E。默认按 checkpoint 内容自动选。")
    parser.add_argument("--device", default=None)

    # ★★ 2026-11-21 新增（成员 B）：把"能真的跑完 20 万步"缺的东西补齐
    parser.add_argument("--resume", action="store_true",
                        help="从 outdir 里的 stage2_last.pt 接着训（云机被抢占后的标准动作）")
    parser.add_argument("--checkpoint", default=None,
                        help="--resume 时要读的断点文件（默认 <outdir>/stage2_last.pt）")
    parser.add_argument("--amp", dest="amp", action="store_true", default=None,
                        help="混合精度（默认按 config.STAGE2['amp']；CPU 上自动关掉）")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="强制关掉混合精度")
    parser.add_argument("--accum-steps", type=int, default=None,
                        help="梯度累积步数（默认 config.STAGE2['accum_steps']）")
    parser.add_argument("--clip-grad", type=float, default=None,
                        help="梯度裁剪阈值（默认 config.STAGE2['clip_grad']）")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="训练时长上限（小时）—— 云机按小时计费，到点自动存盘退出")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="批大小（默认 config.STAGE2['batch_size']）")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="每条裁多少秒（默认 config.STAGE2['max_seconds']）")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader 的 worker 数（★ 本机内存小，最多 2）")
    return parser.parse_args(argv)


def make_scaler(enabled: bool):
    """造一个 GradScaler。

    torch 2.4 推荐 `torch.amp.GradScaler("cuda")`；
    老接口 `torch.cuda.amp.GradScaler` 留个后路（万一环境里 torch 更老）。
    """
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover
        return torch.cuda.amp.GradScaler(enabled=enabled)


def main() -> int:
    args = parse_args()

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    outdir = Path(args.outdir) if args.outdir else paths.WEIGHTS_DIR / "stage2"
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- 把 config 里的三个"写了没人用"的键真正用起来 -------------------
    if args.accum_steps is None:
        args.accum_steps = int(config.STAGE2.get("accum_steps", 1))
    if args.clip_grad is None:
        args.clip_grad = float(config.STAGE2.get("clip_grad", 0.0))
    if args.amp is None:
        args.amp = bool(config.STAGE2.get("amp", False))
    # CPU 上 autocast 没有 fp16 kernel，开了反而更慢 —— 自动关掉并说明
    if args.amp and device.type != "cuda":
        print("  [提示] AMP 只在 CUDA 上有效，本机是 CPU —— 自动关掉。")
        args.amp = False
    batch_size = args.batch_size or config.STAGE2["batch_size"]
    args.speaker_encoder = resolve_speaker_encoder(args)

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
    print(f"  batch/累积  : {batch_size} / {args.accum_steps}")
    print(f"  AMP         : {'开' if args.amp else '关'}")
    print(f"  梯度裁剪    : {args.clip_grad}")
    print(f"  时长上限    : {args.max_hours or '不限'} 小时")
    print(f"  E_spk       : {args.speaker_encoder}")
    if args.speaker_encoder == "upstream":
        print("  ★ 注意：用的是**上游** GE2E。论文里 E_spk 是随 Stage 1 微调过的，")
        print("    所以推理时也必须用上游那个 —— 不然 G 的输入是离分布的，防御会失效。")
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

    # ★ checkpoint 自带"契约"：推理时（infer.py 的 stage2 模式）照着做，
    #   训练与推理就不可能对不上 E_spk 或 G 的结构。
    checkpoint_meta = {
        "speaker_encoder": args.speaker_encoder,
        "alpha": args.alpha,
        "stage1_checkpoint": str(args.stage1_checkpoint) if args.stage1_checkpoint else None,
        "train_args": {
            "lambda_per": args.lambda_per,
            "lambda_de": args.lambda_de,
            "lambda_mask": args.lambda_mask,
            "lambda_gan": args.lambda_gan,
            "batch_size": batch_size,
            "accum_steps": args.accum_steps,
            "lr": config.STAGE2["lr"],
        },
    }
    if args.perturb == "train":
        checkpoint_meta["generator_config"] = {
            "dim": config.SPEAKER_DIM,
            "hidden": args.hidden,
            "layers": args.layers,
        }

    # -- 数据 --------------------------------------------------------------
    loader = build_dataloader(
        "train",
        batch_size=batch_size,
        max_seconds=args.max_seconds,
        num_workers=args.num_workers,
    )
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
    scaler = make_scaler(args.amp)

    # -- 断点续训 ----------------------------------------------------------
    step = 0
    if args.resume:
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else outdir / "stage2_last.pt"
        step = load_checkpoint(checkpoint_path, generator, discriminator,
                               optimizer_g, optimizer_d, scaler, device)
        print(f"  已从 {checkpoint_path} 续训，起始 step = {step}")

    # -- 训练循环 ----------------------------------------------------------
    log_path = outdir / "train_log.jsonl"
    started = time.time()
    deadline = started + args.max_hours * 3600 if args.max_hours else None
    print(f"{'步':>8}  {'total':>9} {'gan':>8} {'de':>8} {'per':>9} "
          f"{'mel':>9} {'mask':>8} {'sim_clone':>10} {'it/s':>7} {'剩余':>8}")
    print("-" * 68)

    generator.train()
    discriminator.train()

    history: list[tuple[int, float]] = []     # (step, L_de) —— 用来做方向自检
    stopped_early = False

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break

            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)

            # ---- 前向 + 反向（可累积）----------------------------------
            last = None
            for _ in range(max(1, args.accum_steps)):
                last = compute_losses(batch, pipeline, generator, discriminator,
                                      args, device, autocast_enabled=args.amp)
                scaled = last["total"] / max(1, args.accum_steps)
                if args.amp:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

            # ---- 判别器的损失（x_adv 要 detach，否则会同时改生成器）--------
            discriminator_loss_value = torch.zeros((), device=device)
            update_discriminator = args.lambda_gan > 0 and last is not None
            if update_discriminator:
                real_scores = discriminator(last["original_mel"])
                fake_scores = discriminator(last["protected_mel"])
                discriminator_loss_value = loss_module.discriminator_gan_loss(
                    real_scores, fake_scores
                )
                if args.amp:
                    scaler.scale(discriminator_loss_value).backward()
                else:
                    discriminator_loss_value.backward()

            # ---- 更新两个优化器（AMP 下要先 unscale 再裁剪）--------------
            if args.amp:
                scaler.unscale_(optimizer_g)
                if update_discriminator:
                    scaler.unscale_(optimizer_d)
            if args.clip_grad and args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in generator.parameters() if p.requires_grad], args.clip_grad
                )
                if update_discriminator:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in discriminator.parameters() if p.requires_grad],
                        args.clip_grad,
                    )
            if args.amp:
                scaler.step(optimizer_g)
                if update_discriminator:
                    scaler.step(optimizer_d)
                scaler.update()          # ★ 一次更新里只调一次
            else:
                optimizer_g.step()
                if update_discriminator:
                    optimizer_d.step()

            metrics = {
                "total": float(last["total"].detach()),
                "gan": float(last["gan"].detach()),
                "de": float(last["de"].detach()),
                "per": float(last["per"].detach()),
                "mel": float(last["mel"].detach()),
                "mask": float(last["mask"].detach()),
                "d": float(discriminator_loss_value.detach()),
                "sim_clone": float(last["sim_clone"].detach()),
            }
            step += 1

            # -- 日志 --------------------------------------------------------
            if step % args.log_every == 0 or step == args.steps:
                elapsed = time.time() - started
                speed = step / max(elapsed, 1e-6)
                remaining = (args.steps - step) / max(speed, 1e-9)
                print(f"{step:>8}  {metrics['total']:>9.4f} {metrics['gan']:>8.4f} "
                      f"{metrics['de']:>8.4f} {metrics['per']:>9.4f} {metrics['mel']:>9.4f} "
                      f"{metrics['mask']:>8.4f} {metrics['sim_clone']:>10.4f} "
                      f"{speed:>7.2f} {_fmt_duration(remaining):>8}")
                metrics["step"] = step
                metrics["elapsed"] = round(elapsed, 1)
                metrics["it_per_s"] = round(speed, 3)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                history.append((step, metrics["de"]))

            # -- 存盘 --------------------------------------------------------
            if args.save_every and (step % args.save_every == 0):
                save_checkpoint(generator, discriminator, optimizer_g, optimizer_d,
                                scaler, step, outdir, meta=checkpoint_meta)

            # -- 时长上限（云机按小时计费，到点就停）--------------------------
            if deadline is not None and time.time() >= deadline:
                print(f"\n  ★ 到达 --max-hours 上限（{args.max_hours} 小时），"
                      f"存盘后退出。")
                stopped_early = True
                break

        if stopped_early or step >= args.steps:
            break

    save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, scaler, step, outdir,
                    meta=checkpoint_meta)

    elapsed = time.time() - started
    print("-" * 68)
    print(f"训练结束，共 {step} 步，用时 {elapsed / 60:.1f} 分钟"
          f"（{step / max(elapsed, 1e-6):.2f} it/s）")
    print(f"checkpoint: {outdir / 'stage2_last.pt'}")
    print(f"日志      : {log_path}")
    if stopped_early:
        print("★ 是被 --max-hours 停的 —— 接着跑用：`--resume`")
    print()
    check_defense_direction(history)
    print("★ 下一步：跑 `evaluate.py` 算 Sim_prot / Sim_clone / DSR")
    print("  ★ L_de 应该在下降。如果它在**上升**，说明符号反了 —— 见 losses.py 的警告。")
    return 0


def _fmt_duration(seconds: float) -> str:
    """把秒数写成 "1d2h" / "3h12m" / "45s" 这种一眼能看懂的格式。"""
    seconds = int(max(seconds, 0))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}d{hours}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def check_defense_direction(history: list[tuple[int, float]]) -> None:
    """★ L_de 的方向自检：**它必须下降**。

    论文式(12)：L_de 是攻击者克隆音与原音的余弦相似度，**要最小化**。
    如果它反而在涨，说明符号反了 —— 而这种错误**不报错、loss 照样降**，
    只能靠看方向发现（A 的 L_dis 就踩过同一个坑，见 docs/experiments.md §4c.2）。

    这里比较"前 10%"与"后 10%"的中位数，步数太少时不做判断。
    """
    if len(history) < 6:
        print("  [L_de 方向自检] 步数太少，不判断。")
        return
    window = max(1, len(history) // 10)
    early = sorted(value for _, value in history[:window])
    late = sorted(value for _, value in history[-window:])
    early_median = early[len(early) // 2]
    late_median = late[len(late) // 2]
    if late_median < early_median:
        print(f"  [L_de 方向自检] ✅ 在下降（{early_median:.4f} → {late_median:.4f}）"
              f" —— 防御方向正确。")
    else:
        print(f"  [L_de 方向自检] ⚠️ **没有下降**（{early_median:.4f} → {late_median:.4f}）"
              f" —— 检查 losses.py::defense_loss 的符号！")


# ---------------------------------------------------------------------------
# 存盘 / 读盘
# ---------------------------------------------------------------------------
def save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, scaler,
                    step: int, outdir: Path, meta: dict | None = None) -> Path:
    """存断点。**必须包含优化器状态**，否则 --resume 只是"接着跑"，不是"接着训"。

    ★ `meta` 里放**这个 checkpoint 自己的契约**（用了哪个 E_spk、G 的结构）——
      推理时（`infer.py` 的 stage2 模式）照着做，两边不可能对不上。
    """
    path = outdir / "stage2_last.pt"
    payload = {
        "step": step,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        # ★ 随机数状态：不存的话，续训后的数据顺序/噪声与上一段不连续，
        #   复现性就没了（论文 §3.1 要求可复现）。
        "rng_python": random.getstate(),
        "rng_numpy": _numpy_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if meta:
        payload.update(meta)
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, generator, discriminator, optimizer_g, optimizer_d,
                    scaler, device) -> int:
    """读断点，返回起始步数。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到断点 {path}\n"
            "（--resume 需要上一次训练存下来的 stage2_last.pt）"
        )
    data = torch.load(path, map_location=device, weights_only=False)
    generator.load_state_dict(data["generator"])
    discriminator.load_state_dict(data["discriminator"])
    if "optimizer_g" in data:
        optimizer_g.load_state_dict(data["optimizer_g"])
    if "optimizer_d" in data:
        optimizer_d.load_state_dict(data["optimizer_d"])
    if scaler is not None and "scaler" in data and getattr(scaler, "is_enabled", lambda: False)():
        scaler.load_state_dict(data["scaler"])
    if "rng_python" in data:
        random.setstate(data["rng_python"])
    if "rng_numpy" in data:
        _set_numpy_state(data["rng_numpy"])
    if "rng_torch" in data:
        torch.set_rng_state(data["rng_torch"])
    if "rng_cuda" in data and data["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(data["rng_cuda"])
    return int(data.get("step", 0))


def _numpy_state():
    import numpy as np
    return np.random.get_state()


def _set_numpy_state(state) -> None:
    import numpy as np
    np.random.set_state(state)


if __name__ == "__main__":
    raise SystemExit(main())
