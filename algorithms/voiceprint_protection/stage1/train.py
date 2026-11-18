"""Stage 1 训练主循环（论文 §2.2 + §3.1）。

【两阶段训练（论文式 6 的实现顺序）】
  第 1 步：训练说话人分类器 W_s
      论文说 W_s "is a single-layer speaker classifier **trained on the speaker
      embedding h**"。所以先用真实说话人嵌入 h 做一个普通的 K 类分类，
      把 W_s 训到能认出说话人 —— 否则后面的熵最大化会退化成
      "让随机 W_s 输出均匀"，起不到"逼内容码丢掉说话人"的作用。

  第 2 步：冻结 W_s，训练解耦-重建网络
      L_S1 = L_rec + λ_cd·L_cd + λ_dis·L_dis
      梯度只回传到 Ec / D / Postnet（以及 GE2E —— 论文说它端到端微调）。

【命令行】
    # 第 1 步：只训分类器（很快，几分钟）
    python algorithms/voiceprint_protection/stage1/train.py --stage classifier --iterations 2000

    # 第 2 步：训解耦网络
    python algorithms/voiceprint_protection/stage1/train.py --stage model --iterations 50000

    # 冒烟（几步，验证流程通不通）
    python algorithms/voiceprint_protection/stage1/train.py --stage model --iterations 5 --smoke

    # 断线续训（云上竞价实例被抢占后用）
    python algorithms/voiceprint_protection/stage1/train.py --stage model --resume

【★ 为什么要断点续训】
云上的训练实例是**竞价实例**，随时可能被抢占（价格波动就下线）。
论文量级是 50 万步，跑几天，断线/被抢占是常态。
所以这里每 `--save-every` 步存一次完整 checkpoint（模型 + 优化器 + 步数），
`--resume` 从最近的 checkpoint 接着跑 —— 丢了最多几分钟。

【★ 为什么要混合精度 + 梯度累积】
本地是 RTX 4060 Laptop（**8 GB 显存**）。
  · AMP（fp16）：省一半显存，还能快 1.5–2 倍；
  · 梯度累积：显存装不下大 batch 时，用"小 batch × 多步"凑出等效大 batch。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ★ 路径引导：让 `python stage1/train.py` 与 `python -m ...stage1.train` 都能跑
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

import config  # noqa: E402
import paths  # noqa: E402

try:                                   # 作为包的子模块运行（python -m ...）
    from . import losses as stage1_losses
    from .dataset import build_dataloader
    from .model import Stage1Model, count_parameters
    from .speaker_encoder import TrainableSpeakerEncoder
except ImportError:                    # 直接当脚本运行（python stage1/train.py）
    from stage1 import losses as stage1_losses
    from stage1.dataset import build_dataloader
    from stage1.model import Stage1Model, count_parameters
    from stage1.speaker_encoder import TrainableSpeakerEncoder

WEIGHTS_DIR = paths.WEIGHTS_DIR


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def set_seed(seed: int = config.SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_weights_dir() -> Path:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return WEIGHTS_DIR


def resolve_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_scaler(enabled: bool) -> "torch.amp.GradScaler":
    """AMP 的梯度缩放器。torch 2.4 用 torch.amp.GradScaler(device, enabled=...)。"""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):        # 老版本 torch 的兜底
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    """统一封装 autocast（CPU 上强制关闭）。"""
    if not enabled or device.type != "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return torch.autocast(device_type="cuda", enabled=True)


def make_writer(log_dir: Path):
    """TensorBoard 写入器；没装 tensorboard 就返回 None（不影响训练）。"""
    try:
        from torch.utils.tensorboard import SummaryWriter

        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(str(log_dir))
    except Exception:  # noqa: BLE001
        return None


class Progress:
    """极简进度打印：平均每 step 多少秒 + 预计还剩多久。"""

    def __init__(self, total: int, enabled: bool = True):
        self.total = total
        self.enabled = enabled
        self.start = time.time()

    def update(self, step: int) -> str:
        if not self.enabled:
            return ""
        elapsed = time.time() - self.start
        speed = step / max(elapsed, 1e-6)
        remaining = (self.total - step) / max(speed, 1e-9)
        return (f"  step {step:>7}/{self.total}  "
                f"{speed:.2f} it/s  已用 {elapsed / 60:.1f} 分  预计还需 {remaining / 60:.1f} 分")


# ---------------------------------------------------------------------------
# 第 1 步：训练说话人分类器 W_s
# ---------------------------------------------------------------------------
def train_classifier(args) -> int:
    """用真实说话人嵌入训练 W_s（K 类分类）。"""
    print("=" * 70)
    print("  Stage 1 · 第 1 步：训练说话人分类器 W_s")
    print("=" * 70)

    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"

    speaker_encoder = TrainableSpeakerEncoder(device=str(device), fine_tune=False)
    classifier = stage1_losses.SpeakerClassifier(
        num_speakers=config.STAGE1["num_speakers"],
        embedding_dim=config.SPEAKER_DIM,
    ).to(device)

    print(speaker_encoder.describe())
    print(f"  分类器输出维度 K = {config.STAGE1['num_speakers']}")
    print(f"  设备 = {device}   AMP = {use_amp}")

    loader = build_dataloader("train", batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr_classifier)
    scaler = make_scaler(use_amp)

    step = 0
    correct = 0
    total = 0
    loss_sum = 0.0
    history: list[dict] = []
    writer = make_writer(WEIGHTS_DIR / "tb" / "stage1_classifier")
    progress = Progress(args.iterations)

    while step < args.iterations:
        for batch in loader:
            if step >= args.iterations:
                break
            mel40 = batch["mel40"].to(device, non_blocking=True)
            speaker_id = batch["speaker_id"].to(device, non_blocking=True)

            with torch.no_grad():
                embedding = speaker_encoder(mel40)          # (B, 256)
            with autocast_context(device, use_amp):
                logits = classifier(embedding)
                loss = F.cross_entropy(logits.float(), speaker_id)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += float(loss.detach())
            correct += int((logits.argmax(-1) == speaker_id).sum())
            total += speaker_id.numel()
            step += 1

            if step % max(1, args.log_every) == 0 or step == args.iterations:
                line = (f"  step {step:>6}  loss {loss_sum / max(1, args.log_every):.4f}  "
                        f"acc {correct / max(1, total):.3f}  "
                        f"({(time.time() - progress.start):.1f}s)")
                print(line)
                history.append({"step": step, "loss": loss_sum / max(1, args.log_every),
                                "acc": correct / max(1, total)})
                if writer is not None:
                    writer.add_scalar("stage1/classifier/loss",
                                      loss_sum / max(1, args.log_every), step)
                    writer.add_scalar("stage1/classifier/acc",
                                      correct / max(1, total), step)
                loss_sum = 0.0

            if args.smoke and step >= args.iterations:
                break

    ensure_weights_dir()
    target = WEIGHTS_DIR / "stage1_speaker_classifier.pt"
    torch.save({
        "model": classifier.state_dict(),
        "num_speakers": config.STAGE1["num_speakers"],
        "embedding_dim": config.SPEAKER_DIM,
        "step": step,
    }, target)
    print(f"\n  已保存分类器：{target}")
    print(f"  训练集准确率 ≈ {correct / max(1, total):.3f}")
    print("  （这个准确率越高，后面 L_dis 的约束越有效）")
    print(f"  训练曲线：{WEIGHTS_DIR / 'stage1_classifier_history.json'}")
    (WEIGHTS_DIR / "stage1_classifier_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    if writer is not None:
        writer.close()
    return 0


# ---------------------------------------------------------------------------
# 第 2 步：训练解耦-重建网络
# ---------------------------------------------------------------------------
def _save_model_checkpoint(model, speaker_encoder, optimizer, scaler, step: int,
                           history: list[dict], path: Path) -> None:
    torch.save({
        "model": model.state_dict(),
        "speaker_encoder": speaker_encoder.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "history": history,
        "config": {
            "dim_neck": model.dim_neck,
            "dim_emb": model.dim_emb,
            "dim_pre": model.dim_pre,
            "freq": model.freq,
            "mel_preset": config.STAGE1.get("mel_preset", config.DEFAULT_PRESET),
        },
    }, path)


@torch.no_grad()
def evaluate(model, speaker_encoder, classifier, loader, device, use_amp: bool,
             max_batches: int = 20) -> dict:
    """在验证集上算平均损失（论文说超参是在验证集上网格搜索选出来的）。"""
    model.eval()
    speaker_encoder.eval()
    sums = {"loss_rec": 0.0, "loss_cd": 0.0, "loss_dis": 0.0, "total": 0.0}
    count = 0

    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        mel80 = batch["mel80"].to(device, non_blocking=True)
        mel40 = batch["mel40"].to(device, non_blocking=True)

        with autocast_context(device, use_amp):
            embedding = speaker_encoder(mel40)
            mel_out, mel_postnet, codes = model(mel80, embedding, embedding)
            codes_out = model.content_codes(mel_out, embedding)
            codes_postnet = model.content_codes(mel_postnet, embedding)
            _, parts = stage1_losses.stage1_loss(
                mel80, mel_out, mel_postnet, codes, codes_out, codes_postnet, classifier)

        for key in sums:
            sums[key] += parts[key]
        count += 1

    model.train()
    speaker_encoder.train()
    if count == 0:
        return {}
    return {key: value / count for key, value in sums.items()}


def train_model(args) -> int:
    print("=" * 70)
    print("  Stage 1 · 第 2 步：训练解耦-重建网络")
    print("=" * 70)

    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"

    speaker_encoder = TrainableSpeakerEncoder(device=str(device), fine_tune=True)
    model = Stage1Model().to(device)
    print(f"  Stage 1 模型参数量：{count_parameters(model):,}")
    print(f"  说话人编码器：{speaker_encoder.describe().splitlines()[3]}")
    print(f"  设备 = {device}   AMP = {use_amp}   梯度累积 = {args.accum_steps}")
    print(f"  模型侧 mel 约定 = {config.STAGE1.get('mel_preset', config.DEFAULT_PRESET)}")

    # -- 分类器（第 1 步的产物）------------------------------------------
    classifier_path = WEIGHTS_DIR / "stage1_speaker_classifier.pt"
    classifier = stage1_losses.SpeakerClassifier(
        num_speakers=config.STAGE1["num_speakers"],
        embedding_dim=config.SPEAKER_DIM,
    ).to(device)
    if classifier_path.exists():
        saved = torch.load(classifier_path, map_location="cpu", weights_only=False)
        classifier.load_state_dict(saved["model"])
        print(f"  已加载训练好的分类器：{classifier_path}")
    else:
        print(f"  [警告] 没找到 {classifier_path}，将用随机初始化的分类器。")
        print("         建议先跑 --stage classifier，否则 L_dis 的约束会失效。")
    classifier.freeze()   # ★ 论文：训练内容编码器时 Ws 固定

    loader = build_dataloader("train", batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True)
    val_loader = None
    try:
        val_loader = build_dataloader("val", batch_size=args.batch_size,
                                      num_workers=0, shuffle=False)
    except Exception as error:  # noqa: BLE001
        print(f"  [警告] 验证集不可用（{error}），跳过验证。")

    parameters = list(model.parameters()) + list(speaker_encoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.STAGE1["lr"])
    scaler = make_scaler(use_amp)

    # -- 断点续训 ---------------------------------------------------------
    start_step = 0
    history: list[dict] = []
    if args.resume:
        checkpoint = _find_latest_checkpoint()
        if checkpoint is None:
            print("  [警告] 没找到可续训的 checkpoint，从头开始。")
        else:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"])
            speaker_encoder.model.load_state_dict(saved["speaker_encoder"])
            if "optimizer" in saved:
                optimizer.load_state_dict(saved["optimizer"])
            if "scaler" in saved and saved["scaler"]:
                scaler.load_state_dict(saved["scaler"])
            start_step = int(saved.get("step", 0))
            history = list(saved.get("history", []))
            print(f"  已从 {checkpoint.name} 续训，起始 step = {start_step}")

    writer = make_writer(WEIGHTS_DIR / "tb" / "stage1_model")
    progress = Progress(args.iterations)

    step = start_step
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    deadline = time.time() + args.max_hours * 3600 if args.max_hours > 0 else None
    stopped_early = False

    while step < args.iterations:
        for batch in loader:
            if step >= args.iterations:
                break
            if deadline is not None and time.time() > deadline:
                print(f"\n  [到达 --max-hours 上限 {args.max_hours}h，停止训练]")
                stopped_early = True
                break

            mel80 = batch["mel80"].to(device, non_blocking=True)
            mel40 = batch["mel40"].to(device, non_blocking=True)

            with autocast_context(device, use_amp):
                embedding = speaker_encoder(mel40)                 # (B, 256)

                mel_out, mel_postnet, codes = model(mel80, embedding, embedding)
                codes_out = model.content_codes(mel_out, embedding)
                codes_postnet = model.content_codes(mel_postnet, embedding)

                total, parts = stage1_losses.stage1_loss(
                    mel80, mel_out, mel_postnet, codes, codes_out, codes_postnet, classifier)

            # 梯度累积：把若干个 micro batch 的损失除以累积步数再回传
            scaler.scale(total / args.accum_steps).backward()
            micro_step += 1

            if micro_step % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % max(1, args.log_every) == 0 or step == args.iterations:
                    print(f"  step {step:>7}  total {parts['total']:.4f}  "
                          f"rec {parts['loss_rec']:.4f}  cd {parts['loss_cd']:.4f}  "
                          f"dis {parts['loss_dis']:.4f}   {progress.update(step)}")
                    history.append({"step": step, **parts})
                    if writer is not None:
                        for key, value in parts.items():
                            writer.add_scalar(f"stage1/{key}", value, step)

                if args.save_every and step % args.save_every == 0:
                    ensure_weights_dir()
                    _save_model_checkpoint(model, speaker_encoder, optimizer, scaler,
                                           step, history,
                                           WEIGHTS_DIR / "stage1_last_ckpt.pt")
                    print(f"    [已保存断点] stage1_last_ckpt.pt（step {step}）")

                if args.val_every and val_loader is not None and step % args.val_every == 0:
                    metrics = evaluate(model, speaker_encoder, classifier, val_loader,
                                       device, use_amp)
                    if metrics:
                        print(f"    [验证] rec {metrics['loss_rec']:.4f}  "
                              f"cd {metrics['loss_cd']:.4f}  "
                              f"dis {metrics['loss_dis']:.4f}  "
                              f"total {metrics['total']:.4f}")
                        if writer is not None:
                            for key, value in metrics.items():
                                writer.add_scalar(f"stage1/val/{key}", value, step)
                        history.append({"step": step, "val": True, **metrics})

            if args.smoke and step >= args.iterations:
                break
        if stopped_early:
            break

    ensure_weights_dir()
    target = WEIGHTS_DIR / "stage1_model.pt"
    _save_model_checkpoint(model, speaker_encoder, optimizer, scaler, step, history, target)
    print(f"\n  已保存模型：{target}")

    history_path = WEIGHTS_DIR / "stage1_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已保存训练曲线：{history_path}")
    print(f"  总步数 {step}，总耗时 {(time.time() - progress.start) / 60:.1f} 分钟")
    if writer is not None:
        writer.close()

    print("\n  下一步：评测 Stage 1（M3 的验收）")
    print("    python algorithms/voiceprint_protection/scripts/eval_stage1.py")
    return 0


def _find_latest_checkpoint() -> Path | None:
    """找最近的可续训 checkpoint。"""
    for name in ("stage1_last_ckpt.pt", "stage1_model.pt"):
        candidate = WEIGHTS_DIR / name
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1 训练")
    parser.add_argument("--stage", choices=["classifier", "model"], default="model",
                        help="classifier = 第 1 步（训 W_s）；model = 第 2 步（训解耦网络）")
    parser.add_argument("--iterations", type=int, default=None,
                        help="训练步数（默认：classifier 2000，model 用 config.STAGE1）")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda / cpu，默认自动选")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=2000,
                        help="每多少步存一次断点（云上被抢占靠它续训）")
    parser.add_argument("--val-every", type=int, default=2000,
                        help="每多少步跑一次验证集")
    parser.add_argument("--accum-steps", type=int, default=None,
                        help="梯度累积步数（默认用 config.STAGE1）")
    parser.add_argument("--lr-classifier", type=float, default=1e-3,
                        help="第 1 步分类器的学习率")
    parser.add_argument("--max-hours", type=float, default=0.0,
                        help="最长训练小时数（0 = 不限）。云上建议设 3~4 小时一轮")
    parser.add_argument("--resume", action="store_true",
                        help="从最近的 checkpoint 续训")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="关闭混合精度")
    parser.add_argument("--smoke", action="store_true", help="只跑几步，验证流程")
    args = parser.parse_args()

    if args.iterations is None:
        args.iterations = 2000 if args.stage == "classifier" else config.STAGE1["iterations"]
    if args.batch_size is None:
        args.batch_size = config.STAGE1["batch_size"]
    if args.accum_steps is None:
        args.accum_steps = config.STAGE1.get("accum_steps", 1)
    if args.amp is None:
        args.amp = config.STAGE1.get("amp", True)

    print("\n路径解析：")
    for line in paths.describe().splitlines():
        print(f"  {line}")
    print()

    if args.stage == "classifier":
        return train_classifier(args)
    return train_model(args)


if __name__ == "__main__":
    raise SystemExit(main())
