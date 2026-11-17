"""端到端冒烟测试 —— 成员 B 主责（任务板 W2-D4-3）。

【本脚本回答一个问题】
"整条推理链到底通不通？"

    wav → mel x (T,80) → GE2E h (256,) → h_adv = h + α·δ
        → 内容码 c (T',32) → 解码 x_adv (T,80) → 声码器 → protected.wav

【与成员 A 的接口约定】
本脚本优先调用 A 写的 `infer.py`，约定它暴露一个函数：

    protect(wav_path: str, alpha: float = 0.10, output_path: str | None = None) -> dict

返回字典至少包含：
    {
      "protected_path": "输出 wav 路径",
      "alpha": 0.10,
      "shapes": {"x": (T,80), "h": (256,), "c": (T',32), "x_adv": (T,80)},
    }

如果 `infer.py` 还不存在，本脚本会退化成"只跑说话人分支"，
把 B 侧能验证的都验证掉，并明确告诉你还差谁的东西。
**它不会假装成功。**

【用法】
    conda activate vpp
    python algorithms/voiceprint_protection/scripts/smoke_test.py
    python algorithms/voiceprint_protection/scripts/smoke_test.py --wav 某条.wav --alpha 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # config / audio_utils / infer
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import audio_utils  # noqa: E402
    import config  # noqa: E402
    import paths  # noqa: E402
except ImportError as error:  # 依赖缺失时给个能看懂的原因（多半是忘了 activate vpp）
    import paths as _paths  # noqa: E402

    _paths.print_env_hint(error)
    raise SystemExit(1) from error

# 方法 ID：组号尚未最终确定（见 TASKS.md §8），先用分支名对应的 group_n005，
# 组号定下来后用环境变量覆盖，或统一改到 config.py 里。
# ⚠️ 必须全小写：平台契约的标识符正则 /^[a-z][a-z0-9_]{1,63}$/ 不接受大写字母。
METHOD_ID = os.environ.get("VPP_METHOD_ID", "group_n005_voiceprint_protection")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def find_input_wav(explicit: str | None) -> Path:
    """按优先级挑一条测试音频。"""
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"指定的音频不存在：{path}")
        return path

    sample = paths.sample_dir(METHOD_ID) / "input.wav"
    if sample.exists():
        return sample

    data_root = paths.data_root()
    manifest = data_root / "test.txt"
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if parts:
                    candidate = data_root / parts[0]
                    if candidate.exists():
                        return candidate

    raise FileNotFoundError(
        "找不到测试音频。请用 --wav 指定一条，或先跑 prepare_data.py 生成 data/test.txt。"
    )


def print_shapes(shapes: dict) -> None:
    """按接口契约的形状打印。"""
    expected = {
        "x": (None, config.FRAME_DIM),
        "h": (config.SPEAKER_DIM,),
        "c": (None, config.CONTENT_DIM),
        "x_adv": (None, config.FRAME_DIM),
    }
    print("\n【张量形状（接口契约）】")
    for key in ("x", "h", "c", "x_adv"):
        actual = shapes.get(key)
        if actual is None:
            print(f"  {key:<6} : 缺失")
            continue
        actual = tuple(actual)
        want = expected.get(key)
        ok = "?" if want is None else ("✓" if (
            (want[0] is None or actual[0] == want[0]) and actual[1:] == want[1:]
        ) else "✗")
        print(f"  {key:<6} : {str(actual):<14} {ok}")


def run_speaker_branch(wav_path: Path, alpha: float) -> int:
    """infer.py 还没就绪时，只跑说话人分支。"""
    print("\n[infer.py 尚未就绪] 退化为「说话人分支」验证。")
    print("  成员 A 的 infer.py 一旦提供 protect()，本脚本会自动走完整链路。\n")

    try:
        from bridge_speaker import SpeakerBridge, SpeakerBridgeError
    except Exception as error:  # noqa: BLE001
        print(f"  [失败] 无法导入 bridge_speaker：{error}")
        return 1

    try:
        bridge = SpeakerBridge()
    except SpeakerBridgeError as error:
        print(f"  [失败] {error}")
        return 1

    wav = audio_utils.load_audio(wav_path)
    import soundfile as sf

    sample_rate = sf.info(str(wav_path)).samplerate
    duration = len(wav) / sample_rate

    embedding = bridge.embed(wav, sample_rate)
    print(f"  输入音频     : {wav_path}")
    print(f"  采样率/时长  : {sample_rate} Hz / {duration:.2f}s")
    print(f"  h.shape      : {tuple(embedding.shape)}")
    print(f"  |h| (归一化后): {float(embedding.norm()):.6f}")

    # L1 阶段不训练 G，用随机方向当扰动（任务板 D4-2）
    rng = np.random.default_rng(config.SEED)
    delta = rng.standard_normal(config.SPEAKER_DIM).astype(np.float32)
    delta /= np.linalg.norm(delta) + 1e-8
    perturbed = embedding.numpy() + alpha * delta
    cosine = float(
        np.dot(embedding.numpy(), perturbed)
        / (np.linalg.norm(embedding.numpy()) * np.linalg.norm(perturbed) + 1e-8)
    )
    print(f"  扰动 α       : {alpha}")
    print(f"  cos(h, h+αδ) : {cosine:.4f}   （α 越大越低，说明扰动确实改动了嵌入）")
    print("\n  [待 A 补齐] infer.py 的 mel / 内容码 / 解码 / 声码器四段尚未串联，")
    print("              因此这里**没有**产出 protected.wav —— 不能算 M1 验收通过。")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="端到端冒烟测试（成员 B）")
    parser.add_argument("--wav", default=None, help="输入音频（不给则自动挑）")
    parser.add_argument("--alpha", type=float, default=config.STAGE2["alpha"],
                        help="扰动强度 α，论文值 0.10")
    parser.add_argument("--out", default=None, help="保护后音频的输出路径")
    parser.add_argument("--report", default=None, help="把结果写成 JSON")
    args = parser.parse_args()

    print("=" * 64)
    print("  端到端冒烟测试（成员 B）")
    print("=" * 64)

    try:
        wav_path = find_input_wav(args.wav)
    except FileNotFoundError as error:
        print(f"\n[失败] {error}")
        return 1

    # -- 优先走完整链路 --------------------------------------------------
    try:
        import infer  # noqa: F401  （成员 A 的模块）
    except ImportError:
        infer = None

    if infer is None or not hasattr(infer, "protect"):
        return run_speaker_branch(wav_path, args.alpha)

    print(f"\n  输入音频 : {wav_path}")
    print(f"  α       : {args.alpha}")

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "algorithms" / "voiceprint_protection" / "results" / "protected.wav"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = infer.protect(str(wav_path), alpha=args.alpha, output_path=str(out_path))
    except Exception as error:  # noqa: BLE001
        print(f"\n  [失败] infer.protect() 抛异常：{error}")
        raise

    print(f"\n  保护后音频 : {result.get('protected_path', out_path)}")
    print_shapes(result.get("shapes", {}))

    protected = result.get("protected_path", out_path)
    if Path(protected).exists():
        original_wav = audio_utils.load_audio(wav_path)
        protected_wav = audio_utils.load_audio(protected)
        print("\n【输出体检】")
        print(f"  原始 RMS      : {audio_utils.rms_energy(original_wav):.6f}")
        print(f"  保护后 RMS    : {audio_utils.rms_energy(protected_wav):.6f}")
        print(f"  频谱距离      : {audio_utils.spectral_distance(original_wav, protected_wav):.4f}")
        if audio_utils.rms_energy(protected_wav) < 1e-4:
            print("  [警告] 输出几乎是静音，检查声码器与 mel 对照")
        else:
            print("  [通过] 输出不是静音")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"  报告已写 : {args.report}")

    print("\n  冒烟测试通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
