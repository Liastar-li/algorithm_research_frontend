"""Stage 2 的数据集：VCTK。

【数据从哪来】
`prepare_data.py`（成员 B，W2-D2）已经把 VCTK 处理成了三份清单：
    data/train.txt   每行：<相对 data/ 的音频路径> <说话人名>
    data/val.txt
    data/test.txt
    data/spk2id.txt  <说话人名> <整数 ID>（只含训练集说话人，K=80）

Stage 2 只用 `train.txt` 训练、`val.txt` 做验证（选 checkpoint 用）。
**测试集 `test.txt` 绝对不能进训练**（评测阶段才用）——
任务板 §1.3 的三条硬要求之一。

【每条样本给什么】
    wav        : (N,) 16 kHz 单声道波形（随机裁到 max_seconds）
    speaker_id : 该说话人的整数 ID（Stage 2 其实用不到，
                 但留着方便以后加 L_dis 之类的分类损失）

【为什么要随机裁】
论文用 VCTK 全句；我们按 `config.STAGE2["max_seconds"]`（3 秒）裁，
理由是显存和训练速度。裁剪是**随机起点**的 —— 同一句话每次裁的位置不同，
相当于数据增强。
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import audio_utils
import config
import paths


class VCTKManifest(Dataset):
    """按清单读音频的 Dataset。

    参数
    ----
    split      : "train" / "val" / "test"
    data_root  : 数据根目录（默认 `paths.data_root()`，受 VPP_DATA_ROOT 控制）
    max_seconds: 每条裁多长（默认 config.STAGE2["max_seconds"] = 3.0）
    sample_rate: 采样率（默认 16000）
    """

    def __init__(self, split: str = "train", data_root: str | Path | None = None,
                 max_seconds: float | None = None, sample_rate: int = 16000,
                 min_seconds: float = 1.0):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split 只能是 train / val / test，收到 {split}")

        self.split = split
        self.data_root = Path(data_root) if data_root else paths.data_root()
        self.sample_rate = sample_rate
        self.max_seconds = (max_seconds if max_seconds is not None
                            else config.STAGE2["max_seconds"])
        self.min_seconds = min_seconds

        # ★ 帧数必须是 freq 的整数倍（契约 §7.1b）—— Stage 1 那边已经这么做了，
        #   Stage 2 之前漏了，结果接上**真实 Stage 1 checkpoint** 时直接崩：
        #       RuntimeError: x 与 x_adv 的帧数必须相等，实际 188 vs 192
        #   原因：AutoVC 的解码器要求 T 是 freq 的整数倍（内部会 pad 到 32 的倍数），
        #   而 Stage 2 裁出来的片段帧数是任意的 ⇒ 重建出来的 mel 比原 mel 长。
        #   这里按 Stage 1 的同一套约定裁：
        #       帧数 frames = int(max_seconds * sr / hop)，向下取整到 freq 的倍数
        #       采样点数 N   = (frames - 1) * hop
        #   ⇒ mel 帧数 = 1 + N // hop = frames（恰好是 freq 的整数倍）
        preset = config.MEL_PRESETS[
            config.STAGE1.get("mel_preset", config.DEFAULT_PRESET)]
        self.hop = preset["hop_length"]
        self.freq = config.CONTENT_DOWNSAMPLE
        frames = int(self.max_seconds * self.sample_rate / self.hop)
        self.crop_frames = max(self.freq, (frames // self.freq) * self.freq)
        self.crop_samples = (self.crop_frames - 1) * self.hop

        manifest = self.data_root / f"{split}.txt"
        if not manifest.exists():
            raise FileNotFoundError(
                f"找不到清单 {manifest}\n"
                "请先跑数据准备：\n"
                "    python algorithms/voiceprint_protection/scripts/prepare_data.py"
            )

        self.speaker_to_id = self._load_speaker_ids()
        self.items: list[tuple[Path, int]] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            path = self.data_root / parts[0]
            speaker = parts[1]
            if not path.exists():
                continue
            # 训练集用 spk2id 的整数 ID；val/test 没在 spk2id 里，退化成 -1
            self.items.append((path, self.speaker_to_id.get(speaker, -1)))

        if not self.items:
            raise RuntimeError(f"清单 {manifest} 里没有可用样本（路径是不是不对？）")

    def _load_speaker_ids(self) -> dict[str, int]:
        spk2id = self.data_root / "spk2id.txt"
        mapping: dict[str, int] = {}
        if spk2id.exists():
            for line in spk2id.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mapping[parts[0]] = int(parts[1])
                    except ValueError:
                        continue
        return mapping

    def __len__(self) -> int:
        return len(self.items)

    def _crop(self, wav: np.ndarray) -> np.ndarray:
        """随机裁一段，长度**恒等于** `self.crop_samples`。

        ★ 长度必须恒定（不能"太短就整条用"）：
          batch 里所有样本的帧数一致，x 与 x_adv 的形状不变式才成立；
          而且 Stage 1 的解码器按 freq 抽样式降采样，长度不齐会直接报错。
        """
        if len(wav) <= self.crop_samples:
            out = np.zeros(self.crop_samples, dtype=np.float32)
            out[: len(wav)] = wav
            return out
        start = random.randint(0, len(wav) - self.crop_samples)
        return wav[start: start + self.crop_samples]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, speaker_id = self.items[index]
        wav = audio_utils.load_audio(path, self.sample_rate)
        wav = self._crop(wav)

        # _crop 已经保证长度恒为 crop_samples（不够就补零），
        # 所以这里不用再按 min_seconds 补一遍 —— 补了反而会破坏帧数不变式。
        assert len(wav) == self.crop_samples, (len(wav), self.crop_samples)

        return torch.from_numpy(np.asarray(wav, dtype=np.float32)), speaker_id


def collate_waveforms(batch: list[tuple[torch.Tensor, int]]):
    """把长度不一的波形拼成 batch。

    返回
    ----
    waveforms : (B, L) 右侧补零
    lengths   : (B,) 每条的真实长度
    speakers  : (B,) 说话人 ID
    """
    waveforms = [item[0].numpy() for item in batch]
    speakers = [item[1] for item in batch]
    lengths = np.array([len(w) for w in waveforms], dtype=np.int64)
    longest = int(lengths.max())
    padded = np.zeros((len(waveforms), longest), dtype=np.float32)
    for index, wave in enumerate(waveforms):
        padded[index, : len(wave)] = wave
    return torch.from_numpy(padded), torch.from_numpy(lengths), torch.tensor(speakers)


def describe_dataset(dataset: VCTKManifest) -> str:
    """打一行数据集摘要（训练脚本启动时用）。"""
    speakers = {speaker for _, speaker in dataset.items}
    seconds = sum(len(path.read_bytes()) for path, _ in dataset.items[:50]) / 50 / 32000
    return (f"{dataset.split}: {len(dataset)} 条 / {len(speakers)} 个说话人 "
            f"/ 平均约 {seconds:.1f} 秒")


def build_dataloader(split: str = "train", batch_size: int | None = None,
                     max_seconds: float | None = None,
                     num_workers: int = 2, shuffle: bool | None = None,
                     seed: int = config.SEED):
    """造一个 DataLoader（默认值全从 config 来）。"""
    from torch.utils.data import DataLoader

    dataset = VCTKManifest(split=split, max_seconds=max_seconds)
    batch_size = batch_size or config.STAGE2["batch_size"]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=num_workers,
        collate_fn=collate_waveforms,
        drop_last=True,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
