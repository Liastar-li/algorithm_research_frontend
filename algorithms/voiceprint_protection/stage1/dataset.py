"""Stage 1 的数据集：从 VCTK 清单读音频，算两套 mel，裁片段。

【清单格式（成员 B 的 prepare_data.py 产出）】
    data/train.txt   每行：<相对 data/ 的音频路径> <说话人名>
    data/val.txt     同上
    data/test.txt    同上
    data/spk2id.txt  每行：<说话人名> <整数 ID>，只含训练集说话人（K = 80）

【每条样本要产出四样东西】
    mel80   : (T, 80)  AutoVC mel   → 给 Ec / D
    mel40   : (T', 40) GE2E mel      → 给 GE2E
    spk_id  : int      说话人 ID      → 给 L_dis 的分类器 W_s
    wav     : (samples,) 原始波形      → 备用（如需重采样）

【为什么要两套 mel】
    AutoVC 的 Ec/D 吃 80 维 mel，论文的说话人编码器 GE2E 吃 40 维 mel。
    两者维度不同、不可互换 —— 见 docs/interface-contract.md §1。

【片段裁剪】
    论文的 AutoVC 训练用固定长度的片段。8 GB 显存下我们按
    `config.STAGE1["max_seconds"]`（默认 3 秒）裁剪，并保证帧数是
    `freq=32` 的整数倍 —— 否则解码器摊开 code 时对不齐。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils import data

import audio_utils
import config
import paths


class VCTKStage1Dataset(data.Dataset):
    """Stage 1 训练集。"""

    def __init__(self, manifest: str | Path, data_root: str | Path | None = None,
                 spk2id: str | Path | None = None, max_seconds: float | None = None,
                 min_seconds: float = 1.0, freq: int | None = None,
                 require_speaker: bool = True, deterministic: bool = False):
        self.manifest = Path(manifest)
        self.data_root = Path(data_root) if data_root else paths.data_root()
        self.max_seconds = max_seconds or config.STAGE1["max_seconds"]
        self.min_seconds = min_seconds
        self.freq = freq or config.AUTOVC_MODEL["freq"]
        # ★ deterministic=True 时不随机裁剪（从第 0 个采样点开始），
        #   评测必须用它 —— 否则同一条音频两次裁到不同片段，结果不可复现。
        self.deterministic = deterministic

        self.samples = self._read_manifest(self.manifest)
        self.speaker_to_id = self._read_spk2id(spk2id)
        self.require_speaker = require_speaker

        # ★ 模型侧 mel 约定（契约字段，见 config.STAGE1["mel_preset"]）
        self.preset = config.MEL_PRESETS[
            config.STAGE1.get("mel_preset", config.DEFAULT_PRESET)]
        self.sample_rate = self.preset["sample_rate"]
        self.hop = self.preset["hop_length"]

        # 训练片段的帧数：取 max_seconds 对应的帧数，并向下取整到 freq 的整数倍
        frames = int(self.max_seconds * self.sample_rate / self.hop)
        self.crop_frames = max(self.freq, (frames // self.freq) * self.freq)

    # -- 清单读取 --------------------------------------------------------
    def _read_manifest(self, path: Path) -> list[tuple[str, str]]:
        if not path.exists():
            raise FileNotFoundError(
                f"找不到清单 {path}\n"
                "请先跑：python algorithms/voiceprint_protection/scripts/prepare_data.py"
            )
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 2:
                    rows.append((parts[0], parts[1]))
        if not rows:
            raise RuntimeError(f"{path} 是空的。")
        return rows

    def _read_spk2id(self, explicit: str | Path | None) -> dict[str, int]:
        candidates = []
        if explicit:
            candidates.append(Path(explicit))
        candidates.append(self.manifest.parent / "spk2id.txt")

        for candidate in candidates:
            if candidate.exists():
                mapping = {}
                with candidate.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        parts = line.split()
                        if len(parts) >= 2:
                            mapping[parts[0]] = int(parts[1])
                if mapping:
                    return mapping
        return {}

    # -- 采样 ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def _resolve(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute() and path.exists():
            return path
        candidate = self.data_root / relative
        if candidate.exists():
            return candidate
        # 清单里可能是 "wav48_silence_trimmed/p225/p225_001.wav" 这种相对 data/raw 的路径
        for base in (paths.raw_root(), self.data_root):
            alternative = base / relative
            if alternative.exists():
                return alternative
        raise FileNotFoundError(f"音频不存在：{relative}（在 {self.data_root} 下找不到）")

    def __getitem__(self, index: int) -> dict:
        relative, speaker = self.samples[index]
        path = self._resolve(relative)

        wav = audio_utils.load_audio(path, self.sample_rate)
        min_samples = int(self.min_seconds * self.sample_rate)
        if len(wav) < min_samples:
            wav = np.pad(wav, (0, min_samples - len(wav)), "constant")

        # 裁到固定帧数（必须是 freq 的整数倍）
        #
        # ★ 这里有个**很容易踩的帧数坑**：
        #   两套 mel 链给的都是 `1 + N // hop` 帧（center 填充的那一帧），
        #   所以要拿到**恰好 crop_frames 帧**，采样点数必须是
        #       N = (crop_frames - 1) * hop
        #   而不是 crop_frames * hop（那样会多出 1 帧）。
        #
        #   为什么帧数必须是 freq 的整数倍？官方 AutoVC 的 Encoder 取
        #   `out_forward[:, i+freq-1, :]`（i = 0, freq, 2*freq, ...），
        #   T 不是 freq 的整数倍时 i+freq-1 会越界；
        #   而且 Decoder 把每个 code 摊回 `int(T / len(codes))` 帧，
        #   摊不满就会**解码输出与输入帧数不等** -> L_rec 直接形状报错。
        #   官方 AutoVC 是在 synthesis.py 里 pad 到 32 的倍数来解决的。
        needed = (self.crop_frames - 1) * self.hop
        if len(wav) >= needed:
            if self.deterministic:
                start = 0
            else:
                start = np.random.randint(0, len(wav) - needed + 1)
            wav = wav[start:start + needed]
        else:
            wav = np.pad(wav, (0, needed - len(wav)), "constant")

        mel80 = audio_utils.model_mel(wav)             # (T, 80)，T = crop_frames
        mel40 = audio_utils.ge2e_mel(wav)             # (T', 40)

        if mel80.shape[0] != self.crop_frames:
            raise RuntimeError(
                f"mel 帧数 {mel80.shape[0]} != 期望 {self.crop_frames}。\n"
                f"hop={self.hop}，采样点数={len(wav)}，crop_frames={self.crop_frames}。\n"
                "两套 mel 链都给 1 + N//hop 帧，采样点数应为 (crop_frames-1)*hop。"
            )

        speaker_id = self.speaker_to_id.get(speaker, -1)
        if self.require_speaker and speaker_id < 0:
            raise KeyError(
                f"说话人 {speaker} 不在 spk2id.txt 里（该文件只含训练集说话人）。\n"
                f"清单：{self.manifest}"
            )

        return {
            "mel80": torch.from_numpy(mel80),
            "mel40": torch.from_numpy(mel40),
            "speaker_id": torch.tensor(speaker_id, dtype=torch.long),
            "speaker": speaker,
            "path": str(path),
            # ★ 原始波形（裁剪后的）。评测要拿它跟重建波形比 —— 没有它就没法算
            #   "重建音频像不像原音"，也没法试听对比。
            "wav": torch.from_numpy(wav.astype(np.float32)),
        }


def collate_stage1(batch: list[dict]) -> dict:
    """把一批样本拼成 batch。

    ★ 变长问题：AutoVC 的解码器把每个 content code 摊回固定帧数，
      所以**所有样本的帧数必须相同** —— 我们在 Dataset 里已经裁成固定帧数，
      这里直接 stack 即可。
    """
    return {
        "mel80": torch.stack([item["mel80"] for item in batch]),
        "mel40": torch.stack([item["mel40"] for item in batch]),
        "speaker_id": torch.stack([item["speaker_id"] for item in batch]),
        "speaker": [item["speaker"] for item in batch],
        "path": [item["path"] for item in batch],
        "wav": [item["wav"] for item in batch],
    }


def build_dataloader(split: str = "train", batch_size: int | None = None,
                     num_workers: int | None = None, shuffle: bool = True,
                     data_root: str | Path | None = None,
                     deterministic: bool = False):
    """建 DataLoader。split ∈ {train, val, test}。

    deterministic=True 时不随机裁剪（评测必须用）。
    """
    batch_size = batch_size or config.STAGE1["batch_size"]
    if num_workers is None:
        # Windows 上多进程 spawn 开销大，默认 2
        num_workers = 2

    data_root = Path(data_root) if data_root else paths.data_root()
    manifest = data_root / f"{split}.txt"

    dataset = VCTKStage1Dataset(
        manifest,
        data_root=data_root,
        spk2id=data_root / "spk2id.txt",
        require_speaker=(split == "train"),
        deterministic=deterministic,
    )

    generator = torch.Generator()
    generator.manual_seed(config.SEED)

    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_stage1,
        drop_last=(split == "train"),
        generator=generator,
        pin_memory=False,   # NPU/Windows 下用 False 更稳
        # ★ persistent_workers=True：否则**每个 epoch 都会重新拉起一批 worker 进程**。
        #   在 Windows 上 worker 是 spawn 模式，每次启动都要重新 import torch
        #   和全部依赖（十几秒）。数据集小的时候（比如冒烟测试）这一步的开销
        #   甚至超过训练本身。开启后 worker 常驻，只启动一次。
        persistent_workers=(num_workers > 0),
    )
