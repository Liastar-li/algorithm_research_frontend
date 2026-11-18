"""桥接：声码器 —— mel 频谱 -> 波形。

【它是干什么的】
Stage 1 的解码器 D 输出的是 **mel 频谱**，不是声音。
要"听"出重建得好不好，必须再过一个**声码器**把 mel 变成波形。

    mel (T, 80)  ──V(·)──►  wav (N,)  @16 kHz

论文 §2.3 原文：
    "The protected mel-spectrogram is reconstructed as x_adv = D(c, h_adv),
     and converted to waveform using **HiFi-GAN vocoder V(·)**."

【★ 为什么这里用的是 WaveRNN 而不是 HiFi-GAN】
HiFi-GAN 是论文指定的声码器，但 **NVIDIA 官方的 `torch.hifi_gan` 仓库
已经下架了**（2026-11 实测：GitHub API 对 `NVIDIA/torch.hifi_gan` 返回 404，
`codeload` / `raw.githubusercontent` 同样 404），NGC 上的 16 kHz 权重也取不到。

而上游 Real-Time-Voice-Cloning 自带的 **WaveRNN 声码器**：
  * 权重可下载（HF `CorentinJ/SV2TTS` 的 `vocoder.pt`，51 MB，已下载并校验）
  * 采样率 16 kHz（与论文一致）
  * 与上游合成器**配套**（同一套 mel 约定）

所以我们用它当 V(·)。这是一处**如实记录的偏离**，写进 docs/experiments.md。

【★ mel 约定必须对上，否则出来是噪音且不报错】
WaveRNN 只认 `vocoder/audio.py::melspectrogram` 那套：
    librosa.stft(n_fft=800, hop=200, win=800) -> 幅度
    -> librosa.filters.mel(16000, 800, n_mels=80, fmin=55)
    -> 20*log10(max(1e-5, .)) -> clip((x+100)/100, 0, 1)

本项目对应的实现是 `audio_utils.rtc_mel()`（预设 `rtc16k`）。
**若 Stage 1 用的是 `model16k`（AutoVC 链）那套 mel，就不能直接喂给它**，
必须把 Stage 1 的 `config.STAGE1["mel_preset"]` 改成 `"rtc16k"`。

【形象比喻】
mel 频谱像乐谱，声码器像演奏乐谱的乐团。
乐团只认自己那本谱子的记谱法 —— 拿另一本谱子（AutoVC 的）给它，
它照样"演奏"，但出来的是噪音，而且**不会报错**。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import paths


class VocoderBridge:
    """上游 WaveRNN 声码器的封装：mel -> 波形。

    用法：
        vocoder = VocoderBridge()
        wav = vocoder.mel_to_wav(mel)          # mel: (T, 80)，取值 [0,1]
    """

    #: 上游 `hp.mel_max_abs_value` = 4.0
    MEL_MAX_ABS_VALUE = 4.0

    def __init__(self, rtc_dir: str | Path | None = None,
                 device: str | None = None,
                 models_dir: str | Path | None = None):
        from bridge_speaker import find_rtc_dir

        self.rtc_dir = find_rtc_dir(str(rtc_dir) if rtc_dir else None)
        self.models_dir = Path(models_dir) if models_dir else paths.models_dir()
        self.device = device

        if str(self.rtc_dir) not in sys.path:
            sys.path.insert(0, str(self.rtc_dir))

        self.vocoder_path = self._find_file("vocoder.pt")

        from vocoder import inference as vocoder_inference

        self._vocoder = vocoder_inference
        self._vocoder.load_model(self.vocoder_path, verbose=False)
        self.sample_rate = 16000

    # -- 内部 ------------------------------------------------------------
    def _find_file(self, name: str) -> Path:
        candidates = [
            self.models_dir / "default" / name,
            self.models_dir / name,
            self.rtc_dir / name,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.stat().st_size > 1024:
                with candidate.open("rb") as handle:
                    if handle.read(1) == b"<":
                        raise FileNotFoundError(
                            f"{candidate} 是一个 HTML 错误页，不是权重文件！\n"
                            "请删掉它，然后重新下载：\n"
                            "    python algorithms/voiceprint_protection/scripts/download_rtc_weights.py"
                        )
                return candidate
        raise FileNotFoundError(
            f"找不到 {name}。请先下载上游权重：\n"
            "    python algorithms/voiceprint_protection/scripts/download_rtc_weights.py\n"
            "已尝试：\n" + "\n".join(f"    {c}" for c in candidates)
        )

    # -- 对外 ------------------------------------------------------------
    def mel_to_wav(self, mel: np.ndarray, scale: float | None = None) -> np.ndarray:
        """mel -> 波形。

        参数
        ----
        mel   : (T, 80) 的 mel 频谱，取值 **[-1, 1]**
                必须是 `rtc16k` 约定（audio_utils.rtc_mel 的输出）
        scale : 送进 WaveRNN 前的缩放，默认 `MEL_MAX_ABS_VALUE`（4.0）。
                上游 `infer_waveform` 内部会再除以 4，两步相消，
                最终 WaveRNN 看到的仍是我们算出来的 mel（∈ [-1,1]）——
                这正是它训练时的量纲（vocoder_dataset.py:28）。
                传 1.0 则相当于让 WaveRNN 看到 mel/4（**错的**）。

        返回 : (N,) 的波形，采样率 16 kHz
        """
        mel = np.asarray(mel, dtype=np.float32)
        if mel.ndim != 2 or mel.shape[1] != 80:
            raise ValueError(f"mel 形状应为 (T, 80)，实际 {mel.shape}")

        scale = self.MEL_MAX_ABS_VALUE if scale is None else scale

        # 上游要的是 (num_mels, frames)，且是**合成器原始输出**的量纲 [-4, 4]
        spec = (mel * scale).T                      # (80, T)
        wav = self._vocoder.infer_waveform(
            spec,
            normalize=True,
            batched=True,
            # ★ 上游写的是 `progress_callback = progress_callback or self.gen_display`，
            #   传 None 等于**启用** tqdm；必须传一个真的可调用对象才能静音。
            progress_callback=lambda *args, **kwargs: None,
        )
        return np.asarray(wav, dtype=np.float32)

    def wav_to_mel(self, wav: np.ndarray) -> np.ndarray:
        """波形 -> mel（上游那套约定），形状 (T, 80)。

        用来做"重建波形 vs 原波形"的对照：两边都转成同一套 mel 再比。
        """
        import audio_utils

        return audio_utils.rtc_mel(np.asarray(wav, dtype=np.float32))

    def describe(self) -> str:
        return (
            "上游 WaveRNN 声码器（论文里是 HiFi-GAN）\n"
            f"  权重      : {self.vocoder_path}\n"
            f"  采样率    : {self.sample_rate}\n"
            "  输入/输出 : mel (T, 80) [0,1]  ->  wav (N,)\n"
            "  ★ mel 必须是 rtc16k 约定（config.STAGE1['mel_preset'] = 'rtc16k'）"
        )


def main() -> int:
    """命令行自检：合成一段随机 mel 看看声码器能不能跑。"""
    import audio_utils

    print(VocoderBridge().describe())
    mel = np.random.RandomState(0).rand(160, 80).astype(np.float32) * 0.5
    wav = VocoderBridge().mel_to_wav(mel)
    print(f"  随机 mel -> 波形 {wav.shape}，RMS = {float(np.sqrt((wav ** 2).mean())):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
