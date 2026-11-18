"""桥接：克隆机 —— 上游的"合成器 + 声码器"，也就是**攻击者**本身。

【它在做什么（比喻）】
你在银行留了一段"声音指纹"。攻击者拿着这段录音，配上一句想说的话，
就能用你的音色把那句话念出来 —— 这就是**声音克隆**。

这个模块扮演的正是那个攻击者：
    输入：一段文本 + 一张 256 维的"声音指纹"
    输出：这段文本用这个音色念出来的音频

【为什么用上游 Real-Time-Voice-Cloning 的合成器】
原计划是用 AutoVC 的解码器当"配音员"，但 **AutoVC 官方权重链已失效**
（IBM Box 链接 404，实测），拿不到主模型与配套声码器。
而上游 RTC 的三个权重全部可下载（HF `CorentinJ/SV2TTS`）：
    encoder.pt (17 MB)   GE2E 说话人编码器
    synthesizer.pt (353 MB) Tacotron 合成器  ← 攻击者的"嗓子"
    vocoder.pt (51 MB)   WaveRNN 声码器      ← 把乐谱变成声音

而且它们**互相配套**（同一套 mel 约定：16 kHz / 80 mel / n_fft 800 / hop 200），
我们完全不需要自己猜 mel 参数 —— 这是最省心的一条路。

【★ 这个模块正是论文 Figure 1 里的红框】
论文的核心主张是：
    "扰动加在说话人嵌入上 -> 攻击者克隆出来的声音就不像本人了"
所以我们用同一个克隆机，跑两次：
    一次用原始嵌入 h      -> 得到"未防御时的克隆音"（像本人）
    一次用被扰动的 h_adv  -> 得到"防御后的克隆音"（不像本人）
两者的差别就是防御效果本身。

【上游接口（third_party/Real-Time-Voice-Cloning）】
    from synthesizer.inference import Synthesizer
    from vocoder import inference as vocoder

    synth = Synthesizer(synthesizer_path)
    synth.load()
    specs = synth.synthesize_spectrograms([text], [embedding])   # -> [(80, Mi)]
    wav = vocoder.infer_waveform(specs[0])                       # -> 1D 波形
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import paths


class ClonerBridge:
    """上游合成器 + 声码器的封装：文本 + 嵌入 -> 音频。"""

    def __init__(self, rtc_dir: str | Path | None = None, device: str | None = None,
                 models_dir: str | Path | None = None,
                 vocoder_target: int = 1000, vocoder_overlap: int = 100):
        from bridge_speaker import SpeakerBridge, find_rtc_dir, find_weights

        self.rtc_dir = find_rtc_dir(str(rtc_dir) if rtc_dir else None)
        self.models_dir = Path(models_dir) if models_dir else paths.models_dir()
        self.device = device

        # ★ 声码器的折叠块长（见 clone() 里的详细说明）。
        #   上游默认 target=8000 / overlap=800，在 CPU 上**极慢**（一次 25 秒），
        #   而平台的请求是同步等待的 —— 这个参数直接决定演示能不能跑完。
        self.vocoder_target = vocoder_target
        self.vocoder_overlap = vocoder_overlap

        # 上游的 synthesizer / vocoder 是"平铺"的包，需要把仓库根目录放进 sys.path。
        # ★ 只放仓库根目录本身（它里面有 synthesizer/ 和 vocoder/ 两个包）。
        if str(self.rtc_dir) not in sys.path:
            sys.path.insert(0, str(self.rtc_dir))

        self.synthesizer_path = self._find_file("synthesizer.pt")
        self.vocoder_path = self._find_file("vocoder.pt")

        from synthesizer.inference import Synthesizer
        from vocoder import inference as vocoder_inference

        self._vocoder = vocoder_inference
        self._synthesizer = Synthesizer(self.synthesizer_path, verbose=False)
        self._synthesizer.load()
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
            f"已尝试：\n" + "\n".join(f"    {c}" for c in candidates)
        )

    # -- 对外 ------------------------------------------------------------
    def clone(self, text: str, embedding: np.ndarray, quiet: bool = True) -> np.ndarray:
        """文本 + 说话人嵌入 -> 音频波形 (N,)，采样率 16 kHz。

        参数
        ----
        text      : 要念的文本（英文最佳，上游是英文 Tacotron）
        embedding : (256,) 说话人嵌入
        quiet     : True 时屏蔽上游 WaveRNN 的 tqdm 进度条（默认屏蔽）

        ⚠️ **text 不能是空字符串**。上游 Tacotron 拿到空文本会吐出一段退化的 mel，
        它自己的"裁尾部静音"循环（`while np.max(m[:, -1]) < stop_threshold: m = m[:, :-1]`）
        会把这段 mel 一路裁空，然后越界报
        `index -1 is out of bounds for axis 1 with size 0`。
        这个失败还是**间歇性**的（同一次调用有时成功有时越界），极难查 ——
        所以这里显式拦下来，给一句人话。
        """
        if not text or not text.strip():
            raise ValueError(
                "克隆机需要非空的参考文本（上游 Tacotron 是英文模型）。\n"
                "空文本会让上游吐出一段退化 mel，然后越界报错。"
            )

        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        specs = self._synthesizer.synthesize_spectrograms([text.strip()], [embedding])
        if not specs:
            raise RuntimeError("上游合成器没有返回结果。")
        spectrogram = np.asarray(specs[0], dtype=np.float32)
        if spectrogram.size == 0:
            raise RuntimeError(
                "上游合成器返回了空频谱（参考文本可能有问题）。\n"
                "换一个非空、英文的句子再试。"
            )

        # ★ 上游 WaveRNN 默认会刷 tqdm 进度条（几百行）。
        #   注意上游写的是 `progress_callback = progress_callback or self.gen_display`，
        #   所以**传 None 等于启用 tqdm**；要静音必须传一个真的可调用对象。
        callback = (lambda *args, **kwargs: None) if quiet else None

        # ★★ 折叠块长：这是**平台能不能用**的关键参数（实测，CPU）
        #
        #   上游 WaveRNN 是自回归的，但会把 mel 折叠成 `target` 长度的块**并行**生成，
        #   然后交叉淡化拼起来。折叠后，Python 层的循环次数 = target − overlap，
        #   **与 mel 长度无关**！所以：
        #     · 默认 target=8000 → 循环 7200 次 → 一次 25 秒（mel 45 帧也是 25 秒）
        #     · 改成 target=1000 → 循环 900 次 → 一次 6.6 秒（★ 3.8 倍）
        #
        #   音质没有损失（实测）：
        #     · 同样参数连跑两次的波形差异 = 0.0944（声码器本身有随机性，mu-law 采样）
        #     · 默认 vs 快速 的波形差异 = 0.0928 → **同量级**，不是音质损失
        #
        #   要追求最高音质可以传 target=8000/overlap=800（实测 25 秒一次）。
        waveform = self._vocoder.infer_waveform(
            spectrogram,
            target=self.vocoder_target,
            overlap=self.vocoder_overlap,
            progress_callback=callback,
        )
        return np.asarray(waveform, dtype=np.float32).reshape(-1)

    def describe(self) -> str:
        return (
            "克隆机桥接就绪（上游 Real-Time-Voice-Cloning：合成器 + 声码器）\n"
            f"  合成器权重 : {self.synthesizer_path}\n"
            f"  声码器权重 : {self.vocoder_path}\n"
            "  接口       : (文本, 256维嵌入) -> 16 kHz 波形\n"
            "  角色       : 论文 Figure 1 里的「攻击者」"
        )
