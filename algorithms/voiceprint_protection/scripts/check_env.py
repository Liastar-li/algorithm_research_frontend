"""环境自检脚本 —— 两名成员都必须跑一次，把输出贴到群里互相比对。

用法：
    conda activate vpp
    python scripts/check_env.py

输出包含：
  1. Python / 操作系统 / Conda 环境信息
  2. PyTorch 与 CUDA 状态（是否真的用上 GPU，而不是 CPU 版）
  3. GPU 型号与显存
  4. 关键依赖版本表
  5. 三个组件的 mel 参数对照表（最容易出问题的地方）
  6. 一段可复现的"环境指纹"字符串 —— 两人指纹一致 = 环境统一
"""

import hashlib
import platform
import subprocess
import sys


def line(char="-", width=64):
    print(char * width)


def section(title):
    print()
    line("=")
    print(f"  {title}")
    line("=")


def check_python():
    section("1. Python 与系统")
    print(f"  Python 版本      : {sys.version.split()[0]}")
    print(f"  Python 完整版本  : {sys.version.replace(chr(10), ' ')}")
    print(f"  解释器路径       : {sys.executable}")
    print(f"  操作系统         : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  平台描述         : {platform.platform()}")

    expected = "3.9"
    if not sys.version.startswith(expected):
        print(f"  [警告] 期望 Python {expected}.x，当前是 {sys.version.split()[0]}。")
        print("         建议 conda create -n vpp python=3.9 -y 重建环境。")
    else:
        print(f"  [OK] Python {expected}.x，符合预期。")

    if "vpp" not in sys.executable:
        print("  [警告] 当前不在 vpp 环境里，可能是系统 Python。")
        print("         执行：conda activate vpp")
    else:
        print("  [OK] 处于 vpp 环境中。")


def check_torch():
    section("2. PyTorch 与 CUDA")
    try:
        import torch
        import torchaudio
    except ImportError as error:
        print(f"  [失败] 无法导入 torch/torchaudio：{error}")
        print("         执行：pip install -r requirements.txt")
        return None

    print(f"  torch 版本       : {torch.__version__}")
    print(f"  torchaudio 版本  : {torchaudio.__version__}")
    print(f"  CUDA 编译版本    : {torch.version.cuda}")

    if torch.version.cuda is None:
        print("  [提示] 当前是 CPU 版 PyTorch（" + torch.__version__ + "）。")
        print()
        print("         这**不阻塞开发**：本项目的 L0（平台打通）、L1（推理链）、")
        print("         评测与文档都可以在 CPU 上完成。只有 Stage 1/2 训练需要 GPU。")
        print()
        print("         若确实要本地跑训练，按下面任一方式换成 CUDA 版：")
        print()
        print("         方式一：先探测国内源（PowerShell 里运行）：")
        print("             $urls = @(")
        print("               'https://mirrors.huaweicloud.com/pytorch-wheels/cu121/torch/',")
        print("               'https://mirrors.cloud.tencent.com/pytorch-wheels/cu121/torch/',")
        print("               'https://mirror.nju.edu.cn/pytorch/whl/cu121/torch/',")
        print("               'https://mirrors.ustc.edu.cn/pytorch-wheels/cu121/torch/',")
        print("               'https://download.pytorch.org/whl/cu121/torch/'")
        print("             )")
        print("             foreach ($u in $urls) {")
        print("               $html = curl.exe -s --max-time 25 $u")
        print("               if (-not $html) { Write-Host \"$u  -> 访问失败\"; continue }")
        print("               $hit = $html | findstr /i \"torch-2.4.1\" | findstr /i \"cp39-cp39-win_amd64\"")
        print("               if ($hit) { Write-Host \"$u  -> [命中] 可用\" }")
        print("               else { Write-Host \"$u  -> 可访问但无目标轮子\" }")
        print("             }")
        print()
        print("         方式二：命中后安装（torchaudio 要单独一个 -f）：")
        print("             pip install --force-reinstall torch==2.4.1+cu121 torchaudio==2.4.1+cu121 `")
        print("               -f https://mirror.nju.edu.cn/pytorch/whl/cu121/torch/ `")
        print("               -f https://mirror.nju.edu.cn/pytorch/whl/cu121/torchaudio/ `")
        print("               -i https://pypi.tuna.tsinghua.edu.cn/simple")
        print("             （南京大学镜像已实测可用，sha256 与官方一致）")
        print()
        print("         方式三：官方索引 + 下载器手动拉（pip 不支持断点续传）：")
        print("             curl.exe -s https://download.pytorch.org/whl/cu121/torch/ `")
        print("               | findstr /i \"torch-2.4.1\" | findstr /i \"cp39-cp39-win_amd64\"")
        print("             用 IDM / Motrix 下载后本地 pip install 该 .whl")
        print()
        print("         方式四（推荐）：训练放云上。华为云 ECS 选「PyTorch 预置镜像」，")
        print("         CUDA 版 torch 开箱即用、零下载。本地只做开发。")
        return None

    print(f"  [OK] 是 CUDA 版（cu{torch.version.cuda.replace('cuda', '')}）。")

    available = torch.cuda.is_available()
    if available:
        print(f"  [OK] CUDA 可用，device_count = {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            total_gb = properties.total_memory / 1024 ** 3
            print(f"       GPU {index}: {properties.name}  显存 {total_gb:.1f} GB  "
                  f"capability {properties.major}.{properties.minor}")
        if torch.cuda.device_count() == 1:
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if total_gb < 10:
                print(f"       [提示] 显存只有 {total_gb:.1f} GB，训练需要收紧 batch / 片段长度。")
                print("              建议：batch_size=4~8，训练片段截断到 2~3 秒，")
                print("                    并开启梯度累积与 AMP 混合精度。")
    else:
        print("  [失败] torch 编译了 CUDA 但不可用 —— 通常是驱动版本过低。")
        print("         检查：nvidia-smi  的 Driver Version 需 >= 525.60.13")

    print()
    print("  实际跑一次张量运算验证：")
    try:
        device = torch.device("cuda" if available else "cpu")
        tensor = torch.randn(512, 512, device=device)
        matrix = torch.randn(512, 512, device=device)
        result = tensor @ matrix
        torch.cuda.synchronize() if available else None
        print(f"       device={device}  输出形状={tuple(result.shape)}  "
              f"有限值={bool(torch.isfinite(result).all())}  [OK]")
    except Exception as error:  # noqa: BLE001
        print(f"       [失败] {error}")


def check_nvidia_smi():
    section("3. 驱动信息（nvidia-smi）")
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if output.returncode == 0 and output.stdout.strip():
            for row in output.stdout.strip().splitlines():
                print(f"  {row}")
            print("  [OK] 驱动可用。")
        else:
            print("  [警告] nvidia-smi 未返回有效结果。")
            if output.stderr.strip():
                print(f"         {output.stderr.strip()}")
    except FileNotFoundError:
        print("  [警告] 未找到 nvidia-smi 命令（可能未安装 NVIDIA 驱动）。")
    except Exception as error:  # noqa: BLE001
        print(f"  [警告] 执行 nvidia-smi 失败：{error}")


def check_packages():
    section("4. 关键依赖版本")
    names = [
        "numpy", "scipy", "librosa", "soundfile", "soxr",
        "tqdm", "tensorboard", "matplotlib",
        "fastapi", "uvicorn", "python_multipart", "pydantic", "requests",
        "huggingface_hub", "tokenizers", "ctranslate2", "faster_whisper",
        "jiwer", "numba", "llvmlite",
    ]
    missing = []
    versions = {}
    for name in names:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", None)
            if not version:
                # 有些包（如 jiwer）不设 __version__，改从包元数据取
                try:
                    from importlib.metadata import version as _meta_version

                    version = _meta_version(name.replace("_", "-"))
                except Exception:  # noqa: BLE001
                    version = "unknown"
            versions[name] = str(version)
            print(f"  {name:<22} {version}")
        except ImportError:
            versions[name] = "MISSING"
            missing.append(name)
            print(f"  {name:<22} [缺失]")

    # python-multipart 的模块名陷阱：
    #   <= 0.0.9 提供模块名 multipart；>= 0.0.10 才提供 python_multipart。
    #   FastAPI 0.110.3 依赖的 starlette 0.37 只会 `import python_multipart`，
    #   所以装了 0.0.9 会表现为"pip 显示已安装、import 却失败"。
    if "python_multipart" in missing:
        legacy = False
        try:
            import multipart as _legacy_multipart

            legacy = True
        except ImportError:
            legacy = False
        print()
        print("  [警告] python_multipart 无法导入。")
        if legacy:
            print("         诊断：当前装的是 python-multipart <= 0.0.9，它提供的模块名是")
            print("               老名字 multipart，而 FastAPI 0.110.3 依赖的 starlette 0.37")
            print("               只会 `import python_multipart` —— 装了也用不上，")
            print("               起服务时会报 'Form data requires \"python-multipart\"'。")
            print("         修复：pip install -U \"python-multipart==0.0.20\"")
        else:
            print("         安装：pip install \"python-multipart==0.0.20\"")
            print("         这是 FastAPI 接收上传文件与表单的必需依赖。")

    if missing:
        print()
        print(f"  [警告] 缺失 {len(missing)} 个包：{', '.join(missing)}")
        print("         或一次装齐：pip install -r requirements.txt")
    else:
        print()
        print("  [OK] 全部依赖就位。")


def check_mel():
    section("5. mel 参数对照（最容易出错的地方）")
    print("""
  论文明确要求：16 kHz、80 维 mel、x ∈ R^(T×80)

  但三个组件的"原生"配置互不相同，必须统一到一份参数表：
  ┌──────────────┬──────────┬────────┬────────┬────────┬────────────┐
  │ 组件          │ 采样率    │ mel 维 │ hop    │ n_fft  │ fmin/fmax  │
  ├──────────────┼──────────┼────────┼────────┼────────┼────────────┤
  │ AutoVC 原版   │ 22050    │ 80     │ 256    │ 1024   │ 40 / 7600  │
  │ GE2E (RTC)    │ 16000    │ 40     │ 160    │ 400    │ 0 / 8000   │
  │ HiFi-GAN 原版 │ 22050    │ 80     │ 256    │ 1024   │ 0 / 8000   │
  │ 论文/本项目   │ 16000    │ 80     │ 256    │ 1024   │ 40 / 7600  │
  └──────────────┴──────────┴────────┴────────┴────────┴────────────┘

  【铁律】
  1. 采样率统一 16000 Hz：GE2E 权重就是 16 kHz 的，改采样率权重直接失效。
  2. Stage 1 训练、Stage 2 训练、推理、评测，mel 参数必须完全一致，
     否则 x_adv 和 x 的帧轴对不上，L_mel 会恒为一个巨大的常数，模型学不到东西。
  3. GE2E 内部自己的 mel（40 维）不要去改——它只用来提 h，
     我们外部的 80 维 mel 只用于 L_mel / L_cd / 解码器。
  4. 声码器权重的采样率必须和你的模型输出对得上：
     HiFi-GAN 官方主推 22.05 kHz 权重；要用 16 kHz 就必须换成对应的 16 k 权重，
     不能拿 22.05 k 的权重去合成 16 k 的 mel。
  5. 全部封装到 algorithms/voiceprint_protection/audio_utils.py，
     任何地方都不许再手写 mel 参数。

  下面的检查已由 scripts/test_audio_utils.py 承担（那里有可自动失败的断言）。
  本脚本只做环境层面的提醒。
""")
    print("  -> 运行 python algorithms/voiceprint_protection/scripts/test_audio_utils.py")
    print("     该脚本会验证：两个预设只差采样率、mel 形状与帧数、")
    print("     与 librosa 原生调用数值完全一致、论文超参值、张量维度约定。")
    print()
    print("  [重要] 模型侧 mel 与声码器侧 mel 约定不同，不要混用：")
    print("     模型侧(AutoVC/本项目): fmin 40 / fmax 7600, power=2, log10(mel)")
    print("     声码器侧(HiFi-GAN)   : fmin 0  / fmax 8000, power=1, log10|STFT|,")
    print("                          再做 (x + 4.5) / 5.0 归一化")
    print("     两者混用会让声码器输出变成噪音，且不报错。")
    print("     见 audio_utils.py 的 mel() 与 vocoder_mel() 两个函数。")


def fingerprint():
    section("6. 环境指纹（两人比对这一行）")
    parts = [f"python={sys.version.split()[0]}"]
    try:
        import torch
        import torchaudio
        parts.append(f"torch={torch.__version__}")
        parts.append(f"torchaudio={torchaudio.__version__}")
        parts.append(f"cuda={torch.version.cuda}")
        parts.append(f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    except ImportError:
        parts.append("torch=MISSING")

    for name in ["numpy", "librosa", "scipy", "faster_whisper", "fastapi"]:
        try:
            module = __import__(name)
            parts.append(f"{name}={getattr(module, '__version__', 'unknown')}")
        except ImportError:
            parts.append(f"{name}=MISSING")

    raw = "|".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    print(f"  明细: {raw}")
    print(f"  指纹: {digest}")
    print()
    print("  两名成员的指纹应当完全一致。若不一致，用以下命令定位差异：")
    print("      pip freeze > my_freeze.txt")
    print("      diff my_freeze.txt requirements.lock.txt")
    print()
    print("  把这一行贴到群里，两人都贴，比对是否一致。")


def main():
    print()
    print("=" * 64)
    print("  语音声纹主动防御 · 环境自检")
    print("=" * 64)
    check_python()
    check_torch()
    check_nvidia_smi()
    check_packages()
    check_mel()
    fingerprint()
    print()
    line("=")
    print("  自检结束。若上面没有 [失败] 项，环境即可用。")
    print("=")
    print()


if __name__ == "__main__":
    main()
