"""Stage 1：内容-说话人解耦-重建网络（论文 §2.2）。

模块划分：
    model.py            —— Ec / D / Postnet（照搬 AutoVC 官方架构）
    losses.py           —— L_rec / L_cd / L_dis（论文式 3/4/5/6）
    dataset.py          —— VCTK 清单 -> 两套 mel + 说话人 ID
    speaker_encoder.py  —— 可微的 GE2E（论文说它端到端微调）
    train.py            —— 两阶段训练主循环

【★ 路径引导（为什么这里要动 sys.path）】
本包里的模块用的是**平铺导入**（`import config` / `import audio_utils`），
这是上游 AutoVC / Real-Time-Voice-Cloning 的风格，也让
`scripts/*.py` 能直接复用同一份实现。

代价是：`algorithms/voiceprint_protection/` 必须在 `sys.path` 上，
否则 `import config` 会失败。两种启动方式各有一处坑：

  1. `python -m algorithms.voiceprint_protection.stage1.train`
     → 先执行本文件的 `__init__.py`，而里面 `from stage1.model import ...`
        找不到顶层的 `stage1` 包（此时 `stage1` 是
        `algorithms.voiceprint_protection.stage1`，不是顶层包）
  2. `python algorithms/voiceprint_protection/stage1/train.py`
     → `sys.path[0]` 是 `stage1/` 目录本身，找不到 `config`

所以这里**先把父目录插进 sys.path**，再用**相对导入**拉子模块。
这样两种方式、以及 `from stage1 import losses` 这种平铺写法都能跑。
"""

from __future__ import annotations

import sys
from pathlib import Path

# ★ 必须放在下面所有导入之前：父目录 = algorithms/voiceprint_protection
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from .losses import (  # noqa: E402,F401
    SpeakerClassifier,
    content_consistency_loss,
    content_windows,
    disentanglement_loss,
    reconstruction_loss,
    stage1_loss,
)
from .model import (  # noqa: E402,F401
    ContentEncoder,
    Postnet,
    SpeakerDecoder,
    Stage1Model,
    count_parameters,
)

__all__ = [
    "ContentEncoder",
    "Postnet",
    "SpeakerDecoder",
    "Stage1Model",
    "count_parameters",
    "SpeakerClassifier",
    "content_windows",
    "content_consistency_loss",
    "disentanglement_loss",
    "reconstruction_loss",
    "stage1_loss",
]
