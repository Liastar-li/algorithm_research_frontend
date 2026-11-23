#!/usr/bin/env bash
# 云机一键脚本：Stage 2 训练（Linux + NVIDIA GPU）
#
# 【与 cloud_train_stage1.sh 的关系】
#   Stage 1 脚本准备的是「Stage 1 需要的东西」；本脚本准备的是「Stage 2 需要的东西」。
#   两者共用同一套环境与数据准备逻辑（--setup 可以直接复用），
#   差别在最后启动的那条命令。
#
# 【★ Stage 2 比 Stage 1 多两样】
#   ① **Stage 1 的 checkpoint**（必须先有）—— 从本机 scp 上去，不要在云机上重训
#   ② **HiFi-GAN 声码器权重**（Stage 2 的梯度要穿过它，缺了 Stage 2 训不了）
#
# 【用法】
#   bash scripts/cloud_train_stage2.sh --setup          # 配环境 + 数据 + 权重（不烧训练时间）
#   bash scripts/cloud_train_stage2.sh --status         # 看进度
#   bash scripts/cloud_train_stage2.sh --start          # 开跑（tmux 里，断线不影响）
#   bash scripts/cloud_train_stage2.sh --start --resume # 断了接着跑
#   bash scripts/cloud_train_stage2.sh --stop           # 停了（别忘了去控制台释放实例）
#
# 【★ 为什么一定要用 --max-hours】
#   云机按小时计费，忘了关就一直烧钱（我们只有 500 元代金券）。
#   --max-hours 到点自动存盘退出 ⇒ 哪怕忘了看，损失也有上限。
#
# 【可调项】
#   --data-root DIR    数据根目录（默认 /root/vpp/data）
#   --steps N          训练步数（默认 config.STAGE2['iterations'] = 200000）
#   --batch-size N     批大小（默认 config.STAGE2['batch_size'] = 8）
#   --accum-steps N    梯度累积（默认 config.STAGE2['accum_steps']）
#   --log-every N      每多少步打一行日志（默认 50；冒烟时设 1）
#   --save-every N     每多少步存一次 checkpoint（默认 1000）
#   --max-hours N      时长上限（小时）—— 到点自动存盘退出
#   --outdir DIR       checkpoint 的输出目录（默认 <仓库>/algorithms/voiceprint_protection/weights/stage2）
#                      ★ 消融实验要用不同的目录，否则会覆盖主实验的权重
#   --lambda-de N      λ_de（论文 5.0；消融 w/o L_de 时设 0）
#   --lambda-per N     λ_per（论文 15.0）
#   --lambda-mask N    λ_mask（论文 0.1；消融 w/o L_mask 时设 0）
#   --lambda-gan N     λ_GAN（论文 1.0；设 0 = 只训 L_per + L_de）
#   --no-mel-loss      消融 w/o L_mel（L_per 里只留 L_mask）
#   --stage1-ckpt PATH Stage 1 权重路径（必须）
#   --session NAME     tmux 会话名（默认 vpp-stage2）
#   --skip-data        跳过数据准备（数据已就位时）
#
# 【环境变量】
#   VPP_DATA_ROOT   数据根目录
#   VPP_VENV        虚拟环境目录（默认 $HOME/vpp-venv）
#   HF_ENDPOINT     HuggingFace 镜像（国内建议 https://hf-mirror.com）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${VPP_DATA_ROOT:-/root/vpp/data}"
VENV_DIR="${VPP_VENV:-$HOME/vpp-venv}"
PY="$VENV_DIR/bin/python"
SESSION="vpp-stage2"
STEPS=""
BATCH_SIZE=""
ACCUM_STEPS=""
LOG_EVERY=""
SAVE_EVERY=""
MAX_HOURS=""
OUTDIR=""
LAMBDA_DE=""
LAMBDA_PER=""
LAMBDA_MASK=""
LAMBDA_GAN=""
NO_MEL_LOSS=0
STAGE1_CKPT=""
SKIP_DATA=0
ACTION=""

export VPP_DATA_ROOT="$DATA_ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

while [ $# -gt 0 ]; do
  case "$1" in
    --setup) ACTION="setup" ;;
    --start) ACTION="start" ;;
    --status) ACTION="status" ;;
    --stop) ACTION="stop" ;;
    --data-root) DATA_ROOT="$2"; shift ;;
    --steps) STEPS="$2"; shift ;;
    --batch-size) BATCH_SIZE="$2"; shift ;;
    --accum-steps) ACCUM_STEPS="$2"; shift ;;
    --log-every) LOG_EVERY="$2"; shift ;;
    --save-every) SAVE_EVERY="$2"; shift ;;
    --outdir) OUTDIR="$2"; shift ;;
    --lambda-de) LAMBDA_DE="$2"; shift ;;
    --lambda-per) LAMBDA_PER="$2"; shift ;;
    --lambda-mask) LAMBDA_MASK="$2"; shift ;;
    --lambda-gan) LAMBDA_GAN="$2"; shift ;;
    --no-mel-loss) NO_MEL_LOSS=1 ;;
    --max-hours) MAX_HOURS="$2"; shift ;;
    --stage1-ckpt) STAGE1_CKPT="$2"; shift ;;
    --session) SESSION="$2"; shift ;;
    --skip-data) SKIP_DATA=1 ;;
    -h|--help)
      sed -n '2,38p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "未知参数：$1（用 --help 看用法）"
      exit 2
      ;;
  esac
  shift
done

say() {
  echo ""
  echo "=============================================================="
  echo "  $*"
  echo "=============================================================="
}

check_gpu() {
  say "检查 GPU"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  else
    echo "  [警告] 没找到 nvidia-smi —— 这台机器可能没有 NVIDIA GPU。"
    echo "         Stage 2 在 CPU 上也能跑，但慢约 100 倍（实测 30 s/step，GPU 约 0.2 s/step）。"
    echo "         ⇒ 200k 步在 CPU 上要跑两个月，别在 CPU 上开长训。"
  fi
}

setup_python() {
  say "准备 Python 环境：$VENV_DIR"
  if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
  fi
  PY="$VENV_DIR/bin/python"
  "$PY" -m pip install --upgrade pip wheel

  if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "  [OK] torch 已就位（带 CUDA）"
  else
    echo "  安装 torch（CUDA 12.1）…"
    if ! "$PY" -m pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121; then
      echo "  官方索引不行，退到 NJU 镜像（requirements.txt 里实测过的那条）…"
      "$PY" -m pip install torch==2.4.1 torchaudio==2.4.1 -f https://mirror.nju.edu.cn/pytorch/whl/cu121/torch/ -f https://mirror.nju.edu.cn/pytorch/whl/cu121/torchaudio/ -i https://pypi.tuna.tsinghua.edu.cn/simple
    fi
    "$PY" -c "import torch; print('  torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  fi

  echo "  安装其余依赖…"
  "$PY" -m pip install -r "$REPO_ROOT/algorithms/voiceprint_protection/requirements.txt"
}

setup_third_party() {
  say "拉取第三方官方组件（固定 commit）"
  bash "$REPO_ROOT/scripts/setup_third_party.sh"
}

fetch_weights() {
  say "下载上游权重（★ Stage 2 的梯度要穿过 HiFi-GAN，缺了训不了）"
  local pkg="$REPO_ROOT/algorithms/voiceprint_protection/scripts"
  "$PY" "$pkg/download_encoder_weights.py"
  "$PY" "$pkg/download_rtc_weights.py"
  "$PY" "$pkg/download_hifigan_weights.py"
  echo "  权重目录：$DATA_ROOT/models/default"
  echo "  HF_ENDPOINT=$HF_ENDPOINT"
}

prepare_data() {
  say "准备 VCTK 数据"
  if [ "$SKIP_DATA" = "1" ]; then
    echo "  [跳过] --skip-data"
    return
  fi
  local raw="$DATA_ROOT/raw"
  if [ ! -d "$raw/wav48_silence_trimmed" ]; then
    echo "  下载 VCTK（只取 wav48_silence_trimmed + txt）…"
    # ★ --per-speaker 100 是**数据契约**：每人均匀取 100 条 = 11,000 条，
    #   与 B 侧逐条一致（差集为空）。
    "$PY" "$REPO_ROOT/scripts/download_vctk.py" --out "$raw" --include wav48_silence_trimmed --include txt --per-speaker 100 --concurrency 32
  else
    echo "  [跳过] 已有 $raw/wav48_silence_trimmed"
  fi

  echo "  转 16 kHz 并生成清单（train/val/test = 80/10/20）…"
  "$PY" "$REPO_ROOT/algorithms/voiceprint_protection/scripts/prepare_data.py" --raw-root "$raw" --out-root "$DATA_ROOT"
}

check_stage1_checkpoint() {
  say "检查 Stage 1 checkpoint（★ Stage 2 的解码器要用它）"
  local path="$STAGE1_CKPT"
  if [ -z "$path" ]; then
    path="$REPO_ROOT/algorithms/voiceprint_protection/weights/stage1_model.pt"
  fi
  if [ ! -f "$path" ]; then
    echo "  [失败] 找不到 Stage 1 权重：$path"
    echo ""
    echo "  ★ Stage 2 训练**必须**有 Stage 1 的权重（冻结的解码器 D(c, h_adv)）。"
    echo "    从本机传上去（358 MB 左右）："
    echo "      scp <本机>/algorithms/voiceprint_protection/weights/stage1_model.pt \\"
    echo "          root@<云机IP>:$REPO_ROOT/algorithms/voiceprint_protection/weights/"
    echo ""
    echo "  或者显式指定：--stage1-ckpt /path/to/stage1_model.pt"
    exit 1
  fi
  echo "  [OK] $path"
  STAGE1_CKPT="$path"
}

start_training() {
  say "启动 Stage 2 训练"
  if [ ! -x "$PY" ]; then
    echo "  [失败] 虚拟环境不存在：$VENV_DIR（先跑 --setup）"
    exit 1
  fi
  check_stage1_checkpoint

  local train_args=(--vocoder hifigan --device cuda
                    --stage1-checkpoint "$STAGE1_CKPT")
  if [ -n "$STEPS" ]; then train_args+=(--steps "$STEPS"); fi
  if [ -n "$BATCH_SIZE" ]; then train_args+=(--batch-size "$BATCH_SIZE"); fi
  if [ -n "$ACCUM_STEPS" ]; then train_args+=(--accum-steps "$ACCUM_STEPS"); fi
  if [ -n "$LOG_EVERY" ]; then train_args+=(--log-every "$LOG_EVERY"); fi
  if [ -n "$SAVE_EVERY" ]; then train_args+=(--save-every "$SAVE_EVERY"); fi
  if [ -n "$MAX_HOURS" ]; then train_args+=(--max-hours "$MAX_HOURS"); fi
  if [ -n "$OUTDIR" ]; then train_args+=(--outdir "$OUTDIR"); fi
  if [ -n "$LAMBDA_DE" ]; then train_args+=(--lambda-de "$LAMBDA_DE"); fi
  if [ -n "$LAMBDA_PER" ]; then train_args+=(--lambda-per "$LAMBDA_PER"); fi
  if [ -n "$LAMBDA_MASK" ]; then train_args+=(--lambda-mask "$LAMBDA_MASK"); fi
  if [ -n "$LAMBDA_GAN" ]; then train_args+=(--lambda-gan "$LAMBDA_GAN"); fi
  if [ "$NO_MEL_LOSS" = "1" ]; then train_args+=(--no-mel-loss); fi

  # ★ 消融实验：不同的 λ / 不同的 outdir —— 别把主实验的权重覆盖了
  if [ -n "$OUTDIR" ]; then
    mkdir -p "$OUTDIR"
    echo "  输出目录   : $OUTDIR"
  fi
  if [ "$NO_MEL_LOSS" = "1" ]; then
    echo "  ★ 消融：w/o L_mel（L_per 里只留 L_mask）"
  fi

  local log_dir="$REPO_ROOT/logs"
  mkdir -p "$log_dir"
  local log_file="$log_dir/stage2.log"

  if command -v tmux >/dev/null 2>&1; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    local cmd="$PY $REPO_ROOT/algorithms/voiceprint_protection/stage2/train.py"
    for a in "${train_args[@]}"; do cmd="$cmd $a"; done
    tmux new-session -d -s "$SESSION" "$cmd 2>&1 | tee -a $log_file"
    echo "  tmux 会话：$SESSION"
    echo "  跟踪：tmux attach -t $SESSION"
  else
    echo "  [提示] 没装 tmux，改用 nohup 后台跑"
    nohup "$PY" "$REPO_ROOT/algorithms/voiceprint_protection/stage2/train.py" "${train_args[@]}" > "$log_file" 2>&1 &
    echo "  跟踪：tail -f $log_file"
  fi
  echo "  日志：$log_file"
  echo ""
  echo "  ★ 断线/被抢占后：加 --resume 接着跑（优化器状态、随机数状态都在 checkpoint 里）"
}

status_training() {
  say "训练状态"
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "  tmux 会话 $SESSION 正在跑，最近日志："
    tmux capture-pane -t "$SESSION" -p | tail -20
  else
    echo "  tmux 会话 $SESSION 不在跑"
  fi
  local log_dir="$REPO_ROOT/logs"
  if [ -d "$log_dir" ]; then
    ls -la "$log_dir" | tail -10
  fi
}

case "$ACTION" in
  setup)
    check_gpu
    setup_python
    setup_third_party
    fetch_weights
    prepare_data
    say "环境就绪（还没开始训练）"
    echo "  自检："
    echo "    source $VENV_DIR/bin/activate"
    echo "    export VPP_DATA_ROOT=$DATA_ROOT"
    echo "    python algorithms/voiceprint_protection/scripts/check_env.py"
    echo ""
    echo "  ★ 别忘了把 Stage 1 的 checkpoint 传上去（见 --start 的提示）"
    echo ""
    echo "  确认无误后再开跑："
    echo "    bash scripts/cloud_train_stage2.sh --start --max-hours 8"
    ;;
  start)
    start_training
    ;;
  status)
    status_training
    ;;
  stop)
    if command -v tmux >/dev/null 2>&1; then
      tmux kill-session -t "$SESSION" 2>/dev/null || true
    fi
    echo "  已停止 $SESSION"
    echo "  ★ 云机记得**释放实例**（不是只关机）—— 见 docs/notes/cloud_ecs_stage2.md §6"
    ;;
  *)
    echo "用法：bash scripts/cloud_train_stage2.sh {--setup|--start|--status|--stop}"
    echo "详细说明：bash scripts/cloud_train_stage2.sh --help"
    exit 2
    ;;
esac
