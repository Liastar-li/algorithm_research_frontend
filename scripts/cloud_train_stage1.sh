#!/usr/bin/env bash
# 云机一键脚本：Stage 1 正式训练（Linux + NVIDIA GPU）
#
# 【为什么要有它】
# 训练要烧钱，所以把"配环境"和"开跑"拆成两条独立的命令：
#   --setup   只把环境、数据、权重准备好，**不启动训练**（云机按小时计费）
#   --start   才真正开跑（默认放 tmux 里，断线不影响）
# 中间可以先跑 check_env.py 自检，确认没问题再 --start。
#
# 【用法】
#   bash scripts/cloud_train_stage1.sh --setup
#   bash scripts/cloud_train_stage1.sh --status
#   bash scripts/cloud_train_stage1.sh --start
#   bash scripts/cloud_train_stage1.sh --start --stage classifier --iterations 200000
#
# 【可调项】
#   --data-root DIR   数据根目录（默认 /root/vpp/data）
#   --per-speaker N   每个说话人取几句（0 = 全部；调试 5，正式 0）
#   --archive FILE    直接用现成数据包（.tar.gz），跳过下载与转换
#   --skip-data       跳过数据准备（数据已就位时）
#   --batch-size N    批大小（默认用 config 里的值）
#   --iterations N    训练步数（默认用 config 里的值）
#   --stage NAME      classifier | model（默认 model）
#   --session NAME    tmux 会话名（默认 vpp-train）
#
# 【环境变量】
#   VPP_DATA_ROOT   数据根目录
#   VPP_VENV        python 虚拟环境目录（默认 $HOME/vpp-venv）
#   HF_ENDPOINT     HuggingFace 镜像（国内建议 https://hf-mirror.com）
#
# 【注意】VCTK 整包 11.75 GiB，本脚本只定向拉 wav48_silence_trimmed/ 与 txt/。
#   国内直连 Edinburgh 约 30~400 KB/s；**云机带宽通常好得多**，
#   若仍很慢，先用 --per-speaker 5 跑通流程，正式训练再补齐。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${VPP_DATA_ROOT:-/root/vpp/data}"
VENV_DIR="${VPP_VENV:-$HOME/vpp-venv}"
PY="$VENV_DIR/bin/python"
SESSION="vpp-train"
STAGE="model"
ITERATIONS=""
BATCH_SIZE=""
PER_SPEAKER=0
ARCHIVE=""
SKIP_DATA=0
ACTION=""

export VPP_DATA_ROOT="$DATA_ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

while [ $# -gt 0 ]; do
  case "$1" in
    --setup) ACTION="setup" ;;
    --start) ACTION="start" ;;
    --status) ACTION="status" ;;
    --stop) ACTION="stop" ;;
    --data-root) DATA_ROOT="$2"; shift ;;
    --per-speaker) PER_SPEAKER="$2"; shift ;;
    --archive) ARCHIVE="$2"; shift ;;
    --batch-size) BATCH_SIZE="$2"; shift ;;
    --iterations) ITERATIONS="$2"; shift ;;
    --stage) STAGE="$2"; shift ;;
    --session) SESSION="$2"; shift ;;
    --skip-data) SKIP_DATA=1 ;;
    -h|--help)
      sed -n '2,40p' "${BASH_SOURCE[0]}"
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
    echo "         Stage 1 在 CPU 上也能跑，但慢约 50 倍（实测 14 s/step，GPU 0.3 s/step）。"
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
    "$PY" -m pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
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
  say "下载上游权重"
  local pkg="$REPO_ROOT/algorithms/voiceprint_protection/scripts"
  "$PY" "$pkg/download_encoder_weights.py"
  "$PY" "$pkg/download_rtc_weights.py"
  "$PY" "$pkg/download_hifigan_weights.py"
  echo "  权重目录：$DATA_ROOT/models/default"
  echo "  HF_ENDPOINT=$HF_ENDPOINT"

  # WER 用的 Whisper（论文用 Whisper-large = faster-whisper 的 large-v3）。
  # 云机带宽通常比家里好；缺了也不影响训练，只是算不了 WER。
  local whisper_model="${VPP_WHISPER_MODEL:-large-v3}"
  echo "  下载 Whisper：$whisper_model"
  if "$PY" -c "from faster_whisper import WhisperModel; WhisperModel('$whisper_model', device='cpu', compute_type='int8')"; then
    echo "  [OK] Whisper 就位"
  else
    echo "  [提示] Whisper 没下成功 —— 不影响训练，只是算不了 WER，回头再补"
  fi
}

prepare_data() {
  say "准备 VCTK 数据"
  if [ "$SKIP_DATA" = "1" ]; then
    echo "  [跳过] --skip-data"
    return
  fi
  if [ -n "$ARCHIVE" ]; then
    echo "  从归档解包：$ARCHIVE"
    mkdir -p "$DATA_ROOT"
    tar -xf "$ARCHIVE" -C "$DATA_ROOT"
    return
  fi

  local raw="$DATA_ROOT/raw"
  if [ ! -d "$raw/wav48_silence_trimmed" ]; then
    echo "  下载 VCTK（只取 wav48_silence_trimmed + txt）…"
    # ★ --per-speaker 100 是**数据契约**：每人均匀取 100 条 = 11,000 条，
    #   与 B 侧逐条一致（差集为空）。不给它会下全部 88,328 个 flac（10.9 GiB），
    #   而且数据跟 B 的那份对不上。
    "$PY" "$REPO_ROOT/scripts/download_vctk.py" --out "$raw" --include wav48_silence_trimmed --include txt --per-speaker 100 --concurrency 32
  else
    echo "  [跳过] 已有 $raw/wav48_silence_trimmed"
  fi

  echo "  转 16 kHz 并生成清单（train/val/test = 80/10/20）…"
  "$PY" "$REPO_ROOT/algorithms/voiceprint_protection/scripts/prepare_data.py" --raw-root "$raw" --out-root "$DATA_ROOT" --limit-per-speaker "$PER_SPEAKER"
}

start_training() {
  say "启动 Stage 1 训练"
  if [ ! -x "$PY" ]; then
    echo "  [失败] 虚拟环境不存在：$VENV_DIR（先跑 --setup）"
    exit 1
  fi
  local train_args=(--stage "$STAGE" --device cuda)
  if [ -n "$ITERATIONS" ]; then train_args+=(--iterations "$ITERATIONS"); fi
  if [ -n "$BATCH_SIZE" ]; then train_args+=(--batch-size "$BATCH_SIZE"); fi

  local log_dir="$REPO_ROOT/logs"
  mkdir -p "$log_dir"
  local log_file="$log_dir/stage1_${STAGE}.log"

  if command -v tmux >/dev/null 2>&1; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    local cmd="$PY $REPO_ROOT/algorithms/voiceprint_protection/stage1/train.py"
    for a in "${train_args[@]}"; do cmd="$cmd $a"; done
    tmux new-session -d -s "$SESSION" "$cmd 2>&1 | tee -a $log_file"
    echo "  tmux 会话：$SESSION"
    echo "  跟踪：tmux attach -t $SESSION"
  else
    echo "  [提示] 没装 tmux，改用 nohup 后台跑"
    nohup "$PY" "$REPO_ROOT/algorithms/voiceprint_protection/stage1/train.py" "${train_args[@]}" > "$log_file" 2>&1 &
    echo "  跟踪：tail -f $log_file"
  fi
  echo "  日志：$log_file"
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
    echo "  确认无误后再开跑："
    echo "    bash scripts/cloud_train_stage1.sh --start"
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
    echo "  已停止 $SESSION（云机记得关机，别按小时空烧）"
    ;;
  *)
    echo "用法：bash scripts/cloud_train_stage1.sh {--setup|--start|--status|--stop}"
    echo "详细说明：bash scripts/cloud_train_stage1.sh --help"
    exit 2
    ;;
esac
