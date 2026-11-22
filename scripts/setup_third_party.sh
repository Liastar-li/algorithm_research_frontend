#!/usr/bin/env bash
# 一键拉取本项目依赖的第三方官方组件（按固定 commit，保证两人/两机环境完全一致）。
#
# 为什么需要这个脚本：
#   third_party/ 里是**第三方官方仓库**，我们不把它提交进成果仓库
#   （避免把别人的代码和许可证混进我们的提交）。
#   代价是每个人/每台机器都要自己拉一次 —— 所以用这个脚本把版本钉死。
#
# 用法：
#   bash scripts/setup_third_party.sh
#
# 幂等：已经是目标版本就跳过；版本不符会删除重拉。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$REPO_ROOT/third_party"

# 组件定义：名称|仓库地址|固定commit(留空则用默认分支)
COMPONENTS=(
  "AutoVC|https://github.com/auspicious3000/autovc.git|c1827a60e83e6d26030b8953843b2f7cf2bd36db"
  "wavenet_vocoder|https://github.com/r9y9/wavenet_vocoder.git|d4333546eccd3b681b7092e4f52160ad7e62dcc9"
  "Real-Time-Voice-Cloning|https://github.com/CorentinJ/Real-Time-Voice-Cloning.git|890f3a03187195b9829db2079b75c2ba2ab0405c"
)

echo "=============================================================="
echo "  第三方组件一键搭建"
echo "=============================================================="
echo "  目标目录: $THIRD_PARTY"
echo ""

mkdir -p "$THIRD_PARTY"

for entry in "${COMPONENTS[@]}"; do
  IFS='|' read -r name url sha <<< "$entry"
  path="$THIRD_PARTY/$name"

  echo "--- $name ---"
  echo "    $url"
  if [ -n "$sha" ]; then echo "    commit: $sha"; fi

  if [ -d "$path/.git" ]; then
    current="$(git -C "$path" rev-parse HEAD 2>/dev/null || echo "")"
    if [ -n "$sha" ] && [ "$current" = "$sha" ]; then
      echo "    [跳过] 已是目标版本"
      continue
    fi
    # ★ 2026-11-22（成员 A）：**断网时不删**。
    #   云机常常连不上 GitHub（实测 22 端口直接超时），
    #   原来「先 rm -rf 再 clone」=> 拉不到就把已有的也弄没了。
    #   现在先备份，拉成功再换；拉失败就**保留原来的**并继续。
    echo "    版本不符，尝试拉取目标版本（失败则保留现有副本）..."
    backup="${path}.bak.$(date +%s)"
    cp -a "$path" "$backup"
    rm -rf "$path"
  fi

  if ! { if [ -n "$sha" ]; then
      # 按 SHA 浅拉取：不需要完整历史
      git init -q "$path"
      git -C "$path" remote add origin "$url"
      git -C "$path" fetch -q --depth 1 origin "$sha"
      git -C "$path" checkout -q FETCH_HEAD
    else
      git clone -q --depth 1 "$url" "$path"
    fi; }; then
    echo "    [警告] 拉取失败（网络不通？）"
    if [ -n "${backup:-}" ] && [ -d "$backup" ]; then
      echo "    [保留] 恢复原有副本：$backup"
      mv "$backup" "$path"
    fi
    echo "    [跳过] $name（用现有副本继续）"
    continue
  fi
  [ -n "${backup:-}" ] && rm -rf "$backup"
  echo "    [OK] 已就绪  $(git -C "$path" rev-parse HEAD)"
done

# --- wavenet_vocoder 的 version.py 是打包时生成的，git 里没有，必须补上 ---
echo ""
echo "--- 生成 wavenet_vocoder/version.py ---"
VERSION_DIR="$THIRD_PARTY/wavenet_vocoder/wavenet_vocoder"
VERSION_FILE="$VERSION_DIR/version.py"
if [ -f "$VERSION_FILE" ]; then
  echo "    [跳过] 已存在"
else
  sha="$(git -C "$THIRD_PARTY/wavenet_vocoder" rev-parse HEAD)"
  version="0.1.1+${sha:0:7}"
  printf "__version__ = '%s'\n" "$version" > "$VERSION_FILE"
  echo "    [OK] 已生成 $VERSION_FILE  ($version)"
fi

echo ""
echo "=============================================================="
echo "  完成。接下来："
echo "  1) 权重放到 algorithms/voiceprint_protection/weights/"
echo "     autovc.ckpt / 3000000-BL.ckpt / checkpoint_step001000000_ema.pth"
echo "  2) 验证："
echo "     python algorithms/voiceprint_protection/scripts/check_env.py"
echo "     python algorithms/voiceprint_protection/scripts/verify_autovc_mel.py"
echo "=============================================================="
echo ""
