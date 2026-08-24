#!/usr/bin/env bash
# llamacpp 一键引导脚本：Debian/Ubuntu 与 Arch 系通用。
#
# 用法：
#   bash scripts/bootstrap.sh                 # 安装工具本体（含面板依赖）
#   bash scripts/bootstrap.sh --with-driver   # 额外自动安装 NVIDIA 驱动并编译运行时
#
# 环境变量：
#   LLAMACPP_REPO   仓库地址（默认 SSH：git@github.com:xiaoxya/llamacpp.git）
#   LLAMACPP_DIR    安装目录（默认 ~/llamacpp）
set -Eeuo pipefail

REPO_URL="${LLAMACPP_REPO:-git@github.com:xiaoxya/llamacpp.git}"
DEST="${LLAMACPP_DIR:-$HOME/llamacpp}"
WITH_DRIVER=false
[[ "${1:-}" == "--with-driver" ]] && WITH_DRIVER=true

log() { printf '[bootstrap] %s\n' "$*"; }
die() { printf '[bootstrap][错误] %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null || die "请先安装 git（sudo apt install git / sudo pacman -S git）"

# ---------- 发行版识别与基础包 ----------
FAMILY=""
if [[ -r /etc/os-release ]]; then
  ID="$(. /etc/os-release && echo "${ID,,}")"
  ID_LIKE="$(. /etc/os-release && echo "${ID_LIKE,,}")"
  case "$ID $ID_LIKE" in
    *arch*) FAMILY=arch ;;
    *debian*) FAMILY=debian ;;
  esac
fi
[[ -n "$FAMILY" ]] || die "未识别受支持的发行版（支持 Arch 系与 Debian/Ubuntu 系）"

log "检测到发行版家族：$FAMILY；安装基础工具…"
if [[ "$FAMILY" == arch ]]; then
  sudo pacman -S --needed --noconfirm python python-pip git >/dev/null
else
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3 python3-venv python3-pip git ca-certificates >/dev/null
fi

# ---------- 获取/更新源码 ----------
if [[ -d "$DEST/.git" ]]; then
  log "已存在 $DEST，拉取更新…"
  git -C "$DEST" pull --ff-only
else
  log "克隆仓库到 $DEST …"
  git clone "$REPO_URL" "$DEST"
fi
cd "$DEST"

# ---------- Python 环境 ----------
log "创建虚拟环境并安装依赖…"
python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e ".[panel]"

mkdir -p "$HOME/.local/bin"
ln -sf "$(cd "$DEST" && realpath .venv/bin/llamacpp)" "$HOME/.local/bin/llamacpp-py"
hash -r

log "安装完成：$(realpath "$HOME/.local/bin/llamacpp-py") --version"
"$HOME/.local/bin/llamacpp-py" --version

# ---------- 可选：驱动 + 运行时编译 ----------
if [[ "$WITH_DRIVER" == true ]]; then
  log "安装 NVIDIA 驱动（可能需要几分钟）…"
  "$HOME/.local/bin/llamacpp-py" driver --type auto || \
    log "驱动步骤未完成，可稍后手动执行：llamacpp-py driver --type auto"
  log "安装构建依赖并编译 llama.cpp…"
  "$HOME/.local/bin/llamacpp-py" install --driver none
  log "全部完成！下一步："
  echo "  llamacpp-py models          # 选择 GGUF 模型"
  echo "  llamacpp-py config API_KEY '你的密钥'"
  echo "  llamacpp-py start           # 启动服务（需要重启过机器加载驱动）"
else
  cat <<TIP

后续步骤：
  1. NVIDIA 驱动（Ubuntu）：llamacpp-py driver --type auto
     （Debian 需先启用 non-free 源：llamacpp-py driver --type nvidia-driver-550）
  2. 编译运行时：            llamacpp-py install
  3. 选模型并启动：          llamacpp-py models && llamacpp-py config API_KEY '密钥'
  4. 启动服务：              llamacpp-py start
TIP
fi
