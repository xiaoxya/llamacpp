#!/usr/bin/env bash
# llama.cpp CUDA deployment and management for Arch Linux.
# Tuned for Intel i7-11700, 64GB RAM, RTX 4070 SUPER 12GB + RTX 3060 12GB.
# Updated: 2026-08-20

set -Eeuo pipefail
shopt -s inherit_errexit nullglob
IFS=$'\n\t'
umask 077

readonly SCRIPT_VERSION="1.3.1"
readonly APP_NAME="llamacpp"
readonly SERVICE_NAME="${APP_NAME}.service"
readonly CONFLICTING_SERVICE="vllm-dual-3060.service"
readonly LEGACY_SERVICE="llama-cpp-dual-gpu.service"
readonly EXPECTED_GPU_COUNT=2
readonly MIN_GPU_MEMORY_MIB=11000
readonly REPOSITORY_URL="https://github.com/ggml-org/llama.cpp.git"

readonly CONFIG_HOME="${XDG_CONFIG_HOME:-${HOME}/.config}"
readonly DATA_HOME="${XDG_DATA_HOME:-${HOME}/.local/share}"
readonly USER_BIN_DIR="${HOME}/.local/bin"
readonly CONFIG_DIR="${CONFIG_HOME}/${APP_NAME}"
readonly SERVER_CONFIG_FILE="${CONFIG_DIR}/server.env"
readonly BUILD_CONFIG_FILE="${CONFIG_DIR}/build.env"
readonly LEGACY_CONFIG_FILE="${CONFIG_DIR}/config.env"
readonly INSTALL_ROOT="${DATA_HOME}/${APP_NAME}"
readonly SOURCE_DIR="${INSTALL_ROOT}/src"
readonly BUILD_DIR="${INSTALL_ROOT}/build"
readonly SERVER_BIN="${BUILD_DIR}/bin/llama-server"
readonly BENCH_BIN="${BUILD_DIR}/bin/llama-bench"
readonly MANAGER_BIN="${USER_BIN_DIR}/${APP_NAME}"
readonly LAUNCHER_BIN="${USER_BIN_DIR}/${APP_NAME}-start"
readonly SYSTEMD_USER_DIR="${CONFIG_HOME}/systemd/user"
readonly SERVICE_FILE="${SYSTEMD_USER_DIR}/${SERVICE_NAME}"
# shellcheck disable=SC2155  # readlink 失败时应立即终止，无需掩藏返回值
readonly SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"

DRY_RUN=false
VERBOSE=false
NON_INTERACTIVE=false

# Conservative defaults for two unequal 12GB GPUs.
MODEL_DIR="${HOME}/models"
MODEL=""
MODEL_ALIAS=""
MM_PROJ=""
HOST="0.0.0.0"
PORT="8080"
API_KEY=""
CTX_SIZE="32768"
N_PARALLEL="2"
BATCH_SIZE="512"
UBATCH_SIZE="256"
THREADS="8"
THREADS_BATCH="16"
N_GPU_LAYERS="all"
SPLIT_MODE="layer"
TENSOR_SPLIT="1,1"
MAIN_GPU="0"
FIT="true"
FIT_TARGET="1536,1536"
FLASH_ATTN="auto"
CACHE_TYPE_K="q8_0"
CACHE_TYPE_V="q8_0"
CONT_BATCHING="true"
CACHE_PROMPT="true"
JINJA="true"
REASONING="auto"
REASONING_FORMAT="auto"
REASONING_EFFORT="default"
REASONING_BUDGET="-1"
METRICS="true"
WEBUI="true"
LOAD_MODE="auto"
CUDA_VISIBLE_DEVICES="0,1"
CUDA_ARCHITECTURES="86;89"
BUILD_JOBS="auto"
LLAMACPP_REF="master"
EXTRA_ARGS=""

readonly SERVER_CONFIG_KEYS=(
  MODEL_DIR MODEL MODEL_ALIAS MM_PROJ HOST PORT API_KEY CTX_SIZE N_PARALLEL
  BATCH_SIZE UBATCH_SIZE THREADS THREADS_BATCH N_GPU_LAYERS SPLIT_MODE
  TENSOR_SPLIT MAIN_GPU FIT FIT_TARGET FLASH_ATTN CACHE_TYPE_K CACHE_TYPE_V
  CONT_BATCHING CACHE_PROMPT JINJA REASONING REASONING_FORMAT
  REASONING_EFFORT REASONING_BUDGET METRICS WEBUI LOAD_MODE
  CUDA_VISIBLE_DEVICES EXTRA_ARGS
)
readonly BUILD_CONFIG_KEYS=(CUDA_ARCHITECTURES BUILD_JOBS LLAMACPP_REF)

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] [INFO] %s\n' "$(timestamp)" "$*" >&2; }
warn() { printf '[%s] [WARN] %s\n' "$(timestamp)" "$*" >&2; }
error() { printf '[%s] [ERROR] %s\n' "$(timestamp)" "$*" >&2; }
debug() { [[ "${VERBOSE}" == true ]] && printf '[%s] [DEBUG] %s\n' "$(timestamp)" "$*" >&2 || true; }
die() { error "$*"; exit 1; }

on_error() {
  local exit_code=$?
  local line_no=${1:-unknown}
  error "命令在第 ${line_no} 行失败（退出码 ${exit_code}）。"
  exit "${exit_code}"
}
trap 'on_error "$LINENO"' ERR

usage() {
  # shellcheck disable=SC2088  # 帮助文本中的 ~ 是给用户看的字面量
  cat <<'USAGE'
Arch Linux 双 NVIDIA 12GB GPU llama.cpp 一键部署/管理器

用法：
  llamacpp.sh [全局选项] <命令> [命令选项]

命令：
  install       安装构建依赖、CUDA Toolkit，编译 llama.cpp 并生成服务
  update        拉取并重新编译最新 llama.cpp，保留配置
  self-install  仅刷新管理脚本、启动器和 user service
  start         启动 llama-server
  stop          停止服务
  restart       重启服务
  status        查看服务、GPU 和配置摘要
  logs          查看日志；logs -f 持续跟踪
  config        管理 server.env；支持 --show/--edit/--edit-build/KEY VALUE
  models        扫描 ~/models 下的 GGUF 并选择
  doctor        检查系统、驱动、CUDA、编译结果与配置
  self-test     检查 HTTP health 和 OpenAI /v1/models
  bench         使用当前模型执行 llama-bench
  driver        可选安装/修复 nvidia-open 或 nvidia-open-dkms
  uninstall     移除程序和 service，默认保留模型与配置
  menu          交互式管理菜单
  run-server    内部命令：以前台方式运行 llama-server

全局选项：
  --dry-run             显示主要安装/删除操作但不执行
  --non-interactive     禁用交互
  --verbose             显示调试信息
  -h, --help            显示帮助
  --version             显示版本

install/update 选项：
  --ref REF             Git 分支、tag 或 commit；默认 master
  --driver TYPE         none、nvidia-open 或 nvidia-open-dkms
  --start               安装/更新后启动
  --enable-linger       退出登录后继续运行 user service（需要 sudo）

driver 选项：
  --type nvidia-open|nvidia-open-dkms
  --repair              强制重新安装驱动包

uninstall 选项：
  --purge               同时删除源码、构建目录和配置
  --yes                 跳过 purge 确认

常用示例：
  ./llamacpp.sh install
  llamacpp models
  llamacpp config API_KEY 'replace-with-a-long-random-key'
  llamacpp config CTX_SIZE 65536
  llamacpp config BUILD_JOBS auto
  llamacpp start

配置文件：
  ~/.config/llamacpp/server.env  llama-server 启动参数，可直接编辑
  ~/.config/llamacpp/build.env   编译参数，不参与服务启动

默认 API：
  OpenAI:    http://HOST:8080/v1
  Anthropic: http://HOST:8080/v1/messages
  Web UI:    http://HOST:8080/

退出码：0 成功；1 一般错误；2 参数或配置错误。
USAGE
}

run() {
  if [[ "${DRY_RUN}" == true ]]; then
    printf '[DRY-RUN]'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

is_tty() { [[ -t 0 && -t 1 && "${NON_INTERACTIVE}" == false ]]; }

confirm() {
  local prompt=$1 answer
  is_tty || return 1
  read -r -p "${prompt} [y/N] " answer
  [[ "${answer,,}" == y || "${answer,,}" == yes ]]
}

expand_home() {
  local value=$1
  # shellcheck disable=SC2088  # 这里的 ~ 是字面量前缀匹配，不是路径展开
  if [[ "${value}" == '~' ]]; then
    printf '%s\n' "${HOME}"
  elif [[ "${value}" == '~/'* ]]; then
    # 模式必须加引号：未加引号的 ~/ 会被 tilde 展开成 $HOME，导致前缀剥不掉。
    printf '%s/%s\n' "${HOME}" "${value#"~/"}"
  else
    printf '%s\n' "${value}"
  fi
}

key_is_in_list() {
  local needle=$1 key
  shift
  for key in "$@"; do
    [[ "${key}" == "${needle}" ]] && return 0
  done
  return 1
}

is_server_config_key() { key_is_in_list "$1" "${SERVER_CONFIG_KEYS[@]}"; }
is_build_config_key() { key_is_in_list "$1" "${BUILD_CONFIG_KEYS[@]}"; }
is_config_key() { is_server_config_key "$1" || is_build_config_key "$1"; }

load_config_file() {
  local file=$1 kind=$2 line key value allowed=false
  [[ -f "${file}" ]] || return 0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line=${line%$'\r'}
    [[ -z "${line}" || "${line}" == \#* || "${line}" != *=* ]] && continue
    key=${line%%=*}
    value=${line#*=}
    if [[ "${key}" == LLAMA_CPP_REF && "${kind}" != server ]]; then
      warn "配置键 LLAMA_CPP_REF 已更名为 LLAMACPP_REF；本次兼容读取旧值。"
      key=LLAMACPP_REF
    fi
    case "${kind}" in
      server) is_server_config_key "${key}" && allowed=true || allowed=false ;;
      build) is_build_config_key "${key}" && allowed=true || allowed=false ;;
      legacy) is_config_key "${key}" && allowed=true || allowed=false ;;
      *) die "内部错误：未知配置类型 ${kind}" ;;
    esac
    if [[ "${allowed}" == true ]]; then
      printf -v "${key}" '%s' "${value}"
    else
      warn "忽略 ${file} 中的未知配置项：${key}"
    fi
  done < "${file}"
}

load_config() {
  if [[ -f "${SERVER_CONFIG_FILE}" || -f "${BUILD_CONFIG_FILE}" ]]; then
    load_config_file "${SERVER_CONFIG_FILE}" server
    load_config_file "${BUILD_CONFIG_FILE}" build
  elif [[ -f "${LEGACY_CONFIG_FILE}" ]]; then
    load_config_file "${LEGACY_CONFIG_FILE}" legacy
  fi
  MODEL_DIR=$(expand_home "${MODEL_DIR}")
  MODEL=$(expand_home "${MODEL}")
  MM_PROJ=$(expand_home "${MM_PROJ}")
}

validate_bool() { [[ "$1" == true || "$1" == false ]]; }

validate_config() {
  local failures=0 variable value
  [[ -n "${MODEL_DIR}" && "${MODEL_DIR}" == /* ]] || { error "MODEL_DIR 必须是绝对路径。"; failures=$((failures + 1)); }
  [[ -z "${MODEL}" || "${MODEL}" == /* ]] || { error "MODEL 必须留空或为绝对 GGUF 路径。"; failures=$((failures + 1)); }
  [[ -z "${MM_PROJ}" || "${MM_PROJ}" == /* ]] || { error "MM_PROJ 必须留空或为绝对路径。"; failures=$((failures + 1)); }
  [[ -n "${HOST}" && "${HOST}" != *[[:space:]]* ]] || { error "HOST 无效。"; failures=$((failures + 1)); }
  [[ "${PORT}" =~ ^[0-9]+$ ]] && (( PORT >= 1 && PORT <= 65535 )) || { error "PORT 必须在 1..65535。"; failures=$((failures + 1)); }
  for variable in CTX_SIZE N_PARALLEL BATCH_SIZE UBATCH_SIZE THREADS THREADS_BATCH; do
    value=${!variable}
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { error "${variable} 必须为正整数。"; failures=$((failures + 1)); }
  done
  [[ "${BUILD_JOBS}" == auto || "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]] || { error "BUILD_JOBS 必须为 auto 或正整数。"; failures=$((failures + 1)); }
  [[ "${MAIN_GPU}" =~ ^[0-9]+$ ]] || { error "MAIN_GPU 必须为非负整数。"; failures=$((failures + 1)); }
  [[ "${N_GPU_LAYERS}" == all || "${N_GPU_LAYERS}" == auto || "${N_GPU_LAYERS}" =~ ^[0-9]+$ ]] || { error "N_GPU_LAYERS 必须为 all、auto 或非负整数。"; failures=$((failures + 1)); }
  [[ "${SPLIT_MODE}" =~ ^(none|layer|row|tensor)$ ]] || { error "SPLIT_MODE 无效。"; failures=$((failures + 1)); }
  [[ "${TENSOR_SPLIT}" =~ ^[0-9]+([.][0-9]+)?(,[0-9]+([.][0-9]+)?)+$ ]] || { error "TENSOR_SPLIT 格式应类似 1,1。"; failures=$((failures + 1)); }
  [[ "${FIT_TARGET}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { error "FIT_TARGET 格式应类似 1536,1536。"; failures=$((failures + 1)); }
  [[ "${FLASH_ATTN}" =~ ^(on|off|auto)$ ]] || { error "FLASH_ATTN 无效。"; failures=$((failures + 1)); }
  [[ "${CACHE_TYPE_K}" =~ ^(f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1)$ ]] || { error "CACHE_TYPE_K 无效。"; failures=$((failures + 1)); }
  [[ "${CACHE_TYPE_V}" =~ ^(f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1)$ ]] || { error "CACHE_TYPE_V 无效。"; failures=$((failures + 1)); }
  [[ "${REASONING}" =~ ^(on|off|auto)$ ]] || { error "REASONING 无效。"; failures=$((failures + 1)); }
  [[ "${REASONING_FORMAT}" =~ ^(auto|none|deepseek|deepseek-legacy)$ ]] || { error "REASONING_FORMAT 无效。"; failures=$((failures + 1)); }
  [[ "${REASONING_EFFORT}" =~ ^(default|minimal|low|medium|high|xhigh|max)$ ]] || { error "REASONING_EFFORT 无效。"; failures=$((failures + 1)); }
  [[ "${REASONING_BUDGET}" =~ ^-1$|^[0-9]+$ ]] || { error "REASONING_BUDGET 必须为 -1 或非负整数。"; failures=$((failures + 1)); }
  [[ "${LOAD_MODE}" =~ ^(auto|none|mmap|mlock|mmap[+]mlock|dio)$ ]] || { error "LOAD_MODE 无效。"; failures=$((failures + 1)); }
  [[ "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { error "CUDA_VISIBLE_DEVICES 无效。"; failures=$((failures + 1)); }
  [[ "${CUDA_ARCHITECTURES}" =~ ^[0-9]+([;][0-9]+)*$ ]] || { error "CUDA_ARCHITECTURES 格式应类似 86;89。"; failures=$((failures + 1)); }
  [[ -n "${LLAMACPP_REF}" && "${LLAMACPP_REF}" != *[[:space:]]* ]] || { error "LLAMACPP_REF 无效。"; failures=$((failures + 1)); }
  for variable in FIT CONT_BATCHING CACHE_PROMPT JINJA METRICS WEBUI; do
    value=${!variable}
    validate_bool "${value}" || { error "${variable} 只能是 true 或 false。"; failures=$((failures + 1)); }
  done
  (( failures == 0 )) || return 2
}

write_config_file() {
  local file=$1 description=$2 tmp key value
  shift 2
  validate_config
  for key in "$@"; do
    value=${!key}
    [[ "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] || die "配置 ${key} 不能包含换行。"
  done
  [[ "${DRY_RUN}" == true ]] && { log "将写入${description}：${file}"; return 0; }
  mkdir -p -- "${CONFIG_DIR}"
  tmp=$(mktemp "${file}.tmp.XXXXXX")
  chmod 600 "${tmp}"
  {
    printf '# %s %s; parsed as data and never sourced.\n' "${APP_NAME}" "${description}"
    for key in "$@"; do
      value=${!key}
      printf '%s=%s\n' "${key}" "${value}"
    done
  } > "${tmp}"
  mv -f -- "${tmp}" "${file}"
  chmod 600 "${file}"
}

write_server_config() {
  local tmp key value
  validate_config
  for key in "${SERVER_CONFIG_KEYS[@]}"; do
    value=${!key}
    [[ "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] || die "配置 ${key} 不能包含换行。"
  done
  [[ "${DRY_RUN}" == true ]] && { log "将写入 Server 启动配置：${SERVER_CONFIG_FILE}"; return 0; }
  mkdir -p -- "${CONFIG_DIR}"
  tmp=$(mktemp "${SERVER_CONFIG_FILE}.tmp.XXXXXX")
  chmod 600 "${tmp}"
  {
    printf '# llama-server startup configuration. Edit values, then run:\n'
    printf '#   systemctl --user restart %s\n' "${SERVICE_NAME}"
    printf '# Parsed as data by %s; never sourced as shell code.\n' "${APP_NAME}"
    printf '\n# Model\n'
    printf 'MODEL_DIR=%s\nMODEL=%s\nMODEL_ALIAS=%s\nMM_PROJ=%s\n' "${MODEL_DIR}" "${MODEL}" "${MODEL_ALIAS}" "${MM_PROJ}"
    printf '\n# OpenAI-compatible HTTP server\n'
    printf 'HOST=%s\nPORT=%s\nAPI_KEY=%s\n' "${HOST}" "${PORT}" "${API_KEY}"
    printf '\n# Context, concurrency and CPU threads\n'
    printf 'CTX_SIZE=%s\nN_PARALLEL=%s\nBATCH_SIZE=%s\nUBATCH_SIZE=%s\n' "${CTX_SIZE}" "${N_PARALLEL}" "${BATCH_SIZE}" "${UBATCH_SIZE}"
    printf 'THREADS=%s\nTHREADS_BATCH=%s\n' "${THREADS}" "${THREADS_BATCH}"
    printf '\n# Dual-GPU offload and memory\n'
    printf 'N_GPU_LAYERS=%s\nSPLIT_MODE=%s\nTENSOR_SPLIT=%s\nMAIN_GPU=%s\n' "${N_GPU_LAYERS}" "${SPLIT_MODE}" "${TENSOR_SPLIT}" "${MAIN_GPU}"
    printf 'FIT=%s\nFIT_TARGET=%s\nFLASH_ATTN=%s\n' "${FIT}" "${FIT_TARGET}" "${FLASH_ATTN}"
    printf 'CACHE_TYPE_K=%s\nCACHE_TYPE_V=%s\nCUDA_VISIBLE_DEVICES=%s\n' "${CACHE_TYPE_K}" "${CACHE_TYPE_V}" "${CUDA_VISIBLE_DEVICES}"
    printf '\n# Chat template, reasoning and server features\n'
    printf 'CONT_BATCHING=%s\nCACHE_PROMPT=%s\nJINJA=%s\n' "${CONT_BATCHING}" "${CACHE_PROMPT}" "${JINJA}"
    printf 'REASONING=%s\nREASONING_FORMAT=%s\nREASONING_EFFORT=%s\nREASONING_BUDGET=%s\n' "${REASONING}" "${REASONING_FORMAT}" "${REASONING_EFFORT}" "${REASONING_BUDGET}"
    printf 'METRICS=%s\nWEBUI=%s\nLOAD_MODE=%s\n' "${METRICS}" "${WEBUI}" "${LOAD_MODE}"
    printf '\n# Additional native llama-server options. Avoid duplicating managed options above.\n'
    printf 'EXTRA_ARGS=%s\n' "${EXTRA_ARGS}"
  } > "${tmp}"
  mv -f -- "${tmp}" "${SERVER_CONFIG_FILE}"
  chmod 600 "${SERVER_CONFIG_FILE}"
}

write_build_config() {
  write_config_file "${BUILD_CONFIG_FILE}" 'build configuration' "${BUILD_CONFIG_KEYS[@]}"
}

write_config() {
  write_server_config
  write_build_config
}

ensure_config() {
  if [[ ! -f "${SERVER_CONFIG_FILE}" && ! -f "${BUILD_CONFIG_FILE}" && -f "${LEGACY_CONFIG_FILE}" ]]; then
    load_config
    validate_config
    write_config
    log "已将旧配置迁移为 ${SERVER_CONFIG_FILE} 和 ${BUILD_CONFIG_FILE}；原文件保留。"
    return 0
  fi
  [[ -f "${SERVER_CONFIG_FILE}" ]] || write_server_config
  [[ -f "${BUILD_CONFIG_FILE}" ]] || write_build_config
  load_config
  validate_config
}

redact() { [[ -z "$1" ]] && printf '(未设置)' || printf '********'; }

show_config() {
  ensure_config
  printf '=== Server 启动配置 ===\n文件: %s\n' "${SERVER_CONFIG_FILE}"
  printf 'MODEL_DIR=%s\nMODEL=%s\n' "${MODEL_DIR}" "${MODEL:-（未选择）}"
  printf 'MODEL_ALIAS=%s\nMM_PROJ=%s\n' "${MODEL_ALIAS:-（自动）}" "${MM_PROJ:-（未设置）}"
  printf 'HOST=%s\nPORT=%s\nAPI_KEY=%s\n' "${HOST}" "${PORT}" "$(redact "${API_KEY}")"
  printf 'CTX_SIZE=%s\nN_PARALLEL=%s\nBATCH_SIZE=%s\nUBATCH_SIZE=%s\n' "${CTX_SIZE}" "${N_PARALLEL}" "${BATCH_SIZE}" "${UBATCH_SIZE}"
  printf 'THREADS=%s\nTHREADS_BATCH=%s\nN_GPU_LAYERS=%s\n' "${THREADS}" "${THREADS_BATCH}" "${N_GPU_LAYERS}"
  printf 'SPLIT_MODE=%s\nTENSOR_SPLIT=%s\nMAIN_GPU=%s\n' "${SPLIT_MODE}" "${TENSOR_SPLIT}" "${MAIN_GPU}"
  printf 'FIT=%s\nFIT_TARGET=%s\nFLASH_ATTN=%s\n' "${FIT}" "${FIT_TARGET}" "${FLASH_ATTN}"
  printf 'CACHE_TYPE_K=%s\nCACHE_TYPE_V=%s\nCONT_BATCHING=%s\nCACHE_PROMPT=%s\n' "${CACHE_TYPE_K}" "${CACHE_TYPE_V}" "${CONT_BATCHING}" "${CACHE_PROMPT}"
  printf 'JINJA=%s\nREASONING=%s\nREASONING_FORMAT=%s\n' "${JINJA}" "${REASONING}" "${REASONING_FORMAT}"
  printf 'REASONING_EFFORT=%s\nREASONING_BUDGET=%s\nMETRICS=%s\nWEBUI=%s\n' "${REASONING_EFFORT}" "${REASONING_BUDGET}" "${METRICS}" "${WEBUI}"
  printf 'LOAD_MODE=%s\nCUDA_VISIBLE_DEVICES=%s\nEXTRA_ARGS=%s\n' "${LOAD_MODE}" "${CUDA_VISIBLE_DEVICES}" "${EXTRA_ARGS}"
  printf '\n=== 编译配置（不参与服务启动）===\n文件: %s\n' "${BUILD_CONFIG_FILE}"
  printf 'CUDA_ARCHITECTURES=%s\nBUILD_JOBS=%s（当前解析为 %s）\nLLAMACPP_REF=%s\n' "${CUDA_ARCHITECTURES}" "${BUILD_JOBS}" "$(resolved_build_jobs)" "${LLAMACPP_REF}"
}

set_config_value() {
  local key=${1:-} value=${2-}
  [[ -n "${key}" ]] || die "缺少配置键。"
  is_config_key "${key}" || die "不支持的配置键：${key}"
  [[ "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] || die "配置值不能包含换行。"
  ensure_config
  printf -v "${key}" '%s' "${value}"
  if [[ "${key}" == MODEL || "${key}" == MODEL_DIR || "${key}" == MM_PROJ ]]; then
    printf -v "${key}" '%s' "$(expand_home "${value}")"
  fi
  if is_server_config_key "${key}"; then
    write_server_config
  else
    write_build_config
  fi
  log "已更新 ${key}。"
}

prompt_value() {
  local variable=$1 label=$2 current=${!1} answer
  read -r -p "${label} [${current}]: " answer
  if [[ -n "${answer}" ]]; then
    printf -v "${variable}" '%s' "${answer}"
  fi
  return 0
}

prompt_secret() {
  local variable=$1 label=$2 answer state="未设置"
  [[ -n "${!variable}" ]] && state="已设置；直接回车保留"
  read -r -s -p "${label} [${state}]: " answer
  printf '\n'
  if [[ -n "${answer}" ]]; then
    printf -v "${variable}" '%s' "${answer}"
  fi
  return 0
}

config_wizard() {
  is_tty || die "交互配置需要终端；自动化请使用 config KEY VALUE。"
  ensure_config
  printf '\n直接回车保留当前值。布尔值使用 true/false。\n\n'
  prompt_value MODEL_DIR "模型目录" "${MODEL_DIR}"
  MODEL_DIR=$(expand_home "${MODEL_DIR}")
  if confirm "现在扫描并选择 GGUF 模型吗？"; then select_model || true; fi
  prompt_value MODEL_ALIAS "API 模型别名（空=文件名）" "${MODEL_ALIAS}"
  prompt_value MM_PROJ "多模态 mmproj 路径（空=无）" "${MM_PROJ}"
  MM_PROJ=$(expand_home "${MM_PROJ}")
  prompt_value HOST "监听地址" "${HOST}"
  prompt_value PORT "端口" "${PORT}"
  prompt_secret API_KEY "API Key（清空可执行 config API_KEY ''）"
  prompt_value CTX_SIZE "上下文长度" "${CTX_SIZE}"
  prompt_value N_PARALLEL "并发 slots" "${N_PARALLEL}"
  prompt_value BATCH_SIZE "逻辑 batch size" "${BATCH_SIZE}"
  prompt_value UBATCH_SIZE "物理 ubatch size" "${UBATCH_SIZE}"
  prompt_value THREADS "生成/解码 CPU 线程数" "${THREADS}"
  prompt_value THREADS_BATCH "提示词处理 CPU 线程数" "${THREADS_BATCH}"
  prompt_value SPLIT_MODE "双卡 split mode: layer/row/tensor/none" "${SPLIT_MODE}"
  prompt_value TENSOR_SPLIT "双卡比例" "${TENSOR_SPLIT}"
  prompt_value FIT_TARGET "每卡预留显存 MiB" "${FIT_TARGET}"
  prompt_value FLASH_ATTN "Flash Attention: auto/on/off" "${FLASH_ATTN}"
  prompt_value CACHE_TYPE_K "K cache 类型" "${CACHE_TYPE_K}"
  prompt_value CACHE_TYPE_V "V cache 类型" "${CACHE_TYPE_V}"
  prompt_value REASONING "Reasoning: auto/on/off" "${REASONING}"
  prompt_value REASONING_EFFORT "Reasoning effort" "${REASONING_EFFORT}"
  prompt_value REASONING_BUDGET "Reasoning budget，-1=不限" "${REASONING_BUDGET}"
  prompt_value EXTRA_ARGS "额外 llama-server 参数" "${EXTRA_ARGS}"
  write_server_config
  log "配置完成。"
}

edit_config_file() {
  local file=$1 label=$2 editor backup
  ensure_config
  editor=${EDITOR:-}
  if [[ -z "${editor}" ]]; then
    if command -v nano >/dev/null 2>&1; then editor="nano"; else editor="vi"; fi
  fi
  backup=$(mktemp "${CONFIG_DIR}/${label}.backup.XXXXXX")
  cp -p -- "${file}" "${backup}"
  "${editor}" "${file}"
  if ! (load_config; validate_config); then
    mv -f -- "${backup}" "${file}"
    die "配置无效，已恢复编辑前版本。"
  fi
  rm -f -- "${backup}"
  chmod 600 "${file}"
  log "${label} 配置有效并已保存。"
}

edit_config() { edit_config_file "${SERVER_CONFIG_FILE}" server; }
edit_build_config() { edit_config_file "${BUILD_CONFIG_FILE}" build; }

assert_arch_linux() {
  [[ "$(uname -s)" == Linux ]] || die "此脚本仅支持 Linux。"
  [[ -r /etc/arch-release ]] || die "此增强版仅支持 Arch Linux 及其直接衍生发行版。"
  [[ "$(uname -m)" == x86_64 ]] || die "需要 x86_64。"
  command -v pacman >/dev/null 2>&1 || die "未找到 pacman。"
}

gpu_count() {
  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | awk 'NF { count++ } END { print count+0 }'
}

detect_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null |
    awk -v expected="${EXPECTED_GPU_COUNT}" -v minimum="${MIN_GPU_MEMORY_MIB}" '
      NF { gsub(/[[:space:]]/, "", $0); count++; if (($0 + 0) < minimum) low++ }
      END { exit !(count == expected && low == 0) }
    '
}

gpu_models_are_mixed() {
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null |
    awk 'NF && !seen[$0]++ { unique++ } END { exit !(unique > 1) }'
}

gpu_low_memory_count() {
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null |
    awk -v minimum="${MIN_GPU_MEMORY_MIB}" 'NF { gsub(/[[:space:]]/, "", $0); if (($0 + 0) < minimum) low++ } END { print low+0 }'
}

print_gpu_table() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu,driver_version --format=csv,noheader
  else
    printf 'nvidia-smi 不可用\n'
  fi
}

check_gpu_or_die() {
  if ! detect_gpus; then
    print_gpu_table >&2
    die "需要 2 张可用 NVIDIA GPU，且每张显存至少约 $((MIN_GPU_MEMORY_MIB / 1024)) GiB。"
  fi
  gpu_models_are_mixed && warn "检测到不同型号 GPU；layer split 可用，但速度受较慢的 RTX 3060 限制。" || true
}

pacman_install() {
  local args=(sudo pacman -S --needed)
  [[ "${NON_INTERACTIVE}" == true ]] && args+=(--noconfirm)
  args+=(-- "$@")
  run "${args[@]}"
}

install_build_dependencies() {
  assert_arch_linux
  log "安装 CUDA Toolkit 和 llama.cpp 构建依赖……"
  pacman_install base-devel cmake ninja git ccache cuda openssl curl python
}

installed_driver_package() {
  local package
  for package in nvidia-open-dkms nvidia-open; do
    pacman -Q "${package}" >/dev/null 2>&1 && { printf '%s\n' "${package}"; return 0; }
  done
  printf '%s\n' none
}

kernel_header_packages() {
  local kernel
  for kernel in linux linux-lts linux-zen linux-hardened; do
    pacman -Q "${kernel}" >/dev/null 2>&1 && printf '%s-headers\n' "${kernel}"
  done
}

manage_driver() {
  local type="nvidia-open-dkms" repair=false
  while (( $# > 0 )); do
    case "$1" in
      --type) shift; type=${1:?--type 需要参数} ;;
      --repair) repair=true ;;
      *) die "driver 未知选项：$1" ;;
    esac
    shift
  done
  [[ "${type}" == nvidia-open || "${type}" == nvidia-open-dkms ]] || die "驱动类型无效：${type}"
  assert_arch_linux
  local -a packages=(nvidia-utils "${type}") headers=()
  if [[ "${type}" == nvidia-open-dkms ]]; then
    mapfile -t headers < <(kernel_header_packages)
    (( ${#headers[@]} > 0 )) || warn "未识别标准 Arch kernel；请自行安装对应 headers。"
    packages+=(dkms "${headers[@]}")
  fi
  log "当前驱动包：$(installed_driver_package)；目标：${type}"
  if [[ "${repair}" == true ]]; then
    local args=(sudo pacman -S)
    [[ "${NON_INTERACTIVE}" == true ]] && args+=(--noconfirm)
    args+=(-- "${packages[@]}")
    run "${args[@]}"
  else
    pacman_install "${packages[@]}"
  fi
  command -v mkinitcpio >/dev/null 2>&1 && run sudo mkinitcpio -P || true
  warn "驱动变更后必须重启；脚本不会自动重启。"
}

source_tree_is_clean() {
  git -C "${SOURCE_DIR}" diff --quiet --ignore-submodules -- &&
    git -C "${SOURCE_DIR}" diff --cached --quiet --ignore-submodules --
}

prepare_source() {
  local ref=$1
  run mkdir -p -- "${INSTALL_ROOT}"
  if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    log "克隆 llama.cpp 官方仓库……"
    run git clone --filter=blob:none "${REPOSITORY_URL}" "${SOURCE_DIR}"
  else
    source_tree_is_clean || die "源码目录有本地修改，拒绝覆盖：${SOURCE_DIR}"
    log "获取 llama.cpp 更新……"
    run git -C "${SOURCE_DIR}" fetch --tags --prune origin
  fi
  if [[ "${ref}" == master ]]; then
    run git -C "${SOURCE_DIR}" checkout master
    run git -C "${SOURCE_DIR}" pull --ff-only origin master
  else
    run git -C "${SOURCE_DIR}" checkout --detach "${ref}"
  fi
}

available_cpu_threads() {
  local count
  if command -v nproc >/dev/null 2>&1; then
    count=$(nproc)
  else
    count=$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')
  fi
  [[ "${count}" =~ ^[1-9][0-9]*$ ]] || count=1
  printf '%s\n' "${count}"
}

resolved_build_jobs() {
  if [[ "${BUILD_JOBS}" == auto ]]; then
    available_cpu_threads
  else
    printf '%s\n' "${BUILD_JOBS}"
  fi
}

build_llama_cpp() {
  local nvcc_path=/opt/cuda/bin/nvcc build_jobs
  [[ "${DRY_RUN}" == true || -x "${nvcc_path}" ]] || die "未找到 ${nvcc_path}；请确认 Arch cuda 包安装成功。"
  export PATH="/opt/cuda/bin:${PATH}"
  export CUDACXX="${nvcc_path}"
  build_jobs=$(resolved_build_jobs)
  log "配置 CUDA 构建：sm_${CUDA_ARCHITECTURES//;/, sm_}，并行任务 ${build_jobs}（BUILD_JOBS=${BUILD_JOBS}）……"
  run cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLAMA_CURL=ON \
    -DLLAMA_OPENSSL=ON
  log "编译 llama-server、llama-cli 和 llama-bench……"
  run cmake --build "${BUILD_DIR}" --target llama-server llama-cli llama-bench --parallel "${build_jobs}"
  [[ "${DRY_RUN}" == true || -x "${SERVER_BIN}" ]] || die "编译结束但未找到 llama-server。"
}

install_manager_copy() {
  run mkdir -p -- "${USER_BIN_DIR}"
  if [[ "${SCRIPT_PATH}" != "$(readlink -m -- "${MANAGER_BIN}")" ]]; then
    run install -m 0755 -- "${SCRIPT_PATH}" "${MANAGER_BIN}"
  fi
}

write_launcher() {
  local tmp
  [[ "${DRY_RUN}" == true ]] && { log "将生成启动脚本：${LAUNCHER_BIN}"; return 0; }
  mkdir -p -- "${USER_BIN_DIR}"
  tmp=$(mktemp "${USER_BIN_DIR}/${APP_NAME}-start.tmp.XXXXXX")
  cat > "${tmp}" <<'LAUNCHER'
#!/usr/bin/env bash
set -Eeuo pipefail
exec "${HOME}/.local/bin/llamacpp" run-server "$@"
LAUNCHER
  chmod 0755 "${tmp}"
  mv -f -- "${tmp}" "${LAUNCHER_BIN}"
}

write_service() {
  local tmp
  [[ "${DRY_RUN}" == true ]] && { log "将生成 user service：${SERVICE_FILE}"; return 0; }
  mkdir -p -- "${SYSTEMD_USER_DIR}"
  tmp=$(mktemp "${SYSTEMD_USER_DIR}/${SERVICE_NAME}.tmp.XXXXXX")
  cat > "${tmp}" <<UNIT
[Unit]
Description=llama.cpp OpenAI/Anthropic API server (dual NVIDIA GPU)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=%h/.local/bin/${APP_NAME}-start
Restart=on-failure
RestartSec=10
TimeoutStopSec=45
KillSignal=SIGTERM
Environment=PATH=/opt/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/bin
Environment=LD_LIBRARY_PATH=/opt/cuda/lib64

[Install]
WantedBy=default.target
UNIT
  chmod 0644 "${tmp}"
  mv -f -- "${tmp}" "${SERVICE_FILE}"
  run systemctl --user daemon-reload
  run systemctl --user enable "${SERVICE_NAME}"
}

scan_models() {
  local root=${1:-${MODEL_DIR}} file name shard
  [[ -d "${root}" ]] || return 0
  find "${root}" -type f -iname '*.gguf' ! -iname 'mmproj*' -print0 2>/dev/null |
    while IFS= read -r -d '' file; do
      name=${file##*/}
      if [[ "${name}" =~ -([0-9]{5})-of-[0-9]{5}[.]gguf$ ]]; then
        shard=${BASH_REMATCH[1]}
        [[ "${shard}" == 00001 ]] || continue
      fi
      printf '%s\0' "${file}"
    done
}

select_model() {
  local -a models=()
  local item selection size index=1
  mkdir -p -- "${MODEL_DIR}"
  mapfile -d '' -t models < <(scan_models "${MODEL_DIR}" | sort -zu)
  if (( ${#models[@]} == 0 )); then
    warn "在 ${MODEL_DIR} 下没有找到可用 GGUF。"
    return 1
  fi
  if [[ "${NON_INTERACTIVE}" == true ]]; then
    if (( ${#models[@]} == 1 )); then
      MODEL=${models[0]}
      write_server_config
      log "自动选择唯一模型：${MODEL}"
      return 0
    fi
    warn "发现多个模型；非交互模式不会擅自选择。"
    return 1
  fi
  printf '\n检测到以下 GGUF 模型：\n'
  for item in "${models[@]}"; do
    size=$(du -h -- "${item}" 2>/dev/null | awk '{print $1}' || printf '?')
    printf '  %d) [%s] %s\n' "${index}" "${size}" "${item}"
    index=$((index + 1))
  done
  while true; do
    read -r -p "请选择模型 [1-${#models[@]}，q 取消]: " selection
    [[ "${selection}" == q ]] && return 1
    if [[ "${selection}" =~ ^[0-9]+$ ]] && (( selection >= 1 && selection <= ${#models[@]} )); then
      MODEL=${models[selection - 1]}
      [[ -n "${MODEL_ALIAS}" ]] || MODEL_ALIAS=$(basename -- "${MODEL}" .gguf)
      write_server_config
      log "已选择：${MODEL}"
      return 0
    fi
    warn "选择无效。"
  done
}

parse_extra_args() {
  local input=$1
  EXTRA_ARGV=()
  [[ -z "${input}" ]] && return 0
  mapfile -d '' -t EXTRA_ARGV < <(python - "${input}" <<'PY'
import shlex
import sys

try:
    values = shlex.split(sys.argv[1])
except ValueError as exc:
    print(f"EXTRA_ARGS parse error: {exc}", file=sys.stderr)
    raise SystemExit(2)
for value in values:
    if "\0" in value:
        raise SystemExit("NUL is not allowed")
    sys.stdout.buffer.write(value.encode() + b"\0")
PY
  )
}

append_bool_flag() {
  local -n target=$1
  local enabled=$2 on_flag=$3 off_flag=$4
  if [[ "${enabled}" == true ]]; then target+=("${on_flag}"); else target+=("${off_flag}"); fi
}

build_server_command() {
  local -n result=$1
  parse_extra_args "${EXTRA_ARGS}"
  result=(
    "${SERVER_BIN}"
    --model "${MODEL}"
    --host "${HOST}"
    --port "${PORT}"
    --ctx-size "${CTX_SIZE}"
    --parallel "${N_PARALLEL}"
    --batch-size "${BATCH_SIZE}"
    --ubatch-size "${UBATCH_SIZE}"
    --threads "${THREADS}"
    --threads-batch "${THREADS_BATCH}"
    --n-gpu-layers "${N_GPU_LAYERS}"
    --split-mode "${SPLIT_MODE}"
    --tensor-split "${TENSOR_SPLIT}"
    --main-gpu "${MAIN_GPU}"
    --flash-attn "${FLASH_ATTN}"
    --cache-type-k "${CACHE_TYPE_K}"
    --cache-type-v "${CACHE_TYPE_V}"
    --load-mode "${LOAD_MODE}"
    --reasoning "${REASONING}"
    --log-timestamps
  )
  [[ -n "${MODEL_ALIAS}" ]] && result+=(--alias "${MODEL_ALIAS}")
  [[ -n "${MM_PROJ}" ]] && result+=(--mmproj "${MM_PROJ}")
  [[ -n "${API_KEY}" ]] && result+=(--api-key "${API_KEY}")
  [[ "${FIT}" == true ]] && result+=(--fit on --fit-target "${FIT_TARGET}") || result+=(--fit off)
  append_bool_flag result "${CONT_BATCHING}" --cont-batching --no-cont-batching
  append_bool_flag result "${CACHE_PROMPT}" --cache-prompt --no-cache-prompt
  append_bool_flag result "${JINJA}" --jinja --no-jinja
  append_bool_flag result "${WEBUI}" --webui --no-webui
  [[ "${METRICS}" == true ]] && result+=(--metrics)
  [[ "${REASONING_FORMAT}" != auto ]] && result+=(--reasoning-format "${REASONING_FORMAT}")
  [[ "${REASONING_EFFORT}" != default ]] && result+=(--reasoning-effort "${REASONING_EFFORT}")
  [[ "${REASONING_BUDGET}" != -1 ]] && result+=(--reasoning-budget "${REASONING_BUDGET}")
  result+=("${EXTRA_ARGV[@]}")
}

check_conflicting_service() {
  if systemctl --user is-active --quiet "${CONFLICTING_SERVICE}" 2>/dev/null; then
    die "${CONFLICTING_SERVICE} 正在占用 GPU；请先运行 vllm-dual-3060 stop。"
  fi
  if systemctl --user is-active --quiet "${LEGACY_SERVICE}" 2>/dev/null; then
    die "旧服务 ${LEGACY_SERVICE} 正在占用 GPU；请先执行 systemctl --user disable --now ${LEGACY_SERVICE}。"
  fi
}

run_server() {
  local -a command=()
  ensure_config
  [[ -x "${SERVER_BIN}" ]] || die "llama-server 未编译；请先运行 install。"
  [[ -n "${MODEL}" ]] || die "尚未选择模型；请运行 models。"
  [[ -f "${MODEL}" ]] || die "模型不存在：${MODEL}"
  [[ "${MODEL,,}" == *.gguf ]] || die "llama.cpp 模型必须为 GGUF 文件。"
  [[ -z "${MM_PROJ}" || -f "${MM_PROJ}" ]] || die "MM_PROJ 不存在：${MM_PROJ}"
  check_gpu_or_die
  check_conflicting_service
  [[ -n "${API_KEY}" || "${HOST}" == 127.0.0.1 || "${HOST}" == localhost ]] || warn "服务监听 ${HOST} 但未配置 API_KEY；局域网内任何设备都可访问。"
  build_server_command command
  export CUDA_VISIBLE_DEVICES
  export PATH="/opt/cuda/bin:${PATH}"
  export LD_LIBRARY_PATH="/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
  log "启动 llama-server：model=${MODEL}, ctx=${CTX_SIZE}, split=${SPLIT_MODE}/${TENSOR_SPLIT}"
  printf '[COMMAND]'
  local arg
  for arg in "${command[@]}"; do
    if [[ -n "${API_KEY}" && "${arg}" == "${API_KEY}" ]]; then printf ' %q' '********'; else printf ' %q' "${arg}"; fi
  done
  printf '\n'
  exec "${command[@]}"
}

service_available() { command -v systemctl >/dev/null 2>&1 && [[ -f "${SERVICE_FILE}" ]]; }
service_is_active() { systemctl --user is-active --quiet "${SERVICE_NAME}" 2>/dev/null; }

health_url() {
  local health_host=${HOST}
  case "${health_host}" in
    0.0.0.0|'') health_host=127.0.0.1 ;;
    ::|'[::]') health_host='[::1]' ;;
    *:*) health_host="[${health_host}]" ;;
  esac
  printf 'http://%s:%s/health' "${health_host}" "${PORT}"
}

wait_for_health() {
  local timeout_seconds=${1:-180} elapsed=0 code
  while (( elapsed < timeout_seconds )); do
    service_is_active || return 1
    code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 "$(health_url)" 2>/dev/null || printf 000)
    if [[ "${code}" == 200 ]]; then return 0; fi
    (( elapsed % 15 == 0 )) && log "模型仍在加载（health HTTP ${code}，已等待 ${elapsed}s）……"
    sleep 3
    elapsed=$((elapsed + 3))
  done
  return 1
}

start_service() {
  ensure_config
  if [[ -z "${MODEL}" ]] && is_tty; then select_model || true; load_config; fi
  [[ -n "${MODEL}" ]] || die "未选择模型；请先运行 models。"
  check_gpu_or_die
  check_conflicting_service
  service_available || die "user service 未安装；请先运行 install。"
  run systemctl --user start "${SERVICE_NAME}"
  if [[ "${DRY_RUN}" == false ]]; then
    sleep 2
    service_is_active || { systemctl --user status "${SERVICE_NAME}" --no-pager || true; die "服务启动失败；运行 logs 查看。"; }
    if wait_for_health 180; then
      log "服务就绪：OpenAI http://127.0.0.1:${PORT}/v1；Web UI http://127.0.0.1:${PORT}/"
    else
      warn "服务仍在运行但 180 秒内未就绪；请运行 logs -f。"
    fi
  fi
}

stop_service() {
  service_available || { warn "service 未安装。"; return 0; }
  if service_is_active; then run systemctl --user stop "${SERVICE_NAME}"; log "服务已停止。"; else log "服务已经停止。"; fi
}

restart_service() {
  stop_service
  start_service
}

show_status() {
  ensure_config
  printf '\n=== GPU ===\n'
  print_gpu_table || true
  printf '\n=== 服务 ===\n'
  if service_available; then systemctl --user status "${SERVICE_NAME}" --no-pager || true; else printf 'service 未安装：%s\n' "${SERVICE_FILE}"; fi
  printf '\n=== 构建 ===\n'
  [[ -x "${SERVER_BIN}" ]] && "${SERVER_BIN}" --version || printf 'llama-server 未编译\n'
  [[ -d "${SOURCE_DIR}/.git" ]] && printf 'Git commit: %s\n' "$(git -C "${SOURCE_DIR}" rev-parse --short HEAD)"
  printf '\n=== 配置 ===\n'
  show_config
  printf '\nOpenAI API: http://%s:%s/v1\nWeb UI: http://%s:%s/\n' "${HOST}" "${PORT}" "${HOST}" "${PORT}"
}

show_logs() {
  local follow=false lines=100
  while (( $# > 0 )); do
    case "$1" in
      -f|--follow) follow=true ;;
      -n|--lines) shift; lines=${1:?--lines 需要数字} ;;
      *) die "logs 未知选项：$1" ;;
    esac
    shift
  done
  [[ "${lines}" =~ ^[1-9][0-9]*$ ]] || die "日志行数必须为正整数。"
  local args=(journalctl --user -u "${SERVICE_NAME}" -n "${lines}" --no-pager)
  [[ "${follow}" == true ]] && args+=(-f)
  "${args[@]}"
}

self_test_command() {
  local -a curl_args=(curl -fsS --max-time 10)
  ensure_config
  service_is_active || die "服务未运行。"
  [[ -n "${API_KEY}" ]] && curl_args+=(-H "Authorization: Bearer ${API_KEY}")
  "${curl_args[@]}" "$(health_url)"
  printf '\n'
  "${curl_args[@]}" "http://127.0.0.1:${PORT}/v1/models"
  printf '\n'
  log "HTTP 自检通过。"
}

bench_command() {
  local bench_ngl bench_tensor_split arg
  local -a command=()
  ensure_config
  [[ -x "${BENCH_BIN}" ]] || die "llama-bench 未编译。"
  [[ -f "${MODEL}" ]] || die "未选择有效模型。"
  service_is_active && die "请先停止 llama.cpp service，避免显存冲突。"
  check_conflicting_service
  check_gpu_or_die
  case "${N_GPU_LAYERS}" in
    all|auto) bench_ngl=-1 ;;
    *) bench_ngl=${N_GPU_LAYERS} ;;
  esac
  bench_tensor_split=${TENSOR_SPLIT//,/\/}
  command=(
    "${BENCH_BIN}"
    -m "${MODEL}"
    -ngl "${bench_ngl}"
    -sm "${SPLIT_MODE}"
    -ts "${bench_tensor_split}"
    -mg "${MAIN_GPU}"
    -t "${THREADS}"
    -b "${BATCH_SIZE}"
    -ub "${UBATCH_SIZE}"
    -ctk "${CACHE_TYPE_K}"
    -ctv "${CACHE_TYPE_V}"
    -fa "${FLASH_ATTN}"
    -lm "${LOAD_MODE}"
    --progress
  )
  export CUDA_VISIBLE_DEVICES PATH="/opt/cuda/bin:${PATH}" LD_LIBRARY_PATH="/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
  printf '[BENCH COMMAND]'
  for arg in "${command[@]}"; do printf ' %q' "${arg}"; done
  printf '\n'
  "${command[@]}"
}

doctor_command() {
  local failures=0 warnings=0 ram_kib cpu_model count low_memory
  printf '=== %s doctor ===\n' "${APP_NAME}"
  [[ -r /etc/arch-release ]] && printf '[OK] Arch Linux\n' || { printf '[FAIL] 不是 Arch Linux\n'; failures=$((failures + 1)); }
  cpu_model=$(awk -F: '/model name/ { sub(/^[[:space:]]+/, "", $2); print $2; exit }' /proc/cpuinfo 2>/dev/null || true)
  printf '[INFO] CPU: %s\n' "${cpu_model:-未知}"
  printf '[INFO] 当前可用逻辑 CPU: %s\n' "$(available_cpu_threads)"
  [[ "${cpu_model}" == *i7-11700* ]] && printf '[OK] 目标 CPU i7-11700\n' || { printf '[WARN] CPU 与目标配置不同\n'; warnings=$((warnings + 1)); }
  ram_kib=$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || printf 0)
  (( ram_kib >= 60000000 )) && printf '[OK] RAM: 约 %d GiB\n' "$((ram_kib / 1024 / 1024))" || { printf '[WARN] RAM: %d GiB\n' "$((ram_kib / 1024 / 1024))"; warnings=$((warnings + 1)); }
  if command -v nvidia-smi >/dev/null 2>&1; then
    count=$(gpu_count); low_memory=$(gpu_low_memory_count); print_gpu_table
    [[ "${count}" -eq 2 ]] && printf '[OK] GPU 数量: 2\n' || { printf '[FAIL] GPU 数量: %s\n' "${count}"; failures=$((failures + 1)); }
    [[ "${low_memory}" -eq 0 ]] && printf '[OK] 每张 GPU 显存不少于约 %d GiB\n' "$((MIN_GPU_MEMORY_MIB / 1024))" || { printf '[FAIL] %s 张 GPU 显存不足\n' "${low_memory}"; failures=$((failures + 1)); }
    gpu_models_are_mixed && { printf '[WARN] 混合 GPU：性能受较慢卡限制\n'; warnings=$((warnings + 1)); } || true
  else
    printf '[FAIL] nvidia-smi 不可用\n'; failures=$((failures + 1))
  fi
  printf '[INFO] 驱动包: %s\n' "$(installed_driver_package)"
  if [[ -x /opt/cuda/bin/nvcc ]]; then printf '[OK] %s\n' "$(/opt/cuda/bin/nvcc --version | tail -n 1)"; else printf '[FAIL] CUDA Toolkit/nvcc 未安装\n'; failures=$((failures + 1)); fi
  command -v cmake >/dev/null 2>&1 && printf '[OK] %s\n' "$(cmake --version | head -n 1)" || { printf '[FAIL] cmake 不可用\n'; failures=$((failures + 1)); }
  if [[ -x "${SERVER_BIN}" ]]; then
    printf '[OK] %s\n' "$("${SERVER_BIN}" --version | head -n 1)"
    if ldd "${SERVER_BIN}" 2>/dev/null | grep -q 'not found'; then printf '[FAIL] llama-server 存在缺失动态库\n'; failures=$((failures + 1)); else printf '[OK] llama-server 动态库完整\n'; fi
    "${SERVER_BIN}" --list-devices || { printf '[FAIL] llama.cpp 无法列出 CUDA 设备\n'; failures=$((failures + 1)); }
  else
    printf '[FAIL] llama-server 未编译：%s\n' "${SERVER_BIN}"; failures=$((failures + 1))
  fi
  if [[ -f "${SERVER_CONFIG_FILE}" ]]; then
    load_config
    if validate_config; then
      printf '[OK] Server 启动配置有效：%s\n' "${SERVER_CONFIG_FILE}"
      printf '[OK] 编译配置：%s；并行任务 %s（BUILD_JOBS=%s）\n' "${BUILD_CONFIG_FILE}" "$(resolved_build_jobs)" "${BUILD_JOBS}"
    else
      printf '[FAIL] 配置无效\n'; failures=$((failures + 1))
    fi
  else
    printf '[WARN] 配置尚未生成\n'; warnings=$((warnings + 1))
  fi
  printf '\n结果：%d 个失败，%d 个警告。\n' "${failures}" "${warnings}"
  (( failures == 0 ))
}

install_command() {
  local ref=master driver=none start_after=false enable_linger=false
  while (( $# > 0 )); do
    case "$1" in
      --ref) shift; ref=${1:?--ref 需要参数} ;;
      --driver) shift; driver=${1:?--driver 需要参数} ;;
      --start) start_after=true ;;
      --enable-linger) enable_linger=true ;;
      *) die "install 未知选项：$1" ;;
    esac
    shift
  done
  [[ "${driver}" =~ ^(none|nvidia-open|nvidia-open-dkms)$ ]] || die "driver 选项无效。"
  assert_arch_linux
  install_build_dependencies
  [[ "${driver}" != none ]] && manage_driver --type "${driver}"
  LLAMACPP_REF=${ref}
  ensure_config
  LLAMACPP_REF=${ref}
  write_build_config
  prepare_source "${ref}"
  build_llama_cpp
  install_manager_copy
  write_launcher
  write_service
  run mkdir -p -- "${MODEL_DIR}"
  if [[ -z "${MODEL}" && "${DRY_RUN}" == false ]]; then select_model || true; fi
  [[ "${enable_linger}" == true ]] && run sudo loginctl enable-linger "${USER}"
  log "安装完成。Server 启动配置：${SERVER_CONFIG_FILE}"
  if [[ "${driver}" != none ]]; then
    warn "驱动已变更，请重启后再启动。"
  elif [[ "${start_after}" == true ]]; then
    start_service
  else
    log "下一步：${MANAGER_BIN} doctor && ${MANAGER_BIN} start"
  fi
}

update_command() {
  local ref start_after=false was_active=false
  ensure_config
  ref=${LLAMACPP_REF}
  while (( $# > 0 )); do
    case "$1" in
      --ref) shift; ref=${1:?--ref 需要参数} ;;
      --start) start_after=true ;;
      *) die "update 未知选项：$1" ;;
    esac
    shift
  done
  service_is_active && was_active=true
  [[ "${was_active}" == true ]] && stop_service
  install_build_dependencies
  LLAMACPP_REF=${ref}
  write_build_config
  prepare_source "${ref}"
  build_llama_cpp
  install_manager_copy
  write_launcher
  write_service
  if [[ "${was_active}" == true || "${start_after}" == true ]]; then start_service; fi
  log "更新完成：$(git -C "${SOURCE_DIR}" rev-parse --short HEAD 2>/dev/null || printf dry-run)"
}

self_install_command() {
  ensure_config
  install_manager_copy
  write_launcher
  write_service
  log "管理脚本与 service 已刷新到版本 ${SCRIPT_VERSION}；启动配置：${SERVER_CONFIG_FILE}"
}

safe_remove_tree() {
  local path=$1 resolved
  resolved=$(readlink -m -- "${path}")
  [[ "${resolved}" == "${HOME}/"* && "${resolved}" != "${HOME}" ]] || die "拒绝删除不安全路径：${resolved}"
  run rm -rf -- "${resolved}"
}

uninstall_command() {
  local purge=false yes=false
  while (( $# > 0 )); do
    case "$1" in
      --purge) purge=true ;;
      --yes) yes=true ;;
      *) die "uninstall 未知选项：$1" ;;
    esac
    shift
  done
  stop_service || true
  command -v systemctl >/dev/null 2>&1 && run systemctl --user disable "${SERVICE_NAME}" || true
  run rm -f -- "${SERVICE_FILE}" "${LAUNCHER_BIN}" "${MANAGER_BIN}"
  command -v systemctl >/dev/null 2>&1 && run systemctl --user daemon-reload || true
  if [[ "${purge}" == true ]]; then
    if [[ "${yes}" != true ]] && ! confirm "将删除 ${INSTALL_ROOT} 和 ${CONFIG_DIR}，确定吗？"; then
      warn "已取消 purge；源码、构建和配置保留。"
      return 0
    fi
    safe_remove_tree "${INSTALL_ROOT}"
    safe_remove_tree "${CONFIG_DIR}"
    log "已删除源码、构建和配置。"
  else
    log "已移除命令与 service；保留源码、构建、配置和模型。"
  fi
  log "模型目录 ${MODEL_DIR} 永远不会被卸载命令删除。"
}

config_command() {
  case "${1:-}" in
    --show) show_config ;;
    --edit) edit_config ;;
    --edit-build) edit_build_config ;;
    --wizard|'') config_wizard ;;
    *)
      (( $# == 2 )) || die "用法：config KEY VALUE；或 config --show/--edit/--edit-build/--wizard"
      set_config_value "$1" "$2"
      ;;
  esac
}

menu_command() {
  local choice
  is_tty || die "menu 需要交互式终端。"
  while true; do
    cat <<'MENU'

==== llama.cpp 双 NVIDIA GPU 管理菜单 ====
1) 安装/编译 llama.cpp
2) 扫描并选择 GGUF
3) 配置向导
4) 启动
5) 停止
6) 重启
7) 状态
8) 日志
9) Doctor
10) HTTP 自检
11) Benchmark
12) 更新 llama.cpp
13) 安装/修复 NVIDIA open 驱动
0) 退出
MENU
    read -r -p '请选择: ' choice
    case "${choice}" in
      1) install_command ;;
      2) ensure_config; select_model || true ;;
      3) config_wizard ;;
      4) start_service ;;
      5) stop_service ;;
      6) restart_service ;;
      7) show_status ;;
      8) show_logs ;;
      9) doctor_command || true ;;
      10) self_test_command ;;
      11) bench_command ;;
      12) update_command ;;
      13) manage_driver ;;
      0) return 0 ;;
      *) warn "无效选择。" ;;
    esac
  done
}

main() {
  local -a args=()
  while (( $# > 0 )); do
    case "$1" in
      --dry-run) DRY_RUN=true ;;
      --non-interactive) NON_INTERACTIVE=true ;;
      --verbose) VERBOSE=true ;;
      -h|--help) usage; return 0 ;;
      --version) printf '%s %s\n' "${APP_NAME}" "${SCRIPT_VERSION}"; return 0 ;;
      --) shift; args+=("$@"); break ;;
      *) args+=("$1") ;;
    esac
    shift
  done
  set -- "${args[@]}"
  local cmd=${1:-menu}
  (( $# > 0 )) && shift || true
  case "${cmd}" in
    install) install_command "$@" ;;
    update) update_command "$@" ;;
    self-install) self_install_command "$@" ;;
    start) start_service "$@" ;;
    stop) stop_service "$@" ;;
    restart) restart_service "$@" ;;
    status) show_status "$@" ;;
    logs) show_logs "$@" ;;
    config) config_command "$@" ;;
    models) ensure_config; select_model ;;
    doctor) doctor_command "$@" ;;
    self-test) self_test_command "$@" ;;
    bench) bench_command "$@" ;;
    driver) manage_driver "$@" ;;
    uninstall) uninstall_command "$@" ;;
    menu) menu_command "$@" ;;
    run-server) run_server "$@" ;;
    help) usage ;;
    *) error "未知命令：${cmd}"; usage >&2; return 2 ;;
  esac
}

if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  die "需要 Bash 4.4 或更高版本。"
fi

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi

