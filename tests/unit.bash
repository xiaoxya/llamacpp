#!/usr/bin/env bash
# llamacpp.sh 纯函数单元测试。
# 通过 source 复用脚本函数；机器相关检查（GPU/pacman/systemd）不在覆盖范围。
set -uo pipefail

TESTS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${TESTS_DIR}/.." && pwd)

# shellcheck source=lib.sh
source "${TESTS_DIR}/lib.sh"

# 被测脚本自带 set -Eeuo pipefail 与 ERR trap，会干扰测试流程；
# source 后关闭，仅保留其函数与变量定义。
# shellcheck source=../llamacpp.sh
source "${ROOT}/llamacpp.sh"
trap - ERR
set +e

FIXTURE_DIR=$(mktemp -d)
trap 'rm -rf -- "${FIXTURE_DIR}"' EXIT

section() { printf '\n[%s]\n' "$1"; }

join_array() {
  local IFS=' '
  printf '%s' "$*"
}

# ---------- expand_home ----------
section "expand_home"
assert_eq "~ 展开为 HOME" "${HOME}" "$(expand_home '~')"
# shellcheck disable=SC2088  # 测的就是字面量 ~/ 的展开行为
assert_eq "~/ 前缀展开" "${HOME}/models" "$(expand_home '~/models')"
assert_eq "相对/绝对路径原样保留" "/opt/data" "$(expand_home '/opt/data')"
assert_eq "普通字符串原样保留" "abc" "$(expand_home 'abc')"

# ---------- 配置键白名单 ----------
section "配置键白名单"
assert_return "server 键被接受" 0 is_server_config_key MODEL
assert_return "build 键不被 server 接受" 1 is_server_config_key BUILD_JOBS
assert_return "build 键被 build 接受" 0 is_build_config_key BUILD_JOBS
assert_return "未知键被拒绝" 1 is_config_key NOT_A_KEY
assert_return "LLAMACPP_REF 属于 build 键" 0 is_build_config_key LLAMACPP_REF

# ---------- validate_config ----------
section "validate_config"
assert_return "默认值通过校验" 0 validate_config

PORT=70000 validate_config >/dev/null 2>&1
assert_eq "超范围 PORT 返回 2" 2 "$?"

FIT=maybe validate_config >/dev/null 2>&1
assert_eq "非法布尔 FIT 返回 2" 2 "$?"

SPLIT_MODE=bogus validate_config >/dev/null 2>&1
assert_eq "非法 SPLIT_MODE 返回 2" 2 "$?"

MODEL=relative/path.gguf validate_config >/dev/null 2>&1
assert_eq "非绝对路径 MODEL 返回 2" 2 "$?"

CTX_SIZE=-5 validate_config >/dev/null 2>&1
assert_eq "负数 CTX_SIZE 返回 2" 2 "$?"

CUDA_ARCHITECTURES="86;89" validate_config >/dev/null 2>&1
assert_eq "合法 CUDA 架构恢复通过" 0 "$?"

# ---------- load_config_file ----------
section "load_config_file"
cat > "${FIXTURE_DIR}/server.env" <<'EOF'
# 注释行
MODEL=/tmp/fake.gguf
PORT=9999
BOGUS_KEY=ignored
EOF
MODEL='' PORT=8080
load_config_file "${FIXTURE_DIR}/server.env" server 2>/dev/null
assert_eq "解析 MODEL" "/tmp/fake.gguf" "${MODEL}"
assert_eq "解析 PORT" "9999" "${PORT}"

cat > "${FIXTURE_DIR}/legacy.env" <<'EOF'
LLAMA_CPP_REF=b456
BUILD_JOBS=4
EOF
LLAMACPP_REF=master BUILD_JOBS=auto
load_config_file "${FIXTURE_DIR}/legacy.env" legacy 2>/dev/null
assert_eq "旧键 LLAMA_CPP_REF 兼容读取" "b456" "${LLAMACPP_REF}"
assert_eq "legacy 同时读取 BUILD_JOBS" "4" "${BUILD_JOBS}"

LLAMACPP_REF=master
load_config_file "${FIXTURE_DIR}/legacy.env" build 2>/dev/null
assert_eq "build 类型同样迁移旧键 LLAMA_CPP_REF" "b456" "${LLAMACPP_REF}"
assert_eq "build 类型读取自身键" "4" "${BUILD_JOBS}"
BUILD_JOBS=auto

# ---------- parse_extra_args ----------
section "parse_extra_args"
if command -v python >/dev/null 2>&1; then
  EXTRA_ARGS='--verbose --chat-template chatml "--top-p 0.9"'
  parse_extra_args "${EXTRA_ARGS}"
  assert_eq "shlex 拆分为 4 个参数" 4 "${#EXTRA_ARGV[@]}"
  assert_eq "引号内空格保留" "--top-p 0.9" "${EXTRA_ARGV[3]}"
  EXTRA_ARGS=''
  parse_extra_args "${EXTRA_ARGS}"
  assert_eq "空串产生零参数" 0 "${#EXTRA_ARGV[@]}"
else
  printf '  SKIP parse_extra_args（未找到 python）\n'
fi

# ---------- health_url ----------
section "health_url"
HOST=0.0.0.0 PORT=8080
assert_eq "0.0.0.0 映射到 127.0.0.1" "http://127.0.0.1:8080/health" "$(health_url)"
HOST='::' PORT=9000
assert_eq ":: 映射到 [::1]" "http://[::1]:9000/health" "$(health_url)"
HOST=192.168.1.5 PORT=8080
assert_eq "IPv4 原样拼接" "http://192.168.1.5:8080/health" "$(health_url)"
HOST=::1 PORT=8080
assert_eq "IPv6 加方括号" "http://[::1]:8080/health" "$(health_url)"
# shellcheck disable=SC2034  # 这些变量由被测函数动态读取，静态分析看不到
HOST=127.0.0.1 PORT=8080

# ---------- append_bool_flag / redact ----------
section "辅助函数"
declare -a arr=()
append_bool_flag arr true --on --off
assert_eq "布尔 true 取 on flag" "--on" "${arr[0]}"
append_bool_flag arr false --on --off
assert_eq "布尔 false 取 off flag" "--off" "${arr[1]}"
unset arr

assert_eq "redact 空值" "(未设置)" "$(redact '')"
assert_eq "redact 非空打码" "********" "$(redact 'secret')"

# ---------- scan_models ----------
section "scan_models"
mkdir -p "${FIXTURE_DIR}/models"
: > "${FIXTURE_DIR}/models/model-a.gguf"
: > "${FIXTURE_DIR}/models/mmproj-model.gguf"
: > "${FIXTURE_DIR}/models/qwen-00001-of-00003.gguf"
: > "${FIXTURE_DIR}/models/qwen-00002-of-00003.gguf"
: > "${FIXTURE_DIR}/models/notes.txt"
mapfile -d '' -t found < <(scan_models "${FIXTURE_DIR}/models")
assert_eq "扫描结果数量（排除 mmproj 与非首分片）" 2 "${#found[@]}"
assert_contains "包含普通 GGUF" "$(join_array "${found[@]@Q}")" "model-a.gguf"
assert_not_contains "排除 mmproj" "$(join_array "${found[@]@Q}")" "mmproj-model.gguf"
assert_contains "保留首分片" "$(join_array "${found[@]@Q}")" "qwen-00001-of-00003.gguf"
assert_not_contains "排除非首分片" "$(join_array "${found[@]@Q}")" "qwen-00002-of-00003.gguf"

# ---------- build_server_command ----------
section "build_server_command"
MODEL="${FIXTURE_DIR}/models/model-a.gguf"
API_KEY='secret-key'
WEBUI=false
EXTRA_ARGS='--verbose'
declare -a argv=()
build_server_command argv
expected_bin="${XDG_DATA_HOME:-${HOME}/.local/share}/llamacpp/build/bin/llama-server"
assert_eq "首元素为 llama-server 绝对路径" "${expected_bin}" "${argv[0]}"
cmdline=$(join_array "${argv[@]}")
assert_contains "包含模型路径" "${cmdline}" "--model ${MODEL}"
assert_contains "包含 API Key" "${cmdline}" "--api-key secret-key"
assert_contains "WEBUI=false 映射为 --no-webui" "${cmdline}" "--no-webui"
assert_contains "FIT=true 展开 fit-target" "${cmdline}" "--fit on --fit-target ${FIT_TARGET}"
assert_contains "EXTRA_ARGS 追加在末尾" "${cmdline}" "--verbose"
assert_not_contains "METRICS 默认开启但 WEBUI 关闭不影响 metrics" "${cmdline}" "--no-metrics"
# shellcheck disable=SC2034  # 由被测函数动态读取
API_KEY=''
EXTRA_ARGS=''
# shellcheck disable=SC2034  # 由被测函数动态读取
WEBUI=true

# ---------- CLI 入口（子进程集成冒烟） ----------
section "CLI 冒烟"
ver=$("${ROOT}/llamacpp.sh" --version 2>/dev/null)
assert_contains "--version 输出名称" "${ver}" "llamacpp"
"${ROOT}/llamacpp.sh" definitely-not-a-command >/dev/null 2>&1
assert_eq "未知命令返回退出码 2" 2 "$?"
"${ROOT}/llamacpp.sh" --help >/dev/null 2>&1
assert_eq "--help 返回退出码 0" 0 "$?"

summary
