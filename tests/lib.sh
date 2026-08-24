#!/usr/bin/env bash
# 极简断言库：零依赖，供 tests/unit.bash 使用。
TESTS_RUN=0
TESTS_FAILED=0

pass() {
  TESTS_RUN=$((TESTS_RUN + 1))
  printf '  ok   %s\n' "$1"
}

fail() {
  local desc=$1
  shift
  TESTS_RUN=$((TESTS_RUN + 1))
  TESTS_FAILED=$((TESTS_FAILED + 1))
  printf '  FAIL %s\n' "${desc}"
  local line
  for line in "$@"; do
    printf '       %s\n' "${line}"
  done
}

assert_eq() {
  local desc=$1 expected=$2 actual=$3
  if [[ "${expected}" == "${actual}" ]]; then
    pass "${desc}"
  else
    fail "${desc}" "期望: ${expected}" "实际: ${actual}"
  fi
}

assert_contains() {
  local desc=$1 haystack=$2 needle=$3
  if [[ "${haystack}" == *"${needle}"* ]]; then
    pass "${desc}"
  else
    fail "${desc}" "应包含: ${needle}" "实际: ${haystack}"
  fi
}

assert_not_contains() {
  local desc=$1 haystack=$2 needle=$3
  if [[ "${haystack}" != *"${needle}"* ]]; then
    pass "${desc}"
  else
    fail "${desc}" "不应包含: ${needle}" "实际: ${haystack}"
  fi
}

assert_return() {
  local desc=$1 expected=$2
  shift 2
  local actual=0
  "$@" >/dev/null 2>&1 || actual=$?
  assert_eq "${desc}" "${expected}" "${actual}"
}

summary() {
  printf '\n=== %d 个用例，%d 个失败 ===\n' "${TESTS_RUN}" "${TESTS_FAILED}"
  (( TESTS_FAILED == 0 ))
}
