.POSIX:

SCRIPT := llamacpp.sh
PY := .venv/bin/python

.PHONY: check lint test test-py format venv install-user help

check: lint test test-py

lint:
	bash -n $(SCRIPT)
	bash -n tests/unit.bash
	shellcheck -S warning $(SCRIPT) scripts/bootstrap.sh

test:
	bash tests/unit.bash

test-py:
	PYTHONPATH=src $(PY) -m pytest

venv:
	python3 -m venv .venv
	.venv/bin/pip install --quiet -e ".[panel]" pytest

format:
	@if command -v shfmt >/dev/null 2>&1; then \
		shfmt -w -i 4 -ci $(SCRIPT) tests/unit.bash; \
	else \
		echo "shfmt 未安装；跳过（pacman -S shfmt）"; \
	fi

# 安装 Python 版 CLI 到 ~/.local/bin（不覆盖 Bash 版的 llamacpp 命令）
install-user: venv
	.venv/bin/pip install --quiet -e ".[panel]"
	ln -sf $$(realpath .venv/bin/llamacpp) ~/.local/bin/llamacpp-py
	@echo "已安装：llamacpp-py（Bash 版 llamacpp 保持不变）"

help:
	@echo "可用目标：check / lint / test（Bash）/ test-py（Python）/ venv / format / install-user"
