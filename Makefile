.POSIX:

SCRIPT := llamacpp.sh

.PHONY: check lint test format help

check: lint test

lint:
	bash -n $(SCRIPT)
	bash -n tests/unit.bash
	shellcheck -S warning $(SCRIPT)

test:
	bash tests/unit.bash

format:
	@if command -v shfmt >/dev/null 2>&1; then \
		shfmt -w -i 4 -ci $(SCRIPT) tests/unit.bash; \
	else \
		echo "shfmt 未安装；跳过（pacman -S shfmt）"; \
	fi

help:
	@echo "可用目标：check（lint+test）/ lint / test / format"
