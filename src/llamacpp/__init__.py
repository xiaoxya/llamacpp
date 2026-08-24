"""llamacpp：Arch Linux 双 NVIDIA GPU llama.cpp 部署与管理器。"""

__version__ = "2.0.0"

APP_NAME = "llamacpp"
SERVICE_NAME = f"{APP_NAME}.service"
CONFLICTING_SERVICE = "vllm-dual-3060.service"
LEGACY_SERVICE = "llama-cpp-dual-gpu.service"
REPOSITORY_URL = "https://github.com/ggml-org/llama.cpp.git"
