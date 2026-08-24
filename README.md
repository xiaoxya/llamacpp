# llamacpp

Arch Linux 双 NVIDIA GPU（12GB × 2）llama.cpp 一键部署/管理器。

针对 Intel i7-11700 / 64GB RAM / RTX 4070 SUPER + RTX 3060 调优，提供从
依赖安装、CUDA 编译、systemd user service 到日常运维（配置、模型选择、
监控、benchmark）的完整闭环。当前包含 **Bash 与 Python 双实现**，共享同一份
`~/.config/llamacpp/*.env` 配置（已由测试验证双向兼容）。

## 功能特性

- **一键安装**：自动安装 CUDA Toolkit 与构建依赖，克隆并编译 llama.cpp
  （llama-server / llama-cli / llama-bench），生成 systemd user service
- **双卡管理**：layer/row/tensor split、显存配比（`TENSOR_SPLIT`）、
  每卡预留显存（`FIT_TARGET`）
- **OpenAI / Anthropic 兼容 API**：`http://HOST:8080/v1` 与
  `http://HOST:8080/v1/messages`，自带 Web UI
- **配置即数据**：`server.env` / `build.env` 以纯数据格式解析，
  绝不作为 shell/Python 代码执行，杜绝注入
- **运维工具集**：doctor 系统自检、HTTP 自检、llama-bench 基准测试、
  日志跟踪、交互式菜单
- **安全默认值**：API Key 支持、卸载永远不碰模型目录、危险路径拒绝删除

## 快速开始（Bash 版）

```bash
./llamacpp.sh install          # 安装依赖 + 编译 + 生成 service
./llamacpp.sh models           # 扫描 ~/models 下的 GGUF 并选择
./llamacpp.sh config API_KEY '替换为长随机密钥'
./llamacpp.sh start            # 启动并等待 health 就绪
./llamacpp.sh doctor           # 系统/驱动/CUDA/编译/配置全面自检
```

## 快速开始（Python 版）

```bash
make install-user              # 安装到 ~/.local/bin/llamacpp-py
llamacpp-py --version
llamacpp-py config --show      # 直接读取 Bash 版写出的配置
llamacpp-py models             # 子命令与 Bash 版一一对应
```

两个版本读写同一份 `~/.config/llamacpp/server.env` 与 `build.env`，
可随时互换接管，无需重新安装或迁移。

## 常用命令

| 命令 | 说明 |
| --- | --- |
| `install` / `update` | 安装编译 / 拉取更新重编译 |
| `start` / `stop` / `restart` | 服务生命周期管理 |
| `status` | 服务、GPU、构建与配置摘要 |
| `logs -f` | journalctl 日志跟踪 |
| `config KEY VALUE` | 修改配置（`config --show` 查看） |
| `models` | 扫描并选择 GGUF 模型 |
| `doctor` / `self-test` | 环境自检 / HTTP 自检 |
| `bench` | llama-bench 双卡基准 |
| `menu` | 交互式管理菜单 |
| `uninstall --purge` | 卸载（purge 连源码与配置一起删） |

完整选项见 `--help`。

## 开发

```bash
make check     # lint + 全部测试（Bash 与 Python）
make lint      # bash -n + shellcheck
make test      # Bash 单元测试（零依赖纯 Bash 实现）
make test-py   # Python pytest 测试
make format    # shfmt 格式化（需自行安装）
make install-user  # 安装 Python 版 CLI 到 ~/.local/bin/llamacpp-py
```

- Bash 测试通过 `source llamacpp.sh` 复用其函数；机器相关检查不在覆盖范围。
- Python 测试覆盖配置往返兼容、启动命令构建、模型扫描、GPU 解析、
  unit 文件渲染及 CLI 子进程冒烟。

### 双实现并存约定

- **Bash 版**（`llamacpp.sh`）：当前生产版本，命令名 `llamacpp`
- **Python 版**（`src/llamacpp/`）：阶段 C 重写，命令名 `llamacpp-py`

## 项目路线

- **阶段 A（已完成）**：工程化整理——仓库结构、lint 门禁、单元测试、文档
- **阶段 C（进行中）**：Python 重写完成核心模块，配置双向兼容；
  待真机验收后切换默认
- **阶段 B'**：Profile 多配置、SQLite 监控告警（Webhook 通知）、
  FastAPI + HTMX Web 管理面板

设计详情见 [docs/specs](docs/specs/)。

## 许可证

[MIT](LICENSE)
