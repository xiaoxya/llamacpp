llamacpp
========

Arch Linux / Debian / Ubuntu 双 NVIDIA GPU（12GB × 2）llama.cpp 一键部署与管理器。

针对 Intel i7-11700 / 64GB RAM / RTX 4070 SUPER + RTX 3060 调优，提供从
依赖安装、CUDA 编译、systemd user service 到日常运维（配置、模型选择、
监控、benchmark）的完整闭环。当前包含 **Bash 与 Python 双实现**：

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

### Python 版新功能（B'）

- **Profile 多配置**：`profile create/use/list/delete`，一套配置快照、
  一键切换；面板同步支持切换
- **监控告警**：采样 nvidia-smi + `/metrics` 存 SQLite；显存/温度/
  health 连续失败阈值告警；通知走面板内 + 日志 + Webhook
  （generic / 企业微信 / 钉钉 / Telegram / Bark 格式适配）
- **Web 管理面板**：FastAPI + HTMX 无构建链；仪表盘（GPU、服务控制、
  告警历史）、模型切换、配置编辑、Profile 切换；
  可选 `PANEL_KEY` 登录认证

## 发行版支持

| 功能 | Arch 系 | Debian/Ubuntu 系 |
| --- | --- | --- |
| 运行时管理（启停/配置/模型/面板/监控/bench） | ✅ | ✅ |
| 依赖安装 + CUDA 编译（`install`/`update`） | ✅ `pacman` | ✅ `apt`（CUDA Toolkit 需 NVIDIA 官方 repo 或 `nvidia-cuda-toolkit`） |
| 驱动管理（`driver`） | ✅ nvidia-open(-dkms) + mkinitcpio | ✅ Ubuntu: `ubuntu-drivers autoinstall`；Debian: non-free 仓库后指定包名如 `nvidia-driver-550` |
| Bash 版 `llamacpp.sh` | ✅ | ❌ 仅 Arch |

CUDA 路径自动探测（`CUDA_HOME` → `/usr/local/cuda` → `/opt/cuda` → PATH），
自定义安装只需 `export CUDA_HOME=...`。Python 版子命令在两种家族完全一致。

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

### 监控与面板（Python 版）

```bash
llamacpp-py monitor config                 # 生成/查看 monitor.env（阈值与 Webhook）
llamacpp-py monitor run                    # 启动采样循环（可注册为 systemd 服务）
llamacpp-py monitor status                 # 最近吞吐与告警

llamacpp-py panel                          # 前台启动面板 http://127.0.0.1:8199/

# 局域网访问 + 后台常驻（推荐）：
#   1. 在 monitor.env 设置 PANEL_KEY="长随机密钥" 启用登录认证
#   2. 安装为 systemd user service（默认监听 0.0.0.0:8199）
llamacpp-py panel install                  # 注册服务并立即后台启动
llamacpp-py panel start|stop|restart       # 日常管理
llamacpp-py panel status / logs -f         # 状态与日志
llamacpp-py panel uninstall                # 移除服务

llamacpp-py profile create myprofile       # 快照当前启动配置
llamacpp-py profile list && llamacpp-py profile use myprofile
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
- **阶段 C（已完成核心）**：Python 重写全部子命令，配置双向兼容；
  待真机验收后切换默认并冻结 Bash 版
- **阶段 B'（已完成核心）**：Profile 多配置、SQLite 监控告警（Webhook）、
  FastAPI + HTMX 面板；同样待真机验收

设计详情见 [docs/specs](docs/specs/)。

## 许可证

[MIT](LICENSE)
