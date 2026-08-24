# llamacpp 项目设计文档

日期：2026-08-24
状态：已获用户批准

## 背景与目标

现有单文件脚本 `llamacpp.sh`（v1.3.1）是面向 Arch Linux 双 NVIDIA 12GB GPU
（RTX 4070 SUPER + RTX 3060）的 llama.cpp 一键部署/管理器。本项目的目标是把
它演进为一个规范的多阶段工程：

1. **阶段 A —— 工程化整理**（Bash）：建立 Git 仓库、目录结构、lint、测试与
   文档；脚本逻辑保持不变。
2. **阶段 C —— Python 重写**：用 Python 全量重写 Bash 版的全部能力
   （安装、编译、驱动、service、配置、模型、doctor、bench），配置文件格式
   与 Bash 版完全兼容，实现平滑接管。完成后 Bash 版冻结，仅接收 bug fix。
3. **阶段 B' —— 新功能**（基于 Python 版）：
   - Profile 多配置管理；
   - 监控采样（nvidia-smi + llama-server `/metrics` → SQLite）与告警
     （面板内 + 日志 + 自定义 Webhook）；
   - Web 管理面板（FastAPI + Jinja2 + HTMX，无前端构建链）。

## 关键决策记录

| 决策点 | 结论 | 理由 |
| --- | --- | --- |
| 推进顺序 | A → C → B' | Web 面板必然需要非 Bash 后端，先做 Bash 版面板会返工 |
| 重写语言 | Python | 生态成熟、typer/rich CLI 体验好、与面板共用核心逻辑 |
| 面板技术栈 | FastAPI + Jinja2 + HTMX | 单语言、无 Node 构建链、部署简单 |
| 告警渠道 | 面板内 + 日志 + Webhook | Webhook 可对接企业微信/钉钉/Telegram/Bark |
| Python 版范围 | 全覆盖（含安装编译） | 一套工具说完就完，Bash 冻结保底 |

## 阶段 A 设计

```
llamacpp/
├── llamacpp.sh              # 现有脚本（逻辑不动，仅修 lint）
├── README.md                # 中文使用文档
├── LICENSE                  # MIT
├── Makefile                 # lint / test / check 目标
├── .gitignore / .editorconfig
├── docs/specs/              # 本设计文档及后续规格
└── tests/
    ├── unit.bash            # 零依赖纯函数测试（sourcing 方式）
    └── lib.sh               # 断言辅助函数
```

- Lint 门禁：`bash -n` + `shellcheck -S warning` 清零。
- 测试策略：通过 `source llamacpp.sh` 复用其"被 source 时不执行 main"的守卫，
  对纯函数做单元测试：`expand_home`、`validate_config`、`load_config_file`、
  `parse_extra_args`、`health_url`、`scan_models`、`append_bool_flag` 等。
  机器相关检查（GPU/pacman/systemd）不进单元测试。
- 本机暂无 bats/shfmt 且无免密 sudo，故测试采用零依赖纯 Bash 实现；
  Makefile 中保留 bats/shfmt 的可选目标（检测到工具才启用）。

## 阶段 C 设计

- 工具链：uv 管理项目；`typer` CLI + `pydantic` 配置模型 + `rich` 输出。
- 子命令与 Bash 版一一对应：
  install / update / self-install / start / stop / restart / status / logs /
  config / models / doctor / self-test / bench / driver / uninstall /
  run-server。
- **配置兼容约束（硬性）**：直接读写 `~/.config/llamacpp/server.env` 与
  `build.env`，键名与解析规则和 Bash 版一致（含 `LLAMA_CPP_REF → LLAMACPP_REF`
  兼容迁移），Python 版可无缝接管 Bash 版装好的环境。

```
src/llamacpp/
├── cli.py          # typer 入口
├── config.py       # env 读写 + 校验
├── gpu.py          # nvidia-smi 探测
├── installer.py    # pacman 依赖、驱动、源码克隆、cmake/ninja 编译
├── service.py      # systemd user unit 写入与管理
├── models.py       # GGUF 扫描/选择
└── server.py       # 启动命令构建、健康检查、self-test
tests/              # pytest
```

## 阶段 B' 设计

1. **Profile 多配置**
   - 存储：`~/.config/llamacpp/profiles/<名字>.env`；
   - 激活指针：一个指针文件记录当前 profile，切换即改写生效配置；
   - CLI：`profile list/use/create/delete`；面板提供一键切换。

2. **监控告警**
   - 采集：定时抓取 nvidia-smi（显存/温度/利用率）+ llama-server `/metrics`
     （token 吞吐等），写入 SQLite；
   - 告警规则：显存超阈值、温度超阈值、服务异常重启、health 检查失败；
   - 通知：面板内告警中心 + 日志 + 自定义 Webhook URL（POST JSON，
     以模板字段适配企业微信/钉钉/Telegram/Bark 的格式差异）。

3. **Web 面板**
   - 技术栈：FastAPI + Jinja2 + HTMX，无前端构建链；
   - 功能：仪表盘（GPU 曲线、吞吐）、服务启停/重启、模型选择切换、
     配置/profile 编辑、告警历史；
   - 安全：支持 API Key 登录保护；默认仅监听 localhost。

## 错误处理与测试原则

- 所有破坏性操作沿用 Bash 版既有防护（路径安全检查、确认提示、--yes）。
- Python 版每个模块有对应 pytest 单元测试；命令构建、配置往返、参数校验
  为必测项；涉及 systemd/nvidia-smi 的部分以 mock 测试。
