# ClaudeAdapter

一个把 Claude Code 的 Anthropic 请求转成 OpenAI 兼容请求的 Rust 代理。你在 `config.yaml` 里配置供应商、默认模型和模型映射，Claude Code 指向本地服务后，请求就会按 `api_mode` 转发到对应供应商的 `/chat/completions` 或 `/responses`。

## 配置

`config.yaml` 示例：

```yaml
bind: 127.0.0.1:8787
incoming_api_key: claude_adapter

activate_provider: bigmodel

providers:
  bigmodel:
    base_url: https://open.bigmodel.cn/api/paas/v4
    api_mode: responses
    api_key: env:BIGMODEL_API_KEY
    headers: {}
    model_default: GLM-4.7-Flash
    model_map:
      claude-sonnet-4-6: GLM-4.6V-Flash
      claude-opus-4-6: GLM-4.6V-Flash
```

说明：

- `activate_provider`：当前启用的供应商
- `model_map`：命中时使用映射模型
- `model_default`：没命中映射时使用的默认模型
- `api_mode`：`chat_completions` 或 `responses`
- `api_key: env:XXX`：从 `.env` 或环境变量读取密钥

`.env` 示例：

```env
BIGMODEL_API_KEY=your_api_key
```

## 使用

默认读取当前目录的 `config.yaml`，如果有 `.env` 会自动加载：

```powershell
cargo run
```

也可以指定文件：

```powershell
cargo run -- -y .\config.yaml -e .\.env
```

`config.yaml` 和 `.env` 修改后会自动重载，不需要重启。只有 `bind` 变化时才需要重启。

## Claude Code

让 Claude Code 指向本地服务：

```powershell
$env:ANTHROPIC_BASE_URL="http://127.0.0.1:8787"
$env:ANTHROPIC_AUTH_TOKEN="claude_adapter"
claude
```

服务入口是 `POST /v1/messages`。

## 快速设置环境

如果你不想手动设置环境变量，可以直接用 `scripts` 里的脚本：

- Windows：双击 `scripts/win-open-claude-env.cmd`
- Windows PowerShell：运行 `scripts/win-open-claude-env.ps1`
- macOS：运行 `scripts/macos-open-claude-env.command`
- Linux：运行 `scripts/linux-open-claude-env.sh`

这些脚本会自动设置：

- `ANTHROPIC_BASE_URL=http://127.0.0.1:8787`
- `ANTHROPIC_AUTH_TOKEN=claude_adapter`

## 发布

推送版本 tag 后会自动编译 Windows、macOS、Linux，并把产物上传到 GitHub Release。

例如：

```powershell
git tag v0.1.0
git push origin v0.1.0
```
