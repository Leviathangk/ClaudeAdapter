# ClaudeAdapter

一个 Rust 后端代理，用来把 Claude Code 发来的 OpenAI 风格请求按模型映射转发到自定义的 OpenAI 兼容供应商。

当前支持：

- `POST /v1/messages`
- `POST /chat/completions`
- `POST /responses`

核心行为：

- 接收 Claude Code 使用的 Anthropic Messages 请求 `POST /v1/messages`
- 从 YAML 读取供应商列表和 Claude 模型映射
- 收到请求后根据 `model` 找到目标供应商和目标模型
- 把请求里的 `model` 改成目标模型
- 按 `api_mode` 转发到对应供应商的 `/chat/completions` 或 `/responses`
- 把上游 OpenAI 响应转换回 Anthropic Messages 响应
- 当请求体包含 `"stream": true` 时，按 SSE 流式透传上游响应

## 配置

项目启动时会自动加载根目录 `.env` 文件，所以推荐这样用：

1. 复制 `config.yaml.example` 为 `config.yaml`
2. 新建 `.env`

示例：

```env
OPENAI_COMPATIBLE_API_KEY=your-provider-key
BIGMODEL_API_KEY=your-bigmodel-key
```

如果你只是临时测试，也可以直接设置环境变量：

```powershell
$env:OPENAI_COMPATIBLE_API_KEY="your-provider-key"
```

`api_key` 和 `incoming_api_key` 支持两种写法：

- 直接写明文
- `env:ENV_NAME`

例如：

```yaml
api_key: env:BIGMODEL_API_KEY
incoming_api_key: claude_adapter
```

`incoming_api_key` 是你本地代理入口的 Bearer Token，不是上游供应商密钥。

如果服务只监听 `127.0.0.1`，固定写成 `claude_adapter` 是可以的；如果后续要暴露到局域网或公网，建议换成随机值。

建议不要把真实供应商密钥直接写进 `config.yaml`。

`config.yaml` 的推荐结构：

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
      claude-sonnet-4.6: GLM-4.6V-Flash
      claude-opus-4-6: GLM-4.6V-Flash
```

字段说明：

- `activate_provider`: 当前启用的供应商名称，必须对应 `providers` 里的某一项
- `providers.<name>.api_mode`: 上游转发模式，决定当前 provider 的请求最终转发到 `/chat/completions` 还是 `/responses`
- `providers.<name>.model_map`: 主动映射表。请求里的 Claude 模型名如果命中这里，就转成这里配置的目标模型名
- `providers.<name>.model_default`: 默认映射。请求模型没有出现在 `model_map` 里时，自动回退到这里配置的模型

`api_mode` 当前支持两种写法：

- `chat_completions`: 上游转发到 `base_url + /chat/completions`
- `responses`: 上游转发到 `base_url + /responses`

例如 `base_url: https://api.openai.com/v1` 且 `api_mode: responses`，实际转发地址是 `https://api.openai.com/v1/responses`。

`api_mode` 只决定上游怎么转发，不限制本地入口必须使用 `/chat/completions` 还是 `/responses`。

`base_url` 建议写供应商根地址，不要自己再手动拼接口路径。

映射优先级：

1. 先查 `model_map`
2. 如果没有匹配项，使用 `model_default`

例如：

- 请求模型是 `claude-sonnet-4.6`，命中 `model_map`，转发成 `GLM-4.6V-Flash`
- 请求模型是 `claude-3-unknown`，没有命中 `model_map`，转发成 `GLM-4.7-Flash`

## Claude Code 接入

Claude Code 应该指向本地 Anthropic 兼容入口：

```powershell
$env:ANTHROPIC_BASE_URL="http://127.0.0.1:8787"
$env:ANTHROPIC_AUTH_TOKEN="claude_adapter"
claude
```

当前最关键的兼容入口是：

- `POST /v1/messages`

这个接口会：

1. 接收 Claude Code 发来的 Anthropic Messages 请求
2. 用 `model_map` / `model_default` 解析目标模型
3. 按当前 provider 的 `api_mode` 转发到上游 OpenAI 接口
4. 再把上游结果转回 Anthropic Messages 响应

目前 `POST /v1/messages` 只支持非流式请求，`stream: true` 还没有实现。

## 运行

```powershell
cargo run
```

## 一键应用 Claude Code 配置

如果你不想再手动使用 `cc switch`，可以直接运行项目里的脚本来覆盖 `~/.claude/settings.json`。

只修复模型，恢复成安全默认值 `sonnet`：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\apply-claude-settings.ps1
```

如果你要同时改 Claude Code 的 `ANTHROPIC_BASE_URL` 和 `ANTHROPIC_AUTH_TOKEN`：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\apply-claude-settings.ps1 \
  -BaseUrl "http://127.0.0.1:8787" \
  -AuthToken "claude_adapter" \
  -Model "sonnet"
```

如果你只想清掉当前显式设置的模型，让 Claude Code 自己回到默认模型：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\apply-claude-settings.ps1 -ClearModel
```

脚本行为：

- 自动备份现有 `C:\Users\<你>\.claude\settings.json`
- 只修改你传入的配置项
- 默认会把 `model` 设成 `sonnet`

也可以指定配置路径：

```powershell
$env:CLAUDE_ADAPTER_CONFIG="E:\Project\MyProject\ClaudeAdapter\config.yaml"
cargo run
```

## Claude Code 接入思路

把 Claude Code 的 OpenAI 兼容请求地址指向这个服务，例如：

- Base URL: `http://127.0.0.1:8787`
- API Key: `claude_adapter`
- Model: 使用当前 `activate_provider` 下 `model_map` 里的 Claude 模型名，例如 `claude-sonnet-4.6`

这样 Claude Code 发到本地 `/chat/completions` 或 `/responses` 的请求，就会被映射后转发到你定义的供应商。

## 测试

```powershell
cargo test
```

当前测试覆盖：

- `/v1/messages` 到 OpenAI `chat/completions` 的转换
- `/v1/messages` 到 OpenAI `responses` 的转换
- `/chat/completions` 模型映射与普通转发
- `/responses` 模型映射与普通转发
- `stream: true` 时的 SSE 透传
- 入站 Bearer Token 校验
