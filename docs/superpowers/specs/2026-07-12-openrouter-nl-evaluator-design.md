# OpenRouter NL 评估器可配置设计

## 背景

Tau2 retail 任务的最终奖励可能同时包含数据库断言和自然语言断言。自然语言断言由 Tau2 的 `NLAssertionsEvaluator` 调用大语言模型判断。目前固定版本 Tau2 在 `tau2.config` 中将该模型硬编码为 `gpt-4.1-2025-04-14`，因此 LiteLLM 会按 OpenAI 官方 provider 路由并读取 `OPENAI_API_KEY`。

本项目需要默认通过 OpenRouter 调用 GPT-4.1，模型标识为 `openrouter/openai/gpt-4.1`，凭证从 `OPENROUTER_API_KEY` 读取。实现不能修改固定版本的 `external/tau2-bench` checkout，也不能把任何 API Key 写入配置、日志、事件或 manifest。

## 目标

- 在项目配置中显式声明 NL assertion 评估器模型、生成参数和凭证环境变量名。
- 默认使用 `openrouter/openai/gpt-4.1`、`temperature: 0.0` 和 `OPENROUTER_API_KEY`。
- 在创建运行产物或开始 episode 前检查凭证并绑定 Tau2 评估器。
- 在 manifest 中记录实际评估器配置，使不同 provider 的基线结果可区分、可追溯。
- 保持 Tau2 checkout 不变，并继续通过现有 commit pin 校验。

## 非目标

- 不改变 Qwen agent 的 vLLM/OpenAI-compatible 服务配置。
- 不改变 DeepSeek 用户模拟器配置。
- 不替换 Tau2 的数据库断言或奖励聚合逻辑。
- 不引入 OpenRouter SDK；继续使用 Tau2 已依赖的 LiteLLM。
- 不把 OpenRouter 结果声明为与 OpenAI 官方端点逐 token 等价。

## 方案比较

### 方案一：项目侧运行时绑定，采用

项目加载配置和已验证 Tau2 runtime 后，显式更新 `tau2.evaluator.evaluator_nl_assertions` 模块中已经导入的模型与参数常量。

优点是外部 checkout 保持干净，配置和运行溯源由项目统一负责；缺点是绑定代码需要对固定 Tau2 版本的模块契约进行验证。

### 方案二：重定向 OpenAI Base URL，不采用

将 OpenRouter Key 放入 OpenAI 兼容客户端并覆盖 base URL。Tau2 当前使用日期版模型名，而 OpenRouter 使用 `openai/gpt-4.1` slug；这种方式会把 provider 身份隐藏在环境变量中，也不利于 manifest 溯源。

### 方案三：修改 Tau2 checkout，不采用

直接修改 `tau2.config` 最简单，但会污染固定 checkout、破坏 commit pin，并增加升级和复现实验的成本。

## 配置模型

项目配置新增顶层 `evaluation`：

```yaml
evaluation:
  nl_assertions:
    model: openrouter/openai/gpt-4.1
    model_args:
      temperature: 0.0
    api_key_env: OPENROUTER_API_KEY
```

对应 Pydantic 模型继续使用 `extra="forbid"`。约束如下：

- `model` 必须是非空字符串。
- `model_args` 是 JSON 可序列化的公开生成参数，不允许放入凭证。
- `api_key_env` 只保存环境变量名称，默认值为 `OPENROUTER_API_KEY`。
- 默认配置即为 OpenRouter GPT-4.1；需要切回官方 OpenAI 时必须显式改配置和凭证变量名。

## 组件与职责

### `EvaluationConfig`

负责解析和校验评估器声明，不读取环境变量，也不导入 Tau2。

### Tau2 NL 评估器绑定器

新增项目侧小型绑定模块，职责为：

1. 检查 `api_key_env` 指定的环境变量存在且非空。
2. 导入固定 checkout 中的 `tau2.evaluator.evaluator_nl_assertions`。
3. 验证模块暴露预期的模型和参数全局变量。
4. 将模型和参数替换为项目配置值。
5. 返回不含凭证值的已解析配置，供 manifest 使用。

之所以绑定 evaluator 模块本身，而不是只修改 `tau2.config`，是因为 Tau2 使用 `from tau2.config import ...` 在模块导入时复制了常量引用。只修改 `tau2.config` 无法可靠影响已经导入的 evaluator 模块。

### `run_baseline`

运行顺序调整为：

1. 解析参数并限制为 train split。
2. 加载项目配置。
3. 校验 Tau2 checkout、commit pin 和任务集合。
4. 加载已验证的 Tau2 gym factory。
5. 检查凭证并绑定 NL assertion 评估器。
6. 创建 probe、Qwen policy 和 immutable manifest。
7. 执行 episodes。

凭证错误必须发生在 manifest 创建和 episode 执行之前，避免留下看似有效但无法完成评分的运行目录。

## Manifest

manifest 新增：

```json
{
  "evaluation_config": {
    "nl_assertions": {
      "model": "openrouter/openai/gpt-4.1",
      "model_args": {"temperature": 0.0},
      "api_key_env": "OPENROUTER_API_KEY"
    }
  }
}
```

只记录环境变量名称，不读取或记录其值。该字段仍经过现有递归脱敏器处理，防止将来错误加入 `api_key`、`token` 等字段时泄漏。

manifest schema version 从 1 提升为 2，因为新增字段改变了运行产物契约。旧 manifest 保持可读，不做迁移或重写。

## 错误处理

- 缺少或空的 `OPENROUTER_API_KEY`：抛出明确错误，只显示环境变量名称，不显示任何凭证值。
- Tau2 evaluator 模块不符合预期契约：快速失败，提示固定 checkout 与项目适配器不兼容。
- `model_args` 不可 JSON 序列化：在配置或 manifest 创建前失败。
- OpenRouter 请求失败：由 Tau2/LiteLLM 原异常保留，并由现有环境适配器补充 task、split 和 episode step 上下文。

## 测试策略

### 配置测试

- 默认配置解析为 `openrouter/openai/gpt-4.1`。
- 默认凭证变量名为 `OPENROUTER_API_KEY`。
- dotted override 可以切换模型、温度和凭证变量名。
- 空模型或空凭证变量名被拒绝。

### 绑定器单元测试

- 使用测试模块验证模型和参数被正确绑定。
- 缺少或空凭证时在导入/调用评估器前失败。
- 返回值只包含公开配置，不包含环境变量值。
- Tau2 模块契约缺失时给出稳定错误。

### Baseline 与 manifest 测试

- `run_baseline` 在创建 manifest 前调用绑定器。
- manifest schema version 为 2，并包含 `evaluation_config`。
- OpenRouter Key 不出现在 manifest、命令或标准输出中。
- 现有 Qwen、用户模拟器、无 adapter、无 memory 的基线约束保持不变。

### 完整验证

- 运行新增的配置、绑定器和 baseline 测试。
- 运行完整 pytest 测试集。
- 在配置了 `OPENROUTER_API_KEY` 的真实机器上执行一个 retail train task，确认 NL assertion 请求通过 OpenRouter 完成。

## 使用方式

PowerShell 当前会话：

```powershell
$env:OPENROUTER_API_KEY = Read-Host "OpenRouter API Key"
python -m scripts.run_baseline <现有参数>
```

Key 仅需配置在实际运行 `scripts.run_baseline` 的机器上，不需要配置在只提供 Qwen vLLM 服务的 AutoDL 机器上。

## 参考

- [LiteLLM OpenRouter provider 文档](https://docs.litellm.ai/docs/providers/openrouter)
- [OpenRouter GPT-4.1 模型页](https://openrouter.ai/openai/gpt-4.1/api)
