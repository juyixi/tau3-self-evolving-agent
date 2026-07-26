# DeepSeek NL 评估器固定设计

## 决策

项目的标准 Tau2 Retail 运行统一使用 `deepseek/deepseek-v4-pro` 承担两个外部
LLM 角色：

- 非 solo 用户模拟器。
- Tau2 `NLAssertionsEvaluator` 的自然语言断言评估器。

两者统一读取运行机器环境变量 `DEEPSEEK_API_KEY`。标准命令始终使用
`configs/default.yaml`，不再维护机器专用的 evaluator 覆盖配置。

## 固定配置

```yaml
tau2:
  user_llm: deepseek/deepseek-v4-pro
  user_llm_args:
    thinking:
      type: disabled
    temperature: 0.0
    max_tokens: 8192

evaluation:
  nl_assertions:
    model: deepseek/deepseek-v4-pro
    model_args:
      temperature: 0.0
      thinking:
        type: disabled
    api_key_env: DEEPSEEK_API_KEY
```

`NLAssertionsConfig` 的代码 fallback 必须与该 YAML 保持一致，避免遗漏配置段时
静默退回其他 provider。默认测试、Fast Loop、baseline 和 Stage 8 A/B/C/D
全部继承同一配置。

## 运行时约束

1. 在创建 run artifact 前检查 `DEEPSEEK_API_KEY` 存在且非空。
2. 通过项目 adapter 将模型和公开参数绑定到固定 Tau2 checkout 的 evaluator
   模块，不修改 `external/tau2-bench`。
3. manifest 记录实际模型、公开参数和凭证环境变量名称，不记录凭证值。
4. 同一组对照实验的所有 cell 必须使用完全相同的 evaluator 配置。
5. 真实集成测试只要求 Qwen 端点、Qwen revision 和 `DEEPSEEK_API_KEY`。

## 历史兼容

项目保留 provider 可配置适配层，以读取历史 OpenRouter run 或执行显式复现实验。
这不构成标准运行路径。任何非 DeepSeek evaluator run 都必须使用显式配置文件、
独立 Run ID，并且不得与标准 Stage 8 结果混合比较。

此前的
[OpenRouter NL 评估器可配置设计](2026-07-12-openrouter-nl-evaluator-design.md)
已被本决策取代。

## 验收

- `configs/default.yaml` 解析为 `deepseek/deepseek-v4-pro` 和
  `DEEPSEEK_API_KEY`。
- `ProjectConfig` 缺省 evaluator 时得到相同结果。
- `scripts.evaluate_retail`、`scripts.run_fast_loop` 与
  `scripts.run_baseline` 均从默认配置绑定 evaluator。
- 默认真实集成门禁不再检查 `OPENROUTER_API_KEY`。
- 完整测试集通过，运行产物中不出现任何 API Key 值。
