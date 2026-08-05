# 项目结构重构共识记录

本文档记录项目结构重构讨论中已经达成的共识。当前阶段只澄清目标架构，
不代表已经开始修改实现。

## 入口设计

### 已确认决策

在线任务执行入口应当收敛为一个。Baseline、带 Memory 的执行、训练任务执行和
测试任务执行不再分别暴露独立入口，而是由同一个执行入口通过显式参数控制。

统一入口至少包含以下两个相互独立的控制维度：

1. **Memory 开关**
   - 启用 Memory：执行任务时允许检索和使用 Memory。
   - 关闭 Memory：按 Baseline 模式执行，不检索或使用 Memory。
2. **运行阶段**
   - `train`：允许在任务结束后积累经验，包括写入和维护 Memory。
   - `test`：只执行任务，不积累经验；如果启用了 Memory，只能读取冻结的
     Memory Snapshot，禁止写入和维护。

两个维度组合后的语义如下：

| 阶段 | Memory | 运行语义 |
| --- | --- | --- |
| `train` | 关闭 | Baseline 训练任务执行，不检索 Memory，也不积累 Memory 经验 |
| `train` | 启用 | 检索并使用 Memory，任务结束后写入和维护经验 |
| `test` | 关闭 | 无 Memory 的测试任务执行 |
| `test` | 启用 | 使用冻结的只读 Memory Snapshot 执行测试，不写入、不维护 |

因此，Baseline 是统一执行入口的一种参数组合，而不是独立的脚本或业务流程；
训练和测试也共享相同的任务执行实现，只通过明确的阶段参数改变经验生命周期
权限。

### 入口范围

这里的“单一执行入口”指 Agent 与 Tau2 环境交互、完成任务并产生 rollout 的
在线执行入口。配置加载、任务选择、模型连接、Memory 权限和运行产物均由该
入口统一组织。

## 命令行参数管理

### 已确认技术方案

项目使用以下组合管理命令行参数：

- 只发布一个名为 `tau3` 的 CLI 程序。
- 使用 Python 标准库 `argparse` 解析命令行。
- 使用 Pydantic 模型承载类型化请求并执行跨字段业务校验。
- 使用 YAML 保存稳定的实验配置。
- 使用运行 Manifest 自动记录模型、Memory、数据和代码的 lineage。

不为本次重构引入 Typer 或其他新的 CLI 框架。`argparse` 只承担输入解析，
不得把 `argparse.Namespace` 传入核心业务层。

CLI 通过 `pyproject.toml` 发布，目标形式为：

```toml
[project.scripts]
tau3 = "tau3_evolver.cli:main"
```

### 命令域

同一个 `tau3` 程序包含两个操作域：

```text
tau3 run ...
tau3 slow-loop ...
```

- `tau3 run` 是唯一在线任务执行入口，统一替代独立的 Baseline、Fast Loop
  和评估执行脚本。
- `tau3 slow-loop` 是人工启动的离线数据构建、审计和权重训练入口，不属于
  在线执行流程，也不会被 `tau3 run` 自动调用。

这里的子命令只区分在线执行和离线训练两个操作域，不重新引入多套在线执行
流程。

### 在线执行参数

`tau3 run` 只暴露需要使用者作出决策的参数。目标参数按职责分为：

| 分组 | 参数 | 说明 |
| --- | --- | --- |
| 执行语义 | `--mode train\|test` | 决定是否允许积累经验 |
| Memory | `--memory / --no-memory`、`--memory-source` | 决定是否使用 Memory 及其读取来源 |
| Benchmark | `--benchmark retail\|airline` | 选择要执行的 Benchmark 任务类别 |
| 调试 | `--debug` | 运行一个可并发的小任务子集；train 经验写入独立 Debug Memory |
| 测试输入 | `--memory-snapshot`、`--checkpoint` | 测试时使用的冻结产物 |
| 基础配置 | `--config`、少量 `--set` | 加载配置并执行显式覆盖 |
| 运行输出 | `--run-id`、`--output-root` | 标识和保存本次执行 |

预期调用形式如下：

```text
tau3 run --benchmark retail --mode train --memory \
  --config configs/default.yaml

tau3 run --benchmark airline --mode test --memory \
  --memory-source retail \
  --memory-snapshot S1 \
  --checkpoint checkpoints/opd \
  --config configs/default.yaml

tau3 run --benchmark retail --mode test --debug --no-memory \
  --run-id retail-debug \
  --config configs/default.yaml

tau3 run --benchmark retail --mode train --debug --memory \
  --run-id retail-debug-train \
  --config configs/default.yaml
```

### Benchmark 与任务选择

执行入口不再接受 `task_id` 作为任务编号标记，也不再暴露 `--task-id` 或
`--all-tasks` 参数。使用者只指定要运行的 Benchmark 任务类别：

```text
--benchmark retail
--benchmark airline
```

`benchmark` 与 `mode` 共同决定实际任务集合：

| Benchmark | Mode | 实际任务集合 |
| --- | --- | --- |
| `retail` | `train` | Retail 官方 train 任务 |
| `retail` | `test` | Retail 官方 test 任务 |
| `airline` | `train` | Airline 官方 train 任务 |
| `airline` | `test` | Airline 官方 test 任务 |

任务目录、任务 ID 集合和执行顺序由对应 Benchmark 的任务目录组件统一解析，
不由 CLI 调用者逐项传递。单个任务 ID 只作为 Benchmark 内部数据和运行产物中
的任务身份存在，不再构成公开执行入口的参数。

为开发期连通性和并发验证，统一入口额外提供 `--debug` 开关，但不重新开放按
任务 ID 选择。Debug 按当前 mode 对应官方 split 的稳定顺序取前
`execution.max_concurrency` 个任务（默认 3 个），并仍通过同一个 `run_domain`
批量执行路径运行。`execution.max_concurrency` 小于 2 时拒绝启动，以保证该模式
确实覆盖并发调度。

Debug 可以搭配 `train` 或 `test`。搭配 `train --memory` 时允许积累经验，但默认
Memory namespace 改为 `<benchmark>-debug`，例如 `retail-debug` 和
`airline-debug`，读写均与正式 Memory 隔离。如果显式读取其他 namespace，仍需
提供冻结 Snapshot。搭配 `test` 时继续遵守测试只读规则。

Debug Run 不代表完整 Benchmark 指标，也不能作为 Slow Loop 数据源；`run.json`
的 `execution.task_scope` 记录为 `debug`，并记录实际 Memory 读写 namespace。
未指定 `--debug` 时仍执行完整 split，并记录为 `full`。

`benchmark` 同时是任务类型和任务执行环境的运行时解析键。执行入口收到该
参数后，必须通过统一的 Benchmark Registry 得到完整的 Benchmark 定义，而不
只是得到一组任务 ID。

Benchmark 接入采用“静态定义 + 准备后运行对象”的两层结构，避免把注册信息、
外部依赖导入结果和本次运行状态混入同一个对象。

#### BenchmarkDefinition：静态定义

`BenchmarkDefinition` 是注册到 Benchmark Registry 的不可变静态定义。注册
定义本身不得提前导入重量级 Benchmark 依赖，也不保存某次运行产生的状态。

每个静态定义至少提供：

```text
benchmark name
task type / task schema loader
task catalog loader
train / test split resolver
environment adapter / environment factory builder
default Memory namespace
```

如果评估或 Slow Loop 存在 Benchmark 专属语义，静态定义还可以提供 evaluator
binder 和 task group resolver，避免这些领域判断重新散落到通用流程中。

Tau2 的自然语言断言评估属于上述 evaluator binder：`Tau2BenchmarkDefinition`
将绑定器交给 `PreparedBenchmark`，通用执行器在 `run_domain` 启动前把项目的
`evaluation.nl_assertions` 配置绑定到当前 Tau2 Runtime。这样任务主体、用户模拟器
和终局断言评估统一受项目配置管理，不会回退到 Tau2 内置的 OpenAI 默认模型。

静态定义不包含：

```text
mode
Memory 开关或 Snapshot
Checkpoint
Run ID
具体任务 ID
已创建的环境实例
OPD 训练状态
```

#### PreparedBenchmark：准备后的运行对象

`PreparedBenchmark` 由静态定义结合本次配置和 `mode` 解析生成，保存本次运行
真正需要使用的、已经导入和解析完成的对象：

```text
benchmark name
imported task type / task schema
resolved task catalog and split
environment factory
resolved external runtime origin
evaluator binding（如适用）
default Memory namespace and task group
```

目标边界如下：

```text
BenchmarkRegistry.resolve(name)
  -> BenchmarkDefinition
  -> definition.prepare(config, mode)
  -> PreparedBenchmark
  -> 通用任务执行器
```

通用任务执行器只依赖 `PreparedBenchmark`，不再自行导入 Benchmark 包、定位任务
文件或判断 Retail/Airline。

#### Retail 与 Airline 共用 Tau2 基础协议

Retail 和 Airline 使用同一套 Tau2 基础协议，不为两个领域分别维护独立的任务、
环境或 Agent 执行框架。

共用能力包括：

```text
Tau2 Runtime binding
Tau2 Task base type
Tau2 Registry task / split / environment resolution
run_domain batch execution
Tau3 Agent Factory
Tool schema projection and Tool Memory injection
Message / Simulation / Evaluation result normalization
Batch Snapshot and experience commit protocol
Event and Manifest schema
```

目标定义关系为：

```text
BenchmarkDefinition
  -> Tau2BenchmarkDefinition（共用实现）
      -> retail（静态注册实例）
      -> airline（静态注册实例）
```

`retail` 和 `airline` 只提供不同的注册值和领域资源，不实现两套执行器。准备
阶段从首次绑定的 Tau2 Registry 中统一解析：

```text
registry.get_tasks_loader(benchmark)
registry.get_task_splits_loader(benchmark)
registry.get_env_constructor(benchmark)
```

通用层把 Tau2 Task 当作不透明任务对象，只依赖内部稳定身份，不解析订单、航班
或其他领域字段。Domain Policy、数据库、原生工具、Memory Namespace、Task Group
和领域配置保持隔离。

共用的是执行协议，不是领域语义。禁止为了共用而统一 Retail/Airline 工具名称、
参数签名、Policy、数据库模型或 Memory 内容。

目标解析关系如下：

```text
--benchmark retail
  -> Retail task type
  -> Retail task catalog
  -> Retail train/test split
  -> Retail execution environment

--benchmark airline
  -> Airline task type
  -> Airline task catalog
  -> Airline train/test split
  -> Airline execution environment
```

统一入口只依赖准备完成的 `PreparedBenchmark`。任务加载和执行主流程中不得分散编写
`if benchmark == "retail"` 或 `if benchmark == "airline"` 等领域分支。新增
Benchmark 时，应通过注册新的 Benchmark 定义接入，不修改通用执行流程。

#### 外部 Benchmark 版本策略

项目不再要求 `tau2-bench` 固定到仓库预置的 commit 或版本，也不把 pin 文件
匹配作为生产执行的前置条件。

本次运行第一次成功导入的 `tau2-bench` 是该运行的权威 Tau2 Runtime：

- 首次导入时解析并缓存实际的包来源路径。
- 同一次运行中的后续 Retail、Airline 或其他 Tau2 Benchmark 复用这份已绑定
  Runtime，不切换到另一份 checkout 或安装来源。
- 如果能够读取 commit、package version 等信息，则将其作为 provenance 写入
  `PreparedBenchmark` 和运行 Manifest，但只用于复现和审计，不作为版本门禁。
- 新的一次独立运行可以根据当时首次成功导入的 `tau2-bench` 建立新的 Runtime
  绑定，不需要与历史运行版本一致。

无法识别的 Benchmark、任务类型正常导入失败或执行环境无法创建时，由
`prepare()` 或环境创建过程报告正常运行错误；生产主流程不额外执行独立的
Benchmark Preflight。

#### Benchmark Preflight 的位置

Benchmark Preflight 只在 `tests/` 目录下管理，例如：

```text
tests/benchmarks/
  test_registry.py
  test_retail.py
  test_airline.py
```

这些测试负责验证：

- Benchmark 是否正确注册；
- 任务类型和任务 Schema 是否能够导入；
- 任务目录及 train/test Split 是否能够解析；
- 环境工厂是否满足公共环境协议；
- 必要时执行最小环境 smoke test。

Preflight 测试不属于生产包，不被 `tau3 run` 调用，不参与生产入口的控制流，
其测试结果也不改变主流程的参数和行为。

### Benchmark 执行后端改为 run_domain

#### 已确认决策

生产任务执行不再通过 Tau2 Gym Adapter 逐任务驱动，统一改为使用 Tau2 官方
`run_domain` 批量执行接口。

目标主流程为：

```text
PreparedBenchmark
  -> Tau2 run_domain
  -> 为每个任务创建独立 Environment
  -> Environment 提供原生 Tools 和 Domain Policy
  -> Tau3 Agent Factory 创建独立 Agent
  -> 并发完成任务
  -> 汇总轨迹、结果和经验变更
```

`PreparedBenchmark` 负责提供当前 Benchmark 对应的 `run_domain` Runtime、任务
Split 和 Agent 注册能力。通用执行器不再直接构造 `Tau2RetailEnv` 或其他 Gym
环境适配器。

Gym 接入不再属于生产主流程。现有 Gym Adapter 是否在迁移期保留为测试工具，
在制定实施计划时单独确认；它不能继续与 `run_domain` 形成两个并列的生产执行
后端。

#### 自定义 Tau3 Agent Factory

`run_domain` 必须使用注册到 Tau2 Registry 的自定义 Tau3 Agent Factory，不能
直接使用不具备 Memory 生命周期能力的 Tau2 内置 Agent。

Agent Factory 接收当前任务环境提供的原生工具、Domain Policy 和任务对象，
并为每个并发任务创建独立的 Agent 与 Agent State：

```text
environment.get_tools()
environment.get_policy()
task
  -> Tau3 Agent Factory
  -> Tau3 Agent
```

Tau3 Agent 内部继续负责：

- Memory Retrieval；
- Memory Selection；
- Action Prompt 构建；
- Qwen Tool Calling；
- Fast Loop 事件记录；
- 任务结束后的经验候选生成。

#### Tool Tier Memory 接入

Tool Tier Memory 与 `run_domain` 兼容。它通过自定义 Tau3 Agent 增强 Agent 可见
的原生工具 Schema，不修改 Tau2 Environment 的实际工具对象和执行逻辑。

```text
Tau2 原生 Tool
  -> 复制 OpenAI Tool Schema
  -> 注入 Selected Tool Memory 到 description
  -> Qwen 生成原生 ToolCall
  -> Tau2 Orchestrator 调用原始 Environment Tool
```

Tool Memory 只能引用当前 Environment 已提供的工具，并保持工具名称和参数签名
不变。它不能凭空创建可执行工具、替换工具实现或修改 Environment 的真实接口。

每个 Agent 使用私有的工具 Schema 副本，禁止修改 Tau2 原始 Tool 对象，避免
并发任务之间发生工具描述污染。

#### Memory 来源与跨领域泛化

执行 Benchmark 与 Memory 读取来源是两个独立概念。启用 Memory 时，调用者可
通过 `--memory-source <namespace>` 指定读取哪一个 Memory Namespace：

```text
tau3 run --benchmark retail --mode test --memory
  -> 默认读取 retail Memory

tau3 run --benchmark airline --mode test --memory --memory-source retail
  -> 执行 Airline 测试任务
  -> 读取 Retail Memory
  -> 观测跨领域泛化指标
```

如果省略 `--memory-source`，使用当前 Benchmark 定义的默认 Memory Namespace，
通常与 Benchmark 同名：

```text
resolved_memory_source = request.memory_source or benchmark.default_memory_namespace
```

`--memory-source` 选择逻辑命名空间，`--memory-snapshot` 在该命名空间内选择确切
的冻结版本。Test Mode 启用 Memory 时仍必须指定或解析出冻结 Snapshot。

跨领域 Memory 使用遵循以下隔离规则：

- 来源 Namespace 与执行 Benchmark 不同时，来源 Memory 始终以只读方式打开。
- 跨领域运行不得修改来源 Memory 的正文、状态、使用计数或 Snapshot。
- Test Mode 无论来源是否跨领域，都禁止任何 Memory 写入和 Maintenance。
- Train Mode 产生的新经验只写入当前执行 Benchmark 的默认 Memory Namespace，
  不写入跨领域来源 Namespace。
- 事件和 Manifest 必须同时记录执行 Benchmark、Memory Source Namespace、输入
  Snapshot ID，以及是否为跨领域组合。

跨领域来源中的 Tool Tier Memory 必须先与目标 Benchmark 当前 Environment 的
工具 Schema 做兼容检查：

- 目标环境不存在同名工具时，该 Tool Memory 不进入 Selector 候选集。
- 同名工具的参数 Schema 不兼容时，该 Tool Memory 不进入 Selector 候选集。
- 兼容检查只针对本次运行的 Agent 私有 Schema，不修改来源 Memory。
- 被过滤的 Memory ID、来源工具、过滤原因和数量写入运行事件与评估报告。
- Tip、Skill 和 Trajectory 等非 Tool Tier 仍可参与跨领域检索与选择，其迁移
  效果由泛化实验指标反映。

跨领域评估报告至少将以下字段作为独立实验维度：

```text
execution_benchmark
memory_source_namespace
memory_snapshot_id
cross_domain_memory
retrieved / selected counts by tier
incompatible_tool_memory_count
```

不同 `execution_benchmark + memory_source_namespace + memory_snapshot_id` 组合不得
在报告中被静默合并。典型的 Airline 泛化对照包括：

```text
Airline + no Memory
Airline + Airline Memory
Airline + Retail Memory
```

因此，Benchmark 的默认 Memory Namespace 属于静态定义，而本次运行选择的
`memory_source` 属于 `ExecutionRequest`。两者不能合并成同一个字段。

#### 并发 Memory 语义

`run_domain` 批量并发后，Memory 可见性从逐任务更新改为批次快照语义：

```text
Batch 开始：确定输入 Memory Snapshot S0
  -> Batch 内所有任务只读取 S0
  -> 各任务独立检索、选择和使用 Memory
  -> 各任务产生待提交的经验变更
Batch 结束：统一校验并提交经验，形成 S1
下一 Batch：读取 S1
```

同一 Batch 中的任务不能读取其他并发任务刚产生的经验。Memory 写入、冲突处理、
Snapshot 发布和 Maintenance 只在 Batch 边界执行。测试模式始终读取冻结 Snapshot，
Batch 结束后也不得提交经验。

并发度属于 YAML 中的稳定运行配置，不作为区分业务流程的新入口参数。

#### run_domain 路径测试要求

`tests/` 下需要覆盖以下行为，但这些测试不进入生产主流程：

- 自定义 Agent Factory 能收到每个任务环境的 Tools、Policy 和 Task；
- 不同并发任务具有独立 Agent State 和工具 Schema；
- Selected Tool Memory 只注入匹配工具的 Agent 私有 Schema；
- 增强后的工具调用仍由 Tau2 原始工具执行；
- 同一 Batch 读取相同输入 Snapshot；
- 经验只在 Batch 成功结束后提交；
- Test Mode 不产生任何 Memory 写入或 Maintenance。

以下信息不再作为在线执行入口要求使用者维护的参数：

- `iteration`；
- `parent_iteration_dir`；
- `completed_train_tasks_before`；
- `stop_after`；
- 可由 checkpoint、配置或 Manifest 确定的模型与 Adapter lineage 字段。

这些信息如果仍有保留价值，应由运行产物、输入 checkpoint 和 Manifest 自动
推导并记录，不得继续构成在线执行入口的编排负担。

### 移除 iteration 概念

`iteration` 不再作为 OPD 训练的标记，也不再作为新架构中的业务参数。

已确认的调整包括：

- 从 `tau3 run` 的命令行参数中移除 `iteration`。
- 从 `tau3 slow-loop` 的命令行参数中移除 `iteration`。
- 从新的 `ExecutionRequest` 和 Slow Loop 请求模型中移除 `iteration`。
- 不使用 `iteration` 决定任务采样、Memory 生命周期、数据构建范围、Slow Loop
  启动时机或训练输出位置。
- 不再通过递增的 iteration 序号表达 OPD 训练批次及其先后关系。

OPD 训练的身份和来源改由显式产物标识表达：

```text
source run IDs
  -> Memory Snapshot ID
  -> dataset build ID
  -> checkpoint / adapter revision
  -> Manifest lineage
```

手动启动 Slow Loop 时，使用者显式选择输入 Source Runs 或已经构建并审计通过的
Dataset。训练输出通过唯一的 run、dataset 和 checkpoint 标识追踪，不依赖
`iteration` 参数。

### 类型化请求边界

CLI 解析结果必须先转换为稳定的类型化请求，再进入核心业务层。目标调用边界
如下：

```text
命令行参数
  -> argparse 解析
  -> Pydantic ExecutionRequest 校验
  -> execute(request)
```

`ExecutionRequest` 至少表达以下领域信息：

```text
mode
memory_enabled
memory_source
benchmark
memory_snapshot
checkpoint
run_id
output_root
```

核心执行逻辑只依赖该请求模型，不依赖命令行解析器。这使同一执行逻辑可以被
CLI、测试或其他 Python 调用方复用。

### 权限模型与组合校验

`mode` 和 Memory 开关的组合校验集中在请求模型或紧邻请求模型的领域校验层，
不得散落在脚本、runner、Memory Repository 和评估 guard 中。

校验完成后生成明确的执行权限：

```text
can_read_memory
can_write_memory
can_run_maintenance
can_use_train_split
can_use_test_split
```

组合约束如下：

| 模式 | 必须满足的约束 |
| --- | --- |
| `train + memory` | 允许读取、写入和维护训练 Memory |
| `train + no-memory` | 禁止读取和写入 Memory |
| `test + memory` | 从指定或默认 Namespace 读取冻结 Snapshot；只读；禁止写入和维护 |
| `test + no-memory` | 禁止传入 Memory Snapshot；禁止写入和维护 |

底层组件检查已经解析完成的权限，不重复解释原始 CLI 参数。

### 配置来源与优先级

所有配置来源使用固定优先级：

```text
命令行显式参数
  > 环境变量
  > YAML 配置文件
  > 程序默认值
```

环境变量只用于部署环境或敏感信息，例如模型服务地址和 API Key。Benchmark、
运行阶段、Memory 开关、Memory Source、Snapshot 和 Checkpoint 等实验语义必须
显式出现在命令或 YAML 配置中，不能由隐藏环境变量改变。

`tau3 run` 在解析 Benchmark、创建 run 目录或启动任务前，固定加载项目根目录的
`.env`。已经存在的进程环境变量优先，`.env` 不覆盖它们。用户模拟器与 NL 断言
评估器分别通过 `tau2.user_api_key_env` 和
`evaluation.nl_assertions.api_key_env` 声明所需凭证；入口统一执行 fail-fast
预检，缺失或空值时直接终止。密钥值不得进入 YAML、运行制品、日志或 Git，仓库只
提交空值模板 `.env.example`。

Qwen vLLM 服务统一通过 `python -m scripts.serve_qwen_vllm` 启动。该启动器从项目
配置读取服务地址、served model name 和 `model.max_context_tokens`，并固定当前
Qwen3.5-9B 上下文为 `131072` tokens；该字段使用静态类型约束，不能通过 `--set`
缩小。运行 Manifest 同步记录这一配置，避免 Memory Selector 首次读取长 trajectory
时因服务仍以 32K 上下文启动而失败。模型实际权重路径仍由部署时的
`--model-path` 指定，不允许启动器隐式选择训练 checkpoint。

每次运行都必须把合并、校验后的最终配置写入运行目录，并在 Manifest 中记录
配置来源、输入产物和关键 revision，保证实验可复现。

## 运行制品契约

### 正式制品收敛为两个文件

每次在线任务执行只发布两个正式制品：

```text
runs/<run-id>/
├── run.json
└── episodes.jsonl
```

`run.json` 是唯一的运行级记录，负责保存本次运行的输入、解析后的配置、
Benchmark 与 Runtime lineage、模型与 Checkpoint、Memory 来源、输入与输出
Snapshot、自动 Maintenance 的压缩记录、完成状态、任务数量、聚合指标以及
`episodes.jsonl` 的 SHA-256。Maintenance 记录只保存维护轮次、触发任务计数、维护前
Snapshot、受检公开诊断、规范化命令和提交结果，不重新发布完整生命周期事件。
它不保存逐任务轨迹、逐任务失败详情或完整 Tau2 Simulation。

`episodes.jsonl` 是唯一的任务级记录，每个 Benchmark 任务恰好占一行。成功行保存
标准化轨迹、终局评估、Memory 检索与选择证据、经验提议及其提交结果、Token 与
延迟；失败行保存失败阶段、错误类型和有限诊断。每行不重复 `run.json` 已经记录的
Benchmark、Mode、模型、Checkpoint、Memory Namespace 和 Snapshot 等运行级字段。

以下文件和目录不再作为正式运行制品：

```text
manifest.json
results.json
evaluation.json
events.jsonl
tau2/
cache/memory/
```

原 `events.jsonl` 中对训练和审计有价值的事件不会丢失，而是按任务聚合为一个稳定的
Episode Schema。Tau2 原始完整 Simulation 不再重复落盘，也不再嵌入 Episode；主流程
只保留后续评估、审计和训练所需的标准化字段。Tau2 `save_to` 如为 Runtime 调用所需，
只能指向运行期临时位置，执行结束后不得发布到 Run 目录。

### 字段唯一归属

- 运行配置、输入选择和 lineage 只属于 `run.json`。
- 逐任务状态、轨迹、评估和 Memory 生命周期证据只属于 `episodes.jsonl`。
- 失败详情只属于对应的失败 Episode；`run.json` 只保存失败数量和整体状态。
- 输入与输出 Memory Snapshot ID 只属于 `run.json`；Episode 通过所属 Run 继承。
- Memory Candidate 详情只记录一次；Selection 只引用被选择的 Memory ID。
- Memory Proposal 与提交结果合并记录，每个 Proposal 明确标记为新建、重放或丢弃。
- 聚合指标保存在 `run.json` 的摘要中；它们是从 Episode 派生的运行级视图，不是第二份
  任务结果。
- `run.json` 记录 `episodes.jsonl` 的字节数、行数和 SHA-256，使 Slow Loop 能在消费前
  验证任务证据未被修改。

### Slow Loop 同步切换

Slow Loop 不再读取旧的 `manifest.json + results.json + events.jsonl` 三文件组合。
Source Run Loader 只接受 `run.json + episodes.jsonl`，直接把每个 Episode 行转换为
`EpisodeEvidence`，不再按 `EpisodeStarted`、`DecisionMade`、`EnvironmentStepped`、
`EpisodeFinished` 等事件重新拼装任务。

Dataset Manifest 中的 Source Run lineage 使用 `run_sha256` 和 `episodes_sha256`，移除
`manifest_sha256`、`results_sha256`、`events_sha256` 和事件行范围。旧三文件协议不在
生产主流程中保留兼容分支；历史数据如需继续使用，应通过主流程之外的一次性迁移工具
转换。

### Audit 是发布门禁

Audit 是校验动作，不是一层新的训练数据。目标流程为：

```text
run.json + episodes.jsonl
        -> 在临时目录构建 OPD Dataset
        -> Audit
             -> 通过：原子发布 Dataset，允许 Slow Loop Training
             -> 失败：终止构建并丢弃临时 Dataset
```

Audit 至少验证：Source Run 是否全部来自官方 Train Split；Run、模型、Checkpoint、
Memory Snapshot lineage 是否一致；制品哈希、Schema、任务覆盖和计数是否正确；Memory
检索、选择和写入证据能否与 Snapshot 对应；Attribution 与 OPD Examples 能否重算；
以及是否混入凭证、测试答案或其他禁止训练的数据。

Source Run 可保留终局 evaluator 的诊断结果，但 Slow Loop Evidence 只继承可训练的
`final_reward` 与公开轨迹；`nl_assertions`、rubric、golden action 等 evaluator 细节
不得进入 Dataset，即使它们只准备提供给 Teacher 也不允许。

通过结果不再单独发布 `audit_report.json`。Dataset Manifest 只记录审计契约版本和
`passed: true`；失败详情直接返回调用者，并且失败的数据集不发布。每次开始训练前仍
必须重新执行 Audit，不能仅信任已保存的通过标记。

## Slow Loop 启动方式

Slow Loop 属于离线训练操作，不由在线任务执行入口自动触发。

已确认的启动原则如下：

- Slow Loop 训练由使用者显式、手动启动。
- `run_iteration` 不再负责启动 Slow Loop 训练。
- 不在 `run_iteration` 中根据已完成任务数、迭代次数或其他计数条件自动触发
  Slow Loop。
- 在线执行阶段只负责产生并持久化后续 Slow Loop 所需的训练轨迹、Memory
  变化和来源信息。
- 使用者在确认数据范围和质量后，再单独启动数据构建、审计与 Slow Loop
  训练。

这里的手动启动边界只约束离线的 Dataset/Audit/LoRA 训练，不等同于 Memory
Maintenance 的触发方式。`memory.maintenance_period` 表示在线 train 任务累计到一定
数量时，对 Memory 仓库执行一次整理并记录 Maintainer 证据；它不负责启动 Slow
Loop，也不能触发权重训练。

触发边界确定如下：Memory Maintenance 属于 Fast Loop，必须在启用 Memory 的 train
执行中按 `memory.maintenance_period` 自动触发，持续整理经验并留下 Maintainer 证据；
Slow Loop 始终由管理员手动启动。管理员根据经验积累程度决定何时构建 Dataset、执行
Audit，并将经验内化为模型权重，Fast Loop 的维护周期不得自动启动任何 Slow Loop
操作。

当前实现已经在成功的 train+memory Batch 边界重新接入 `run_due_maintenance`：经验
候选先统一提交，随后按仓库累计完成任务数判断维护周期，并按轮次顺序执行当前边界
前所有尚未完成的 Maintenance，最后再发布输出 Snapshot。维护记录压缩写入
`run.json`，可由手动 Slow Loop 校验并转换为 Maintainer Evidence；test、关闭
Memory 或任务 Batch 失败时不执行 Maintenance。
该自动化只属于 Fast Loop，不为 Slow Loop 增加周期计数或后台调度器。

### Slow Loop 调试模式

开发期允许显式使用 `tau3 slow-loop build --debug` 消费 `debug train` 制品，但必须
保持以下隔离：

- 只有显式 `--debug` 才接受 `execution.task_scope=debug` 的 Source Run；正式构建仍
  拒绝 debug 制品。
- Debug Dataset 继续引用 `<benchmark>-debug` Memory namespace，并在
  `source_context.task_scope` 中记录 `debug`，不能混入正式 Dataset。
- `tau3 slow-loop train --debug` 只能消费上述 Debug Dataset。某类样本为零时，不
  伪造训练样本；该阶段加载原始基座、创建零影响 LoRA 并发布
  `step-00000000` checkpoint，明确记录 `debug_initialized_without_examples=true`。
- 有真实样本的阶段仍执行正常的生成、OPD loss、反向传播和优化步骤。Debug Bundle
  只用于验证数据、审计、模型加载、训练编排与四类 checkpoint 发布链路，不得作为
  正式训练产物或评测模型使用。

这一边界将在线经验采集与离线权重更新解耦：在线入口决定任务如何执行以及
是否积累 Memory 经验，手动 Slow Loop 决定何时将已积累的数据内化到模型
权重中。
