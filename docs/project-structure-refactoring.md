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
default Memory namespace
task group
train / test split definition
online credential requirements
prepare(config, mode) -> PreparedBenchmark
```

`BenchmarkDefinition` 是 Benchmark 注册信息的唯一事实来源。默认 Registry、CLI
的合法选项、请求解析后的定义查找和运行前凭据需求都由静态定义集合派生。除定义
文件外，生产代码不得再次声明 Benchmark 名称列表，也不得通过枚举、`Literal` 或
`if benchmark == ...` 复制注册信息。

如果评估、任务解析或运行时存在 Benchmark 专属语义，应由静态定义创建对应的
`BenchmarkExecutor`，避免这些领域判断重新散落到通用流程中。Tau2 的自然语言
断言绑定由 `Tau2BenchmarkExecutor` 在 `run_domain` 启动前完成；通用执行层既不
接触 Tau2 Runtime，也不感知断言实现。

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

`PreparedBenchmark` 由静态定义结合本次配置和 `mode` 解析生成。它只公开通用
运行元数据和一个遵守公共协议的不透明执行器，不公开 Benchmark 的加载、注册、
环境创建和执行细节：

```text
benchmark name
resolved task IDs
split name / split hash
resolved external runtime origin
default Memory namespace and task group
executor: BenchmarkExecutor
```

目标边界如下：

```text
BenchmarkRegistry.resolve(name)
  -> BenchmarkDefinition
  -> definition.prepare(config, mode)
  -> PreparedBenchmark
       -> executor.execute(BenchmarkExecutionRequest)
       -> BenchmarkExecutionResult
```

通用执行层只依赖 `BenchmarkExecutor` 的请求和结果协议，不再自行导入 Benchmark
包、定位任务文件或判断 Retail/Airline。每一类 Benchmark 负责创建自己的执行器；
共享同一运行协议的 Retail 与 Airline 各自持有一个 `Tau2BenchmarkExecutor` 实例，
但复用同一个实现。

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
  -> BenchmarkExecutor
  -> Tau2BenchmarkExecutor
  -> Tau2 run_domain
  -> 为每个任务创建独立 Environment
  -> Environment 提供原生 Tools 和 Domain Policy
  -> Tau3 Agent Factory 创建独立 Agent
  -> 并发完成任务
  -> 汇总轨迹、结果和经验变更
```

`PreparedBenchmark` 只负责公开任务 ID、Split 身份、运行来源和通用 Executor。
`Tau2BenchmarkExecutor` 独占 `run_domain` Runtime、Agent 注册、`TextRunConfig`
构造、结果归一化和 Tau2 断言绑定。通用执行层不再直接构造 Tau2 环境适配器，
也不访问 Tau2 Registry。

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

## `src` 包边界审计

### 记录性质

本节记录 2026-08-06 对当前 `src/tau3_evolver` 实现进行的结构审计。这里首先描述
当前代码的真实依赖关系，再记录建议的目标边界。除前文已经确认的设计决策外，本节
中的目录移动、模块合并和删除候选仍属于待确认方案，不表示已经完成实现。

当前代码已经具备统一入口、Benchmark 两层定义、`run_domain` 批量执行、两文件在线
制品、自动 Memory Maintenance 和手动 Slow Loop 等主能力，但包名层面的分组尚未
形成真正的分层。依赖图中以下七个包互相可达，实际构成一个大型循环依赖区域：

```text
agent
artifacts
benchmarks
evaluation
execution
memory
models
```

这意味着这些包目前不是可以独立理解和替换的层。修改其中一个包的协议，往往会反向
影响多个名义上的下层包。后续重构的重点不应是简单增加目录，而应先确定概念的唯一
所有者，并建立单向依赖。

目标依赖方向为：

```text
CLI / Config
     |
     v
Execution Application Layer -------------------- Slow Loop
     |                                               |
     +--> Benchmark Adapter                          +--> Artifact Contracts
     +--> Fast Loop                                  +--> Fast Loop Contracts
     +--> Inference                                  +--> Memory Schema
     +--> Memory                                     +--> Offline Modeling
     +--> Artifact Projection
```

底层 Memory、Artifact Schema 和 Fast Loop Contract 不得反向依赖执行入口或具体
Tau2 Runtime。Slow Loop 可以消费在线制品和 Fast Loop 协议，但在线 Fast Loop 不得
依赖 Slow Loop。

### Benchmark 与 Execution 边界

`BenchmarkDefinition + PreparedBenchmark + BenchmarkExecutor` 边界已在本轮
Benchmark 重构中落地。

审计时发现并已处理的问题：

- 已删除 `execution.request.BenchmarkName`，请求使用经过安全校验的字符串。
- CLI 的 `--benchmark` choices 直接读取 `benchmark_registry.names()`。
- 默认 Registry 只从静态 `BenchmarkDefinition` 集合创建；新增 Benchmark 不再修改
  请求类型和通用执行流程。
- `PreparedBenchmark` 已收紧为通用元数据和 `BenchmarkExecutor`，不再公开 Tau2
  Runtime、Registry、环境工厂或 evaluator binder。
- Tau2 Agent Factory 注册与注销、`TextRunConfig`、`run_domain`、断言绑定、
  Simulation 收尾和结果归一化已全部迁入 `benchmarks/tau2/`。
- `execution.batch` 只消费标准化的 `BenchmarkExecutionResult`，不再导入 Tau2。

目标边界如下：

```text
BenchmarkRegistry.resolve(name)
  -> BenchmarkDefinition
  -> PreparedBenchmark
       name
       split_name / split_hash
       task_ids
       default_memory_namespace
       task_group
       runtime_origin
       executor: BenchmarkExecutor
```

`BenchmarkExecutor` 对通用执行层暴露稳定的批量执行协议。Tau2 Registry 注册、
`TextRunConfig` 构造、`run_domain` 调用、Agent Factory 注销和 Tau2 Result 归一化
全部封装在 `benchmarks/tau2/` 内部。通用 `execution` 不再访问 Tau2 Registry 或
Runtime 的具体类型和私有属性。

`ExecutionRequest.benchmark` 应使用经过安全校验的字符串，CLI 的合法 Benchmark
集合由 Registry 提供。请求模型只表达调用者选择，Benchmark 是否存在由 Registry
统一验证。

本轮之后的新增约束是：生产代码中的 Benchmark 名称只能出现在对应静态定义中；
面向用户的名称集合和运行时行为都必须从 Registry 解析，不允许建立第二份映射表。

### Fast Loop 领域包（已建立）

原顶层 `agent/` 已取消，Benchmark 无关的在线经验循环正式收敛为：

```text
fast_loop/
  contracts.py
  settings.py
  prompts.py
  selection.py
  decision.py
  writing.py
  maintenance.py
  context.py
  tools.py
```

当前边界为：

- `contracts.py` 保存 Decision、Policy Protocol、Lifecycle Response 和 Episode Result
  等在线、离线共同使用的稳定协议。
- `settings.py` 保存 Fast Loop 运行视图；`selection.py`、`decision.py`、`writing.py`
  和 `maintenance.py` 各自拥有明确的服务边界。
- `prompts.py` 是在线执行与 Slow Loop 数据共同使用的 Prompt 契约。
- `context.py` 通过窄协议隔离 ExecutionContext，`tools.py` 提供通用 Tool Schema
  复制和规范化。
- Slow Loop 单向依赖 Fast Loop 的 Contract 和 Prompt，保证训练数据与在线推理使用
  同一协议。
- Fast Loop 可以调用 Memory Repository 和 Operations，但 Memory 不得反向导入
  Fast Loop。

### Fast Loop 与 Tau2 Adapter 的边界

当前目录关系为：

```text
fast_loop/（Benchmark 无关）
  contracts.py
  settings.py
  context.py
  prompts.py
  selection.py
  decision.py
  writing.py
  maintenance.py
  tools.py

benchmarks/tau2/（Tau2 专属）
  agent.py
  action_codec.py
  actions.py
  agent_state.py
  episodes.py
  tool_schemas.py
  executor.py
  assertions.py
```

Tau2 Half Duplex Agent、Message、ToolCall、Agent State、Simulation、Action Codec 和
Registry 私有 API 现在只允许出现在 `benchmarks/tau2/`。在线 Fast Loop Policy
只负责生成通用 `ActionDecision`；Tau2 工具名、
参数格式和停止动作的校验由 Tau2 Agent Adapter 完成，并继续使用统一的修复流程。

Tau2 Agent 与 Episode Adapter 通过 `fast_loop.selection`、`fast_loop.decision` 和
`fast_loop.writing` 的公开函数复用公共流程，不再跨包导入以下划线开头的私有实现。
项目中不再保留顶层 `agent/` 兼容包，防止新代码继续写入旧边界。

`actions.py` 与 `action_codec.py` 还分别实现 Tool Action 的解析和校验。后续应由一个
结构化 Action Codec 同时完成清理、解析、工具名校验和 Tau2 Message 转换，避免两套
解析规则发生偏差。

### Memory 包的目标纯度

Memory 包只负责 Memory 数据模型、仓储、检索和原子领域操作。本轮已经完成执行期
职责与通用基础设施的迁移；Memory Schema 的内部拆分仍作为后续独立重构处理。

#### 执行期 Memory 解析

原 `memory/snapshots.py` 处理的是“本次执行选择哪个 Memory Namespace 和 Snapshot”，
而不是 Snapshot 数据结构本身，现已移动为：

```text
execution/memory_resolution.py
```

Memory 包只提供创建、打开和校验 Snapshot 的原语。Memory generation、已提交批次、
已完成任务数等运行进度也已从 `memory/batches.py` 移入
`execution/memory_state.py`，原文件不再保留。

#### 自动 Maintenance

自动 Maintenance 的编排、Prompt、模型决策、修复和事件记录已经移动到
`fast_loop/maintenance.py`。Episode 是否允许生成 Memory 的判定也已经移动到
`fast_loop/outcomes.py`。Memory 只提供 Lookup、Merge、Delete、Repository 和检索资格
等领域能力。

#### Memory Schema 循环

`memory/types.py` 在 Pydantic Validator 中反向导入 `memory/tier_contracts.py`，而
`tier_contracts.py` 又导入 `MemoryTier`。建议整理为：

```text
memory/schema/
  base.py
  tiers.py
  item.py
```

- `base.py`：MemoryTier、MemoryStatus 和稳定 ID。
- `tiers.py`：Tip、Skill、Tool、Trajectory Payload Contract。
- `item.py`：MemoryItem、MemorySnapshot 及组合校验。

#### 通用持久化能力

原实现中 `memory/json_store.py` 从 `artifacts/jsonl.py` 导入私有
`_fsync_directory`，而 `artifacts/run.py` 又从 `memory/json_store.py` 导入原子写入
函数，形成双向依赖。

原子写入、目录 fsync、文件锁和 JSONL 读写现已统一归属：

```text
persistence/
  atomic.py
  jsonl.py
  locking.py
  layout.py
  embedding_cache.py
```

Memory 与 Artifacts 不再相互导入。项目路径布局从 `memory/paths.py` 移入
`persistence/layout.py`；Embedding 缓存移入 `persistence/embedding_cache.py`；具体
Qwen Embedding 实现和配置工厂移入 `models/embeddings.py`。Memory 只保留
`EmbeddingProvider` 与向量校验契约。不建立没有明确边界的 `common.py` 或 `utils.py`。

### Artifact Contract 与投影边界

原 `artifacts/episodes.py` 直接导入 `fast_loop.contracts.EpisodeResult` 和
`execution.results.BatchFailure`。本轮已建立 `artifacts/contracts.py`，由 Artifacts
定义 `CompletedEpisodeProjection` 和 `FailedEpisodeProjection`，Execution 负责把内部
结果显式映射成投影输入。Artifacts 不再导入 Execution、Fast Loop 或 Memory。

Run、Episode 和 Maintenance 制品输出目前仍主要通过 `dict[str, Any]` 表达。Slow Loop
因此需要在 `source_runs.py`、`evidence.py` 和 `audit.py` 中重复手动校验字段。

后续类型化输出的目标结构仍为：

```text
artifacts/
  schemas.py
  readers.py
  writers.py
  hashing.py
```

Artifact 包拥有带 Schema Version 的不可变类型，例如 `RunArtifact`、
`EpisodeArtifact` 和 `MaintenanceArtifact`，但不导入 Agent 或 Execution 的具体结果
类型。执行层负责把内部结果投影为 Artifact Schema，再交给 Writer 发布。

Credential 和 URL 脱敏不属于制品投影专属能力，原 `artifacts/sanitize.py` 已移动为
`security/redaction.py`。Benchmark、Fast Loop 和 Artifacts 共同依赖该公共安全边界。

`execution/events.py` 当前只保存运行期间的内存生命周期事件，不再发布旧的
`events.jsonl`。为避免它与旧正式制品混淆，建议改名为 `execution/trace.py`，并明确
它只是构建 `run.json + episodes.jsonl` 的内部临时证据。

Slow Loop Dataset 当前的 `evidence/episodes.jsonl` 同时包含 Episode Evidence 和
Maintenance Evidence，文件名不能准确表达内容。后续应选择以下一种形式：

```text
evidence/episodes.jsonl
evidence/maintenance.jsonl
```

或统一改名为 `evidence/records.jsonl`。在正式迁移前需先确认是分文件还是统一记录流。

### 在线推理与离线模型边界

当前 `models/` 混合三类职责：

```text
openai_compatible.py  在线 HTTP Client 和 Fast Loop Policy
policy.py             旧通用 Policy 接口
qwen35.py             离线模型和 Tokenizer 加载
lora.py               LoRA 构造、校验和 Checkpoint 发布
```

`models/openai_compatible.py` 当前仍同时包含 HTTP Transport、
`OpenAICompatibleFastLoopPolicy`、OpenAI-compatible ToolCall 解析和多个输出规范化
函数，边界仍然偏宽。旧 `OpenAICompatibleQwenPolicy` 已移到
`benchmarks/tau2/baseline_policy.py`，通用模型模块不再导入 Tau2。

建议拆分为：

```text
inference/
  openai_client.py
  fast_loop_policy.py

modeling/
  qwen35.py
  lora.py
```

旧 `models.policy.Policy`、`DecisionRequest`、`DecisionResponse` 和 Tau2 下的
`OpenAICompatibleQwenPolicy` 当前没有接入在线主链路，主要由兼容测试使用。它们属于
删除候选，但需要先确认是否仍承担外部 Python API 兼容责任。

当前在线入口的 `--checkpoint` 只写入 Run Lineage 和内部 Trace，没有参与 Endpoint、
Served Model 或 Adapter 的实际选择。因此传入该参数不会切换在线权重。后续必须二选一：

- 将它接入明确的在线 Adapter/Model 选择协议；或
- 从 `tau3 run` 删除该参数，改由真实模型服务配置记录 Model Revision。

在该语义明确前，不应把 `--checkpoint` 描述为能够加载权重的执行参数。

### Slow Loop 内部结构

当前 `slow_loop/` 同时包含数据构建、独立 Audit、单 Adapter 训练和四 Adapter Suite，
总代码规模已经明显超过其他单一包。建议拆为两个子域：

```text
slow_loop/
  data/
    source_runs.py
    evidence.py
    attribution.py
    examples.py
    audit.py
    builder.py

  training/
    suite.py
    worker.py
    trainer.py
    alignment.py
    loss.py
    opd_step.py
```

当前调用关系是：

```text
slow_loop.runner
  -> training_suite
      -> subprocess: train_command
          -> training.OPDTrainer
```

四个 LoRA 使用独立子进程有利于分阶段释放模型和显存，该进程隔离可以保留；但
`train_command` 实际是内部 Worker，`training` 实际是 Trainer，`training_suite` 才是
公开的 `tau3 slow-loop train` 服务。重命名后应明确只有 Suite 是用户入口。

Audit 中部分哈希、规范化和重算逻辑与 Dataset Builder 重复，是为了避免生产器和
校验器共享同一个错误实现，不应为了形式上的 DRY 全部合并。只抽取纯 I/O 原语，
关键审计判断保持独立。

### Evaluation 的职责拆分

当前 `evaluation/` 中三个文件属于不同层：

| 当前文件 | 实际职责 | 建议归属 |
| --- | --- | --- |
| `metrics.py` | BatchResult 指标汇总 | `reporting/metrics.py` |
| `comparisons.py` | 多个 Run 报告比较 | `reporting/comparisons.py` |
| `tau2_nl_assertions.py` | Tau2 Runtime 断言绑定 | `benchmarks/tau2/assertions.py` |

`comparisons.py` 当前只被测试调用，没有接入执行主链路；它是可复用报告能力，不是在线
Evaluation 阶段。拆分完成后可以取消含义过宽的顶层 `evaluation/` 包。

### CLI 与配置的重复边界

当前只有一个发布命令 `tau3`，但参数解析分散在：

```text
cli.py
slow_loop/runner.py
slow_loop/training_suite.py
slow_loop/train_command.py
```

`cli.py` 对 Slow Loop 只解析 `action + REMAINDER`，再由下层模块重新解析参数。后续应
选择统一方案：由顶层 CLI 构建完整 Subparser，或由各命令模块向顶层注册 Parser；
无论采用哪一种，解析后的 `argparse.Namespace` 都不得进入服务层，每个公开操作都应
转换为独立的 Pydantic/Dataclass Request。

`config.MemoryConfig` 已经定义 Fast Loop Memory 参数，`fast_loop.settings.FastLoopConfig`
又复制一套相同字段，并出现真实默认值不一致：

```text
MemoryConfig.maintenance_tip_capacity = 200
FastLoopConfig.maintenance_tip_capacity = 240
```

正式入口手动复制配置，因此当前主链路使用 200；直接构造 FastLoopConfig 的测试或
其他调用路径会使用 240。后续只保留一个配置事实来源，运行时视图不得重新声明另一组
默认值。

### 尚未接入主链路的实现候选

以下实现不是立即删除项，但当前没有主链路消费者，应在迁移时逐一确认：

- `evaluation.comparisons.compare_reports`：只有测试调用。
- `memory.factory.open_training_memory`：只有测试和包导出调用。
- `models.policy` 与 `OpenAICompatibleQwenPolicy`：只有兼容测试调用。
- `load_qwen35_processor`：当前是 Tokenizer 兼容别名。

判断标准不是“测试是否覆盖”，而是该接口是否仍属于目标架构的正式能力。如果只是旧
实现的兼容测试，应连同测试一起移除；如果是正式 Python API，则应移动到明确的包并
写入公开契约。

### 建议的目标目录草案

在上述边界全部确认后，`src/tau3_evolver` 的目标结构建议为：

```text
tau3_evolver/
  cli.py
  config.py

  execution/
    request.py
    capabilities.py
    runner.py
    batch.py
    memory_resolution.py
    trace.py

  benchmarks/
    contracts.py
    registry.py
    tau2/
      definition.py
      runtime.py
      executor.py
      agent_adapter.py
      episode_adapter.py
      assertions.py

  fast_loop/
    contracts.py
    settings.py
    prompts.py
    selection.py
    decision.py
    writing.py
    maintenance.py

  memory/
    schema/
    repository.py
    storage.py
    retrieval.py
    embeddings.py
    operations.py

  inference/
    openai_client.py
    fast_loop_policy.py

  modeling/
    qwen35.py
    lora.py

  artifacts/
    schemas.py
    readers.py
    writers.py
    hashing.py

  persistence/
    atomic.py
    jsonl.py

  reporting/
    metrics.py
    comparisons.py

  slow_loop/
    data/
    training/

  serving/
    vllm.py
```

该草案不要求机械地为每个文件建立一个目录。实施时应根据稳定协议和代码规模决定是否
合并小文件，但必须遵守以下依赖规则：

1. `execution` 是应用编排层，可以依赖其他领域；其他领域不得反向依赖它。
2. Tau2 特定类型和私有 Registry API 只能出现在 `benchmarks/tau2/`。
3. `fast_loop` 拥有 Selection、Action、Write、Maintenance 的协议和流程。
4. `memory` 不得导入 Agent、Execution、Artifact Builder 或在线模型实现。
5. `artifacts` 拥有制品 Schema 和 I/O，不导入具体执行结果类型。
6. `slow_loop` 只消费已发布 Artifact、Fast Loop Contract 和 Memory Schema，不参与
   在线执行。
7. 在线 HTTP 推理与离线 Qwen/LoRA 模型加载分离。
8. 通用原子 I/O 有明确的 `persistence` 所有者，不建立无边界的 Utils 包。

### 后续确认与实施顺序

建议按以下顺序继续澄清，确认一层后再制定文件迁移计划：

1. ~~正式建立 `fast_loop`，确认当前 `agent` 是取消还是只保留通用 Controller。~~
   本轮已完成：取消顶层 `agent`，公共能力按 Fast Loop 阶段拆分。
2. ~~收紧 `PreparedBenchmark`，通过 `BenchmarkExecutor` 封装全部 Tau2 执行细节。~~
   本轮已完成；后续新增 Benchmark 必须沿用该协议。
3. 让 Memory 回归纯领域，并将执行期 Snapshot 解析移出。自动 Maintenance 编排
   已迁入 `fast_loop/maintenance.py`。
4. 建立类型化 Artifact Contract，确认内部 Trace 和 Slow Loop Evidence 的文件边界。
5. 拆分在线 Inference、离线 Modeling、Reporting 和 Slow Loop 子域。
6. 最后处理未接入主链路的兼容接口和历史测试，避免过早删除仍需迁移的数据契约。
