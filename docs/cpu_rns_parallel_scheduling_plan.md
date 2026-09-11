# CPU RNS 并行调度探索计划：NTT/INTT 最小实验

2026-09-07 后续范围已限定为 **EvalMod 完整并行策略**，设计见
[EvalMod 覆盖与调度方案](cpu_evalmod_parallel_strategy.md)。SlotToCoeff 不在新一轮
优化范围内。下文保留已完成的局部 NTT 实验，不能将其视为完整 EvalMod 策略。

2026-09-08：完整候选已尝试，[B 在本轮降低完整 BTS 延迟约 17.5%](cpu_evalmod_parallel_results.md)，
CPU 成本增加；保留显式选项，未替换默认。下述负面结论仅针对先前局部 NTT 试验。

状态：三步均已完成；当前候选没有端到端净收益，保持实验开关。日期：2026-09-06。

正式结果见 [第三步报告](cpu_rns_parallel_scheduling_step3.md)：16 线程中位数为
Legacy 23.195 秒、Serial 32.702 秒、LimbParallel 28.618 秒。局部改善不足以
抵消取消旧双分支并行的损失，默认仍为 `decomp_ntt_threads=0`。

实现和验证见 [第二步报告](cpu_rns_parallel_scheduling_step2.md)：新入口已接通，
58 个相关测试和三种模式共 6 次完整 BTS 冒烟通过；实际观察到最多 16 个 worker。

定位结果见 [第一步报告](cpu_rns_parallel_scheduling_step1.md)：选定编译器生成的
`POLY.decomp_modup` → runtime 批量 NTT/INTT 路径。该路径约占 EvalMod 用户态
CPU 样本的 18%，不能把它视为全部 NTT 开销；关闭旧双分支并行后可能没有净收益。

- 探索分支：`explore/cpu-rns-parallel-scheduling`
- 起点：`feature/dsl-bts-linear-transform`，提交 `619de690ffc9ea1f06cfeeb15b19bdfd22ead068`
- 起点包含未提交的 DSL/OpenFHE 配置、benchmark 和 profiling 工作。复现基线
  还需参考现有 `build.json` 的产物哈希，不能只依赖上述提交号。

## 1. 本轮只回答一个问题

**让实际 EvalMod 调用链中的批量 NTT/INTT 按 limb 并行，能否在保持 FHE
参数和精度的条件下改善整次 Bootstrap？**

[Profile 报告](../ace_edsl/benchmarks/cpu_bootstrap/PROFILE.md) 显示：16 线程时
DSL EvalMod 为 8.774 秒，OpenFHE 为 5.519 秒；DSL 此阶段的 CPU/墙钟约 1.98，
NTT/INTT 占用户态 CPU 样本的 44.5%。这些数据用于选择实验对象，不代表已经
证明并行化一定能提速。新增线程的调度、同步和访存成本仍需实测。

本轮仅做 CPU/ANT 的一条真实热点路径，不新增公开 DSL 语法。规则基于不同
limb 的 NTT 独立性，不匹配 Bootstrap 函数名、生成变量名或 Python 源码。

## 2. 最小实现方案

### 接入实际热点

先通过 IR dump 和调用链确定 EvalMod 的 NTT 在哪里执行。当前
`Linear_transform_only()` 是混合 lowering：线性变换走 HPOLY，周围的
EvalMod 仍走既有 CKKS→SPOLY 路径。仅修改 `h2lpoly.cxx` 不能保证覆盖热点。

第一步已确认 NTT 隐藏在 runtime 中。本轮从 `POLY.decomp_modup` 的调用发射
位置增加带线程预算的入口，内部仅调度 part2 INTT 和 part1/part3 NTT。保留旧
入口及其串行行为，不展开整个 key-switch 或换基，也不同时接入 H2LPOLY。

保留现有单 limb NTT/INTT 算法，仅调整独立 limb 工作的执行顺序。当前生成产物
中的该 opcode 调用都位于 EvalMod，但选择依据是算子语义，不是 Bootstrap
名称。具体 buffer、prime 子列表和统计状态约束见第一步报告。

### 只提供两种新执行方式

实验模式下关闭 DSL 的 `parallel_eval_mod`，两个分支顺序执行，再选择：

- **Serial**：选定的批量 NTT/INTT 串行执行。
- **LimbParallel**：选定的批量 NTT/INTT 按 limb 并行执行。

旧的“两分支并行、内部串行”保持可用，作为主性能对照。新模式默认关闭。
实验配置通过现有 pipeline/benchmark 配置传递，不要求用户修改数学算法。

第一版使用配置给定的线程预算，工作线程数不超过预算和独立 limb 数。遇到
已有外层并行团队时明确串行回退，保留线性变换现有 sections；不创建嵌套团队，
也不无差别删除旧并行标记。未知或不支持的情况回退串行。

复用必要的循环属性、已有 IR dump 和简单日志，说明选中的模式或回退原因。
暂不建设 `ParallelPlan`、候选搜索器、专用计划导出或成本模型。强制实验模式
也不能绕过合法性检查。

## 3. 必须保留的正确性条件

对选定 NTT 路径，明确验证以下条件即可，不先实现通用别名/依赖分析框架：

- 不同 limb 的输出区域互不重叠；输入、twiddle 和 key 在并行区内只读。
- 允许已确认安全的原地 NTT；部分重叠或无法判断的别名回退串行。
- 正确区分活跃 limb 数与分配 stride，涉及 P 部分时验证其真实起点。
- loop index、modulus cursor 和临时 scratch 为每个任务独有。
- 输出分配及缓存初始化在并行区前完成；NTT 标志等整体元数据在任务完成后
  更新；最后一次使用前不能释放 buffer。
- 保持单 limb 内部运算和归约次序，不改写换基中的共享累加。

只审计受影响调用链的共享状态。尤其是 runtime 计时：`Rtm_stack` 为
threadprivate，但 `Append_rtlib_timing` 更新共享数组。若路径会并发更新，
需要消除竞争，并让新旧对照使用相同的计时实现与设置。

## 4. 三步推进

| 步骤 | 工作 | 完成条件 |
| --- | --- | --- |
| 1. 定位与约束（已完成） | 选定 Decomp_modup 内的 NTT/INTT 批次，检查 buffer、scratch、缓存和统计状态 | 具体修改位置与安全条件已记录于第一步报告 |
| 2. 最小实现（已完成） | 接入 Serial/LimbParallel、线程预算及回退，协调外层双分支开关 | 真实 BTS 命中新入口，逐系数与全槽位检查通过，详见第二步报告 |
| 3. 对照实验（已完成） | 1/16 线程主测及独立阶段诊断，每组预热一次、正式调用三次 | 36 次正确性检查通过；得到负面的端到端结论，详见第三步报告 |

如果只有孤立 NTT 加速而整次 BTS 没有收益，仍是有用的实验结论，但不能
宣称 Bootstrap 已优化。若收益不足，先用 profile 判断覆盖不足、线程启动
还是访存竞争造成限制；不立即扩展成通用调度框架。

## 5. 验证与验收

正确性检查：

- 同一输入下，串行/并行 NTT 和 INTT 的全部系数精确一致。
- 少量小尺寸测试覆盖原地执行、线程数大于 limb 数、非整齐分块，以及所选
  路径实际支持的 Q/P 布局；不支持的布局验证串行回退。
- 覆盖首次调用、重复调用和已有外层团队，检查 scratch、元数据及生命周期。
- 完整 BTS 检查所有 32768 槽位的复数误差，同时检查实际 Q 数、scale 和
  scale degree，并与新 Serial 参考比较精度。

实验设置：

- 沿用 [现有 benchmark](../ace_edsl/benchmarks/cpu_bootstrap/README.md)：
  N=65536，degree=44、K=28、相同系数，Q=31/P=11，输入 Q=2、输出 Q=14，
  输入输出 scale=2^56、scale degree=1；其它 FHE 参数也保持原配置。
- 不同时调整额外 Q limb、平方 lowering、allocator、内存池或 GPU 实现。
- 使用同机、同 Release 编译器与运行库，固定 affinity、OpenMP、allocator
  设置。共享线程安全修复同时用于对照，避免混入无关差异。
- 对照旧策略、新 Serial、新 LimbParallel；线程预算先测 1 和现有 16，必要
  时增加一个中间点解释扩展行为，不做大规模参数扫描。
- 每组一次预热、至少三次正式调用，顺序运行，报告全部样本和中位数。
  重测旧策略，不直接与历史耗时相减；波动掩盖差异时再增加重复。
- 同时报告 EvalMod 和完整 BTS 时间、CPU 时间、次缺页及峰值 RSS。
  主性能数据不带 perf 采样，perf 用于解释变化。

需要分别回答：所选 NTT 路径是否加速？关闭旧分支并行后的净效果如何？整次
BTS 是否获益？收益是否伴随明显内存或 CPU 成本？只满足第一项不等于验收提速。

本轮交付小范围代码、相关测试、复现命令及结果报告。负面结果如实记录；
在完整 BTS 收益可重复确认前，新模式保持实验开关，不替换默认行为。

## 6. 有收益后再讨论

- 使用已测量的简单工作量阈值选择串行/并行，之后才考虑成本表或 tuning。
- 扩展到模乘等其它独立 limb 算子，以及另一条 lowering 路径。
- 线程团队复用、分支＋limb 联合调度、通用调度表示与更复杂的依赖分析。
- 额外 Q limb、平方优化和工作区复用，分别立项测量，避免混淆收益来源。

这些不是本轮完成条件。目前已完成本轮探索，保留旧策略，不自动扩大实现范围。
