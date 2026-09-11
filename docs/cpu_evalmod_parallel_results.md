# EvalMod 完整策略试验结果

日期：2026-09-08。分支：`explore/cpu-rns-parallel-scheduling`。

**候选 B（双分支＋算子任务共享团队）得到正向结果：EvalMod 阶段中位数由
8.569 秒降到 4.196 秒，约快 51%；完整 BTS 由 23.849 秒降到 19.678 秒，
约快 17.5%。** CPU 时间明显增加，因此保持显式实验选项，默认仍为 Legacy。

## 1. 已实现的范围

- 仅在编译器标记的 EvalMod 区域内发射新入口；区域外保留原调用。
- A：一个团队内顺序推进两个分支，算子工作通过分块任务执行。
- B：同一个团队内同时推进两个分支，共享算子任务；不增加嵌套团队。
- 两者共用 `Evalmod_decomp`、`Evalmod_mod_down`、`Evalmod_rescale` 和
  `Evalmod_hw_mod{mul,add,sub}`，保留原数学求值顺序及 FHE 参数。
- DecompModUp 覆盖 INTT、coefficient 块换基及补集 NTT；ModDown 将每个输出
  limb 的换基、NTT、修正合并；Rescale 先准备公共 last-limb，再按输出 limb
  使用分块私有 scratch。
- 配置 0/1/2 分别代表 Legacy/A/B。旧 `decomp_ntt_threads` 试验与新模式互斥。

相对于设计的实际调整：现有 SPOLY RNS 循环复用单 limb scratch，首版没有直接
并行这些循环。逐元素算术在每次 Hw 调用内部使用至少 16384 coefficients 的
分块，同步完成后才执行下一调用。这保留了 key 累加和 weighted-sum 顺序，
但尚未实现跨 Hw 调用的输出-limb 融合，任务数量较多，是后续优化空间。
该分块值是本轮固定实验参数，并非自动调优结果。

## 2. 主结果：无诊断计数、无采样

同机、同库、默认 allocator、相同 affinity；每组一次预热、三次正式调用。
单位：秒，样本不含预热。

| 线程数 | 策略 | 正式样本 | 完整 BTS 中位数 |
| ---: | --- | --- | ---: |
| 16 | Legacy | 23.848883, 24.245071, 22.935565 | **23.848883** |
| 16 | A | 22.418140, 21.732352, 22.483406 | **22.418140** |
| 16 | B | 19.096800, 19.677534, 19.684240 | **19.677534** |
| 1 | Legacy | 34.151, 35.684, 35.454 | **35.453949** |
| 1 | A | 35.673421, 36.545806, 35.908857 | **35.908857** |
| 1 | B | 34.751030, 35.245870, 35.489763 | **35.245870** |

单线程三组相近，不据此宣称单线程优化。16 线程下 A 相对 Legacy 约快 6.0%，
B 约快 17.5%；B 也比 A 快约 12.2%。这次没有再为局部并行牺牲整个分支层优势。

16 线程的资源代价：

| 策略 | CPU 秒/调用中位数 | 进程峰值 RSS GiB |
| --- | ---: | ---: |
| Legacy | 64.586 | 14.414 |
| A | 140.541 | 14.421 |
| B | 110.731 | 14.395 |

B 的 CPU 时间比 Legacy 增加约 71.5%，不能说 CPU 效率也提高了。CPU 时间包含
执行、调度和等待，未在本轮单独量化其比例。峰值 RSS 包含 setup、keys 和缓存
常量，三组接近；它不代表 EvalMod scratch 的大小。

## 3. 独立阶段测量

阶段诊断另跑，每组同样一次预热、三次正式调用，不作为上表主测的替代。

| 阶段 | Legacy | A | B |
| --- | ---: | ---: | ---: |
| CoeffToSlot | 8.845 | 8.845 | 9.024 |
| Split | 0.443 | 0.442 | 0.451 |
| **EvalMod** | **8.569** | **6.063** | **4.196** |
| SlotToCoeff | 5.337 | 5.345 | 5.369 |

收益集中于 EvalMod。区域外没有做算术或调度优化；去除新头文件和文件路径
差异后，A/B 在 EvalMod 区域外的生成 C 与 Legacy 完全一致。
独立诊断与主测仍有运行波动，阶段中位数不应直接与主测总中位数相加减。

## 4. 覆盖与正确性

A/B 各有额外两次启用诊断的完整调用，记录的每次区域数据一致：

| 类别 | runtime 调用数/区域 | 分块工作数/区域 |
| --- | ---: | ---: |
| 逐元素算术 | 29002 | 116008 |
| DecompModUp | 90 | 3660 |
| ModDown | 64 | 1728 |
| Rescale | 92 | 1472 |

四类 fallback 计数均为 0，thread mask 覆盖 16 个线程，最大同时活动的叶子工作
为 16。观察器属于区域上下文，支持两个分支同时执行，不再使用旧探针的全局
active 标记。父操作不使用跨 taskgroup 的线程栈计时，现有叶子内核的计时不跨
任务调度点。

- 6 组主测共 24 次、3 组阶段诊断共 12 次、覆盖诊断 4 次，总计 **40 次 BTS
  全部通过** 32768 槽位误差、输入输出 Q 和 scale 检查。
- 所有完整调用的最大误差为 **4.5091e-4**，低于共同门限 0.02。
- 18 个 runtime 测试通过：6 个新执行内核/并发测试及 12 个原有相关测试。
- 43 个 Python 配置、区域隔离、codegen 与相关回归测试通过；4 个既有 benchmark
  结果检查测试通过。修改的 Python 通过语法检查，新 C runtime 通过非 OpenMP
  编译检查；未宣称完成非 OpenMP 的完整 BTS 测试。
- compiler 检查未闭合/嵌套区域及缺失区域；新策略只允许 ANT 的当前已验证
  linear-transform lowering 路径，不接受与旧局部 NTT 模式冲突的配置。

## 5. 使用与构建

Python pipeline 选项为 `evalmod_schedule=1` 或 `2`；原生选项为
`-P2C:evalmod_schedule=2`。它们只应用于带 `cpu_evalmod` 属性的内部区域。
Bootstrap demo 的 `ACE_BOOTSTRAP_EVALMOD_SCHEDULE` 同时协调区域生成与 codegen，
不需要修改用户数学代码。

```sh
python3 ace_edsl/benchmarks/cpu_bootstrap/run.py \
  --ace-prefix /path/to/updated/install --bindings-dir /path/containing/ace_bindings \
  --openfhe-prefix tmp_openfhe_cpu_compare/openfhe-install \
  --openfhe-source tmp_openfhe_cpu_compare/openfhe-src \
  --work-dir tmp_evalmod_b --evalmod-schedule 2 --build-only

OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
OMP_MAX_ACTIVE_LEVELS=1 ACE_EVALMOD_DIAGNOSTICS=0 tmp_evalmod_b/ant_bench 3
```

`ACE_EVALMOD_DIAGNOSTICS=1` 仅用于覆盖验证，正式计时保持关闭。新的区域属性
要求同步重建 CKKS opcode registry、FHEpoly 和 bindings；不能混用旧 registry
与新头文件。bindings 已补充相关静态 archive 的链接依赖，避免库更新后模块
没有重新链接而加载过期的 opcode 元信息。

本机仍使用隔离安装前缀的增量重建，没有全量重建整个仓库；ONNX loader 沿用
此前 bootstrap 实验的禁用桩，ONNX 功能未在此次验证。

## 6. 原始证据和结论边界

关键原始日志、构建哈希及汇总已归档到
[benchmark results](../ace_edsl/benchmarks/cpu_bootstrap/results/evalmod-2026-09-08/README.md)，
可随 Git 迁移。新服务器使用
[README 复现步骤](../ace_edsl/benchmarks/cpu_bootstrap/README.md#continuing-on-another-server)
重新生成和测量。

本机完整工作目录仍为 `tmp_evalmod_full/`（不随 Git 迁移）：

- `build_overlay.py`、`run_cases.py`、`run_phases.py`、`run_single.py`：本机复现脚本。
- `bts{0,1,2}/{build.json,run.log,single.log,phase.log}`：构建哈希与测量。
- `bts{1,2}/coverage.log`：覆盖与 worker 观察。
- `final-results.json`、`summary16.json`、`phases.json`：汇总。
- `unit-regression.log`、`python-tests.log`：验证记录。

本轮参数仍为 N=65536、degree=44、K=28、Q=31/P=11、输入 Q=2、输出 Q=14、
scale=2^56/scale degree=1，运行于相同 16-vCPU KVM 环境。没有重新测量
OpenFHE，不把旧 OpenFHE 数据与本轮数据拼接成新的胜负结论。

当前结论是 B 在该工作负载上降低了延迟，同时增加 CPU 成本。保持默认模式 0，
提供 B 作为可选择的试验策略。后续可在 EvalMod 范围内检查逐元素任务粒度及
等待成本；此次没有扩展 SlotToCoeff 优化，也没有引入自动成本模型。
