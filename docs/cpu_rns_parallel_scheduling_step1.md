# CPU RNS 并行探索：第一步定位与约束

状态：第一步已完成；尚未实现并行。分支：`explore/cpu-rns-parallel-scheduling`。
本次基于当前源码、生成 AIR/C 和既有 perf 调用栈，选定一条最小接入路径。

## 1. 已选定的路径

```text
DSL EvalMod 的密文乘法 / 平方
  → relinearize
  → CKKS2POLY::Handle_kswitch
  → POLY.decomp_modup
  → 生成 C：Decomp_modup(res, c2, q_part_idx)
  → Conv_ntt2poly_with_primes(part2) → Ftt_inv → Inverse_transform
  → 现有串行换基
  → Conv_poly2ntt_inplace_with_primes(part1 / part3)
      → Ftt_fwd → Forward_transform
```

**选择 runtime 批量入口方案。** 编译器为 `POLY.decomp_modup` 的 CPU 调用
传递显式 NTT 线程预算，runtime 仅在上述两个转换 helper 内分割独立 limb。
首版不展开整个 key-switch，不改 H2LPOLY，不并行换基或 decomposition 外层循环。

实际产物证据：

- `bootstrap_full_poly_driver.air:41363` 保留 `POLY.decomp_modup`，没有展开成
  单 limb NTT；`bootstrap_full.c:9082` 是对应生成调用的第一个位置。
- 生成 C 中 32 个 `Decomp_modup` 调用位置全部位于 EvalMod；该区域还包含
  64 个 `Mod_down`、92 个 `Rescale` 位置，没有直接的 `Hw_ntt/Hw_intt` 调用。
  这些是静态位置数，不是运行次数；`Decomp_modup` 还处于 `Num_decomp` 循环内。
- 线性变换采用 `Precomp` 等原有 runtime 调用。保留旧入口的串行行为，新增
  编译器选择的入口，可避免用全局开关把所有 native 调用一并改变。

上述生成文件在 `tmp_openfhe_cpu_compare/benchmark/generator/output/` 下。

## 2. 用真实调用栈确认覆盖范围

重新分析既有 16 线程 perf 记录的第一个正式 EvalMod 区间：
`4375.291051–4384.118303`，共 1627 个用户态 CPU 样本。
以下按叶函数及其调用栈中的 runtime 父函数归类，所有 NTT/INTT 样本均有归属。

| runtime 父函数 | Forward 样本 | Inverse 样本 | 占全部 EvalMod 用户态样本 |
| --- | ---: | ---: | ---: |
| **Decomp_modup，首版选择** | 221 | 74 | **18.13%** |
| Mod_down，暂不接入 | 142 | 67 | 12.85% |
| Rescale，暂不接入 | 210 | 10 | 13.52% |
| 合计 | 573 | 151 | 44.50% |

例如已采到 `Forward_transform ← Ftt_fwd ← Conv_poly2ntt_inplace_with_primes
← Decomp_modup ← dsl_bts_bootstrap_full._omp_fn.3`，与源码链一致。

**收益预期需要收窄。** 这条路径只覆盖约 40.7% 的 NTT/INTT 样本，并非全部
44.5% 的 EvalMod 用户态工作。关闭外层双分支并行又会影响其余计算，所以这个
最小实验可能改善局部 NTT，却不能抵消取消旧策略的损失。当前证据不支持承诺
端到端提速，也不能将样本占比直接换算为可节省的墙钟秒数。第二、三步必须同时
比较旧双分支策略、新 Serial 和新 LimbParallel，允许得到负面结论。

## 3. 并行边界与必要约束

| 对象 | 当前行为 / 约束 | 第二步需要做的最小处理 |
| --- | --- | --- |
| `q_part_idx` 外层循环 | 复用 `_pgen_ext`，累加到 `_pgen_swk_c0/c1` | 保持串行；不能直接加 parallel-for |
| part2 INTT | 输入是借用的 `res` 切片；输出是单独分配的 `part2_poly_intt` | 先分配，再按独立 limb 写输出；全部完成后才能换基 |
| part1/part3 NTT | 各自是 `res` 的借用切片，原地转换 | 按切片的实际长度与起点执行，不分配或释放借用数据 |
| `VALUE_LIST` 游标 | 当前在循环外创建，循环中递增 `_vals._i` | 每个迭代建立局部 view，地址为固定基址加 `i * degree` |
| prime / twiddle | `Get_vlprime_at(primes, i)` 指向初始化好的 NTT context | 共享只读；使用传入的 prime 列表，不用全局 prime 下标代替 |
| polynomial 元数据 | helper 尾部更新 `_is_ntt` | 所有任务完成后，由调用线程更新一次 |
| 换基 scratch `sum[]` | `Decomp_modup` 中串行清零、累加和写出 | 保持现有顺序，排除在并行区之外 |
| 生命周期 | view 和 prime 子列表在调用栈上，INTT buffer 在调用尾释放 | 同步完成后返回；不产生逃逸任务 |

布局细节不能省略：

- `part2_primes` 的列表可能比最后一个 partition 的活跃 limb 数长。INTT 的
  循环上限是 `Poly_level(part2)`，不能改成 `LIST_LEN(part2_primes)`。
- `part1` 在首个 partition 中可能长度为 0，应保留原有零次循环语义。
- `Extract_poly` 已把切片表示为连续的局部 view。part3 的 prime 列表可能
  混合剩余 Q 和 P；批量函数按 view 和对应列表处理，不假定它从全局 Q0 开始。
- 当前生成路径在调用前为 `_pgen_ext` 按活跃 Q+P 分配空间，输入 c2 与该输出
  分离。新入口首版只接受这种已知布局与别名关系；不泛化到未知重叠或带空洞
  的 Q/P buffer。不支持的情况使用原串行路径。

## 4. 单 limb 内核与共享状态审计

- `Ftt_fwd/Ftt_inv` 在当前 CPU 路径调用 `Forward_transform/Inverse_transform`。
  二者只修改当前输出 view，读取 twiddle、modulus 和 inverse-degree 常量。
  没有随机采样、key 修改或运行时 twiddle 初始化。
- 表在 `Set_primes → Init_crtprime → Init_nttcontext → Precompute_ntt` 中准备。
  `Get_ntt` 只是返回现有 context 的地址；context 初始化/销毁不能与执行并发。
- `Ftt_*` 的非原地分支调用 `Init_i64_value_list`。这里使用已初始化、大小足够
  的输出 view 时只复制当前 limb，不分配共享输出；局部 view 必须在迭代内建立。
- **明确的共享写入是计时统计。** `Ftt_*` 调用的 `Rtm_stack` 已 threadprivate，
  但 `Append_rtlib_timing` 无同步地更新三个全局数组。这是已存在的潜在数据竞争，
  不能把新增 worker 放进去后仍称线程安全。第二步须做最小同步修复，例如原子
  累加，并让旧策略和新策略使用同一实现；不同时重构整个统计体系。

以上是源码约束审计，不代表新并行实现已经通过并发测试；相关测试属于第二步。

## 5. 第二步的明确修改位置

| 文件 | 计划改动 |
| --- | --- |
| `fhe-cmplr/include/fhe/poly/ir2c_core.h:232` | 对已选模式下的 `POLY.decomp_modup` 发射带线程预算的调用；旧模式仍发射旧符号 |
| POLY 配置及现有 pipeline/benchmark 配置入口 | 传递实验开关和预算，CPU/ANT 限定；关闭实验中的 `parallel_eval_mod` |
| `fhe-cmplr/rtlib/ant/include/poly/rns_poly.h` | 声明新增入口，保留 `Decomp_modup` ABI |
| `fhe-cmplr/rtlib/ant/poly/src/rns_poly.c:161` | 共用原有 decomposition/换基实现，仅将预算传到两个 NTT 批次位置 |
| `fhe-cmplr/rtlib/ant/poly/src/rns_poly_impl.c:164,236` | 实现带预算的局部 view 循环；旧 helper 保持串行调用方式 |
| `fhe-cmplr/rtlib/common/linux/rtlib_timing.c:26` | 修复共享累计的并发写入 |

入口的示意形式是 `Decomp_modup_with_ntt_threads(res, src, part, threads)`，
名称尚未实现。旧入口等价于调用共同实现的串行模式。新入口预算不大于调用方
给定上限和批次长度；预算 <=1、批次长度 <=1、处于外层并行团队或条件不支持
时走串行路径。保留隐式同步，不设置 `nowait`。

该方案只需要现有语义 opcode 的调用选择和一个 runtime 执行参数，不新增
公开 DSL 语法、调度 IR、成本模型或通用循环分析。

## 6. 本步验证和可复现证据

已核对 benchmark 记录中的原始产物哈希，二者仍匹配：

- `libFHErt_ant.a`：`5a512983560e2edf77e47d6a460eee543fd6b228444a3f616938ad2bc9af128a`
- `bootstrap_full.c`：`347eadb424afff4f46fefff65080e77b1c081b446abbad5657ba7959277576ab`

本步复用了已经通过正确性检查的 profile 数据，没有重新运行 BTS，也没有新增
并行实现。调用栈归类、静态位置统计分别保存为：

- `tmp_openfhe_cpu_compare/step1/ntt-attribution.json`
- `tmp_openfhe_cpu_compare/step1/static-sites.json`
- `tmp_openfhe_cpu_compare/step1/evalmod-stacks.txt`

调用栈导出命令：

```sh
perf script -i tmp_openfhe_cpu_compare/profile/threads16/dsl_cpu.perf.data \
  --time 4375.291051,4384.118303 -F time,period,event,ip,sym,dso
```

时间窗口来自对应 `dsl_cpu.log` 中 `sample=1, stage=evalmod` 的记录，重跑 profile
后应读取新窗口。第二步优先测试零长度批次、最后一个不完整 partition、独立
输出/原地 NTT、首次调用和外层团队回退，再验证完整 BTS。
