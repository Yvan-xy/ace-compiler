# DSL CPU bootstrap / OpenFHE profiling — 2026-09-06

结论：在既定的 16 线程对比中，主要差距是 **EvalMod 的并行粒度不足，以及
SlotToCoeff 中串行多项式运算和临时缓冲区开销**。单线程时 DSL 总耗时反而
更短；提高线程数后 OpenFHE 获得了更大的加速。当前证据不支持将差距笼统归为
“Python DSL 开销”或“ANT 的 NTT 算法很慢”。Python 不在计时路径上。

## 测量方式与正确性

沿用 [RESULTS.md](RESULTS.md) 的全部 FHE 参数、输入、输出规范和现有生成产物。
只在独立副本中加入阶段边界探针；没有改动原始 benchmark、编译器优化或
OpenFHE 数学实现。profiling 构建仍为 `-O3 -DNDEBUG`，增加调试信息。

- 每个阶段记录 `CLOCK_MONOTONIC`、进程累计 CPU/系统时间及缺页次数。
- 基准组：16 线程，每边一次预热、三次正式测量。
- `perf record -e cpu-clock:u -F 99 --call-graph dwarf,4096 --clockid mono`。
  热点报告仅选择第一次正式调用及其阶段时间窗口，不包含 keygen、预热、解密。
- 两个诊断组分别改变线程数、glibc 分配器设置。每组每边一次预热、一次正式调用，
  只用于定位原因，不作为新的稳定性能排名。
- 三组共 **16 次完整 bootstrap 全部通过** 32768 槽位误差检查及 Q/scale 检查。
  阶段序列和时间窗口连续性也通过检查；`perf` 报告丢失样本数为 0。

## 1. 时间究竟差在哪里

单位为秒，各列为三次正式调用的中位数。

| 阶段 | DSL | OpenFHE | DSL − OpenFHE |
| --- | ---: | ---: | ---: |
| ModRaise / 前处理 | 0.080 | 0.045 | +0.035 |
| CoeffToSlot | 8.386 | 8.920 | **−0.534** |
| 共轭拆分 | 0.460 | 0.158 | +0.302 |
| EvalMod，两分支合计 | 8.774 | 5.519 | **+3.255** |
| 重组 | 0.030 | 0.007 | +0.023 |
| SlotToCoeff | 4.979 | 3.352 | **+1.628** |
| Post-scale | 0.012 | 0.010 | +0.002 |
| 输出规范化 | <0.001 | 0.007 | −0.007 |
| 整次调用 | **22.779** | **18.013** | **+4.766** |

各阶段中位数不要求严格相加等于整次调用中位数。两实现把少量归一化工作放在
不同位置，因此细小的前后处理阶段不宜独立排名。EvalMod 和 SlotToCoeff 的
差距约 4.88 秒，CoeffToSlot 的优势抵消了其中一部分。

本次完整耗时与无探针的 22.675 / 17.983 秒相比均相差不到 0.5%，没有出现
profiling 把原来的性能关系显著改变的迹象。

## 2. 主因：多核扩展能力不足

| 阶段 | DSL 1 线程 | DSL 16 线程 | 加速比 | OpenFHE 1 线程 | OpenFHE 16 线程 | 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CoeffToSlot | 9.997 | 8.386 | 1.19× | 14.978 | 8.920 | 1.68× |
| EvalMod | 16.532 | 8.774 | 1.88× | 13.700 | 5.519 | 2.48× |
| SlotToCoeff | 6.029 | 4.979 | 1.21× | 7.535 | 3.352 | 2.25× |
| 整次调用 | **33.106** | **22.779** | **1.45×** | **36.994** | **18.013** | **2.05×** |

单线程总时间关系反转。DSL 的两次线性变换在单线程下均较快，却没有像 OpenFHE
那样从更多线程中获益。EvalMod 的单线程实现仍有差距，多核扩展不足又进一步
拉大了 16 线程时的差距。

源码与实测对应：

- [bootstrap_decomposition.py](../../edsl/core/bootstrap_decomposition.py) 的
  `fullpacked_bootstrap_primitive` 为两个 EvalMod 分支生成两个 OpenMP sections。
  分支内部的生成 C 仍逐个处理 RNS limb，`Conv_poly2ntt*` 也串行遍历素数。
- 16 线程时，DSL EvalMod 的进程 CPU 时间为 **17.413 秒**，墙钟 **8.774 秒**，
  CPU/墙钟约 **1.98**，与两个工作分支吻合。
- OpenFHE 的 `dcrtpoly-impl.h` 在多项式加减乘、格式转换、换基等操作中使用
  `omp parallel for`，而不仅是在外层并行两个 EvalMod 分支。
- DSL SlotToCoeff 的 CPU/墙钟约 **3.42**；其中 `Multiply_add`、`Mac_poly`、
  `Add_poly` 等 Q/P 系数循环仍是串行，外层旋转并行没有覆盖这些运算。

**不能把 CPU/墙钟直接等同于有效工作核数。** OpenFHE EvalMod 此比值约 11.67，
但约 46.7% 的该阶段用户态样本落在 libgomp 的两个等待地址。反汇编可见 `pause`
自旋循环。它确实有更细的并行，但也付出了较高等待开销；不是 16 核线性加速。

## 3. 热点落到具体函数

下表是第一次正式调用中相应阶段的 **用户态 CPU self 样本占比**，不是墙钟占比。

| DSL EvalMod 热点 | 占比 |
| --- | ---: |
| `Forward_transform` | 35.22% |
| `Hw_modmul` | 13.95% |
| `Decomp_modup` | 10.63% |
| `Inverse_transform` | 9.28% |
| `Fast_base_conv` | 7.01% |
| `Hw_modadd` | 6.58% |
| `memset` | 5.65% |

约 44.5% 的 EvalMod 用户态样本在 NTT/INTT。这说明应优先让独立 limb 的
NTT/INTT 和换基获得并行，而不是先微调 Python tracing 或外层函数调用。
这些比例本身不能证明 ANT 的单次 NTT 比 OpenFHE 更慢。

| DSL SlotToCoeff 热点 | 占比 |
| --- | ---: |
| `Multiply_add` | 33.94% |
| `Mac_poly` | 11.79% |
| `Automorphism_transform` | 10.68% |
| `memset` | 10.04% |
| `Add_poly` | 9.64% |

源码位置：
[rns_poly_impl.c](../../../fhe-cmplr/rtlib/ant/poly/src/rns_poly_impl.c) 的
`Multiply_add`、NTT 转换循环；
[rns_poly.c](../../../fhe-cmplr/rtlib/ant/poly/src/rns_poly.c) 的多项式运算；
[hal.c](../../../fhe-cmplr/rtlib/ant/hal/src/hal.c) 的 limb 级算术。

## 4. 内存开销确实存在，但不是完整解释

默认分配器下，SlotToCoeff 每次调用的阶段中位数：

| 指标 | DSL | OpenFHE |
| --- | ---: | ---: |
| minor faults | 557621 | 55952 |
| 系统 CPU 时间 | 4.240 秒 | 0.507 秒 |
| major faults | 0 | 0 |

这是次缺页和系统时间，不应解释成磁盘读取或直接当作分配字节数。
[rns_poly.h](../../../fhe-cmplr/rtlib/ant/include/poly/rns_poly.h) 中
`Alloc_poly_data` 是连续 Q/P 大块内存的 `malloc + memset`，`Init_poly_by_size`
先释放再分配。生成代码又会在临时量最后使用后释放它们，缺少跨调用的工作区复用。

给 **两边同时** 设置以下环境变量，以减少大块 mmap/trim 并保留可复用的堆页：

```sh
MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=-1
```

| 指标 | DSL 默认 | DSL 调整后 | OpenFHE 默认 | OpenFHE 调整后 |
| --- | ---: | ---: | ---: | ---: |
| 总耗时 | 22.779 | 21.394 | 18.013 | 13.452 |
| EvalMod | 8.774 | 8.454 | 5.519 | 4.438 |
| SlotToCoeff | 4.979 | 4.491 | 3.352 | 3.312 |
| SlotToCoeff minor faults | 557621 | 181 | 55952 | 384 |
| SlotToCoeff 系统 CPU 秒 | 4.240 | 0.062 | 0.507 | 0.102 |

该诊断组只有一次正式调用。它验证了内存分配/页复用确有影响，也显示移除绝大
多数次缺页后 SlotToCoeff 差距仍在。OpenFHE 同样受益，且总时间改善更大。
因此不能把“换分配器”当成 DSL 追平 OpenFHE 的完整方案。这些设置仅用于诊断，
没有写入原 benchmark 的默认配置。

## 5. 两项次要的源码差异

**额外活跃 Q limb。** 外部配置和最终输出一致，但内部调度并不完全相同：

| 阶段入口 | DSL Q / scale degree | OpenFHE Q / scale degree |
| --- | --- | --- |
| CoeffToSlot | 31 / 1 | 30 / 1 |
| EvalMod | 28 / 1 | 27 / 1 |
| SlotToCoeff | 19 / 1 | 18 / 1 |

OpenFHE 在 ModRaise 后做一次归一化乘法和 rescale；DSL 将相应缩放并入变换
常量，没有同样消耗一层，后续便一直携带多一个 Q limb。它最后也被丢弃到共同
的输出 Q=14。这个额外 limb 会增加工作量，但本轮未单独测量它的耗时贡献，
不能将全部差距归给它。下一步可在保持输出能力与精度的条件下调整层数调度。

**平方没有专门降低。** 生成的 EvalMod 有 32 个 ciphertext multiplication
位置，其中 16 个是平方。
[CKKS2POLY::Handle_mul_ciph](../../../fhe-cmplr/poly/src/ckks2poly.cxx) 仍生成四次
多项式乘法，分别计算两次相同的交叉项 `c0*c1` 和 `c1*c0`；OpenFHE 的
`LeveledSHEBase::EvalSquareCore` 使用三次乘法和一次 doubling。这是可消除的
工作，但本轮只核实了源码与生成产物，没有为它宣称独立的实测加速数值。

## 建议的优化顺序

1. **EvalMod 的 RNS/NTT 并行调度。** 当前只有两个分支并行，实际约两核。
   应将 limb 级工作纳入统一线程预算；避免简单开启嵌套 OpenMP 导致线程过量。
2. **线性变换中的串行 Q/P 运算与工作区。** 优先检查 `Multiply_add`、MAC、
   `Add_poly` 的并行覆盖，并复用大块临时缓冲区、避免无用清零。
3. **平方专门 lowering、最后不会使用的 Q limb。** 在前两项之后做局部消除，
   继续使用相同输入输出状态和全槽位误差检查。

这里给出的是 profiling 结论与优化方向，尚未把这些优化应用到编译器或运行库。

## 复现及原始证据

从仓库根目录执行；依赖沿用原 benchmark：

```sh
python3 ace_edsl/benchmarks/cpu_bootstrap/profile.py \
  --ace-prefix tmp_cpu_bts_openfhe_sparse_k28/install-current \
  --openfhe-source tmp_openfhe_cpu_compare/openfhe-src --perf

python3 ace_edsl/benchmarks/cpu_bootstrap/profile.py \
  --ace-prefix tmp_cpu_bts_openfhe_sparse_k28/install-current \
  --openfhe-source tmp_openfhe_cpu_compare/openfhe-src \
  --skip-build --threads 1 --repetitions 1 --run-name threads1

MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=-1 \
python3 ace_edsl/benchmarks/cpu_bootstrap/profile.py \
  --ace-prefix tmp_cpu_bts_openfhe_sparse_k28/install-current \
  --openfhe-source tmp_openfhe_cpu_compare/openfhe-src \
  --skip-build --threads 16 --repetitions 1 --run-name allocator128m
```

输出目录：`tmp_openfhe_cpu_compare/profile/`。

- `threads16/{summary.json,phases.json,*.hotspots.txt,*.perf.data,*.log}`
- `threads1/{summary.json,phases.json,*.log}`
- `allocator128m/{summary.json,phases.json,run-settings.json,*.log}`
- `source-hashes.json` 记录原始生成 C 和插桩副本的哈希。

本轮仍是同一台 16-vCPU KVM 虚拟机、一个确定性输入和每进程一组密钥，原先
关于原生素数/噪声采样与安全等级的限制仍然适用。单线程和分配器诊断数据的
重复次数少，不能据此推广到其他机器或所有 bootstrap 参数。
