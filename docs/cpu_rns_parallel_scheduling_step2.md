# CPU RNS 并行探索：第二步实现与正确性验证

状态：第二步完成，待第三步正式性能对照。分支：`explore/cpu-rns-parallel-scheduling`。

## 实现范围

打通了第一步选定的 `POLY.decomp_modup` → runtime NTT/INTT 批量执行路径。
未增加通用调度 pass、成本模型或公开 DSL 语法；本轮是显式配置控制的
CPU codegen/runtime 最小实验。

| 配置 `decomp_ntt_threads` | 生成入口 | Bootstrap demo 的外层 EvalMod |
| --- | --- | --- |
| 0，默认 | 原 `Decomp_modup` | 保留原 `parallel_eval_mod` 设置 |
| 1 | `Decomp_modup_with_ntt_threads(..., 1)` | 关闭双分支 sections，验证新串行路径 |
| >1 | `Decomp_modup_with_ntt_threads(..., budget)` | 关闭双分支 sections，使用 NTT limb 预算 |

编译器通过 POLY2C 配置、Python binding 和两种 pipeline 接入该参数。CLI 对应
`-P2C:ntt_threads=16`，Python 配置为 `configure_fhe(decomp_ntt_threads=16)`。
Python 配置拒绝非 uint32 整数及非 ANT/POLY 使用方式。

demo/benchmark 使用 `ACE_BOOTSTRAP_DECOMP_NTT_THREADS` 协调 tracing 和 codegen，
在新模式下关闭 `parallel_eval_mod`。通用 pipeline 的 codegen 参数不会删除已经
存在的并行 IR；这种情况下 runtime 的外层团队检查提供串行回退。

## Runtime 行为

- 旧入口及旧转换 helper 保持可用。新入口共用原 decomposition/换基实现，
  只将线程预算传给 part2 INTT 和 part1/part3 NTT。
- 按显式 prime 列表处理连续 limb view；循环上限使用活跃长度，支持最后一个
  partition 比 prime 列表短的情形。
- 每个迭代建立局部 `VALUE_LIST` view，不共享递增游标；输出、表和 scratch
  在创建团队前准备好，整体 NTT 标志在完成同步后更新。
- worker 数不超过请求预算、批次长度、`omp_get_max_threads()` 和 OpenMP
  thread limit。已有外层团队、零/单 limb、预算 <=1 时串行，不创建嵌套团队。
- 对不满足新并行路径布局/别名条件的调用保留旧串行方式；旧接口的有效输入
  先决条件仍然适用，不承诺任意非法 buffer 都可计算。
- `Append_rtlib_timing` 的三个共享累计使用 OpenMP 原子更新；计时栈仍保持
  threadprivate，统计在并行工作完成后读取。

主要文件：

- [调用发射](../fhe-cmplr/include/fhe/poly/ir2c_core.h)
- [runtime 入口](../fhe-cmplr/rtlib/ant/poly/src/rns_poly.c)
- [NTT 批量循环](../fhe-cmplr/rtlib/ant/poly/src/rns_poly_impl.c)
- [统计竞争修复](../fhe-cmplr/rtlib/common/linux/rtlib_timing.c)

独立构建 Python bindings 时增加 OpenMP runtime 链接，解决静态 offline encoder
包含新 NTT 批量实现后的 `GOMP_parallel` 符号依赖。

## 验证结果

共 58 个相关测试通过：

- 8 个新增 runtime 测试：逐系数 round-trip、重复调用、原地操作、空批次、
  prime 列表前缀、全部 decomposition partitions、非 NTT 输入、重叠 view
  回退、线程预算/嵌套回退及并发统计累计（部分测试覆盖多个条件）。
- 4 个既有 CKKS extended-op runtime 测试。
- 6 个新增 Python 测试：配置验证、两种 pipeline 传参、外层 sections 协调，
  以及 0/1/3 三种预算的真实 square kernel codegen。
- 36 个相关既有 Python 回归测试和 4 个 benchmark 结果检查测试。

runtime 测试用链接器包装 `Ftt_fwd/Ftt_inv` 观察真实 team size，确认新入口
实际到达并行 NTT；另验证 32000 次并发统计更新没有丢失计数。
修改的 C 源码通过非 OpenMP 编译检查；没有宣称完成非 OpenMP 的完整运行测试。

### 完整尺寸 BTS 冒烟

三种模式均使用同一隔离 runtime 安装前缀，含相同统计修复。每模式一次预热、
一次后续调用，全部 6 次通过 32768 槽位复数误差及输入输出 Q/scale 检查。
参数保持原比较配置：N=65536、degree=44、K=28、Q=31/P=11、输入 Q=2、
输出 Q=14、输入输出 scale=2^56 / scale degree=1。

| 模式 | 最大误差 | 新入口调用数 | 入口内 NTT limb 调用数 | 多线程 team 内的 limb 调用数 | 最大 team size |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy，0 | 4.4399e-4 | 0 | 0 | 0 | 不适用 |
| Serial，1 | 4.4020e-4 | 180 | 6408 | 0 | 1 |
| LimbParallel，16 | 4.3895e-4 | 180 | 6408 | 6400 | 16 |

调用数为每模式两次 BTS 的合计。新模式的生成 C 包含 32 个新入口位置，预算
参数正确，保留 6 个线性变换 sections 区域；旧模式还有第 7 个双 EvalMod 区域。

这些运行启用了计数探针，重复次数也不足以得出正式性能结论。第三步须关闭
探针，重新测量旧策略、新 Serial、新 LimbParallel，至少三次正式调用。

## 复现与构建说明

需要重新构建 runtime、FHEpoly 和 `air_builder` binding，不能只替换 Python
文件。标准工程构建需启用 `BUILD_WITH_OPENMP=ON`；runtime 单测已注册为
`ut_fhert_ant_ntt_threads`。

本机验证采用现有安装前缀的隔离副本，重编相关 runtime 对象、全部 POLY 对象
和 `air_builder`，没有做整个仓库的全量构建。ONNX loader 沿用此前实验 binding
的禁用桩，ONNX 功能不在本步验证范围内。具体构建/冒烟脚本与日志保存在：

- `tmp_cpu_ntt_step2/{build_overlay.py,run_smokes.py}`
- `tmp_cpu_ntt_step2/{unit-regression.log,python-tests.log,regression-tests.log}`
- `tmp_cpu_ntt_step2/bts{0,1,16}/{build.json,smoke.log}`
- `tmp_cpu_ntt_step2/step2-results.json`

以 16 线程模式为例，使用已更新的编译器和 binding：

```sh
python3 ace_edsl/benchmarks/cpu_bootstrap/run.py \
  --ace-prefix /path/to/updated/install --bindings-dir /path/containing/ace_bindings \
  --openfhe-prefix tmp_openfhe_cpu_compare/openfhe-install \
  --openfhe-source tmp_openfhe_cpu_compare/openfhe-src \
  --work-dir tmp_ntt_smoke16 --decomp-ntt-threads 16 --ntt-probe --build-only
OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
OMP_MAX_ACTIVE_LEVELS=1 tmp_ntt_smoke16/ant_bench 1
```

`--ntt-probe` 是正确性验证专用观察器，新入口内无并发调用者时使用；不能将
带该探针的结果作为正式性能数据。正式比较应重新构建不带探针的可执行文件。
