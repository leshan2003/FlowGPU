# repro/ — 复现与诊断的本地工作区

这不是包的一部分，是为了"跑一跑、调一调"而加的工作台。
所有产出（原始控制台输出 `.txt`、结构化结果 `.json`、剖析记录）都留在这里，
便于别人只用读文件就能复核结论。总报告见 `../../FlowGPU_复现与调参报告.md`。

## 证据文件

| 文件 | 内容 |
|---|---|
| `devices.txt` / `models.txt` / `placements.txt` / `fabrics.txt` | 器件库、模型库、放置策略、互联 fabric 清单 |
| `grounded_run.txt` / `grounded_capacity.txt` | `grounded_published_hw.yaml` 的固定负载运行与容量搜索 |
| `pda_capacity.txt` | `pd_a_reference.yaml` 全 6 臂容量搜索（**修复前**） |
| `regimes_run.txt` / `fixed_regimes.txt` | `regimes.yaml` 负载型态（**修复前 / 修复后**） |
| `v41_run.txt` | `deepseek_v41_flash.yaml` |
| `multimodal.txt` / `quickstart.txt` / `make_figures.txt` | 三个脚本 |
| `bucket_gpu.txt` | `batch_bucket` 1/2/4/8/16 的敏感性（本报告最关键的量化） |
| `profile_pda.txt` / `profile_pda192.txt` | cProfile 输出 |
| （控制台，未落盘） | `capacity_timing.py` 每个 probe 的耗时；结果见总报告 §4.6 |
| `eda_dryrun.txt` | `flowgpu eda --dry-run --local-only` |
| `fixed_grounded_capacity.txt` | **修复后**的 `grounded_published_hw.yaml` 容量搜索 |
| `hot_cold_probe.txt` | 热/冷专家分层的七节探针（A–G），见下 |
| `hot_cold_capacity.txt` | 同一批配置在**服务仿真**下的 SLO 容量对比 |

## 脚本

全部只读地使用 `flowgpu` 包，不修改包的行为；用 `sys.path.insert(0, ".")`
从仓库根目录导入。请在 `项目/FlowGPU-main/` 下运行。

```
python repro/cache_stats.py        <config> <arm> [requests] [rate]
python repro/bucket_sensitivity.py <config> <arm>
python repro/capacity_timing.py    <config> <arm|ALL> [requests] [iters]
python repro/profile_probe.py      <config> <arm> [requests] [rate]
python repro/hot_cold_probe.py
python repro/hot_cold_capacity.py
```

- `cache_stats.py`：一次 probe 里建了多少个图键（prefill 用精确 batch+chunk，
  所以键多；decode 用桶，所以键少）。用来判断某个 probe 慢在图构建还是仿真循环。
- `bucket_sensitivity.py`：clone 出同一负载，只改 `batch_bucket`，看 TPOT/SLO/吞吐怎么动。
  它是发现"decode 成本取自桶而非真实 batch"这个缺陷的工具。
- `capacity_timing.py`：给 `find_capacity` 的每个 probe 打时间戳，并支持
  `FLOWGPU_WATCHDOG=<秒>`（Windows 没有 `signal.alarm`，用守护线程硬退出），
  用来把"某个臂病态慢"和"整机被占满"区分开。
- `profile_probe.py`：cProfile 排序打印。

## 热/冷专家分层（`hot_cold_*`）

这两个脚本回答的是"把一部分专家从类脑的 SRAM 里赶出去，收益能不能
抵掉代价"。它们**不修改包的行为**，但 `hot_cold_probe.py` 依赖包里
新增的 `hot_frac` 支持。

- `hot_cold_probe.py`：七节，逐步定位问题
  - A 单步时间按瓶颈归因（compute / memory / network / bridge / collective）
  - B 类脑池到底装不装得下专家栈（`spill_frac` 对芯片数）
  - C 冷专家搬到 GPU HBM 的**容量**可行性
  - D 越过"装得下"这个点之后，再加芯片还有没有用
  - E 多 node 的代价（same chips, different node granularity）
  - F hot_frac × 偏斜 p 的**手算**相图（注明是手算，不是流水线结果）
  - G 真实构建拆分后的图，对比 `pd_a` 与 `hot_cold`
- `hot_cold_capacity.py`：同一组配置走 `ServingSimulator`，比 SLO 容量。
  **注意**：当前负载下 `find_capacity` 的二分上界顶到天花板（`req/s`
  四行都是 1984），容量点没找到；有意义的列是同一负载下的吞吐、
  TTFT/TPOT p99 和 J/token。

配套的**包内改动**（不属于 `repro/`，需单独 review）：

| 文件 | 改动 |
|---|---|
| `flowgpu/workload/llm.py` | `_moe_block` 支持 `hot_frac` / `hot_traffic`，拆出 `moe_hot` / `moe_cold` / `moe_combine` |
| `flowgpu/mapping/placement.py` | 新增命名策略 `hot_cold` |
| `flowgpu/hardware/dataflow.py` | 专家权重按**不可分单位**计价（见下） |
| `flowgpu/hardware/gpu.py` | 同上，保持两侧一致 |

### ⚠️ 关于 `_byte_map` 的改动

原实现是 `wb = op.weight_bytes / shards`，即把带 `groups` 个专家的权重
均摊到池里所有设备上，等价于让一次访问**骑上整个池的聚合带宽**。

但专家是不可分的：一个专家整体坐在**一颗**芯片上，命中它只能拿到那
一颗芯片的带宽。修正后是

```
wb = ceil(groups / shards) × (weight_bytes / groups)
```

影响很大：`pd_a @512`（溢出 47%）在 batch 1 下从 3.14 ms 变成 9.01 ms。
这是**我们自己引入的建模假设**，需要独立复核——它假设专家均匀分布、
每次命中落在不同芯片上；真实的路由偏斜只会让"最忙的那颗芯片"更忙，
也就是让分层收益**更好**，而不是更差。

## 微批流水与集合通信重叠（`pipeline_overlap_probe.py`）

回答两个问题，它们的答案不一样。

**一、微批流水到底赚不赚？** 它是两笔账相乘，只报净时间看不出谁在起作用：

```
overlap_gain   = (m × 单片) ÷ 流水时间     ← 两池真的并行了吗
microbatch_tax = (m × 单片) ÷ 串行时间     ← 切碎让总工作量涨了多少
net            = overlap_gain ÷ microbatch_tax
```

税来自"权重不跟着切"：decode 是访存受限的，切成 m 片还是要各流一遍权重。
增益来自"稳态周期 = 最忙的那个池，而不是两者之和"。

实测（`co=0.6`）：batch 8 时税 1.63–3.48、净 0.33–0.74（亏）；batch 256
时税 1.15、净 **1.04（pd_a）/ 1.13（hot_cold）**（小赚）。**hot_cold 比
pd_a 更划算**，因为它把两个池的负载配得更平。

**二、集合通信重叠还剩多少？** `collective_overlap` 是把 MoE 的 all-to-all
藏到计算背后的比例（DualPipe / SGLang 两批重叠做的事）。扫描 0 → 0.9：

| 臂 | batch | 随 `co` 的加速 |
|---|---|---|
| pd_a @256 | 8 | **1.63×** |
| pd_a @256 | 64 | **2.23×** |
| hot_cold @256 | 8 | **1.36×** |
| hot_cold @256 | 64 | **1.79×** |

**这是本项目里单项最大的收益**，量级远超微批那 ±5%。而且 hot_cold 的
collective 占比更低（40–54% vs pd_a 的 53–65%），所以它对调度质量的依赖
也更小——吞吐更高的同时更"抗调度差"。

### 这个脚本的两点定位

- A 节**刻意重写了** `execute_pipelined` 的算术，因为包只暴露乘积、不暴露
  两半。它用了私有的 `_scale_graph`，所以只当诊断，不算包 API。
- B 节扫的是**单个 decode step 的图**，不是服务仿真——结论是"这一步能快
  多少"，不是"SLO 容量能涨多少"。

## 注意事项

- `capacity_timing.py` 里 `_probe` 是 monkey-patch（只在本进程内），
  目的是不改包就能计时。
- 这些脚本刻意写得很直白，没有做参数校验；参数错了会直接抛异常。
- 机器上如果同时跑着编辑器/浏览器，长命令放到后台会话会长时间拿不到 CPU
  （见总报告 §4.6），建议前台短跑并开 `-v`。
