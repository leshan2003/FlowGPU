# MoE 实验结论速览

## 一句话结论

在当前的合成 workload 和投影硬件仿真中：

> HBM 保留全部专家、类脑 SRAM 额外存放热点专家，是最稳定的方案。
> 单纯优先复制热点、类脑独占专家、共激活聚类和动态刷新，尚未得到确定收益。

当前实验是：

```text
合成 workload + 投影/估计硬件
```

不是 KA200 真机实验，也不是真实业务 trace。

## 实验完成了吗

计划内的仿真实验已经完成：

- 4 个模型：Mixtral、DeepSeek-V3、Qwen3、gpt-oss
- 完整 9 点 node × batch 矩阵
- 9 点 C 隔离实验
- 6 个关键场景，每个 20 个 paired seeds
- HBM overlay、类脑独占、Brain DRAM staging
- 副本策略、聚类、迁移和版本切换
- 138 个测试通过
- 34 个最终结果目录，源码哈希差异为 0

尚未完成的是：

- 真实 request-ordered router trace
- KA200 真机 kernel、链路、并发和功耗测量
- 完整模型、长窗口的动态刷新收益实验
- 真实 TCO

## 三种策略是什么意思

| 名称 | 通俗解释 |
|---|---|
| Coverage | 先把更多不同的专家各放一份到类脑，优先扩大覆盖面 |
| Replica | 先给最热的少量专家增加第二份、第三份副本 |
| Joint | 每增加一份副本，都比较“放一个新专家”和“给热专家再加副本”哪个收益高 |

所以：

- Coverage 关心“能覆盖多少不同专家”；
- Replica 关心“热点会不会排队”；
- Joint 关心“下一份 SRAM 空间放哪里最划算”。

## 主要结果

### 1. Coverage 最稳定

20-seed 结果：

| 场景 | Coverage 静态加速 | Overlay serving 加速 |
|---|---:|---:|
| Mixtral Zipf 0.5 | 1.053× | 1.026× |
| Mixtral Zipf 1.3 | 1.085× | 1.033× |
| Mixtral Zipf 2.0 | 1.081× | 1.036× |
| Qwen 四层 | 1.112× | 1.085× |
| DeepSeek 四层 | 1.081× | 1.074× |
| gpt-oss 四层 | 1.176× | 1.051× |

结论：

- HBM + 少量类脑 SRAM 副本有稳定正收益；
- 收益主要来自减少热点专家的 HBM/bridge 访问；
- 类脑不是替代 HBM，而是作为高速热点层更合理。

### 2. 单纯复制热点收益较小

Replica-first 在大多数场景只有约 `0.3%–3%` 的收益，明显弱于 Coverage。

不能只因为专家很热就一直复制。还要看：

- 复制后能减少多少远端访问；
- 是否造成额外排队；
- 是否挤掉了其他专家的覆盖机会。

### 3. Joint 不一定最好

Joint 理论上更精细，但当前实验中：

- Mixtral：Joint 与 Coverage 基本相同；
- DeepSeek：基本相同；
- gpt-oss：Coverage 更好；
- Qwen：Joint 均值略高，但置信区间重叠。

因此目前不能声称 Joint 一定优于 Coverage。

### 4. 类脑独占不是普遍更快

H-exclusive 表示：

- 热专家只放类脑；
- 其他专家只放 GPU；
- 不再保留另一层 fallback。

结果是收益依赖模型和 workload，部分场景置信区间跨过 1。
因此当前最稳妥的选择仍是：

```text
GPU/HBM 保留全部专家
类脑/SRAM 额外存放热点副本
```

### 5. Node 和 Batch 不是越多越好

实验结果有明显非单调性。例如：

| Nodes / Batch | 静态加速 | Serving 加速 |
|---|---:|---:|
| 2 / 2 | 1.103× | 1.032× |
| 2 / 4 | 1.058× | 0.981× |
| 4 / 1 | 1.078× | 1.046× |

增加 node 会带来更多远端通信和调度开销；增加 batch 会改变排队、分组计算和副本负载。
因此必须同时比较吞吐、P99、SLO 和 node efficiency。

### 6. C=1 没有额外串行负担

在相同 64 copies / 64 chips 资源下：

- C=1：静态加速约 `1.275×`；
- C=2 及以上：静态加速约 `1.262×`。

C=1 没有退化，说明“每 chip 只放一个专家”本身不会增加串行负担。
在当前 workload 下，C≥2 之后每 chip 专家数不是主瓶颈。

### 7. 聚类目前没有证明收益

在相同副本数、C=1、相同 node 和相同 workload 下：

- unique placement：`19.2007 µs`
- pair clustering：`19.2007 µs`
- full-hyperedge clustering：`19.2007 µs`

结论：

- 聚类路径不会额外增加时间；
- 当前合成 trace 没有产生明显的 node 通信差异；
- 尚不能证明聚类有效；
- 也不能据此否定真实 trace 中的聚类收益。

### 8. GPU 和类脑可以并行，但粒度有限

在 HBM overlay 模式下：

- 同一专家的不同 token/assignment 可以分别由 GPU 和类脑处理；
- 一个 token 的一次 expert 计算不会拆成 GPU/类脑两半。

也就是说，并行粒度是：

```text
不同 assignment 分到不同设备
```

不是：

```text
同一个 expert kernel 做跨设备张量并行
```

## 刷新实验做到什么程度

已经实现：

- 历史统计窗口；
- drift 检测；
- placement 重算；
- source-aware 迁移；
- migration 和 serving 共享网络带宽；
- 新 placement 完成后再激活；
- 在途请求继续使用旧版本。

当前迁移成本：

| 模型 | 迁移数据量 | 迁移时间 |
|---|---:|---:|
| Mixtral | 2688 MiB | 34.409 ms |
| DeepSeek | 1344 MiB | 13.765 ms |
| Qwen | 576 MiB | 5.900 ms |
| gpt-oss | 403.42 MiB | 4.133 ms |

但目前只重放了前两个 serving layer 的代表流量，没有完成完整模型的长期 refresh 实验。

因此当前只能说：

- 刷新机制已经跑通；
- 迁移成本可以计算；
- 还不能回答多久刷新一次最好；
- 还不能给出完整模型 refresh 的净收益和 P99。

## 最终建议

当前最值得继续投入的方向：

1. 以 `HBM fallback + selective SRAM overlay` 为主方案；
2. 以 Coverage 作为主 baseline；
3. 保留 Joint 作为可调策略，但不能预先假定它最好；
4. 选 node 和 batch 时必须联合看 SLO，不做单调外推；
5. 引入真实 router trace 后，再重点研究聚类与动态刷新；
6. 拿到 KA200 实测数据后，再做最终硬件与 TCO 判断。

## 详细资料

- 完整审计：
  [`MOE_最终实验执行与审计_v1.md`](MOE_最终实验执行与审计_v1.md)
- 实验目录：
  [`experiments/README.md`](../experiments/README.md)
- 结果矩阵：
  [`MOE_仿真完整矩阵结果_v1.md`](MOE_仿真完整矩阵结果_v1.md)
