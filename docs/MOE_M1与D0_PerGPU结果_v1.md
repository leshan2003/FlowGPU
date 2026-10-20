# M1 全模型与 D0-per-GPU 结果 v1

> 日期：2026-10-20
>
> 证据类型：synthetic routing + projected hardware

## 1. D0-per-GPU 的有效语义

早期尝试把 16 个请求入口映射到 16 个 GPU endpoint，但每个 expert 只有一份
GPU 副本，导致远端 GPU 没有合法数据路径，结果无效。

本轮采用物理一致的 owner-GPU 模型：

```text
4 个 GPU
每个 GPU 都保存全部 Mixtral experts
Mixtral routed weights: 42 GiB / GPU
H100 usable HBM: 64 GB / GPU after 20% reserve
请求 request_id 绑定 owner GPU
GPU-direct: owner GPU → expert node → owner GPU
```

对比：

| Variant | GPU expert copies | GPU ingress |
|---|---|---|
| D0-shared-minimal | 全模型每 expert 一份，跨 4 GPU 分布 | 所有请求共用一个 endpoint |
| D0-shared-replicated | 每个 GPU 持有全部 experts | 所有请求共用一个 endpoint |
| D0-per-gpu | 每个 GPU 持有全部 experts | 每个请求绑定独立 GPU endpoint |

三者使用相同 4 个 GPU、2 个 Brain nodes、8 个 Brain hot copies 和同一 trace。

## 2. D0-per-GPU 结果

Mixtral 完整 32 层，5 paired seeds，batch 1/4/16。

相对 GPU-only 的 completion 加速：

| Batch | Shared-minimal | Shared-replicated | Per-GPU |
|---:|---:|---:|---:|
| 1 | **1.074** | 1.060 | 1.062 |
| 4 | 1.054 | 1.039 | 1.051 |
| 16 | **1.071** | 1.046 | 1.047 |

GPU busy fraction：

| Batch | Shared-minimal | Shared-replicated | Per-GPU |
|---:|---:|---:|---:|
| 1 | 51.4% | 45.0% | 25.0% |
| 4 | 56.5% | 70.1% | 83.9% |
| 16 | 54.7% | 95.7% | 92.7% |

### 结论

1. **合法 per-GPU owner 模型可以运行，且方向为正。**

2. **per-GPU 相比 shared-replicated 有小幅收益。**
   batch 1 约 `1.062 / 1.060`，batch 4 约 `1.051 / 1.039`，batch 16 基本相同。
   说明分散 GPU ingress 能减少单 endpoint bridge 排队，但幅度有限。

3. **per-GPU 不一定优于 shared-minimal。**
   三个 batch 中 shared-minimal 仍然总体最好。

4. **GPU busy fraction 高不等于性能高。**
   per-GPU 在 batch 4/16 的 busy fraction 达到 84%–93%，但由于请求被绑定到
   owner GPU，负载分配不如全局调度灵活，completion 反而没有提高。

5. **request-to-GPU 绑定会引入负载不均衡。**
   需要 owner-aware load balancing 或动态 request migration，否则 per-GPU
   只是增加 HBM 副本并降低全局调度自由度。

## 3. M1 全模型 B-stage

M1 语义：

```text
全部 experts 在 Brain DRAM
hot experts 在 Brain SRAM
cold expert 按 layer 调用 grouped fetch，最多一次/expert/layer
计 DRAM bandwidth + latency
不使用 GPU expert fallback
```

实验：

```text
Mixtral 完整 32 层
5 paired seeds
1 Brain node
hot experts = 8 / 16 / 24
batch = 1 / 4，另测 hot24 的 16 / 64
```

### 结果

相对 GPU-only 的 completion 加速：

| Hot experts | Batch | M1 / GPU speedup | Hot assignment fraction |
|---:|---:|---:|---:|
| 8 | 1 | 0.0194× | 8.9% |
| 8 | 4 | 0.0286× | 8.5% |
| 16 | 1 | 0.0198× | 16.3% |
| 16 | 4 | 0.0299× | 16.7% |
| 24 | 1 | 0.0199× | 23.0% |
| 24 | 4 | 0.0313× | 24.9% |
| 24 | 16 | 0.0587× | 26.4% |
| 24 | 64 | 0.1573× | 26.5% |

### 结论

1. 在当前 projected profile 下，M1 完整 B-stage 比 GPU HBM 路径慢约
   `32×–52×`（batch 1/4），batch 64 仍慢约 `6.4×`。

2. 提高 hot coverage 只能带来有限改善。
   hot=8→24 时，batch 1 从 `0.0194×` 变为 `0.0199×`，batch 4 从
   `0.0286×` 变为 `0.0313×`。

3. batch 增大会摊薄 grouped fetch：
   batch 64、hot=24 时达到 `0.1573×`，但仍远低于 GPU-only。

4. 该比较是 **mode-only hardware frontier**：
   M1 使用 Brain DRAM/SRAM，GPU-only 使用 16 张 H100。  
   因此不能据此声称相同成本、相同功耗下 GPU 更优。

5. 结论严格依赖 projected profile：

```text
Brain DRAM 200 GB/s
DRAM latency 800 ns
expert 168 MiB
source-aware grouped full-expert staging
无 prefetch/streaming overlap
```

如果真实 DRAM 带宽更高、支持 block streaming/overlap，或 GPU/bridge 饱和，
M1 的相对位置可能改变。

## 4. 对“全部放类脑”的最终判断

当前全模型实验可以明确回答：

> 在现有 projected HP300 profile 和 grouped full-expert B-stage 模型下，
> 全部专家由 Brain DRAM/SRAM 服务显著慢于 GPU/HBM。

反过来说：

> “HBM + GPU fallback”以及“HBM + Brain hot overlay + GPU 并行”在当前条件下
> 明显优于全类脑 B-stage。

但该结论不是永远的硬件定律：

- M1 没有完整 streaming、prefetch 和 overlap；
- GPU-only 使用更多 GPU 硬件；
- 没有真实 KA200 DRAM/NoC 测量；
- 没有相同成本/功耗约束；
- 没有考虑 GPU/bridge 真实饱和区间。

## 5. 当前证据等级

可以支持：

- D0 owner-GPU 全专家副本模型可运行且无非法远端路由；
- 分散 GPU ingress 相对单共享 endpoint 有有限收益；
- request-owner 固定绑定可能导致负载不均；
- M1 full B-stage 在 projected profile 下显著落后 GPU/HBM；
- 提高 hot coverage 与 batch 能改善 M1，但尚不足以逆转。

不能支持：

- D0-per-GPU 在所有 GPU 数和 batch 下更优；
- M1 在所有真实硬件上更差；
- 相同成本/功耗下 GPU 或类脑的优劣；
- per-GPU 模型可外推到 DeepSeek/全模型 HBM 容量不足场景。

## 6. Artifacts

- D0-per-GPU：
  [`moe_2026_d0_pergpu_v1`](../experiments/moe_2026_d0_pergpu_v1/)
- M1 hot=8：
  [`moe_2026_m1_full_v1`](../experiments/moe_2026_m1_full_v1/)
- M1 hot=16：
  [`moe_2026_m1_full_hot16_v1`](../experiments/moe_2026_m1_full_hot16_v1/)
- M1 hot=24：
  [`moe_2026_m1_full_hot24_v1`](../experiments/moe_2026_m1_full_hot24_v1/)
- M1 high-batch：
  [`moe_2026_m1_full_hot24_highbatch_v1`](../experiments/moe_2026_m1_full_hot24_highbatch_v1/)
