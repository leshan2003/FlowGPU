# 已知限制

> **状态：** 本文描述早期通用 v2 MVP。静态副本、无迁移等限制不适用于后续 MoE
> 目标架构；MoE 当前语义、实现缺口和验收合同见
> [`docs/MOE_高置信度实验设计_v2_冻结版.md`](../MOE_高置信度实验设计_v2_冻结版.md)。

## 1. MVP 固有限制

- flow-level 而非 packet-level，不能预测微观 head-of-line blocking；
- kernel 使用查表/解析模型，不模拟指令和 cycle；
- 不做单专家跨芯片 TP；
- 静态副本，不做在线权重迁移；
- decode 主模型；prefill 仅作为外部 block window 在迭代/层边界让行，已启动 kernel
  不抢占，不能代表真实 chunked-prefill 调度；
- 不建模故障、重试和网络丢包；
- synthetic routing 不能替代真实 router trace。

## 2. 可能改变结论的因素

- 真机 GPU↔brain 接线与 ingress 数；
- runtime 是否支持 dispatch 去重、多播和 local combine；
- 真实专家路由偏斜、node affinity 与漂移；
- shared expert 的实现方式；
- 芯片能否并发执行多个专家；
- SRAM 实际可用容量和 buffer 需求；
- 链路并发与交换机 oversubscription；
- expert kernel 在小 `M` 下的真实性能；
- continuous serving 对同层 assignment 的聚合窗口。
- 当前多 seed 区间传播的是 arrival/routing/placement 随机性；在取得校准重复样本前，
  不伪造 kernel/link 参数分布，因此尚无“真机参数 Monte Carlo 置信区间”。

## 3. 不允许的外推

在缺少真实拓扑、kernel 曲线和 routing trace 时：

- 不得声称某真机获得确定加速；
- 不得给出伪精确最优副本数；
- 不得把 hypothetical profile 写成厂商配置；
- 不得把单层固定 batch 结果写成服务吞吐；
- 不得把某个 synthetic skew 的收益推广到真实模型。

## 4. 后续扩展

MVP 验证后可依次加入：

- 单专家 TP/分片；
- 动态副本缓存和迁移；
- prefill/decode 联合调度；
- speculative decoding/MTP；
- packet-level hotspot refinement；
- failure-aware placement；
- 多租户与模型复用；
- 在线路由漂移检测和再放置。

扩展必须保留当前结果契约、provenance 和验证阶梯。
