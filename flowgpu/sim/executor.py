"""Graph executor: prices one graph on one system under one placement.

This is the analytic core.  It walks the op sequence, charges each op to the
pool it was placed on, charges collectives to the right fabric, and charges a
bridge traversal every time consecutive ops change pool.

It returns a :class:`StepResult` carrying not just a latency but a full
attribution -- per pool, per role, per bound (compute / memory / network /
bridge) -- because "the heterogeneous system is 2x faster" is only interesting
if you can say *why*, and only trustworthy if the breakdown adds up.

Overlap model
-------------
Within one graph the dependency chain is strictly sequential, so latency is
the sum of op costs.  Pool-level *concurrency* comes from micro-batching: with
``microbatches = m``, the GPU works on micro-batch ``i+1``'s attention while
the dataflow pool is still on micro-batch ``i``'s FFN.  That turns a serial
sum into a pipeline whose period is the slowest stage, and it is how a real
PD+A system would be built.  :func:`execute_pipelined` models it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..hardware.base import OpCost
from ..mapping.placement import PlacementPolicy
from ..system import DevicePool, System
from ..units import dtype_bytes
from ..workload import graph as G


@dataclass
class StepResult:
    time: float = 0.0
    energy: float = 0.0
    # attribution
    t_by_pool: dict = field(default_factory=dict)
    t_by_role: dict = field(default_factory=dict)
    t_by_bound: dict = field(default_factory=dict)
    e_by_pool: dict = field(default_factory=dict)
    e_by_source: dict = field(default_factory=dict)
    busy_by_pool: dict = field(default_factory=dict)
    bytes_crossed: float = 0.0
    n_crossings: int = 0
    t_bridge: float = 0.0
    t_collective: float = 0.0
    flops: float = 0.0
    warnings: list = field(default_factory=list)
    op_trace: list = field(default_factory=list)

    @property
    def achieved_flops(self) -> float:
        return self.flops / self.time if self.time > 0 else 0.0

    @property
    def avg_power(self) -> float:
        return self.energy / self.time if self.time > 0 else 0.0

    def merge(self, other: "StepResult", serial: bool = True) -> "StepResult":
        out = StepResult()
        out.time = self.time + other.time if serial else max(self.time,
                                                             other.time)
        out.energy = self.energy + other.energy
        out.flops = self.flops + other.flops
        out.bytes_crossed = self.bytes_crossed + other.bytes_crossed
        out.n_crossings = self.n_crossings + other.n_crossings
        out.t_bridge = self.t_bridge + other.t_bridge
        out.t_collective = self.t_collective + other.t_collective
        for attr in ("t_by_pool", "t_by_role", "t_by_bound", "e_by_pool",
                     "e_by_source", "busy_by_pool"):
            d = {}
            for src in (getattr(self, attr), getattr(other, attr)):
                for k, v in src.items():
                    d[k] = d.get(k, 0.0) + v
            setattr(out, attr, d)
        out.warnings = self.warnings + other.warnings
        return out


def _bump(d: dict, k, v):
    d[k] = d.get(k, 0.0) + v


# =========================================================================
def _collective_cost(op: G.Op, pool: DevicePool) -> tuple:
    """Charge a TP all-reduce or EP all-to-all to the right fabric.

    The graph is built before placement is known, so it carries the *model's*
    parallelism degree.  The rank count that actually participates is a
    property of the pool the op landed on: an FFN placed on an expert-parallel
    dataflow pool with ``tp=1`` does no tensor-parallel all-reduce at all, and
    charging one would invent traffic that the hardware never moves.
    """
    meta = op.meta
    coll = meta.get("collective")
    nbytes = meta.get("bytes", op.input_bytes)
    world = int(meta.get("world", 1))
    if coll == "allreduce":
        world = min(world, max(1, pool.parallel.tp))
    elif coll == "all2all":
        world = min(world, max(1, pool.parallel.ep))
    if world <= 1 or not coll:
        return 0.0, 0.0
    link = pool.link_for(world)
    if link is None:
        return 0.0, 0.0
    if coll == "allreduce":
        t = link.allreduce_time(nbytes, world)
    elif coll == "all2all":
        t = link.all2all_time(nbytes, world)
    elif coll == "allgather":
        t = link.allgather_time(nbytes, world)
    else:
        t = link.p2p_time(nbytes)
    wire = link.collective_bytes(nbytes, coll, world)
    e = wire * link.energy_per_byte * world
    return t, e


def execute(graph: G.Graph, system: System, policy: PlacementPolicy,
            n_layers: int | None = None, trace: bool = False,
            residency_override: dict | None = None,
            collective_overlap: float = 0.0) -> StepResult:
    """Price ``graph`` on ``system`` under ``policy``.  Serial (no pipelining).

    ``collective_overlap`` is the fraction of *overlappable* collective time
    (MoE dispatch/combine) that a production engine hides behind compute --
    DeepSeek's DualPipe and SGLang's two-batch overlap both do this.  0.0 is a
    naive engine; 0.5-0.8 is what a tuned one achieves.  TP all-reduces are
    marked non-overlappable because they sit on the critical path of the very
    GEMM that produced their input.
    """
    n_layers = n_layers or (max((o.layer for o in graph.ops), default=0) + 1)
    res = StepResult()
    prev_pool = None
    prev_out_bytes = 0.0

    for op in graph.ops:
        pname = op.meta.get("pool") or policy.place(op, n_layers)
        op.meta["pool"] = pname
        pool = system.pools.get(pname)
        if pool is None:
            res.warnings.append(
                f"op {op.name} placed on unknown pool {pname!r}; "
                f"falling back to {policy.default_pool!r}")
            pool = system.pools[policy.default_pool]
            pname = policy.default_pool

        # ---- bridge crossing -------------------------------------------
        if prev_pool is not None and pname != prev_pool:
            link = system.bridge(prev_pool, pname)
            xb = prev_out_bytes
            if policy.crossing_dtype:
                # quantise the hidden state before it goes on the wire
                src_dt = op.inputs[0].dtype if op.inputs else "bf16"
                xb *= dtype_bytes(policy.crossing_dtype) / dtype_bytes(src_dt)
            if link is None:
                res.warnings.append(
                    f"no bridge between {prev_pool!r} and {pname!r}; "
                    f"crossing charged at 0 -- results will be optimistic")
                tb = 0.0
            else:
                tb = link.p2p_time(xb)
                res.energy += link.p2p_energy(xb)
                _bump(res.e_by_source, "bridge", link.p2p_energy(xb))
            res.time += tb
            res.t_bridge += tb
            res.bytes_crossed += xb
            res.n_crossings += 1
            _bump(res.t_by_bound, "bridge", tb)

        # ---- the op itself ---------------------------------------------
        if op.kind in (G.ALLREDUCE, G.A2A):
            t, e = _collective_cost(op, pool)
            if op.meta.get("overlappable") and collective_overlap > 0:
                # TP all-reduces sit on the critical path of the GEMM that
                # produced their input, so only a fraction can be hidden
                # (Megatron async-TP / chunked overlap); MoE dispatch and
                # combine can be hidden almost entirely (DualPipe).
                sc = float(op.meta.get("overlap_scale", 1.0))
                t *= max(0.0, 1.0 - collective_overlap * sc)
            res.time += t
            res.t_collective += t
            res.energy += e
            _bump(res.t_by_pool, pname, t)
            _bump(res.t_by_role, op.role, t)
            _bump(res.t_by_bound, "collective", t)
            _bump(res.e_by_pool, pname, e)
            _bump(res.e_by_source, "collective", e)
            _bump(res.busy_by_pool, pname, t)
            if trace:
                res.op_trace.append((op.name, pname, t, 0.0, "collective"))
            prev_pool, prev_out_bytes = pname, op.output_bytes
            continue

        residency = (residency_override or {}).get(pname) or pool.residency
        if residency is None:
            residency = pool.device.plan_residency(0.0, 0.0, 0.0)
        shards = pool.stage_devices
        cost: OpCost = pool.device.cost_op(op, residency, shards=shards)

        pool_energy = cost.energy * shards
        res.time += cost.time
        res.energy += pool_energy
        res.flops += op.flops
        _bump(res.t_by_pool, pname, cost.time)
        _bump(res.t_by_role, op.role, cost.time)
        _bump(res.t_by_bound, cost.bound, cost.time)
        _bump(res.e_by_pool, pname, pool_energy)
        _bump(res.busy_by_pool, pname, cost.time)
        for k, v in cost.energy_by_source.items():
            _bump(res.e_by_source, k, v * shards)
        if trace:
            res.op_trace.append((op.name, pname, cost.time, cost.util,
                                 cost.bound))

        prev_pool, prev_out_bytes = pname, op.output_bytes

    return res


# =========================================================================
def execute_pipelined(graph: G.Graph, system: System, policy: PlacementPolicy,
                      microbatches: int = 1, n_layers: int | None = None,
                      trace: bool = False,
                      collective_overlap: float = 0.0) -> StepResult:
    """Model micro-batch pipelining across pools.

    With ``m`` micro-batches the per-pool work is ``1/m`` of the full step and
    the pools run concurrently.  Steady-state period is the busiest pool; the
    whole step costs ``fill + m * period`` where ``fill`` is one serial pass.

    Pipelining does not reduce energy -- it converts serial pool idle time into
    overlap -- so energy is taken from the serial execution.
    """
    if microbatches <= 1:
        return execute(graph, system, policy, n_layers=n_layers, trace=trace,
                       collective_overlap=collective_overlap)

    mb = _scale_graph(graph, 1.0 / microbatches)
    sub = execute(mb, system, policy, n_layers=n_layers, trace=trace,
                  collective_overlap=collective_overlap)

    # Micro-batching only pays when there are at least two pools to overlap.
    # On a single pool it is strictly worse for a weight-bound decode step:
    # halving the tokens barely halves the time (the weights still have to be
    # streamed) but you now do it twice.  A real scheduler would not choose
    # that, so neither does the simulator -- fall back to the serial cost and
    # say so.
    if len(sub.busy_by_pool) < 2:
        serial = execute(graph, system, policy, n_layers=n_layers,
                         trace=trace, collective_overlap=collective_overlap)
        serial.warnings.append(
            f"microbatches={microbatches} ignored: the workload touches only "
            f"pool {list(sub.busy_by_pool) or ['?']} , so there is nothing to "
            f"overlap with and micro-batching would only re-stream weights")
        return serial

    # steady-state period = slowest pool's share, plus the serial glue
    # (bridge + collectives cannot overlap with themselves)
    pool_times = dict(sub.busy_by_pool)
    period = max(list(pool_times.values()) + [0.0])
    period = max(period, sub.t_bridge + sub.t_collective)
    fill = sub.time - period

    if fill + microbatches * period >= sub.time * microbatches:
        # pipelining bought nothing -- the serial schedule is no worse and is
        # what a real scheduler would run
        serial = execute(graph, system, policy, n_layers=n_layers,
                         trace=trace, collective_overlap=collective_overlap)
        if serial.time <= fill + microbatches * period:
            return serial

    out = StepResult()
    out.time = max(sub.time, fill + microbatches * period)
    out.energy = sub.energy * microbatches
    out.flops = sub.flops * microbatches
    out.bytes_crossed = sub.bytes_crossed * microbatches
    out.n_crossings = sub.n_crossings * microbatches
    out.t_bridge = sub.t_bridge * microbatches
    out.t_collective = sub.t_collective * microbatches
    for attr in ("t_by_pool", "t_by_role", "t_by_bound", "e_by_pool",
                 "e_by_source", "busy_by_pool"):
        setattr(out, attr, {k: v * microbatches
                            for k, v in getattr(sub, attr).items()})
    out.warnings = list(sub.warnings)
    out.op_trace = sub.op_trace
    out.t_by_bound["pipeline_fill"] = fill
    return out


def _scale_graph(graph: G.Graph, f: float) -> G.Graph:
    """Shrink a graph's token dimension by ``f`` (for micro-batching)."""
    import copy
    g = copy.deepcopy(graph)
    for op in g.ops:
        op.flops *= f
        op.state_read_bytes *= f
        op.state_write_bytes *= f
        for t in op.inputs + op.outputs:
            if t.shape:
                t.shape = (max(1.0, t.shape[0] * f),) + tuple(t.shape[1:])
        op.m = max(1, int(round(op.m * f)))
        if op.role == G.R_MOE and "experts_touched" in op.meta:
            # Fewer tokens per micro-batch -> fewer *distinct* experts touched
            # -> less weight traffic, but sub-linearly so.  Skipping this would
            # make micro-batching look free, when in fact splitting a decode
            # step in two makes each half re-read most of the same experts:
            # that is precisely why micro-batching a weight-bound step is a
            # bad trade unless it buys cross-pool overlap.
            from ..workload.llm import experts_touched
            E = op.meta["n_experts"]
            k = op.meta["top_k"]
            T_new = max(1.0, op.meta.get("tokens", op.m) * f)
            op.meta["tokens"] = T_new
            nt = experts_touched(E, k, T_new)
            old = op.meta["experts_touched"]
            if old > 0 and op.weights:
                op.weights[0].shape = (op.weights[0].shape[0] * nt / old,)
            op.meta["experts_touched"] = nt
            op.meta["groups"] = nt + op.meta.get("shared_experts", 0)
    g.batch = max(1, int(round(g.batch * f))) if g.batch * f >= 1 else 1
    return g
