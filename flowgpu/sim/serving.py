"""Request-level serving simulator: TTFT, TPOT, throughput, energy.

Models a continuous-batching inference server (vLLM/SGLang style):

* chunked prefill with a token budget per iteration
* iteration-level scheduling -- finished sequences leave, waiting ones join
* paged KV cache with real capacity admission control
* optional prefill/decode disaggregation across pools
* micro-batch pipelining across heterogeneous pools

Latency of each iteration comes from :mod:`flowgpu.sim.executor`, memoised on
``(stage, batch bucket, context bucket)`` so a 10-minute trace costs a few
thousand analytic evaluations rather than millions.

Scheduling jitter is modelled explicitly: each iteration's cost is perturbed
by a lognormal draw with per-pool sigma.  GPUs get ``jitter_sigma ~ 0.06-0.12``
(CTA launch skew, cache-state variance, kernel-launch bubbles); statically
scheduled dataflow chips get ``~0.001-0.005``.  This is the "调度抖动" the
reference article blames for GPU tail latency, and it is why p99 TPOT and mean
TPOT can tell different stories.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

from ..mapping.placement import PlacementPolicy
from ..power import model as pmodel
from ..system import System
from ..workload import llm as llm_builder
from ..workload.graph import ModelSpec, R_ATTN
from ..units import dtype_bytes
from .executor import StepResult, execute, execute_pipelined


# =========================================================================
@dataclass
class Request:
    rid: int
    prompt_len: int
    output_len: int
    arrival: float = 0.0
    # filled by the simulator
    start_prefill: float = -1.0
    first_token: float = -1.0
    finish: float = -1.0
    prefilled: int = 0
    generated: int = 0
    token_times: list = field(default_factory=list)

    @property
    def ttft(self) -> float:
        return self.first_token - self.arrival if self.first_token >= 0 else -1

    @property
    def e2e(self) -> float:
        return self.finish - self.arrival if self.finish >= 0 else -1

    @property
    def tpot(self) -> float:
        """Mean inter-token latency after the first token."""
        if self.generated <= 1 or self.first_token < 0 or self.finish < 0:
            return -1.0
        return (self.finish - self.first_token) / (self.generated - 1)


@dataclass
class WorkloadSpec:
    name: str = "workload"
    n_requests: int = 64
    prompt_len: int = 2048
    output_len: int = 256
    prompt_sigma: float = 0.0        # lognormal sigma on prompt length
    output_sigma: float = 0.0
    arrival_rate: float = 0.0        # req/s; 0 = all present at t=0 (offline)
    max_batch: int = 64
    max_running: int = 256
    chunk_tokens: int = 2048         # chunked-prefill token budget
    seed: int = 0

    def generate(self) -> list:
        rng = random.Random(self.seed)
        reqs = []
        t = 0.0
        for i in range(self.n_requests):
            p = self.prompt_len
            o = self.output_len
            if self.prompt_sigma > 0:
                p = max(1, int(p * math.exp(rng.gauss(0, self.prompt_sigma))))
            if self.output_sigma > 0:
                o = max(1, int(o * math.exp(rng.gauss(0, self.output_sigma))))
            if self.arrival_rate > 0:
                t += rng.expovariate(self.arrival_rate)
            reqs.append(Request(rid=i, prompt_len=p, output_len=o, arrival=t))
        return reqs


@dataclass
class SLO:
    ttft: float = 2.0        # seconds
    tpot: float = 0.05       # seconds per output token (20 tok/s/user)


# =========================================================================
@dataclass
class SimResult:
    requests: list = field(default_factory=list)
    wall_time: float = 0.0
    energy: float = 0.0
    device_energy: float = 0.0
    idle_energy: float = 0.0
    host_energy: float = 0.0
    n_prefill_iters: int = 0
    n_decode_iters: int = 0
    busy_by_pool: dict = field(default_factory=dict)
    e_by_pool: dict = field(default_factory=dict)
    e_by_source: dict = field(default_factory=dict)
    t_by_role: dict = field(default_factory=dict)
    t_by_bound: dict = field(default_factory=dict)
    bytes_crossed: float = 0.0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    kv_capacity_tokens: float = 0.0
    peak_kv_tokens: float = 0.0
    warnings: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ---- derived metrics ------------------------------------------------
    def _p(self, values, q):
        v = sorted(x for x in values if x >= 0)
        if not v:
            return float("nan")
        k = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
        return v[k]

    def ttft_stats(self) -> dict:
        v = [r.ttft for r in self.requests]
        return dict(mean=_mean(v), p50=self._p(v, .5), p90=self._p(v, .9),
                    p99=self._p(v, .99), max=max(v) if v else float("nan"))

    def tpot_stats(self) -> dict:
        v = [r.tpot for r in self.requests]
        return dict(mean=_mean(v), p50=self._p(v, .5), p90=self._p(v, .9),
                    p99=self._p(v, .99), max=max(v) if v else float("nan"))

    def e2e_stats(self) -> dict:
        v = [r.e2e for r in self.requests]
        return dict(mean=_mean(v), p50=self._p(v, .5), p90=self._p(v, .9),
                    p99=self._p(v, .99), max=max(v) if v else float("nan"))

    @property
    def output_throughput(self) -> float:
        return self.decode_tokens / self.wall_time if self.wall_time else 0.0

    @property
    def total_throughput(self) -> float:
        n = self.prefill_tokens + self.decode_tokens
        return n / self.wall_time if self.wall_time else 0.0

    @property
    def request_throughput(self) -> float:
        done = sum(1 for r in self.requests if r.finish >= 0)
        dp = max(1, int(self.meta.get("dp", 1)))
        return done * dp / self.wall_time if self.wall_time else 0.0

    @property
    def avg_power(self) -> float:
        return self.energy / self.wall_time if self.wall_time else 0.0

    @property
    def energy_per_output_token(self) -> float:
        return self.energy / self.decode_tokens if self.decode_tokens else 0.0

    @property
    def energy_per_token(self) -> float:
        n = self.prefill_tokens + self.decode_tokens
        return self.energy / n if n else 0.0

    @property
    def tokens_per_joule(self) -> float:
        e = self.energy_per_output_token
        return 1.0 / e if e else 0.0

    def goodput(self, slo: SLO) -> float:
        ok = [r for r in self.requests
              if r.finish >= 0 and 0 <= r.ttft <= slo.ttft
              and (r.tpot < 0 or r.tpot <= slo.tpot)]
        dp = max(1, int(self.meta.get("dp", 1)))
        return len(ok) * dp / self.wall_time if self.wall_time else 0.0

    def slo_attainment(self, slo: SLO) -> float:
        done = [r for r in self.requests if r.finish >= 0]
        if not done:
            return 0.0
        ok = sum(1 for r in done if 0 <= r.ttft <= slo.ttft
                 and (r.tpot < 0 or r.tpot <= slo.tpot))
        return ok / len(done)


def _mean(v):
    v = [x for x in v if x >= 0]
    return sum(v) / len(v) if v else float("nan")


# =========================================================================
class ServingSimulator:
    """Continuous-batching server over a heterogeneous system."""

    def __init__(self, model: ModelSpec, system: System,
                 policy: PlacementPolicy, workload: WorkloadSpec,
                 microbatches: int = 1, jitter: bool = True,
                 tp_world: int | None = None, ep_world: int | None = None,
                 prefill_pool: str | None = None, seed: int = 0,
                 ctx_bucket: int = 512, batch_bucket: int = 8,
                 collective_overlap: float = 0.0, verbose: bool = False):
        self.model = model
        self.system = system
        self.policy = policy
        self.workload = workload
        self.microbatches = max(1, microbatches)
        self.jitter = jitter
        self.rng = random.Random(seed + 1)
        self.ctx_bucket = ctx_bucket
        self.batch_bucket = batch_bucket
        self.collective_overlap = collective_overlap
        self.verbose = verbose
        self._cache = {}
        self.warnings = []

        # parallelism seen by the graph builder
        attn_pool = self._pool_for_role(R_ATTN)
        ffn_pool = self._pool_for_role("moe") or self._pool_for_role("ffn")
        self.tp_world = tp_world if tp_world is not None else (
            system.pools[attn_pool].parallel.tp if attn_pool else 1)
        self.ep_world = ep_world if ep_world is not None else (
            system.pools[ffn_pool].parallel.ep if ffn_pool else 1)
        self.prefill_pool = prefill_pool
        self._plan_memory()

        # Data parallelism: replicas are independent and symmetric, so one
        # replica is simulated and its throughput/active-energy scaled by dp.
        # (Simulating all replicas would give identical per-request latency
        # and dp-times the wall clock -- the same answer, dp times slower.)
        self.dp = max(1, self.system.pools[self.attn_pool].parallel.dp)
        for nm, pl in self.system.pools.items():
            if pl.parallel.dp != self.dp:
                self.warnings.append(
                    f"pool '{nm}' has dp={pl.parallel.dp} but the attention "
                    f"pool has dp={self.dp}; data-parallel replicas must be "
                    f"consistent across pools or the fleet is mis-counted")
        if self.dp > 1:
            self.workload = replace(
                self.workload,
                n_requests=max(1, self.workload.n_requests // self.dp),
                arrival_rate=self.workload.arrival_rate / self.dp)

    # ------------------------------------------------------------------
    def _pool_for_role(self, role: str) -> str | None:
        from ..workload.graph import Op
        probe = Op(name=f"L1.probe_{role}", kind="gemm", role=role,
                   stage="decode", layer=1)
        try:
            return self.policy.place(probe, self.model.n_layers)
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _plan_memory(self):
        """Split the model's weights across pools according to the policy.

        Weight residency is a *static* property of the placement, so it is
        computed once from a representative decode graph.
        """
        s = self.model
        g = llm_builder.build_decode(s, batch=1, ctx=1,
                                     ep_world=self.ep_world,
                                     tp_world=self.tp_world)
        self.policy.annotate(g, s.n_layers)

        w_by_pool = {}
        seen = set()
        for op in g.ops:
            pool = op.meta["pool"]
            for t in op.weights:
                if t.name in seen:
                    continue
                seen.add(t.name)
                nb = t.nbytes
                if op.role == "moe" and "experts_touched" in op.meta:
                    # the decode-1 graph only names the experts it touched;
                    # the *pool* must hold all of them
                    et = max(op.meta["experts_touched"], 1e-9)
                    if "expert_pool" in t.name:
                        nb = nb / et * op.meta["n_experts"]
                if op.role == "embed":
                    nb = s.vocab * s.d_model * dtype_bytes(s.w_dtype)
                w_by_pool[pool] = w_by_pool.get(pool, 0.0) + nb

        self.weights_by_pool = w_by_pool

        # KV cache lives with attention
        attn_pool = self._pool_for_role(R_ATTN) or self.policy.default_pool
        self.attn_pool = attn_pool
        kv_per_tok = s.kv_bytes_per_token()
        pool = self.system.pools[attn_pool]
        w_here = w_by_pool.get(attn_pool, 0.0)
        per_dev_free = max(
            0.0, pool.device.capacity - w_here / max(1, pool.stage_devices)
            - 2 * 2**30)                       # runtime/activation reserve
        total_free = per_dev_free * pool.count / max(1, pool.replicas)
        self.kv_capacity_tokens = total_free / max(kv_per_tok, 1.0)

        for name, p in self.system.pools.items():
            p.plan(w_by_pool.get(name, 0.0), 0.0, 0.0)
            per_dev = w_by_pool.get(name, 0.0) / max(1, p.stage_devices)
            if per_dev > p.device.capacity:
                self.warnings.append(
                    f"pool '{name}': needs {per_dev/2**30:.1f} GiB/device but "
                    f"{p.device.name} has {p.device.capacity/2**30:.1f} GiB "
                    f"-- weights spill "
                    f"({p.residency.spill_frac*100:.0f}% off-chip)")
        if self.kv_capacity_tokens < self.workload.prompt_len:
            self.warnings.append(
                f"KV capacity {self.kv_capacity_tokens:.0f} tokens < one "
                f"prompt ({self.workload.prompt_len}) -- system cannot serve "
                f"this workload")

    # ------------------------------------------------------------------
    def _bucket(self, batch: int, ctx: int, stage: str = "decode") -> tuple:
        """Memoisation key.

        Prefill batches are small and the token count per iteration is already
        capped by ``chunk_tokens``, so rounding the batch *up* there would
        inflate FLOPs by up to 8x.  Prefill therefore uses the exact batch;
        only decode -- where the batch is large and the cost is dominated by
        batch-independent weight streaming -- is bucketed.
        """
        c = max(1, int(math.ceil(ctx / self.ctx_bucket)) * self.ctx_bucket)
        if stage == "prefill" or self.batch_bucket <= 1:
            return batch, c
        b = max(self.batch_bucket,
                int(round(batch / self.batch_bucket)) * self.batch_bucket)
        return b, c

    def step_cost(self, stage: str, batch: int, ctx: int,
                  chunk: int = 1) -> StepResult:
        key = (stage, *self._bucket(batch, ctx, stage), chunk)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        b, c = key[1], key[2]
        if stage == "prefill":
            g = llm_builder.build_prefill(self.model, b, chunk,
                                          ctx_before=max(0, c - chunk),
                                          ep_world=self.ep_world,
                                          tp_world=self.tp_world)
        else:
            g = llm_builder.build_decode(self.model, b, c,
                                         ep_world=self.ep_world,
                                         tp_world=self.tp_world)
        self.policy.annotate(g, self.model.n_layers)
        if self.microbatches > 1 and stage == "decode":
            r = execute_pipelined(g, self.system, self.policy,
                                  microbatches=self.microbatches,
                                  n_layers=self.model.n_layers,
                                  collective_overlap=self.collective_overlap)
        else:
            r = execute(g, self.system, self.policy,
                        n_layers=self.model.n_layers,
                        collective_overlap=self.collective_overlap)
        # scale from the bucketed batch back to the requested one
        self._cache[key] = r
        return r

    def _scaled(self, r: StepResult, batch: int, bucket_batch: int
                ) -> tuple:
        """Linear interpolation between buckets for time and energy."""
        f = batch / max(1, bucket_batch)
        # decode time is dominated by weight streaming (batch-independent)
        # plus a per-token term; interpolate conservatively on energy only
        return r.time, r.energy * (0.35 + 0.65 * f)

    def _jittered(self, t: float) -> float:
        if not self.jitter:
            return t
        sig = 0.0
        for p in self.system.pools.values():
            sig = max(sig, getattr(p.device, "jitter_sigma", 0.0))
        if sig <= 0:
            return t
        return t * math.exp(self.rng.gauss(-0.5 * sig * sig, sig))

    # ------------------------------------------------------------------
    def _plan_prefill(self, running: list, w: WorkloadSpec):
        need = [r for r in running if r.prefilled < r.prompt_len]
        if not need:
            return None
        budget = w.chunk_tokens
        batch, chunk = [], 0
        for r in need:
            take = min(r.prompt_len - r.prefilled, budget)
            if take <= 0:
                break
            batch.append((r, take))
            chunk = max(chunk, take)
            budget -= take
            if budget <= 0 or len(batch) >= w.max_batch:
                break
        if not batch:
            return None
        ctx = int(max(r.prefilled + t for r, t in batch))
        cost = self.step_cost("prefill", len(batch), ctx, chunk=chunk)
        return dict(batch=batch, cost=cost,
                    pools=set(cost.busy_by_pool) or {self.attn_pool})

    def _plan_decode(self, running: list, w: WorkloadSpec, now: float):
        ready = [r for r in running
                 if r.prefilled >= r.prompt_len and r.finish < 0
                 and r.first_token >= 0]
        if not ready:
            return None
        batch = ready[: w.max_batch]
        ctx = int(max(r.prompt_len + r.generated for r in batch))
        cost = self.step_cost("decode", len(batch), ctx)
        return dict(batch=batch, cost=cost,
                    pools=set(cost.busy_by_pool) or {self.attn_pool})

    # ------------------------------------------------------------------
    def run(self, max_iters: int = 2_000_000) -> SimResult:
        w = self.workload
        reqs = w.generate()
        pending = list(reqs)
        running = []      # prefilling or decoding
        done = []
        now = 0.0
        res = SimResult(requests=reqs)
        res.kv_capacity_tokens = self.kv_capacity_tokens
        res.warnings = list(self.warnings)
        kv_used = 0.0
        it = 0

        # Per-pool availability clocks.  A prefill iteration and a decode
        # iteration may run *concurrently* iff they touch disjoint pools --
        # which is exactly what prefill/decode disaggregation buys you, and
        # what a single global clock would hide.
        pool_free = {name: 0.0 for name in self.system.pools}
        pf_ready, dc_ready = 0.0, 0.0     # engine (not pool) availability

        while (pending or running) and it < max_iters:
            it += 1
            # ---- admit -------------------------------------------------
            while pending and pending[0].arrival <= now and \
                    len(running) < w.max_running:
                r = pending[0]
                need = r.prompt_len + r.output_len
                if kv_used + need > self.kv_capacity_tokens:
                    break
                pending.pop(0)
                kv_used += need
                r.start_prefill = now
                running.append(r)
            if not running:
                if pending and pending[0].arrival > now:
                    now = pending[0].arrival
                    continue
                if pending:
                    # nothing running and nothing admissible: the KV cache
                    # cannot even hold a single request on this system
                    res.warnings.append(
                        f"INFEASIBLE: KV capacity {self.kv_capacity_tokens:.0f} "
                        f"tokens cannot hold one request of "
                        f"{pending[0].prompt_len + pending[0].output_len} "
                        f"tokens; {len(pending)} requests never admitted")
                    break
                break

            # ---- build both candidate iterations -----------------------
            pf = self._plan_prefill(running, w)
            dc = self._plan_decode(running, w, now)

            # An iteration cannot start before the current simulated time --
            # if the machine has been idle waiting for arrivals, its pools are
            # free but the clock has moved on.  Omitting `now` here lets work
            # execute "before" the request that caused it arrived, which shows
            # up as negative TTFT.
            cand = []
            if pf is not None:
                start = max([now, pf_ready] + [pool_free[p] for p in pf["pools"]])
                cand.append(("prefill", start, pf))
            if dc is not None:
                start = max([now, dc_ready] + [pool_free[p] for p in dc["pools"]])
                cand.append(("decode", start, dc))
            if not cand:
                break
            # earliest-start wins; prefill breaks ties so TTFT is not starved
            cand.sort(key=lambda c: (c[1], c[0] != "prefill"))
            kind, start, plan = cand[0]

            sr = plan["cost"]
            dt = self._jittered(sr.time)
            end = start + dt
            for p in plan["pools"]:
                pool_free[p] = end
            _accumulate(res, sr, dt)

            if kind == "prefill":
                pf_ready = end
                res.n_prefill_iters += 1
                for r, take in plan["batch"]:
                    r.prefilled += take
                    res.prefill_tokens += take
                    if r.prefilled >= r.prompt_len:
                        r.first_token = end
                        r.generated = 1
                        r.token_times.append(end)
                        res.decode_tokens += 1
            else:
                dc_ready = end
                res.n_decode_iters += 1
                for r in plan["batch"]:
                    r.generated += 1
                    r.token_times.append(end)
                    res.decode_tokens += 1
                    if r.generated >= r.output_len:
                        r.finish = end

            # advance to the earliest moment any engine could start again
            nxt = [v for v in (pf_ready, dc_ready) if v > 0]
            now = max(now, min(nxt) if nxt else end)
            res.peak_kv_tokens = max(res.peak_kv_tokens, kv_used)

            finished = [r for r in running if r.finish >= 0]
            for r in finished:
                running.remove(r)
                done.append(r)
                kv_used -= (r.prompt_len + r.output_len)

        now = max([now] + list(pool_free.values()))
        res.wall_time = now
        # ---- power accounting -------------------------------------------
        # one replica was simulated; scale the work it did up to the fleet
        if self.dp > 1:
            res.energy *= self.dp
            res.prefill_tokens = int(res.prefill_tokens * self.dp)
            res.decode_tokens = int(res.decode_tokens * self.dp)
            res.bytes_crossed *= self.dp
            for d in (res.e_by_pool, res.e_by_source):
                for k in list(d):
                    d[k] *= self.dp
            res.n_prefill_iters *= self.dp
            res.n_decode_iters *= self.dp
        res.device_energy = res.energy
        idle = pmodel.idle_energy(self.system, now, res.busy_by_pool)
        res.idle_energy = sum(idle.values())
        res.host_energy = pmodel.host_energy(self.system, now)
        res.energy = (res.device_energy + res.idle_energy
                      + res.host_energy) * self.system.pue
        for k, v in idle.items():
            res.e_by_pool[k] = res.e_by_pool.get(k, 0.0) + v
        res.e_by_source["idle"] = res.idle_energy
        res.e_by_source["host"] = res.host_energy
        res.meta.update(dict(
            model=self.model.name, system=self.system.name,
            placement=self.policy.name, microbatches=self.microbatches,
            tp=self.tp_world, ep=self.ep_world, dp=self.dp,
            weights_by_pool={k: v / 2**30
                             for k, v in self.weights_by_pool.items()},
            iters=it))
        return res


def _accumulate(res: SimResult, sr: StepResult, dt: float):
    scale = dt / sr.time if sr.time > 0 else 1.0
    res.energy += sr.energy
    res.bytes_crossed += sr.bytes_crossed
    for k, v in sr.busy_by_pool.items():
        res.busy_by_pool[k] = res.busy_by_pool.get(k, 0.0) + v * scale
    for k, v in sr.e_by_pool.items():
        res.e_by_pool[k] = res.e_by_pool.get(k, 0.0) + v
    for k, v in sr.e_by_source.items():
        res.e_by_source[k] = res.e_by_source.get(k, 0.0) + v
    for k, v in sr.t_by_role.items():
        res.t_by_role[k] = res.t_by_role.get(k, 0.0) + v * scale
    for k, v in sr.t_by_bound.items():
        res.t_by_bound[k] = res.t_by_bound.get(k, 0.0) + v * scale
    for wmsg in sr.warnings:
        if wmsg not in res.warnings:
            res.warnings.append(wmsg)
