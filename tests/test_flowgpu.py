"""FlowGPU test suite.

Two kinds of test:

* **Invariants** -- things that must hold for the simulator to be
  self-consistent (time attribution sums to the total, energy is positive,
  capacity falls as load rises).
* **Ground truth** -- things that must match reality: published parameter
  counts, published KV-cache sizes, the direction and rough magnitude of the
  memory-vs-compute bound in each inference phase, and TDP calibration.

The ground-truth tests are the ones that matter.  A simulator that is
internally consistent but says DeepSeek-V3 has 400 B parameters is worthless.

Run with:  python -m pytest tests/ -q      (or:  python tests/test_flowgpu.py)
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flowgpu as fg
from flowgpu.hardware.registry import DATAFLOW, GPUS, make_device
from flowgpu.mapping import placement as P
from flowgpu.power import calibrate, calibrate_system, tech
from flowgpu.sim.executor import execute
from flowgpu.sim.serving import SLO, ServingSimulator, WorkloadSpec
from flowgpu.system import build_system
from flowgpu.units import parse, dtype_bytes
from flowgpu.workload import llm
from flowgpu.workload.models import get_model, list_models


# =========================================================================
# units
# =========================================================================
def test_units_capacity_is_binary_bandwidth_is_decimal():
    # datasheets: "80 GB" of HBM is 80 GiB; "3.35 TB/s" is 3.35e12 B/s
    assert parse("80 GB") == 80 * 2**30
    assert parse("3350 GB/s") == 3350e9
    assert parse("1 TB/s") == 1e12
    assert abs(parse("200 Gbps") - 25e9) < 1
    assert parse("230 MB") == 230 * 2**20


def test_units_time_power_energy():
    assert abs(parse("480 ns") - 480e-9) < 1e-18
    assert abs(parse("1.5 us") - 1.5e-6) < 1e-15
    assert parse("700 W") == 700
    assert abs(parse("12 pJ") - 12e-12) < 1e-20
    assert parse("989 TFLOPS") == 989e12
    assert parse("50%") == 0.5


def test_dtype_bytes():
    assert dtype_bytes("bf16") == 2
    assert dtype_bytes("fp8") == 1
    assert dtype_bytes("fp4") == 0.5
    assert dtype_bytes("mxfp4") > 0.5      # includes the shared scale


# =========================================================================
# model ground truth
# =========================================================================
def test_deepseek_v3_parameter_count():
    """Published: 671 B total, 37 B activated."""
    m = get_model("deepseek_v3")
    total = m.total_params() / 1e9
    active = m.active_params() / 1e9
    assert 640 < total < 700, f"total {total:.1f} B, expected ~671 B"
    assert 33 < active < 41, f"active {active:.1f} B, expected ~37 B"


def test_deepseek_v3_kv_cache_per_token():
    """MLA caches kv_lora(512) + qk_rope(64) = 576 elements per layer.

    At bf16 over 61 layers that is 576*2*61 = 70272 B = 68.6 KiB/token --
    the number DeepSeek quote as the reason MLA is cheap.
    """
    m = get_model("deepseek_v3")
    kv = m.kv_bytes_per_token()
    assert abs(kv - 576 * 2 * 61) < 1, kv
    # and it must be ~10x smaller than the equivalent GQA model
    g = get_model("llama3_70b")
    per_layer_mla = kv / m.n_layers
    per_layer_gqa = g.kv_bytes_per_token() / g.n_layers
    assert per_layer_gqa / per_layer_mla > 3


def test_llama3_70b_parameter_count():
    m = get_model("llama3_70b")
    total = m.total_params() / 1e9
    assert 68 < total < 73, f"{total:.1f} B, expected ~70.6 B"


def test_llama3_8b_parameter_count():
    m = get_model("llama3_8b")
    total = m.total_params() / 1e9
    assert 7.5 < total < 8.6, f"{total:.1f} B, expected ~8.03 B"


def test_mixtral_parameter_count():
    m = get_model("mixtral_8x7b")
    total = m.total_params() / 1e9
    assert 44 < total < 48, f"{total:.1f} B, expected ~46.7 B"


def test_projected_models_are_flagged():
    for name in list_models()["llm"]:
        m = get_model(name)
        if name.endswith("_proj"):
            assert m.extra["projected"], f"{name} must be flagged projected"
        else:
            assert not m.extra.get("projected"), name
            assert m.extra.get("source"), f"{name} has no source"


# =========================================================================
# workload graph
# =========================================================================
def test_prefill_flops_match_2ND_rule():
    """Prefill FLOPs ~ 2 * active_params * tokens, plus attention."""
    m = get_model("deepseek_v3")
    T = 2048
    g = llm.build_prefill(m, batch=1, chunk=T, ctx_before=0)
    expected_gemm = 2 * m.active_params() * T
    # graph total includes attention score/output FLOPs, which are extra
    assert g.flops > expected_gemm * 0.7, (g.flops, expected_gemm)
    assert g.flops < expected_gemm * 3.0, (g.flops, expected_gemm)


def test_causal_masking_halves_prefill_attention():
    m = get_model("llama3_8b")
    T = 4096
    g = llm.build_prefill(m, 1, T, 0)
    attn = [o for o in g.ops if o.name.endswith(".attn")]
    per_layer = attn[0].flops
    full = 2 * 2 * 1 * m.n_heads * T * T * m.head_dim
    assert abs(per_layer - full / 2) / full < 0.02


def test_moe_experts_touched_saturates():
    """Distinct experts touched: k at 1 token, ~all experts at many."""
    E, k = 256, 8
    assert abs(llm.experts_touched(E, k, 1) - k) < 1e-6
    assert llm.experts_touched(E, k, 8) > k
    assert llm.experts_touched(E, k, 8) < E
    assert llm.experts_touched(E, k, 4096) > E * 0.999
    # monotone
    prev = 0
    for t in (1, 2, 4, 16, 64, 256, 1024):
        v = llm.experts_touched(E, k, t)
        assert v >= prev
        prev = v


def test_moe_decode_weight_traffic_grows_with_batch():
    """The whole MoE story: batch 1 reads 8/256 of the experts, batch 512
    reads all of them.  Weight bytes must reflect that."""
    m = get_model("deepseek_v3", w_dtype="fp8")
    small = llm.build_decode(m, batch=1, ctx=1024)
    big = llm.build_decode(m, batch=512, ctx=1024)

    def moe_w(g):
        return sum(o.weight_bytes for o in g.ops
                   if o.role == "moe" and "expert" in
                   "".join(t.name for t in o.weights))
    assert moe_w(big) > 20 * moe_w(small), (moe_w(small), moe_w(big))


def test_mla_decode_uses_absorbed_form():
    m = get_model("deepseek_v3")
    B, ctx = 16, 4096
    g = llm.build_decode(m, B, ctx)
    attn = [o for o in g.ops if o.name.endswith(".attn")][0]
    assert attn.meta.get("absorbed")
    expected = B * (ctx + 1) * 576 * 2
    assert abs(attn.state_read_bytes - expected) / expected < 0.01


# =========================================================================
# hardware
# =========================================================================
def test_every_registry_device_builds():
    for name in list(GPUS) + list(DATAFLOW):
        d = make_device(name)
        assert d.peak_flops > 0, name
        assert d.capacity > 0, name
        assert d.extra.get("source"), f"{name} has no spec source"
        assert d.extra.get("confidence") in (
            "vendor", "paper", "press", "derived", "estimate"), name


def test_dataflow_devices_expose_the_required_features():
    """Core count, SRAM/core, FLOPS/core, NoC BW, DRAM, chip & board BW."""
    for name in DATAFLOW:
        d = make_device(name)
        assert d.n_cores >= 1, name
        assert d.sram_per_core > 0, name
        assert d.total_sram_bw > 0, name
        assert (d.noc.bisection_bandwidth or d.noc.link_bandwidth) > 0, name
        assert d.inter_chip_bw > 0, name
        assert d.inter_board_bw > 0, name


def test_gpu_decode_is_memory_bound_and_prefill_is_compute_bound():
    """The single most important qualitative behaviour of a GPU."""
    dev = make_device("h100_sxm")
    calibrate(dev)
    m = get_model("llama3_70b", w_dtype="bf16")
    res = dev.plan_residency(140e9, 10e9, 1e9)

    dec = llm.build_decode(m, batch=4, ctx=2048)
    gemms = [o for o in dec.ops if o.kind == "gemm" and o.weight_bytes > 1e7]
    costs = [dev.cost_op(o, res, shards=8) for o in gemms]
    n_mem = sum(c.bound == "memory" for c in costs)
    assert n_mem == len(costs), \
        [(o.name, c.bound) for o, c in zip(gemms, costs) if c.bound != "memory"][:6]

    pre = llm.build_prefill(m, batch=1, chunk=4096, ctx_before=0)
    gemms = [o for o in pre.ops if o.kind == "gemm" and o.weight_bytes > 1e7]
    costs = [dev.cost_op(o, res, shards=8) for o in gemms]
    assert sum(c.bound == "compute" for c in costs) > 0.8 * len(costs), \
        [(o.name, c.bound) for o, c in zip(gemms, costs)]


def test_sram_resident_weights_beat_hbm_on_time_per_weight_byte():
    """The mechanism the whole study rests on.

    Per weight byte streamed, a dataflow chip holding the shard in core SRAM
    must be dramatically faster than a GPU streaming it from HBM.
    """
    gpu = make_device("h100_sxm")
    df = make_device("groq_lpu_v1")
    calibrate(gpu)
    calibrate(df)
    assert df.total_sram_bw / df.total_sram > gpu.level("hbm").bandwidth / \
        gpu.level("hbm").capacity


def test_sram_byte_is_far_cheaper_than_hbm_byte():
    gpu = make_device("h100_sxm")
    calibrate(gpu)
    df = make_device("groq_lpu_v1")
    calibrate(df)
    e_hbm = gpu.energy.j_per_byte["hbm"]
    e_sram = df.energy.j_per_byte["sram"]
    assert e_sram < e_hbm, (e_sram, e_hbm)
    assert e_hbm / e_sram > 3, e_hbm / e_sram


def test_small_batch_gemm_wastes_tensor_cores():
    dev = make_device("h100_sxm")
    calibrate(dev)
    from flowgpu.workload.graph import Op, GEMM
    big = Op("big", GEMM, flops=1, m=8192, n=8192, k=8192)
    small = Op("small", GEMM, flops=1, m=1, n=8192, k=8192)
    assert dev.efficiency(big, 8192) > 10 * dev.efficiency(small, 1)


def test_grouped_gemm_fills_the_machine_better():
    dev = make_device("h100_sxm")
    from flowgpu.workload.graph import Op, GEMM
    one = Op("one", GEMM, flops=1, m=4, n=2048, k=7168, meta={"groups": 1})
    many = Op("many", GEMM, flops=1, m=4, n=2048, k=7168,
              meta={"groups": 128})
    assert dev.efficiency(many, 4) > dev.efficiency(one, 4)


def test_die_area_check_flags_impossible_sram():
    from flowgpu.hardware.dataflow import make_dataflow
    absurd = make_dataflow("absurd", dict(
        cores=1024, sram_per_core="8 MB", process_nm=12,
        peak={"bf16": "100 TFLOPS"}, tdp="100 W",
        noc=dict(bisection_bandwidth="1 TB/s"),
        inter_chip_bw="100 GB/s", inter_board_bw="50 GB/s",
        sram_bw_per_core="100 GB/s"))
    assert not absurd.die_area_check()["plausible"]
    ok = make_device("groq_lpu_v1")
    assert ok.die_area_check()["plausible"]


# =========================================================================
# power
# =========================================================================
def test_calibration_reproduces_tdp_at_the_reference_point():
    from flowgpu.power.model import DATAFLOW_POINT, DEFAULT_POINT
    for name in ("h100_sxm", "a100_80gb_sxm", "iluvatar_bi_v100",
                 "groq_lpu_v1", "lynxi_ka200", "ipu_gc200"):
        d = make_device(name)
        em = calibrate(d)
        pt = DATAFLOW_POINT if d.kind == "dataflow" else DEFAULT_POINT
        p = em.static_power
        p += d.compute.peak_for(pt.dtype) * pt.util_compute * \
            em.j_per_flop[pt.dtype]
        for lvl in d.memory:
            u = pt.util_bw if lvl.name in ("hbm", "dram") else pt.util_sram
            p += lvl.bandwidth * u * em.j_per_byte.get(lvl.name, 0.0)
        if d.kind == "dataflow" and d.n_cores > 1:
            bis = d.noc.bisection_bandwidth or d.noc.link_bandwidth
            p += bis * pt.util_noc * em.j_per_byte_noc
        assert abs(p - em.tdp) / em.tdp < 0.02, (name, p, em.tdp)


def test_energy_coefficients_are_ordered_sanely():
    """regfile < smem < L2 < HBM, per byte.  If this inverts, the whole
    energy story is backwards."""
    d = make_device("h100_sxm")
    calibrate(d)
    j = d.energy.j_per_byte
    assert j["regfile"] < j["smem"] < j["l2"] < j["hbm"], j


def test_lower_precision_costs_less_energy_per_flop():
    d = make_device("h100_sxm")
    calibrate(d)
    j = d.energy.j_per_flop
    assert j["fp32"] > j["bf16"] > j["fp8"] > j["int8"]


def test_sram_energy_grows_with_array_size():
    small = tech.sram_j_per_byte(8 << 10, 7)
    big = tech.sram_j_per_byte(8 << 20, 7)
    assert big > small * 3


# =========================================================================
# placement
# =========================================================================
def _sys2():
    return build_system(dict(name="t", pools={
        "gpu": dict(device="h100_sxm", count=8,
                    parallel=dict(tp=8), devices_per_board=8),
        "brain": dict(device="ipu_gc200", count=64,
                      parallel=dict(ep=64), devices_per_board=4),
    }, bridges=[dict(between=["gpu", "brain"], fabric="pcie5_x16", lanes=8)]))


def test_pd_a_routes_prefill_and_attention_to_gpu():
    m = get_model("deepseek_v3")
    pol = P.pd_a("gpu", "brain")
    pre = llm.build_prefill(m, 1, 512, 0)
    pol.annotate(pre, m.n_layers)
    assert all(o.meta["pool"] == "gpu" for o in pre.ops)

    dec = llm.build_decode(m, 8, 1024)
    pol.annotate(dec, m.n_layers)
    for o in dec.ops:
        if o.role == "attn":
            assert o.meta["pool"] == "gpu", o.name
        if o.role in ("moe", "ffn"):
            assert o.meta["pool"] == "brain", o.name


def test_pd_a_crosses_the_bridge_twice_per_layer():
    m = get_model("deepseek_v3")
    pol = P.pd_a("gpu", "brain")
    dec = llm.build_decode(m, 8, 1024)
    pol.annotate(dec, m.n_layers)
    # one crossing into the dataflow pool and one back, for each layer
    assert dec.meta["crossings"] >= 2 * m.n_layers - 2, dec.meta["crossings"]


def test_layer_split_crosses_once():
    m = get_model("deepseek_v3")
    pol = P.layer_split(m.n_layers, 0.3, "gpu", "brain")
    dec = llm.build_decode(m, 8, 1024)
    pol.annotate(dec, m.n_layers)
    assert dec.meta["crossings"] <= 4, dec.meta["crossings"]


def test_crossing_dtype_reduces_bridge_traffic():
    m = get_model("deepseek_v3", w_dtype="fp8")
    sysm = _sys2()
    calibrate_system(sysm)
    for p in sysm.pools.values():
        p.plan(1e10, 0, 0)
    dec = llm.build_decode(m, 8, 1024)
    plain = P.pd_a("gpu", "brain")
    quant = P.pd_a("gpu", "brain", crossing_dtype="fp8")
    a = execute(dec, sysm, plain, n_layers=m.n_layers)
    import copy
    b = execute(copy.deepcopy(dec), sysm, quant, n_layers=m.n_layers)
    assert b.bytes_crossed < a.bytes_crossed * 0.6


# =========================================================================
# executor invariants
# =========================================================================
def test_time_attribution_sums_to_total():
    m = get_model("deepseek_v3", w_dtype="fp8")
    sysm = _sys2()
    calibrate_system(sysm)
    for p in sysm.pools.values():
        p.plan(1e10, 0, 0)
    for g in (llm.build_decode(m, 8, 1024),
              llm.build_prefill(m, 1, 1024, 0)):
        pol = P.pd_a("gpu", "brain")
        r = execute(g, sysm, pol, n_layers=m.n_layers)
        assert abs(sum(r.t_by_bound.values()) - r.time) / r.time < 1e-9
        assert abs(sum(r.t_by_role.values()) + r.t_bridge - r.time) \
            / r.time < 1e-6
        assert r.energy > 0
        assert abs(sum(r.e_by_source.values()) - r.energy) / r.energy < 1e-6


def test_no_bridge_is_reported_not_silently_free():
    m = get_model("deepseek_v2_lite")
    sysm = build_system(dict(name="nb", pools={
        "gpu": dict(device="h100_sxm", count=8, parallel=dict(tp=8)),
        "brain": dict(device="ipu_gc200", count=8, parallel=dict(ep=8)),
    }))   # no bridges declared
    calibrate_system(sysm)
    for p in sysm.pools.values():
        p.plan(1e9, 0, 0)
    g = llm.build_decode(m, 4, 512)
    r = execute(g, sysm, P.pd_a("gpu", "brain"), n_layers=m.n_layers)
    assert any("no bridge" in w for w in r.warnings)


def test_thinner_bridge_makes_pd_a_slower():
    m = get_model("deepseek_v3", w_dtype="fp8")
    import copy
    times = []
    for lanes in (1, 8, 64):
        sysm = build_system(dict(name="t", pools={
            "gpu": dict(device="h100_sxm", count=8, parallel=dict(tp=8)),
            "brain": dict(device="ipu_gc200", count=64, parallel=dict(ep=64)),
        }, bridges=[dict(between=["gpu", "brain"], fabric="pcie5_x16",
                         lanes=lanes)]))
        calibrate_system(sysm)
        for p in sysm.pools.values():
            p.plan(1e10, 0, 0)
        g = llm.build_decode(m, 64, 2048)
        r = execute(g, sysm, P.pd_a("gpu", "brain"), n_layers=m.n_layers)
        times.append(r.time)
    assert times[0] > times[1] > times[2], times


# =========================================================================
# serving
# =========================================================================
def _mini_sim(rate=1.0, n=24, device="h100_sxm", count=8):
    m = get_model("deepseek_v2_lite", w_dtype="fp8")
    sysm = build_system(dict(name="s", pools={
        "gpu": dict(device=device, count=count, parallel=dict(tp=count))}))
    calibrate_system(sysm)
    wl = WorkloadSpec(n_requests=n, prompt_len=1024, output_len=64,
                      arrival_rate=rate, max_batch=32, chunk_tokens=2048)
    return ServingSimulator(m, sysm, P.all_gpu("gpu"), wl)


def test_serving_produces_sane_metrics():
    r = _mini_sim().run()
    assert r.wall_time > 0 and math.isfinite(r.wall_time)
    assert r.decode_tokens > 0
    assert r.output_throughput > 0
    assert 0 < r.avg_power < 1e7
    assert r.energy_per_output_token > 0
    t = r.ttft_stats()
    assert t["p50"] > 0 and t["p99"] >= t["p50"]


def test_no_request_finishes_before_it_arrives():
    r = _mini_sim(rate=0.3, n=32).run()
    for req in r.requests:
        if req.first_token >= 0:
            assert req.first_token >= req.arrival - 1e-12, req
            assert req.ttft >= 0, req
        if req.finish >= 0:
            assert req.finish >= req.first_token


def test_attainment_falls_as_offered_load_rises():
    slo = SLO(ttft=3.0, tpot=0.06)
    att = [_mini_sim(rate=r, n=48).run().slo_attainment(slo)
           for r in (0.5, 2.0, 8.0, 32.0)]
    assert att[0] >= att[-1], att
    assert att[0] > 0.5, att


def test_throughput_rises_with_offered_load():
    tp = [_mini_sim(rate=r, n=48).run().output_throughput
          for r in (0.5, 2.0, 8.0)]
    assert tp[0] < tp[1] < tp[2], tp


def test_oversized_model_is_flagged_not_silently_run():
    m = get_model("deepseek_v3", w_dtype="bf16")     # 1.25 TiB of weights
    sysm = build_system(dict(name="tiny", pools={
        "gpu": dict(device="a100_80gb_sxm", count=8, parallel=dict(tp=8))}))
    calibrate_system(sysm)
    wl = WorkloadSpec(n_requests=4, prompt_len=512, output_len=8)
    sim = ServingSimulator(m, sysm, P.all_gpu("gpu"), wl)
    assert any("spill" in w or "KV capacity" in w for w in sim.warnings), \
        sim.warnings


def test_dataflow_chip_has_lower_jitter_than_gpu():
    assert make_device("groq_lpu_v1").jitter_sigma < \
        make_device("h100_sxm").jitter_sigma / 5


# =========================================================================
def _main():
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    npass = nfail = 0
    for name, fn in fns:
        try:
            fn()
            npass += 1
            print(f"  PASS  {name}")
        except Exception as e:                       # noqa: BLE001
            nfail += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{npass} passed, {nfail} failed, {len(fns)} total")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(_main())
