"""Profile a single capacity probe so the simulation hot path is measurable.

Run:  python repro/profile_probe.py [config] [arm] [requests]

This is a local working script, not part of the flowgpu package.  It exists
because `flowgpu capacity configs/experiments/pd_a_reference.yaml` took about
3.4 h of wall clock on this host, which makes iterating on it impractical.
"""

import cProfile
import io
import pstats
import sys
import time
from dataclasses import replace

sys.path.insert(0, ".")

from flowgpu.capacity import _probe
from flowgpu.experiment import Experiment, _resolve_system, load_config
from flowgpu.power import calibrate_system
from flowgpu.sim.serving import ServingSimulator
from flowgpu.system import build_system


def build(cfg_path, arm_name):
    cfg = load_config(cfg_path)
    exp = Experiment(cfg)
    run = [r for r in cfg["runs"] if r.get("name") == arm_name][0]
    model = exp._model(run)
    syscfg = _resolve_system(run["system"], exp.systems, ".")
    syscfg.setdefault("name", arm_name)
    system = build_system(syscfg)
    calibrate_system(system, eda_db=exp.eda_db)
    policy = exp._placement(run, model, system)
    wl = exp._workload(run)
    opt = dict(exp.defaults)
    keep = ("microbatches", "jitter", "collective_overlap",
            "tp_world", "ep_world", "seed")
    opt.update({k: v for k, v in run.items() if k in keep})
    return model, system, policy, wl, opt, exp.slo


def main():
    default_cfg = "configs/experiments/pd_a_reference.yaml"
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else default_cfg
    arm = sys.argv[2] if len(sys.argv) > 2 else "pd_a"
    n_req = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    rate = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    model, system, policy, wl, opt, slo = build(cfg_path, arm)
    print("arm=%s model=%s devs=%d n_requests=%d rate=%g"
          % (arm, model.name, system.total_devices, n_req, rate))

    def make_sim(rate, n):
        w = replace(wl, arrival_rate=rate, n_requests=n)
        return ServingSimulator(model, system, policy, w, **opt)

    t0 = time.time()
    pr = cProfile.Profile()
    pr.enable()
    pt = _probe(make_sim, rate, slo, n_req)
    pr.disable()
    dt = time.time() - t0
    print("one probe: %.2f s  att=%.3f tok/s=%.1f wall=%.1fs"
          % (dt, pt.attainment, pt.output_throughput,
             pt.result.wall_time))
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(22)
    print(s.getvalue())


if __name__ == "__main__":
    main()
