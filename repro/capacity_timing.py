"""Time every probe of one arm's capacity search.

`flowgpu capacity configs/experiments/pd_a_reference.yaml` took ~3.4 h on this
host.  This script shows which probes eat the time.

Run:  python repro/capacity_timing.py [config] [arm] [requests] [iters]
"""

import os
import sys
import threading
import time
from dataclasses import replace

sys.path.insert(0, ".")

import flowgpu.capacity as C
from flowgpu.experiment import Experiment, _resolve_system, load_config
from flowgpu.power import calibrate_system
from flowgpu.sim.serving import ServingSimulator
from flowgpu.system import build_system

_real_probe = C._probe


def timed_probe(make_sim, rate, slo, n_requests, warmup_frac=0.2):
    t0 = time.time()
    out = _real_probe(make_sim, rate, slo, n_requests, warmup_frac)
    dt = time.time() - t0
    print("    probe rate=%8.3f  %7.2f s   att=%5.1f%%  tok/s=%8.1f  "
          "wall=%8.1f s  iters=%d/%d"
          % (rate, dt, 100 * out.attainment, out.output_throughput,
             out.result.wall_time, out.result.n_prefill_iters,
             out.result.n_decode_iters))
    return out


C._probe = timed_probe


def _watchdog(seconds):
    """Hard-kill the process if a single arm runs past its budget.

    No signal.alarm on Windows, and the point is only to tell "this arm is
    pathological" apart from "this arm is slow".
    """
    if seconds <= 0:
        return

    def _kill():
        time.sleep(seconds)
        print("!! WATCHDOG: arm exceeded %d s -- killing" % seconds,
              flush=True)
        sys.stdout.flush()
        os._exit(3)

    t = threading.Thread(target=_kill, daemon=True)
    t.start()


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else \
        "configs/experiments/pd_a_reference.yaml"
    arm_arg = sys.argv[2] if len(sys.argv) > 2 else "pd_a"
    n_req = int(sys.argv[3]) if len(sys.argv) > 3 else 192
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    cfg = load_config(cfg_path)
    exp = Experiment(cfg)
    arms = [r.get("name") for r in cfg["runs"]]
    if arm_arg.upper() != "ALL":
        arms = [arm_arg]

    _watchdog(int(os.environ.get("FLOWGPU_WATCHDOG", "0")))
    grand = time.time()
    for arm in arms:
        run = [r for r in cfg["runs"] if r.get("name") == arm][0]
        model = exp._model(run)
        syscfg = _resolve_system(run["system"], exp.systems, ".")
        syscfg.setdefault("name", arm)
        system = build_system(syscfg)
        calibrate_system(system, eda_db=exp.eda_db)
        policy = exp._placement(run, model, system)
        wl = exp._workload(run)
        opt = dict(exp.defaults)
        keep = ("microbatches", "jitter", "collective_overlap",
                "tp_world", "ep_world", "seed")
        opt.update({k: v for k, v in run.items() if k in keep})

        def make_sim(rate, n, _m=model, _s=system, _p=policy, _w=wl, _o=opt):
            return ServingSimulator(_m, _s, _p,
                                    replace(_w, arrival_rate=rate,
                                            n_requests=n), **_o)

        print("arm=%s  n_requests=%d  iters=%d  microbatches=%s"
              % (arm, n_req, iters, opt.get("microbatches")), flush=True)
        t0 = time.time()
        cap = C.find_capacity(model, system, policy, wl, slo=exp.slo,
                              n_requests=n_req, iters=iters, name=arm,
                              sim_kwargs=opt, verbose=False)
        print("  ARM TOTAL %.1f s   max_rate=%.3f  tok/s=%.1f"
              % (time.time() - t0, cap.max_rate,
                 cap.best.output_throughput), flush=True)
    print("GRAND TOTAL %.1f s" % (time.time() - grand), flush=True)


if __name__ == "__main__":
    main()
