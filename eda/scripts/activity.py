#!/usr/bin/env python3
"""Per-bit switching activity of quantised transformer activations.

Feeds the `set_switching_activity` numbers in ptpx.tcl.  Run it to
regenerate the table in that file's comment block, or with a different
sigma / correlation to test sensitivity.

    python3 eda/scripts/activity.py [--sigma 24] [--width 8]
"""
import argparse
import random


def bits(v, w):
    return [(v >> i) & 1 for i in range(w)]


def toggle_rates(rho, sigma, width, n=400_000, seed=11):
    rng = random.Random(seed)
    lim = (1 << (width - 1)) - 1
    mask = (1 << width) - 1
    prev_f = rng.gauss(0, sigma)
    prev = max(-lim, min(lim, round(prev_f)))
    tog = [0] * width
    for _ in range(n):
        cur_f = rho * prev_f + (1 - rho ** 2) ** 0.5 * rng.gauss(0, sigma)
        cur = max(-lim, min(lim, round(cur_f)))
        a, b = bits(prev & mask, width), bits(cur & mask, width)
        for i in range(width):
            tog[i] += a[i] ^ b[i]
        prev, prev_f = cur, cur_f
    return [t / n for t in tog]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=24.0)
    ap.add_argument("--width", type=int, default=8)
    a = ap.parse_args()
    print(f"per-bit toggle rate vs lag-1 correlation rho "
          f"(int{a.width}, sigma={a.sigma:g}):")
    print(f"{'rho':>6} " + " ".join(f"b{i}" for i in range(a.width)) + "   mean")
    for rho in (0.0, 0.5, 0.9, 0.98):
        tr = toggle_rates(rho, a.sigma, a.width)
        print(f"{rho:6.2f} " + " ".join(f"{t:.2f}" for t in tr)
              + f"   {sum(tr)/a.width:.3f}")
    print()
    print("Note: at rho=0 every bit toggles at 0.50, i.e. quantisation alone")
    print("does NOT reduce switching relative to uniform-random data -- the")
    print("sign bit and its extension dominate the high bits and are as")
    print("random as the values.  The reduction comes from temporal")
    print("correlation, which streaming activations do have.")


if __name__ == "__main__":
    main()
