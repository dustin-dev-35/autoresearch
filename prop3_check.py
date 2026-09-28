"""
Finite-n bias of the D_eff estimator, at the exact operating point of the
seed-variance experiment (n = 10000 passages, d = 384, L = 12, K = L - 1).

The paper's Proposition 3 fixes D_eff = 2L/(L+1) for the idealised
orthogonal-update construction, in the population limit.  The estimator is
computed on finite samples, so it carries a bias that depends on n/d.  This
script measures that bias at the experiment's own n and d by running the same
estimator on the same synthetic construction, and sweeps n to show the
convergence.  No GPU and no training.

It is a control, not a baseline: it establishes the floor below which a
measured seed spread cannot be interpreted, and it bounds how much of any
observed deviation from F_L is estimator bias rather than model property.
"""

from __future__ import annotations

import json
import os
import sys

import torch

from deff import d_eff_from_layers, reference_depth

L = 12
D = 384
NS = [int(x) for x in os.environ.get("AXV_PRO3_NS", "1000,4000,10000,20000").split(",")]
SEEDS = [int(x) for x in os.environ.get("AXV_PRO3_SEEDS", "11,22,33").split(",")]


def run(L: int, d: int, n: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    f = [torch.randn(n, d, generator=g, dtype=torch.float64) for _ in range(L)]
    pooled = []
    acc = torch.zeros(n, d, dtype=torch.float64)
    for i in range(L):
        acc = acc + f[i]
        pooled.append(acc.clone())
    return d_eff_from_layers(pooled, L=L)


def main() -> None:
    out = {"L": L, "d": D, "seeds": SEEDS, "sweep": []}
    for n in NS:
        vals = [float(run(L, D, n, s)["D_eff"]) for s in SEEDS]
        f_l = reference_depth(L)
        mean = sum(vals) / len(vals)
        spread = max(vals) - min(vals)
        out["sweep"].append(
            {
                "n": n,
                "D_eff": vals,
                "mean_D_eff": mean,
                "F_L": f_l,
                "rel_bias_pct": (mean - f_l) / f_l * 100.0,
                "seed_spread_D_eff": spread,
                "seed_sd_D_eff": (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5,
            }
        )
        print(
            f"prop3 n={n:6d} mean_D_eff={mean:.6f} F_L={f_l:.6f} "
            f"rel_bias={(mean - f_l) / f_l * 100.0:+.4f}% "
            f"seed_sd={out['sweep'][-1]['seed_sd_D_eff']:.6f}",
            flush=True,
        )
    at = [r for r in out["sweep"] if r["n"] == 10000]
    if at:
        out["operating_point_n"] = 10000
        out["operating_point_rel_bias_pct"] = at[0]["rel_bias_pct"]
        out["operating_point_seed_sd_D_eff"] = at[0]["seed_sd_D_eff"]
    out["operating_point_n_requested"] = 10000
    out["operating_point_measured"] = bool(at)
    print("AXV_RUN_BEGIN prop3-finite-n-bias")
    print("AXV_METRICS_JSON " + json.dumps(out, sort_keys=True, separators=(",", ":")))
    print("AXV_RUN_END prop3-finite-n-bias status=ok")


if __name__ == "__main__":
    sys.exit(main())
