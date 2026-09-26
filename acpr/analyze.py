"""Pre-registered paired analysis (fixed before any pilot data exists).

    python -m acpr.analyze --prereg configs/pilot_memory_prereg.json --runs runs/<id> [runs/<id2> ...]

For each comparison (variant, arm_a, arm_b) the per-seed difference of the
primary metric is computed on seeds where BOTH arms have status OK. Seeds
where either arm FAILED or was NOT_RUN are reported, never dropped silently;
if fewer than `required_seeds` paired seeds remain the verdict is
INCOMPLETE regardless of the numbers.

Verdicts (percentile bootstrap CI over seeds):
  SUPPORTED      mean >= MIE and CI lower bound > 0
  NOT_SUPPORTED  CI upper bound < MIE (an effect of MIE size is excluded)
  INCONCLUSIVE   otherwise
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _metric(rec: dict, name: str) -> Optional[float]:
    node = rec
    for part in name.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return None if node is None else float(node)


def bootstrap_ci(values: Sequence[float], resamples: int, alpha: float, seed: int):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(resamples))
    lo = means[int((alpha / 2) * resamples)]
    hi = means[min(int((1 - alpha / 2) * resamples), resamples - 1)]
    return lo, hi


def verdict(mean: float, lo: float, hi: float, mie: float) -> str:
    if mean >= mie and lo > 0:
        return "SUPPORTED"
    if hi < mie:
        return "NOT_SUPPORTED"
    return "INCONCLUSIVE"


def compare(results: List[dict], variant: str, arm_a: str, arm_b: str, analysis: dict) -> dict:
    metric = analysis["primary_metric"]
    cells: Dict[str, Dict[int, dict]] = {arm_a: {}, arm_b: {}}
    for r in results:
        if r["variant"] == variant and r["arm"] in cells:
            cells[r["arm"]][r["seed"]] = r
    seeds = sorted(set(cells[arm_a]) | set(cells[arm_b]))
    diffs, excluded = [], []
    for s in seeds:
        ra, rb = cells[arm_a].get(s), cells[arm_b].get(s)
        status = (ra or {}).get("status", "MISSING"), (rb or {}).get("status", "MISSING")
        va = _metric(ra, metric) if ra and ra.get("status") == "OK" else None
        vb = _metric(rb, metric) if rb and rb.get("status") == "OK" else None
        if va is None or vb is None:
            excluded.append({"seed": s, "status": status})
            continue
        diffs.append(va - vb)
    out = {
        "variant": variant, "arm_a": arm_a, "arm_b": arm_b, "metric": metric,
        "n_paired_seeds": len(diffs), "excluded_seeds": excluded, "diffs": diffs,
    }
    if len(diffs) < analysis["required_seeds"]:
        out["verdict"] = "INCOMPLETE"
        return out
    mean = sum(diffs) / len(diffs)
    lo, hi = bootstrap_ci(diffs, analysis["bootstrap_resamples"], analysis["alpha"], analysis["bootstrap_seed"])
    out.update(mean_diff=mean, ci_low=lo, ci_high=hi, verdict=verdict(mean, lo, hi, analysis["mie"]))
    return out


def analyze(results: List[dict], analysis: dict) -> dict:
    comps = {c["name"]: compare(results, c["variant"], c["arm_a"], c["arm_b"], analysis) for c in analysis["comparisons"]}
    flags = []
    h1 = comps.get("H1_acp_vs_msp_branch_symmetric", {}).get("verdict")
    nb = comps.get("S2_acp_vs_msp_nobranch_symmetric", {}).get("verdict")
    if h1 == "SUPPORTED" and nb != "SUPPORTED":
        flags.append("STOP_EXPANSION: advantage not shown without simulator restores (branching privilege)")
    if h1 == "SUPPORTED" and comps.get("C1_acp_vs_msp_branch_asymmetric", {}).get("verdict") == "SUPPORTED":
        flags.append("MECHANISM_CHECK: ACP also wins where action-marginal futures carry the clue; effect may not be specific to action conditioning")
    if h1 is None or h1 == "INCOMPLETE":
        flags.append("PRIMARY_INCOMPLETE")
    return {"comparisons": comps, "flags": flags}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args(argv)
    analysis = json.loads(Path(args.prereg).read_text())["analysis"]
    results = []
    for d in args.runs:
        results.extend(json.loads((Path(d) / "metrics.json").read_text()))
    print(json.dumps(analyze(results, analysis), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
