"""Pre-registered analysis logic on synthetic records (not real results)."""

import json
import unittest
from pathlib import Path

from acpr.analyze import analyze, bootstrap_ci, compare, verdict

PREREG = json.loads((Path(__file__).resolve().parent.parent / "configs" / "pilot_memory_prereg.json").read_text())
ANALYSIS = PREREG["analysis"]


def rec(seed, arm, value, variant="symmetric", status="OK"):
    r = {"seed": seed, "arm": arm, "variant": variant, "status": status}
    if status == "OK":
        r["planning"] = {"forced_commit_accuracy": value}
    return r


class TestAnalysis(unittest.TestCase):
    def test_verdict_rules(self):
        self.assertEqual(verdict(0.2, 0.05, 0.3, 0.1), "SUPPORTED")
        self.assertEqual(verdict(0.2, -0.01, 0.3, 0.1), "INCONCLUSIVE")
        self.assertEqual(verdict(0.02, -0.03, 0.08, 0.1), "NOT_SUPPORTED")
        self.assertEqual(verdict(0.08, 0.01, 0.15, 0.1), "INCONCLUSIVE")

    def test_bootstrap_ci_brackets_mean(self):
        vals = [0.1, 0.2, 0.15, 0.3, 0.25, 0.05, 0.2, 0.1, 0.15, 0.2]
        lo, hi = bootstrap_ci(vals, 2000, 0.05, 0)
        self.assertLess(lo, sum(vals) / len(vals))
        self.assertGreater(hi, sum(vals) / len(vals))

    def test_failed_and_missing_seeds_are_reported_and_force_incomplete(self):
        results = [rec(s, "acp_branch", 0.9) for s in range(10)] + [rec(s, "msp_branch", 0.5) for s in range(9)]
        results[3] = rec(3, "acp_branch", None, status="FAILED")
        out = compare(results, "symmetric", "acp_branch", "msp_branch", ANALYSIS)
        self.assertEqual(out["verdict"], "INCOMPLETE")
        self.assertEqual(out["n_paired_seeds"], 8)
        self.assertEqual({e["seed"] for e in out["excluded_seeds"]}, {3, 9})

    def test_full_pipeline_and_stop_flag(self):
        results = []
        for s in range(10):
            results += [rec(s, "acp_branch", 0.9 + 0.001 * s), rec(s, "msp_branch", 0.5),
                        rec(s, "acp_nobranch", 0.5), rec(s, "msp_nobranch", 0.5), rec(s, "random_frozen", 0.5)]
            results += [rec(s, "acp_branch", 0.9, "asymmetric"), rec(s, "msp_branch", 0.88, "asymmetric")]
        out = analyze(results, ANALYSIS)
        self.assertEqual(out["comparisons"]["H1_acp_vs_msp_branch_symmetric"]["verdict"], "SUPPORTED")
        self.assertEqual(out["comparisons"]["S2_acp_vs_msp_nobranch_symmetric"]["verdict"], "NOT_SUPPORTED")
        self.assertEqual(out["comparisons"]["C1_acp_vs_msp_branch_asymmetric"]["verdict"], "NOT_SUPPORTED")
        self.assertTrue(any(f.startswith("STOP_EXPANSION") for f in out["flags"]))


if __name__ == "__main__":
    unittest.main()
