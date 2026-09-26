"""CPU entry point.

    python -m acpr.run --config configs/smoke.json --out runs/<run_id>

Arms (all share encoder/predictor/head initialisations for a given seed):
  acp_branch    action-conditional multi-step objective, branched data
  msp_branch    action-marginal multi-step objective, the SAME branched data
  acp_nobranch  action-conditional objective, non-branched data, SAME total
                transition budget (secondary: is branching worth its cost?)
  msp_nobranch  action-marginal objective, the same non-branched data
                (does any ACP advantage survive without simulator restores?)
  random_frozen untrained encoder (reference lower bound, not a competitor)

Every arm is evaluated with the same frozen-encoder protocol: a fresh
action-conditional reward head is trained on the same train windows, then
used by the same planner on the same test episodes.

Configs with run_kind != "smoke" are refused unless pilot_approved is true.
Failed arms are recorded as FAILED with a traceback; arms skipped because a
resource cap was hit are recorded as NOT_RUN. Neither counts as a pass.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path

from . import autodiff as ad
from .collect import CollectConfig, collect_split
from .data import dataset_hash
from .env import N_ACTIONS, OBS_JUNCTION, MazeConfig
from .evaluate import EvalConfig, clue_probe, collect_probe_histories, planning_eval
from .manifest import (
    Timer,
    canonical_hash,
    environment_info,
    git_state,
    peak_rss_mb,
    source_hash,
)
from .models import (
    LatentPredictor,
    RecurrentEncoder,
    effective_param_count,
    param_count,
    params_hash,
)
from .splits import content_overlap, derive_seed
from .train import TrainConfig, build_windows, evaluate_predictor, train_predictor

REPO = Path(__file__).resolve().parent.parent

ARMS = {
    "acp_branch": {"data": "branch", "stage1": True, "action_conditional": True},
    "msp_branch": {"data": "branch", "stage1": True, "action_conditional": False},
    "acp_nobranch": {"data": "nobranch", "stage1": True, "action_conditional": True},
    "msp_nobranch": {"data": "nobranch", "stage1": True, "action_conditional": False},
    "random_frozen": {"data": "branch", "stage1": False, "action_conditional": True},
}


class JsonlLog:
    def __init__(self, path: Path):
        self.f = path.open("a")

    def __call__(self, rec: dict) -> None:
        self.f.write(json.dumps(rec, sort_keys=True) + "\n")
        self.f.flush()


def _junction_prefix(observations):
    for i, o in enumerate(observations):
        if o == OBS_JUNCTION:
            return tuple(observations[: i + 1])
    return tuple(observations)


def _fresh_models(seed, spec, n_obs, hidden):
    encoder = RecurrentEncoder(n_obs, N_ACTIONS, hidden, random.Random(derive_seed("enc-init", seed)))
    predictor = LatentPredictor(
        "pred", hidden, n_obs, N_ACTIONS, spec["action_conditional"],
        random.Random(derive_seed("pred-init", seed)),
    )
    return encoder, predictor


def run_arm(arm, spec, seed, maze_cfg, cfg, data, dev, probe_calib, log):
    n_obs, hidden = maze_cfg.n_obs, cfg["model"]["hidden"]
    episodes, windows = data
    dev_episodes, dev_windows = dev
    encoder, predictor = _fresh_models(seed, spec, n_obs, hidden)
    out = {
        "arm": arm,
        "spec": spec,
        "encoder_params": param_count(encoder.params()),
        "stage1_predictor_params": param_count(predictor.params()),
        "stage1_predictor_effective_params": effective_param_count(predictor),
    }
    mac0 = ad.COUNTERS["mac"]
    if spec["stage1"]:
        # Each arm selects its learning rate from the same fixed grid by its OWN
        # stage-1 dev objective; test data is never used for selection.
        s1 = cfg["stage1"]
        candidates = []
        for lr in s1.get("lr_grid", [s1["lr"]]):
            enc_c, pred_c = _fresh_models(seed, spec, n_obs, hidden)
            tc = TrainConfig(s1["epochs"], lr, s1["batch_episodes"], derive_seed("s1-order", seed))
            hist = train_predictor(enc_c, pred_c, episodes, windows, tc, True, log, f"{arm}:stage1:lr={lr}")
            dev_metrics = evaluate_predictor(enc_c, pred_c, dev_episodes, dev_windows)
            objective = tc.obs_weight * dev_metrics["obs_nll"] + tc.reward_weight * dev_metrics["reward_mse"]
            candidates.append((objective, lr, enc_c, pred_c, hist, dev_metrics))
        best = min(candidates, key=lambda c: c[0])  # ties keep grid order
        _, lr, encoder, predictor, out["stage1"], out["stage1_dev"] = best
        out["stage1_selected_lr"] = lr
        out["stage1_lr_candidates"] = [
            {"lr": c[1], "dev_objective": c[0], "dev": c[5]} for c in candidates
        ]
    out["stage1_mac"] = ad.COUNTERS["mac"] - mac0

    # Stage 2: identical fresh action-conditional reward head on the frozen encoder.
    mac1 = ad.COUNTERS["mac"]
    head = LatentPredictor("head", hidden, n_obs, N_ACTIONS, True, random.Random(derive_seed("head-init", seed)))
    s2 = cfg["stage2"]
    tc2 = TrainConfig(s2["epochs"], s2["lr"], s2["batch_episodes"], derive_seed("s2-order", seed), obs_weight=0.0)
    out["stage2"] = train_predictor(encoder, head, episodes, windows, tc2, False, log, f"{arm}:stage2")
    out["stage2_dev"] = evaluate_predictor(encoder, head, dev_episodes, dev_windows)
    out["stage2_head_params"] = param_count(head.params())
    out["stage2_mac"] = ad.COUNTERS["mac"] - mac1

    ev = EvalConfig(**cfg["eval"])
    plan = planning_eval(encoder, head, maze_cfg, seed, ev)
    calib_h, calib_c = probe_calib
    out["probe"] = clue_probe(encoder, calib_h, calib_c, plan.pop("_junction_histories"), plan.pop("_junction_clues"), ev)
    out["planning"] = plan
    out["encoder_hash"] = params_hash(encoder.params())
    out["stage2_head_hash"] = params_hash(head.params())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", help="comma-separated subset of the config seeds (for parallel processes)")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text())
    seeds = cfg["seeds"]
    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",")]
        if not set(seeds) <= set(cfg["seeds"]):
            print("REFUSED: --seeds must be a subset of the config seeds", file=sys.stderr)
            return 2
    if cfg.get("run_kind") != "smoke" and not cfg.get("pilot_approved", False):
        print(f"REFUSED: run_kind={cfg.get('run_kind')!r} requires pilot_approved=true", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    log = JsonlLog(out_dir / "raw_log.jsonl")
    timer = Timer()
    caps = cfg["resource_caps"]
    manifest = {
        "run_kind": cfg["run_kind"],
        "command": " ".join(["python3", "-m", "acpr.run", *(sys.argv[1:] if argv is None else argv)]),
        "git": git_state(REPO),
        "source_sha256": source_hash(REPO),
        "config_sha256": canonical_hash(cfg),
        "environment": environment_info(),
        "resource_caps": caps,
        "seeds_run": seeds,
        "data": {},
    }
    results = []
    cap_hit = None
    log({"event": "start", "config_sha256": manifest["config_sha256"], "git": manifest["git"]["commit"]})

    for seed in seeds:
        for variant in cfg["variants"]:
            maze_cfg = MazeConfig(symmetric=(variant == "symmetric"), **cfg["maze"])
            key = f"seed{seed}/{variant}"
            datasets, ledgers = {}, {}
            for mode, bp in (("branch", cfg["branch_prob"]), ("nobranch", 0.0)):
                cc = CollectConfig(cfg["train_budget"], cfg["horizon"], bp, cfg["behavior_forward_prob"])
                res = collect_split(maze_cfg, cc, seed, "train")
                datasets[mode] = (res.episodes, build_windows(res.episodes, cfg["horizon"]))
                ledgers[mode] = {"ledger": res.ledger, "stats": res.stats, "sha256": dataset_hash(res.episodes)}
                # res.labels (evaluator-only) is intentionally discarded here.
            dev_res = collect_split(
                maze_cfg, CollectConfig(cfg["dev_budget"], cfg["horizon"], cfg["branch_prob"], cfg["behavior_forward_prob"]),
                seed, "dev",
            )
            dev = (dev_res.episodes, build_windows(dev_res.episodes, cfg["horizon"]))
            strat = cfg["eval"].get("stratify_clue", True)
            calib_h, calib_c, calib_ledger = collect_probe_histories(maze_cfg, seed, "calibration", cfg["eval"]["n_calibration"], strat)
            test_prefix_h, _, _ = collect_probe_histories(maze_cfg, seed, "test", cfg["eval"]["n_test"], strat)
            overlap = content_overlap(
                (_junction_prefix(e.observations) for e in datasets["branch"][0]),
                (h.observations for h in test_prefix_h),
            )
            manifest["data"][key] = {
                "train": ledgers,
                "dev": {"ledger": dev_res.ledger, "stats": dev_res.stats, "sha256": dataset_hash(dev_res.episodes)},
                "calibration_probe_ledger": calib_ledger,
                "test_junction_prefix_obs_overlap_with_train": overlap,
            }
            log({"event": "data", "key": key, **manifest["data"][key]})

            for arm in cfg["arms"]:
                spec = ARMS[arm]
                rec = {"seed": seed, "variant": variant, "arm": arm}
                if cap_hit is None:
                    t = timer.read()
                    if t["wall_s"] > caps["max_wall_s"]:
                        cap_hit = f"max_wall_s exceeded ({t['wall_s']:.1f}s)"
                    elif peak_rss_mb() > caps["max_peak_rss_mb"]:
                        cap_hit = f"max_peak_rss_mb exceeded ({peak_rss_mb():.1f}MB)"
                if cap_hit is not None:
                    rec.update(status="NOT_RUN", reason=cap_hit)
                else:
                    arm_timer = Timer()
                    try:
                        rec.update(
                            status="OK",
                            **run_arm(arm, spec, seed, maze_cfg, cfg, datasets[spec["data"]], dev, (calib_h, calib_c), log),
                        )
                        rec["train_transitions_spent"] = ledgers[spec["data"]]["ledger"]["transitions_total"]
                        rec["train_restores_used"] = ledgers[spec["data"]]["ledger"]["restores"]
                    except Exception:  # preserve failures instead of hiding them
                        rec.update(status="FAILED", traceback=traceback.format_exc())
                    rec["time"] = arm_timer.read()
                    rec["peak_rss_mb_after"] = peak_rss_mb()
                results.append(rec)
                log({"event": "arm_done", **{k: v for k, v in rec.items() if k not in ("stage1", "stage2")}})

    manifest["time"] = timer.read()
    manifest["peak_rss_mb"] = peak_rss_mb()
    manifest["total_mac"] = ad.COUNTERS["mac"]
    manifest["status_counts"] = {
        s: sum(r["status"] == s for r in results) for s in ("OK", "FAILED", "NOT_RUN")
    }
    manifest["software_status"] = (
        "TECHNICAL_RUN_COMPLETE" if manifest["status_counts"]["OK"] == len(results) else "TECHNICAL_RUN_INCOMPLETE"
    )
    manifest["science_status"] = "SCIENCE_NOT_EVALUATED"
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    log({"event": "end", "status_counts": manifest["status_counts"], "time": manifest["time"]})
    print(json.dumps({"out": str(out_dir), **manifest["status_counts"], "wall_s": manifest["time"]["wall_s"]}))
    return 0 if manifest["status_counts"]["FAILED"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
