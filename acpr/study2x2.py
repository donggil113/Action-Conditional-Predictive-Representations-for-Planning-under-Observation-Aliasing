"""Limited 2x2 study: action conditioning x branching collection (symmetric maze).

    python3 -m acpr.study2x2 --config configs/study_2x2_symmetric.json --out runs/<id>

Arms per seed:
  acp_branch / msp_branch      share ONE branching pretraining dataset exactly
  acp_nobranch / msp_nobranch  share ONE no-branching pretraining dataset exactly
  random_recurrent             untrained encoder (same init as learned encoders)
  informative_control          history-only hand-designed features

Both pretraining datasets spend the same transition cap. Their difference is a
collection-method difference, not identical data. Every arm is read out
through the same restore-free pool (head_train / head_dev): train-pool
standardization -> fresh action-conditional reward head -> 20/40/80-epoch
checkpoint chosen on head_dev reward MSE -> planner on clue-balanced test
episodes with restores forbidden.

Caps (RLIMIT_CPU, RLIMIT_AS, pre-cell projection) and status preservation
follow acpr.diagnose. The verdict never uses a missing cell.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

from . import autodiff as ad
from .collect import CollectConfig, collect_split
from .controls import fit_standardizer
from .data import dataset_hash
from .diagnose import CpuCapExceeded, _on_sigxcpu, make_encoder
from .env import N_ACTIONS, OBS_JUNCTION, MazeConfig
from .evaluate import EvalConfig, clue_probe, collect_probe_histories, planning_eval
from .manifest import canonical_hash, environment_info, git_state, peak_rss_mb, source_hash
from .models import LatentPredictor, RecurrentEncoder, copy_predictor, effective_param_count, param_count, params_hash
from .splits import content_overlap, derive_seed, episode_id, episode_seed
from .train import TrainConfig, build_windows, evaluate_predictor, train_predictor

REPO = Path(__file__).resolve().parent.parent
ARMS = {
    "informative_control": {"learned": False},
    "random_recurrent": {"learned": False},
    "acp_branch": {"learned": True, "action_conditional": True, "data": "pretrain_branch"},
    "acp_nobranch": {"learned": True, "action_conditional": True, "data": "pretrain_nobranch"},
    "msp_branch": {"learned": True, "action_conditional": False, "data": "pretrain_branch"},
    "msp_nobranch": {"learned": True, "action_conditional": False, "data": "pretrain_nobranch"},
}
# (name, arm_a, arm_b, pre-registered MIE contrast?)
CONTRASTS = [
    ("acp_branch_minus_acp_nobranch", "acp_branch", "acp_nobranch", True),        # S1
    ("msp_branch_minus_msp_nobranch", "msp_branch", "msp_nobranch", False),
    ("acp_branch_minus_msp_branch", "acp_branch", "msp_branch", True),            # H1
    ("acp_nobranch_minus_msp_nobranch", "acp_nobranch", "msp_nobranch", True),    # S2
    ("acp_branch_minus_random", "acp_branch", "random_recurrent", True),          # R1
    ("acp_nobranch_minus_random", "acp_nobranch", "random_recurrent", False),
    ("msp_branch_minus_random", "msp_branch", "random_recurrent", False),
    ("msp_nobranch_minus_random", "msp_nobranch", "random_recurrent", False),
]


def _junction_prefix(observations):
    for i, o in enumerate(observations):
        if o == OBS_JUNCTION:
            return tuple(observations[: i + 1])
    return tuple(observations)


def split_checks(seed: int, datasets: dict, cfg: dict) -> dict:
    """Root-episode, branch-family and latent-seed separation across splits."""
    ev = cfg["eval"]
    seeds_by_split = {
        "train": {episode_seed(seed, "train", i) for i in range(max(len(datasets["pretrain_branch"].episodes),
                                                                    len(datasets["pretrain_nobranch"].episodes)))},
        "head_train": {episode_seed(seed, "head_train", i) for i in range(len(datasets["head_train"].episodes))},
        "head_dev": {episode_seed(seed, "head_dev", i) for i in range(len(datasets["head_dev"].episodes))},
        "calibration": {episode_seed(seed, "calibration", i) for i in range(ev["n_calibration"])},
        "test": {episode_seed(seed, "test", i) for i in range(ev["n_test"])},
    }
    names = list(seeds_by_split)
    overlaps = {f"{a}&{b}": len(seeds_by_split[a] & seeds_by_split[b])
                for i, a in enumerate(names) for b in names[i + 1:]}
    ids = {k: {e.episode_id for e in d.episodes} for k, d in datasets.items()}
    for key, d in datasets.items():
        split = {"pretrain_branch": "train", "pretrain_nobranch": "train"}.get(key, key)
        for i, e in enumerate(d.episodes):
            if e.split != split or e.episode_id != episode_id(seed, split, i):
                raise AssertionError(f"{key}: episode {i} has id/split {e.episode_id}/{e.split}")
            for b in e.branches:  # a branch family lives inside its root episode
                if not 0 <= b.t < len(e.actions):
                    raise AssertionError(f"{key}: branch outside its root episode")
    for key in ("pretrain_nobranch", "head_train", "head_dev"):
        if datasets[key].ledger["restores"] != 0 or any(e.branches for e in datasets[key].episodes):
            raise AssertionError(f"{key} must be restore-free")
    if any(v for v in overlaps.values()):
        raise AssertionError(f"latent episode seeds shared across splits: {overlaps}")
    shared_roots = len(ids["pretrain_branch"] & ids["pretrain_nobranch"])
    # Evaluator-side replay of the scripted test prefixes (counted below; no restores).
    test_h, _, test_prefix_ledger = collect_probe_histories(
        MazeConfig(symmetric=True, **cfg["maze"]), seed, "test", ev["n_test"], ev["stratify_clue"])
    test_prefixes = [h.observations for h in test_h]
    return {
        "test_prefix_check_ledger": test_prefix_ledger,
        "latent_seed_overlap_between_splits": overlaps,
        "pretrain_branch_nobranch_shared_root_episodes": shared_roots,
        "test_junction_prefix_obs_overlap": {
            k: content_overlap((_junction_prefix(e.observations) for e in datasets[k].episodes), test_prefixes)
            for k in ("pretrain_branch", "pretrain_nobranch", "head_train")
        },
        "restore_free": ["pretrain_nobranch", "head_train", "head_dev"],
    }


def prepare_seed(seed: int, maze_cfg: MazeConfig, cfg: dict) -> dict:
    k, fwd = cfg["horizon"], cfg["behavior_forward_prob"]
    pt = cfg["pretraining"]
    hp = cfg["readout_pool"]
    datasets = {
        "pretrain_branch": collect_split(maze_cfg, CollectConfig(pt["budget"], k, pt["branch_prob"], fwd), seed, "train"),
        "pretrain_nobranch": collect_split(maze_cfg, CollectConfig(pt["budget"], k, 0.0, fwd), seed, "train"),
        "head_train": collect_split(maze_cfg, CollectConfig(hp["train_budget"], k, 0.0, fwd), seed, "head_train"),
        "head_dev": collect_split(maze_cfg, CollectConfig(hp["dev_budget"], k, 0.0, fwd), seed, "head_dev"),
    }
    ev = cfg["eval"]
    calib_h, calib_c, calib_ledger = collect_probe_histories(maze_cfg, seed, "calibration", ev["n_calibration"], ev["stratify_clue"])
    checks = split_checks(seed, datasets, cfg)
    record = {key: {"ledger": d.ledger, "stats": d.stats, "sha256": dataset_hash(d.episodes)} for key, d in datasets.items()}
    record["calibration_probe_ledger"] = calib_ledger
    record["split_checks"] = checks
    record["prefix_replays"] = 0  # branching uses snapshot/restore, never prefix replay
    # Evaluator labels (datasets[*].labels) are intentionally not used below.
    return {
        "eps": {key: d.episodes for key, d in datasets.items()},
        "win": {key: build_windows(d.episodes, k) for key, d in datasets.items()},
        "calib": (calib_h, calib_c),
        "record": record,
    }


def _weights(params) -> dict:
    return {p.name: {"rows": p.rows, "cols": p.cols, "w": list(p.w)} for p in params}


def run_cell(seed, arm, maze_cfg, cfg, data, log, weights_dir: Path) -> dict:
    spec = ARMS[arm]
    hidden = cfg["hidden"]
    out = {"spec": spec}
    t_cpu, mac0 = time.process_time(), ad.COUNTERS["mac"]
    if spec["learned"]:
        encoder = RecurrentEncoder(maze_cfg.n_obs, N_ACTIONS, hidden, random.Random(derive_seed("enc-init", seed)))
        predictor = LatentPredictor("pred", hidden, maze_cfg.n_obs, N_ACTIONS, spec["action_conditional"],
                                    random.Random(derive_seed("pred-init", seed)))
        pt = cfg["pretraining"]
        tc = TrainConfig(pt["epochs"], pt["lr"], pt["batch_episodes"], derive_seed("s1-order", seed),
                         obs_weight=pt.get("obs_weight", 1.0), reward_weight=pt.get("reward_weight", 1.0),
                         clip_norm=pt.get("clip_norm", 5.0))
        hist = train_predictor(encoder, predictor, data["eps"][spec["data"]], data["win"][spec["data"]], tc, True,
                               log, f"{seed}:{arm}:pretrain")
        out["stage1_train_loss_first_last"] = [hist["epochs"][0]["train_loss_per_target"],
                                               hist["epochs"][-1]["train_loss_per_target"]]
        out["stage1_head_dev_prediction"] = evaluate_predictor(encoder, predictor, data["eps"]["head_dev"], data["win"]["head_dev"])
        out["pretrain_dataset_sha256"] = data["record"][spec["data"]]["sha256"]
        out["stage1_predictor_params"] = param_count(predictor.params())
        out["stage1_predictor_effective_params"] = effective_param_count(predictor)
        out["encoder_sha256"] = params_hash(encoder.params())
        base = encoder
        (weights_dir / f"{seed}_{arm}_encoder.json").write_text(json.dumps(_weights(encoder.params())))
    else:
        base = make_encoder(arm, seed, maze_cfg.n_obs, hidden)
        if isinstance(base, RecurrentEncoder):
            out["encoder_sha256"] = params_hash(base.params())
            (weights_dir / f"{seed}_{arm}_encoder.json").write_text(json.dumps(_weights(base.params())))
    out["stage1_cpu_s"] = time.process_time() - t_cpu
    out["stage1_mac"] = ad.COUNTERS["mac"] - mac0

    t_cpu, mac0 = time.process_time(), ad.COUNTERS["mac"]
    enc = fit_standardizer(base, data["eps"]["head_train"], split="head_train")
    head = LatentPredictor("head", hidden, maze_cfg.n_obs, N_ACTIONS, True, random.Random(derive_seed("head-init", seed)))
    hc = cfg["head"]
    cands = sorted(hc["epoch_candidates"])
    ckpts = {}

    def on_epoch_end(n, pred):
        if n in cands:
            ckpts[n] = {"dev": evaluate_predictor(enc, pred, data["eps"]["head_dev"], data["win"]["head_dev"]),
                        "head": copy_predictor(pred)}
            log({"event": "checkpoint", "seed": seed, "arm": arm, "epoch": n, "dev": ckpts[n]["dev"]})

    tc2 = TrainConfig(cands[-1], hc["lr"], hc["batch_episodes"], derive_seed("s2-order", seed), obs_weight=0.0)
    train_predictor(enc, head, data["eps"]["head_train"], data["win"]["head_train"], tc2, False, log,
                    f"{seed}:{arm}:readout", on_epoch_end=on_epoch_end)
    selected = min(cands, key=lambda n: (ckpts[n]["dev"]["reward_mse"], n))
    sel_head = ckpts[selected]["head"]
    (weights_dir / f"{seed}_{arm}_head.json").write_text(json.dumps(_weights(sel_head.params())))
    ev = EvalConfig(**cfg["eval"])
    plan = planning_eval(enc, sel_head, maze_cfg, seed, ev)
    probe = clue_probe(enc, *data["calib"], plan.pop("_junction_histories"), plan.pop("_junction_clues"), ev)
    out.update(
        epoch_candidates={n: {"dev": ckpts[n]["dev"]} for n in cands},
        selected_epochs=selected,
        selected_at_max_candidate=selected == cands[-1],
        head_sha256=params_hash(sel_head.params()),
        standardizer_source_split=enc.source_split,
        planning=plan,
        probe=probe,
        test_transitions=plan["ledger"]["transitions_total"],
        readout_cpu_s=time.process_time() - t_cpu,
        readout_mac=ad.COUNTERS["mac"] - mac0,
    )
    return out


def judge(cells: list, cfg: dict) -> dict:
    dec = cfg["decision"]
    metric = dec["metric"]
    mie = dec["mie"]
    by = {(c["seed"], c["arm"]): c for c in cells}
    complete = [s for s in cfg["seeds"] if all(by.get((s, a), {}).get("status") == "OK" for a in ARMS)]
    missing = [{"seed": s, "arm": a, "status": by.get((s, a), {}).get("status", "MISSING")}
               for s in cfg["seeds"] for a in ARMS if by.get((s, a), {}).get("status") != "OK"]
    out = {"metric": metric, "mie": mie, "complete_seeds": complete, "missing_or_failed": missing}
    val = lambda s, a: by[(s, a)]["planning"][metric]
    if complete:
        out["contrasts"] = {}
        for name, a, b, prereg in CONTRASTS:
            per = [val(s, a) - val(s, b) for s in complete]
            out["contrasts"][name] = {"per_seed": per, "mean": sum(per) / len(per),
                                      "mie_applies": prereg,
                                      "all_seeds_ge_mie": (all(d >= mie for d in per) if prereg else "n/a")}
        inter = [(val(s, "acp_branch") - val(s, "msp_branch")) - (val(s, "acp_nobranch") - val(s, "msp_nobranch"))
                 for s in complete]
        out["contrasts"]["interaction"] = {"per_seed": inter, "mean": sum(inter) / len(inter),
                                           "mie_applies": False, "all_seeds_ge_mie": "n/a"}
    if len(complete) < dec["required_complete_seeds"]:
        out["verdict"] = "INCOMPLETE"
        return out
    if not all(val(s, "informative_control") >= dec["positive_control_min"] for s in complete):
        out["verdict"] = "EVALUATOR_UNRESOLVED"
        return out
    con = out["contrasts"]
    flags = []
    if con["acp_branch_minus_random"]["all_seeds_ge_mie"] is not True:
        out["verdict"] = "LEARNING_BENEFIT_NOT_SUPPORTED"
        out["flags"] = flags
        return out
    learned = ("acp_branch", "acp_nobranch", "msp_branch", "msp_nobranch")
    if all(val(s, a) >= dec["saturation_min"] for s in complete for a in learned):
        out["verdict"] = "SATURATED_COMPARISON"
        out["flags"] = flags
        return out
    collection = con["acp_branch_minus_acp_nobranch"]["all_seeds_ge_mie"] is True
    conditioning = (con["acp_branch_minus_msp_branch"]["all_seeds_ge_mie"] is True
                    or con["acp_nobranch_minus_msp_nobranch"]["all_seeds_ge_mie"] is True)
    if all(d <= -mie for d in con["acp_branch_minus_acp_nobranch"]["per_seed"]):
        flags.append("BRANCHING_WORSE_IN_ALL_SEEDS")
    if conditioning:
        flags.append("ACTION_CONDITIONING_EFFECT (not distinguishable from the existing action-conditioning explanation)")
    if collection:
        out["verdict"] = "PRELIMINARY_COLLECTION_BENEFIT"
    elif conditioning:
        out["verdict"] = "ACTION_CONDITIONING_ONLY"
    else:
        out["verdict"] = "NO_CONSISTENT_MIE_CONTRAST"
    out["flags"] = flags
    return out


def _transition_totals(data_records: dict, cells: list) -> dict:
    tot = {}
    for rec in data_records.values():
        if "pretrain_branch" not in rec:
            continue
        for key in ("pretrain_branch", "pretrain_nobranch", "head_train", "head_dev"):
            led = rec[key]["ledger"]
            for kind in ("main", "branch"):
                tot[f"{key}_{kind}"] = tot.get(f"{key}_{kind}", 0) + led["transitions_by_kind"][kind]
            tot[f"{key}_restores"] = tot.get(f"{key}_restores", 0) + led["restores"]
        tot["calibration_eval"] = tot.get("calibration_eval", 0) + rec["calibration_probe_ledger"]["transitions_total"]
        tot["split_check_test_prefix_eval"] = (tot.get("split_check_test_prefix_eval", 0)
                                               + rec["split_checks"]["test_prefix_check_ledger"]["transitions_total"])
    tot["test_eval"] = sum(c["test_transitions"] for c in cells if c.get("status") == "OK")
    tot["test_restores"] = sum(c["planning"]["ledger"]["restores"] for c in cells if c.get("status") == "OK")
    tot["all"] = sum(v for k, v in tot.items() if not k.endswith("restores"))
    return tot


def _run_cells(cfg, maze_cfg, caps, cpu_cap, manifest, cells, log, flush, weights_dir):
    stop = None
    costs = {}
    for seed in cfg["seeds"]:
        data = None
        if stop is None:
            try:
                t = time.process_time()
                data = prepare_seed(seed, maze_cfg, cfg)
                manifest["data"][str(seed)] = {**data["record"], "prep_cpu_s": time.process_time() - t}
                log({"event": "data", "seed": seed, **data["record"]})
            except CpuCapExceeded as e:
                stop = f"CAP_EXCEEDED during data prep: {e}"
            except MemoryError:
                stop = "OOM during data prep"
            except Exception:
                manifest["data"][str(seed)] = {"status": "FAILED", "traceback": traceback.format_exc()}
        for arm in cfg["arms"]:
            cell = {"seed": seed, "arm": arm}
            kind = "learned" if ARMS[arm]["learned"] else "fixed"
            est = max(costs.get(kind, [caps["initial_cell_cost_estimate_s"][kind]]))
            used = time.process_time()
            if stop is not None or data is None:
                cell.update(status="NOT_RUN", reason=stop or "data preparation failed")
            elif used + est > cpu_cap - caps["reserve_s"]:
                stop = f"projected CPU {used + est:.0f}s exceeds cap {cpu_cap}s minus reserve"
                cell.update(status="NOT_RUN", reason=stop)
            else:
                t_cpu, t_wall, mac0 = time.process_time(), time.perf_counter(), ad.COUNTERS["mac"]
                try:
                    cell.update(status="OK", **run_cell(seed, arm, maze_cfg, cfg, data, log, weights_dir))
                except CpuCapExceeded as e:
                    cell.update(status="CAP_EXCEEDED", reason=str(e))
                    stop = "CPU cap exceeded"
                except MemoryError:
                    cell.update(status="OOM", traceback=traceback.format_exc())
                    stop = "OOM"
                except Exception:
                    cell.update(status="FAILED", traceback=traceback.format_exc())
                cost = time.process_time() - t_cpu
                costs.setdefault(kind, []).append(cost)
                cell.update(cpu_s=cost, wall_s=time.perf_counter() - t_wall, mac=ad.COUNTERS["mac"] - mac0,
                            peak_rss_mb_after=peak_rss_mb())
            cells.append(cell)
            log({"event": "cell", **{k: v for k, v in cell.items() if k not in ("epoch_candidates",)}})
            flush()
    manifest["_stop"] = stop


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    caps = cfg["resource_caps"]
    cpu_cap = caps["cpu_seconds_hard"]
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_cap, cpu_cap + caps["cpu_kill_grace_s"]))
    as_bytes = int(caps["address_space_gib"] * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))
    signal.signal(signal.SIGXCPU, _on_sigxcpu)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=False)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir()
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    raw = (out_dir / "raw_log.jsonl").open("a")

    def log(rec):
        raw.write(json.dumps(rec, sort_keys=True) + "\n")
        raw.flush()

    wall0, cpu0 = time.perf_counter(), time.process_time()
    manifest = {
        "run_kind": cfg["run_kind"],
        "command": " ".join(["python3", "-m", "acpr.study2x2", *(sys.argv[1:] if argv is None else argv)]),
        "git": git_state(REPO),
        "source_sha256": source_hash(REPO),
        "config_sha256": canonical_hash(cfg),
        "environment": environment_info(),
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "python_threads_at_start": threading.active_count(),
        "limits": {"RLIMIT_CPU": resource.getrlimit(resource.RLIMIT_CPU), "RLIMIT_AS": resource.getrlimit(resource.RLIMIT_AS)},
        "data": {},
    }
    if cfg["variant"] != "symmetric":
        raise ValueError("this study is approved for the symmetric variant only")
    maze_cfg = MazeConfig(symmetric=True, **cfg["maze"])
    cells: list = []
    log({"event": "start", "config_sha256": manifest["config_sha256"], "git": manifest["git"]["commit"]})

    def flush():
        (out_dir / "cells.json").write_text(json.dumps(cells, indent=2, sort_keys=True))

    try:
        _run_cells(cfg, maze_cfg, caps, cpu_cap, manifest, cells, log, flush, weights_dir)
    except CpuCapExceeded as e:
        done = {(c["seed"], c["arm"]) for c in cells}
        for seed in cfg["seeds"]:
            for arm in cfg["arms"]:
                if (seed, arm) not in done:
                    cells.append({"seed": seed, "arm": arm, "status": "NOT_RUN", "reason": f"CPU cap: {e}"})
        manifest["stop_reason_outer"] = str(e)
        flush()
    stop = manifest.pop("_stop", None) or manifest.get("stop_reason_outer")
    verdict = judge(cells, cfg)
    manifest.update(
        time={"wall_s": time.perf_counter() - wall0, "process_cpu_s": time.process_time() - cpu0},
        peak_rss_mb=peak_rss_mb(),
        total_mac=ad.COUNTERS["mac"],
        status_counts={s: sum(c["status"] == s for c in cells) for s in ("OK", "FAILED", "OOM", "CAP_EXCEEDED", "NOT_RUN")},
        stop_reason=stop,
        env_transitions=_transition_totals(manifest["data"], cells),
    )
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    log({"event": "end", "verdict": verdict["verdict"], "status_counts": manifest["status_counts"]})
    print(json.dumps({"out": str(out_dir), "verdict": verdict["verdict"], **manifest["status_counts"],
                      "cpu_s": manifest["time"]["process_cpu_s"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
