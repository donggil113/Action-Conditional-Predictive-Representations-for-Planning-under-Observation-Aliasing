"""Evaluator-discriminability diagnostic (single process, single thread).

    python3 -m acpr.diagnose --config configs/diag_eval_discriminability.json --out runs/<id>

Question: can the current environment plus the frozen-encoder reward head and
planner separate a learned representation from an untrained one? Three fixed
encoders go through the identical downstream path (train-split
standardization, fresh action-conditional reward head with the same init,
data, optimizer and epoch-candidate rule, same planner and test episodes):

  informative_control  ClueCurrentObsEncoder (history-only upper reference)
  random_recurrent     untrained RecurrentEncoder (pilot's random_frozen init)
  memoryless_control   CurrentObsEncoder (negative control)

Resource caps are enforced: RLIMIT_CPU (SIGXCPU -> CAP_EXCEEDED) and
RLIMIT_AS, plus a pre-cell projected-cost check (-> NOT_RUN). Every cell's
status is preserved; the verdict never uses a missing cell.
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
from .controls import ClueCurrentObsEncoder, CurrentObsEncoder, fit_standardizer
from .data import dataset_hash
from .env import N_ACTIONS, MazeConfig
from .evaluate import EvalConfig, clue_probe, collect_probe_histories, planning_eval
from .manifest import canonical_hash, environment_info, git_state, peak_rss_mb, source_hash
from .models import LatentPredictor, RecurrentEncoder, copy_predictor, param_count, params_hash
from .splits import derive_seed
from .train import TrainConfig, build_windows, evaluate_predictor, train_predictor

REPO = Path(__file__).resolve().parent.parent
ENCODERS = ("informative_control", "random_recurrent", "memoryless_control")


class CpuCapExceeded(RuntimeError):
    pass


def _on_sigxcpu(signum, frame):
    # Linux re-sends SIGXCPU every second past the soft limit; raise once and
    # let the hard limit (soft + grace) kill the process if it keeps running.
    signal.signal(signal.SIGXCPU, signal.SIG_IGN)
    raise CpuCapExceeded("RLIMIT_CPU soft limit reached")


def make_encoder(name: str, seed: int, n_obs: int, hidden: int):
    if name == "informative_control":
        return ClueCurrentObsEncoder(n_obs, hidden)
    if name == "memoryless_control":
        return CurrentObsEncoder(n_obs, hidden)
    if name == "random_recurrent":
        return RecurrentEncoder(n_obs, N_ACTIONS, hidden, random.Random(derive_seed("enc-init", seed)))
    raise ValueError(name)


def judge(cells: list, cfg: dict) -> dict:
    dec = cfg["decision"]
    metric = dec["metric"]
    by_seed = {}
    for c in cells:
        by_seed.setdefault(c["seed"], {})[c["encoder"]] = c
    complete, incomplete = [], []
    for seed in cfg["seeds"]:
        row = by_seed.get(seed, {})
        if all(row.get(e, {}).get("status") == "OK" for e in ENCODERS):
            complete.append(seed)
        else:
            incomplete.append({"seed": seed, "status": {e: row.get(e, {}).get("status", "MISSING") for e in ENCODERS}})
    out = {"complete_seeds": complete, "incomplete_seeds": incomplete, "metric": metric}
    if len(complete) < dec["required_complete_seeds"]:
        out["verdict"] = "INCOMPLETE"
        return out

    def val(seed, enc, key=metric):
        return by_seed[seed][enc]["planning"][key]

    per_seed = []
    for s in complete:
        a, b, m = val(s, "informative_control"), val(s, "random_recurrent"), val(s, "memoryless_control")
        per_seed.append({
            "seed": s, "informative": a, "random": b, "memoryless": m,
            "headroom_informative_minus_random": a - b, "random_minus_memoryless": b - m,
            "full_action_success": {e: val(s, e, "success_rate") for e in ENCODERS},
            "full_action_timeout": {e: val(s, e, "timeout_rate") for e in ENCODERS},
            "full_action_mean_return": {e: val(s, e, "mean_return") for e in ENCODERS},
        })
    out["per_seed"] = per_seed
    pos_ok = all(r["informative"] >= dec["positive_control_min"] for r in per_seed)
    neg_ok = all(abs(r["memoryless"] - 0.5) <= dec["negative_control_max_abs_dev_from_half"] for r in per_seed)
    out["positive_control_ok"] = pos_ok
    out["negative_control_ok"] = neg_ok
    out["full_action_positive_control"] = (
        "OK" if all(r["full_action_success"]["informative_control"] >= dec["positive_control_min"] for r in per_seed)
        else "FULL_ACTION_UNRESOLVED"
    )
    if not (pos_ok and neg_ok):
        out["verdict"] = "EVALUATOR_UNRESOLVED"
        return out
    heads = [r["headroom_informative_minus_random"] for r in per_seed]
    if all(h < dec["r1_mie"] for h in heads):
        out["verdict"] = "CURRENT_BENCHMARK_NONDISCRIMINATIVE"
    elif all(h >= dec["r1_mie"] for h in heads):
        out["verdict"] = "HEADROOM_PRESENT"
    else:
        out["verdict"] = "HEADROOM_MIXED"
    return out


def prepare_seed(seed, maze_cfg, cfg):
    cc = CollectConfig(cfg["train_budget"], cfg["horizon"], cfg["branch_prob"], cfg["behavior_forward_prob"])
    train = collect_split(maze_cfg, cc, seed, "train")
    dev = collect_split(
        maze_cfg, CollectConfig(cfg["dev_budget"], cfg["horizon"], cfg["branch_prob"], cfg["behavior_forward_prob"]),
        seed, "dev",
    )
    ev = cfg["eval"]
    calib_h, calib_c, calib_ledger = collect_probe_histories(maze_cfg, seed, "calibration", ev["n_calibration"], ev["stratify_clue"])
    # Evaluator labels (train.labels / dev.labels) are intentionally not used.
    return {
        "train": (train.episodes, build_windows(train.episodes, cfg["horizon"])),
        "dev": (dev.episodes, build_windows(dev.episodes, cfg["horizon"])),
        "calib": (calib_h, calib_c),
        "record": {
            "train": {"ledger": train.ledger, "stats": train.stats, "sha256": dataset_hash(train.episodes)},
            "dev": {"ledger": dev.ledger, "stats": dev.stats, "sha256": dataset_hash(dev.episodes)},
            "calibration_probe_ledger": calib_ledger,
        },
    }


def run_cell(seed, name, maze_cfg, cfg, data, log):
    hidden = cfg["hidden"]
    train_eps, train_ws = data["train"]
    dev_eps, dev_ws = data["dev"]
    enc = fit_standardizer(make_encoder(name, seed, maze_cfg.n_obs, hidden), train_eps)
    head = LatentPredictor("head", hidden, maze_cfg.n_obs, N_ACTIONS, True, random.Random(derive_seed("head-init", seed)))
    hc = cfg["head"]
    cands = sorted(hc["epoch_candidates"])
    ckpts = {}

    def on_epoch_end(n, pred):
        if n in cands:
            ckpts[n] = {"dev": evaluate_predictor(enc, pred, dev_eps, dev_ws), "head": copy_predictor(pred)}
            log({"event": "checkpoint", "seed": seed, "encoder": name, "epoch": n, "dev": ckpts[n]["dev"]})

    tc = TrainConfig(cands[-1], hc["lr"], hc["batch_episodes"], derive_seed("s2-order", seed), obs_weight=0.0)
    train_predictor(enc, head, train_eps, train_ws, tc, False, log, f"{seed}:{name}", on_epoch_end=on_epoch_end)
    selected = min(cands, key=lambda n: (ckpts[n]["dev"]["reward_mse"], n))
    sel_head = ckpts[selected]["head"]
    ev = EvalConfig(**cfg["eval"])
    plan = planning_eval(enc, sel_head, maze_cfg, seed, ev)
    probe = clue_probe(enc, *data["calib"], plan.pop("_junction_histories"), plan.pop("_junction_clues"), ev)
    return {
        "epoch_candidates": {n: {"dev": ckpts[n]["dev"]} for n in cands},
        "selected_epochs": selected,
        "selected_at_max_candidate": selected == cands[-1],
        "head_params": param_count(sel_head.params()),
        "head_sha256": params_hash(sel_head.params()),
        "encoder_params_trainable": 0,
        "standardizer_source_split": enc.source_split,
        "planning": plan,
        "probe": probe,
    }


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
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    raw = (out_dir / "raw_log.jsonl").open("a")

    def log(rec):
        raw.write(json.dumps(rec, sort_keys=True) + "\n")
        raw.flush()

    wall0, cpu0 = time.perf_counter(), time.process_time()
    manifest = {
        "run_kind": cfg["run_kind"],
        "command": " ".join(["python3", "-m", "acpr.diagnose", *(sys.argv[1:] if argv is None else argv)]),
        "git": git_state(REPO),
        "source_sha256": source_hash(REPO),
        "config_sha256": canonical_hash(cfg),
        "environment": environment_info(),
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "python_threads_at_start": threading.active_count(),
        "limits": {"RLIMIT_CPU": resource.getrlimit(resource.RLIMIT_CPU), "RLIMIT_AS": resource.getrlimit(resource.RLIMIT_AS)},
        "data": {},
    }
    maze_cfg = MazeConfig(symmetric=(cfg["variant"] == "symmetric"), **cfg["maze"])
    cells = []
    cell_costs = []
    log({"event": "start", "config_sha256": manifest["config_sha256"], "git": manifest["git"]["commit"]})

    def flush():
        (out_dir / "cells.json").write_text(json.dumps(cells, indent=2, sort_keys=True))

    try:
        _run_cells(cfg, maze_cfg, caps, cpu_cap, manifest, cells, cell_costs, log, flush)
    except CpuCapExceeded as e:
        # Raised between cells (outside a cell's own try): stop and account for the rest.
        done = {(c["seed"], c["encoder"]) for c in cells}
        for seed in cfg["seeds"]:
            for name in ENCODERS:
                if (seed, name) not in done:
                    cells.append({"seed": seed, "encoder": name, "status": "NOT_RUN", "reason": f"CPU cap: {e}"})
        manifest["stop_reason_outer"] = str(e)
        flush()
    stop_reason = manifest.pop("_stop_reason", None) or manifest.get("stop_reason_outer")

    verdict = judge(cells, cfg)
    manifest.update(
        time={"wall_s": time.perf_counter() - wall0, "process_cpu_s": time.process_time() - cpu0},
        peak_rss_mb=peak_rss_mb(),
        total_mac=ad.COUNTERS["mac"],
        status_counts={s: sum(c["status"] == s for c in cells) for s in ("OK", "FAILED", "OOM", "CAP_EXCEEDED", "NOT_RUN")},
        stop_reason=stop_reason,
        env_transitions=_transition_totals(manifest["data"], cells),
    )
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    log({"event": "end", "verdict": verdict["verdict"], "status_counts": manifest["status_counts"]})
    print(json.dumps({"out": str(out_dir), "verdict": verdict["verdict"], **manifest["status_counts"],
                      "cpu_s": manifest["time"]["process_cpu_s"]}))
    return 0


def _run_cells(cfg, maze_cfg, caps, cpu_cap, manifest, cells, cell_costs, log, flush):
    stop_reason = None
    for seed in cfg["seeds"]:
        data = None
        if stop_reason is None:
            try:
                t = time.process_time()
                data = prepare_seed(seed, maze_cfg, cfg)
                manifest["data"][str(seed)] = {**data["record"], "prep_cpu_s": time.process_time() - t}
                log({"event": "data", "seed": seed, **data["record"]})
            except CpuCapExceeded as e:
                stop_reason = f"CAP_EXCEEDED during data prep: {e}"
            except MemoryError:
                stop_reason = "OOM during data prep"
            except Exception:
                manifest["data"][str(seed)] = {"status": "FAILED", "traceback": traceback.format_exc()}
        for name in ENCODERS:
            cell = {"seed": seed, "encoder": name}
            est = max(cell_costs) if cell_costs else caps["initial_cell_cost_estimate_s"]
            used = time.process_time()
            if stop_reason is not None or data is None:
                cell.update(status="NOT_RUN", reason=stop_reason or "data preparation failed")
            elif used + est > cpu_cap - caps["reserve_s"]:
                stop_reason = f"projected CPU {used + est:.0f}s exceeds cap {cpu_cap}s minus reserve"
                cell.update(status="NOT_RUN", reason=stop_reason)
            else:
                t_cpu, t_wall, mac0 = time.process_time(), time.perf_counter(), ad.COUNTERS["mac"]
                try:
                    cell.update(status="OK", **run_cell(seed, name, maze_cfg, cfg, data, log))
                except CpuCapExceeded as e:
                    cell.update(status="CAP_EXCEEDED", reason=str(e))
                    stop_reason = "CPU cap exceeded"
                except MemoryError:
                    cell.update(status="OOM", traceback=traceback.format_exc())
                    stop_reason = "OOM"
                except Exception:
                    cell.update(status="FAILED", traceback=traceback.format_exc())
                cost = time.process_time() - t_cpu
                cell_costs.append(cost)
                cell.update(cpu_s=cost, wall_s=time.perf_counter() - t_wall, mac=ad.COUNTERS["mac"] - mac0,
                            peak_rss_mb_after=peak_rss_mb())
            cells.append(cell)
            log({"event": "cell", **{k: v for k, v in cell.items() if k not in ("epoch_candidates",)}})
            flush()

    manifest["_stop_reason"] = stop_reason


def _transition_totals(data: dict, cells: list) -> dict:
    tot = {"train_main": 0, "train_branch": 0, "train_restores": 0, "dev_main": 0, "dev_branch": 0,
           "dev_restores": 0, "calibration_eval": 0, "test_eval": 0, "test_restores": 0}
    for rec in data.values():
        if "train" not in rec:
            continue
        for split in ("train", "dev"):
            led = rec[split]["ledger"]
            tot[f"{split}_main"] += led["transitions_by_kind"]["main"]
            tot[f"{split}_branch"] += led["transitions_by_kind"]["branch"]
            tot[f"{split}_restores"] += led["restores"]
        tot["calibration_eval"] += rec["calibration_probe_ledger"]["transitions_total"]
    for c in cells:
        if c.get("status") == "OK":
            tot["test_eval"] += c["planning"]["ledger"]["transitions_total"]
            tot["test_restores"] += c["planning"]["ledger"]["restores"]
    tot["all"] = sum(v for k, v in tot.items() if not k.endswith("restores"))
    return tot


if __name__ == "__main__":
    sys.exit(main())
