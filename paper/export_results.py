"""Export manuscript numbers and tables from raw run outputs.

    python3 paper/export_results.py

Reads the run directories named in paper/result_sources.json and writes
  paper/generated/numbers.tex         (\\newcommand macros used in the text)
  paper/tables/*.tex                  (tables included by the manuscript)
  paper/generated/export_manifest.json (source files, sha256, fields used)

Every number in the manuscript that comes from an experiment must be one of
these macros or table cells. Nothing here is typed in by hand.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
ENC_LABEL = {
    "informative_control": "History-only informative control",
    "random_recurrent": "Untrained recurrent encoder",
    "memoryless_control": "Memoryless control",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(run_dir: Path, name: str, used: dict):
    path = run_dir / name
    used[str(path.relative_to(REPO))] = sha(path)
    return json.loads(path.read_text())


def f3(x) -> str:
    if x is None:
        return "--"
    return f"$-${-x:.3f}" if x < -0.0005 else f"{abs(x) if abs(x) < 0.0005 else x:.3f}"


def f2(x) -> str:
    return "--" if x is None else f"{x:.2f}"


def intc(x: int) -> str:
    return f"{int(x):,}".replace(",", "{,}")


def signed(x) -> str:
    return "--" if x is None else (f"$+${x:.3f}" if x > 0 else (f"$-${-x:.3f}" if x < 0 else "0.000"))


class Macros:
    def __init__(self):
        self.lines = []

    def add(self, name: str, value: str, source: str):
        if not name.isalpha():
            raise ValueError(f"macro name must be letters only: {name}")
        self.lines.append(f"\\newcommand{{\\{name}}}{{{value}}} % {source}")


def export_diag(run_dir: Path, m: Macros, used: dict) -> None:
    cells = load(run_dir, "cells.json", used)
    verdict = load(run_dir, "verdict.json", used)
    man = load(run_dir, "manifest.json", used)
    seeds = sorted({c["seed"] for c in cells})
    rel = str(run_dir.relative_to(REPO))
    m.add("DiagVerdict", "\\texttt{" + verdict["verdict"].replace("_", "\\_") + "}", f"{rel}/verdict.json:verdict")
    m.add("DiagSeeds", ", ".join(str(s) for s in seeds), f"{rel}/cells.json:seed")
    m.add("DiagNumOK", str(sum(c["status"] == "OK" for c in cells)), f"{rel}/cells.json:status")
    m.add("DiagNumCells", str(len(cells)), f"{rel}/cells.json")
    by = {(c["seed"], c["encoder"]): c for c in cells}
    heads = [r["headroom_informative_minus_random"] for r in verdict["per_seed"]]
    rm = [r["random_minus_memoryless"] for r in verdict["per_seed"]]
    m.add("DiagHeadroomMin", f3(min(heads)), f"{rel}/verdict.json:per_seed[].headroom_informative_minus_random")
    m.add("DiagHeadroomMax", f3(max(heads)), f"{rel}/verdict.json:per_seed[].headroom_informative_minus_random")
    m.add("DiagRandMinusMemMin", f3(min(rm)), f"{rel}/verdict.json:per_seed[].random_minus_memoryless")
    m.add("DiagRandMinusMemMax", f3(max(rm)), f"{rel}/verdict.json:per_seed[].random_minus_memoryless")
    for key, short in (("informative_control", "Info"), ("random_recurrent", "Rand"), ("memoryless_control", "Mem")):
        vals = [by[(s, key)]["planning"]["forced_commit_accuracy"] for s in seeds]
        m.add(f"DiagForced{short}Min", f3(min(vals)), f"{rel}/cells.json:planning.forced_commit_accuracy[{key}]")
        m.add(f"DiagForced{short}Max", f3(max(vals)), f"{rel}/cells.json:planning.forced_commit_accuracy[{key}]")
        probes = [by[(s, key)]["probe"]["test_accuracy"] for s in seeds]
        m.add(f"DiagProbe{short}Min", f2(min(probes)), f"{rel}/cells.json:probe.test_accuracy[{key}]")
        m.add(f"DiagProbe{short}Max", f2(max(probes)), f"{rel}/cells.json:probe.test_accuracy[{key}]")
        succ = [by[(s, key)]["planning"]["success_rate"] for s in seeds]
        m.add(f"DiagSuccess{short}Min", f3(min(succ)), f"{rel}/cells.json:planning.success_rate[{key}]")
        m.add(f"DiagSuccess{short}Max", f3(max(succ)), f"{rel}/cells.json:planning.success_rate[{key}]")
        tout = [by[(s, key)]["planning"]["timeout_rate"] for s in seeds]
        m.add(f"DiagTimeout{short}Max", f3(max(tout)), f"{rel}/cells.json:planning.timeout_rate[{key}]")
    tr = man["env_transitions"]
    for k, name in (("all", "All"), ("train_main", "TrainMain"), ("train_branch", "TrainBranch"),
                    ("train_restores", "TrainRestores"), ("dev_main", "DevMain"), ("dev_branch", "DevBranch"),
                    ("dev_restores", "DevRestores"), ("calibration_eval", "Calib"), ("test_eval", "Test"),
                    ("test_restores", "TestRestores")):
        m.add(f"DiagTransitions{name}", intc(tr[k]), f"{rel}/manifest.json:env_transitions.{k}")
    m.add("DiagCPUSeconds", intc(round(man["time"]["process_cpu_s"])), f"{rel}/manifest.json:time.process_cpu_s")
    m.add("DiagPeakRSSMB", f"{man['peak_rss_mb']:.1f}", f"{rel}/manifest.json:peak_rss_mb")
    m.add("DiagTotalGMAC", f"{man['total_mac'] / 1e9:.2f}", f"{rel}/manifest.json:total_mac")
    m.add("DiagCommit", man["git"]["commit"][:7], f"{rel}/manifest.json:git.commit")
    cpus = [c["cpu_s"] for c in cells]
    m.add("DiagCellCPUMin", f"{min(cpus):.0f}", f"{rel}/cells.json:cpu_s")
    m.add("DiagCellCPUMax", f"{max(cpus):.0f}", f"{rel}/cells.json:cpu_s")

    def trip(key, field, fmt=f3, sub="planning"):
        return " / ".join(fmt(by[(s, key)][sub][field]) for s in seeds)

    rows = []
    for key in ("informative_control", "random_recurrent", "memoryless_control"):
        sel = " / ".join(str(by[(s, key)]["selected_epochs"]) for s in seeds)
        rows.append(
            f"{ENC_LABEL[key]} & {trip(key, 'forced_commit_accuracy')} & {trip(key, 'success_rate')} & "
            f"{trip(key, 'timeout_rate')} & {trip(key, 'test_accuracy', f2, 'probe')} & {sel} \\\\"
        )
    (PAPER / "tables" / "diag_controls.tex").write_text(
        "% Generated by paper/export_results.py from " + rel + "/cells.json. Do not edit.\n"
        "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
        "Encoder & Forced-commit & Full-action success & Timeout & Probe (test) & Epochs \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )


def export_study(run_dir: Path, m: Macros, used: dict) -> None:
    cells = load(run_dir, "cells.json", used)
    verdict = load(run_dir, "verdict.json", used)
    man = load(run_dir, "manifest.json", used)
    rel = str(run_dir.relative_to(REPO))
    seeds = sorted({c["seed"] for c in cells})
    by = {(c["seed"], c["arm"]): c for c in cells}
    arms = ["acp_branch", "acp_nobranch", "msp_branch", "msp_nobranch", "random_recurrent", "informative_control"]
    label = {
        "acp_branch": "A: ACP + branching", "acp_nobranch": "B: ACP + no branching",
        "msp_branch": "C: MSP + branching", "msp_nobranch": "D: MSP + no branching",
        "random_recurrent": "E: untrained recurrent", "informative_control": "F: informative control",
    }
    m.add("StudyVerdict", "\\texttt{" + verdict["verdict"].replace("_", "\\_") + "}", f"{rel}/verdict.json:verdict")
    m.add("StudySeeds", ", ".join(str(s) for s in seeds), f"{rel}/cells.json:seed")
    m.add("StudyNumOK", str(sum(c["status"] == "OK" for c in cells)), f"{rel}/cells.json:status")
    m.add("StudyNumCells", str(len(cells)), f"{rel}/cells.json")

    def val(s, a, field, sub="planning"):
        c = by.get((s, a))
        if not c or c.get("status") != "OK":
            return None
        return c[sub][field] if sub else c[field]

    def trip(a, field, fmt=f3, sub="planning"):
        return " / ".join(fmt(val(s, a, field, sub)) for s in seeds)

    rows = []
    for a in arms:
        st = {by.get((s, a), {}).get("status", "MISSING") for s in seeds}
        status = "OK" if st == {"OK"} else "/".join(sorted(st))
        rows.append(f"{label[a]} & {trip(a, 'forced_commit_accuracy')} & {trip(a, 'test_accuracy', f2, 'probe')} & "
                    f"{' / '.join(str(val(s, a, 'selected_epochs', None) or '--') for s in seeds)} & {status} \\\\")
    (PAPER / "tables" / "study_forced.tex").write_text(
        "% Generated by paper/export_results.py from " + rel + "/cells.json. Do not edit.\n"
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        "Arm & Forced-commit & Probe (test) & Head epochs & Status \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    rows = []
    for a in arms:
        rows.append(f"{label[a]} & {trip(a, 'success_rate')} & {trip(a, 'mean_return')} & {trip(a, 'timeout_rate')} & "
                    f"{' / '.join(str(val(s, a, 'forced_commit_ties') if val(s, a, 'forced_commit_ties') is not None else '--') for s in seeds)} \\\\")
    (PAPER / "tables" / "study_fullaction.tex").write_text(
        "% Generated by paper/export_results.py from " + rel + "/cells.json. Do not edit.\n"
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        "Arm & Full-action success & Mean return & Timeout & Forced ties \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    con = verdict.get("contrasts", {})
    names = {
        "acp_branch_minus_acp_nobranch": "ACP$_b$ $-$ ACP$_n$ (collection, ACP)",
        "msp_branch_minus_msp_nobranch": "MSP$_b$ $-$ MSP$_n$ (collection, MSP)",
        "acp_branch_minus_msp_branch": "ACP$_b$ $-$ MSP$_b$ (conditioning, branch data)",
        "acp_nobranch_minus_msp_nobranch": "ACP$_n$ $-$ MSP$_n$ (conditioning, no-branch data)",
        "interaction": "(ACP$_b$$-$MSP$_b$) $-$ (ACP$_n$$-$MSP$_n$)",
        "acp_branch_minus_random": "ACP$_b$ $-$ untrained",
        "acp_nobranch_minus_random": "ACP$_n$ $-$ untrained",
        "msp_branch_minus_random": "MSP$_b$ $-$ untrained",
        "msp_nobranch_minus_random": "MSP$_n$ $-$ untrained",
    }
    rows = []
    for k, lab in names.items():
        c = con.get(k)
        if c is None:
            rows.append(f"{lab} & -- & -- & -- \\\\")
            continue
        per = " / ".join(signed(v) for v in c["per_seed"])
        rows.append(f"{lab} & {per} & {signed(c['mean'])} & {c['all_seeds_ge_mie']} \\\\")
    (PAPER / "tables" / "study_contrasts.tex").write_text(
        "% Generated by paper/export_results.py from " + rel + "/verdict.json. Do not edit.\n"
        "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
        "Paired contrast (forced-commit) & Per seed & Mean & $\\geq$ MIE in all seeds \\\\\n\\midrule\n"
        + "\n".join(rows).replace("True", "yes").replace("False", "no") + "\n\\bottomrule\n\\end{tabular}\n"
    )
    for k, short in (("acp_branch_minus_acp_nobranch", "CollACP"), ("msp_branch_minus_msp_nobranch", "CollMSP"),
                     ("acp_branch_minus_msp_branch", "CondBranch"), ("acp_nobranch_minus_msp_nobranch", "CondNoBranch"),
                     ("interaction", "Interaction"), ("acp_branch_minus_random", "LearnRand")):
        c = con.get(k)
        if c:
            m.add(f"Study{short}Mean", signed(c["mean"]), f"{rel}/verdict.json:contrasts.{k}.mean")
            m.add(f"Study{short}Min", signed(min(c["per_seed"])), f"{rel}/verdict.json:contrasts.{k}.per_seed")
            m.add(f"Study{short}Max", signed(max(c["per_seed"])), f"{rel}/verdict.json:contrasts.{k}.per_seed")
    if con:
        c_lr = con["acp_branch_minus_random"]["per_seed"]
        for i, v in enumerate(c_lr):
            m.add(f"StudyLearnRandSeed{ROMAN[i]}", signed(v), f"{rel}/verdict.json:contrasts.acp_branch_minus_random.per_seed[{i}]")
        allc = [v for c in con.values() for v in c["per_seed"]]
        m.add("StudyAllContrastMin", signed(min(allc)), f"{rel}/verdict.json:contrasts.*.per_seed")
        m.add("StudyAllContrastMax", signed(max(allc)), f"{rel}/verdict.json:contrasts.*.per_seed")
    for a, short in (("acp_branch", "ACPb"), ("acp_nobranch", "ACPn"), ("msp_branch", "MSPb"), ("msp_nobranch", "MSPn"),
                     ("random_recurrent", "Rand"), ("informative_control", "Info")):
        vals = [v for v in (val(s, a, "forced_commit_accuracy") for s in seeds) if v is not None]
        if vals:
            m.add(f"StudyForced{short}Min", f3(min(vals)), f"{rel}/cells.json:planning.forced_commit_accuracy[{a}]")
            m.add(f"StudyForced{short}Max", f3(max(vals)), f"{rel}/cells.json:planning.forced_commit_accuracy[{a}]")
            m.add(f"StudyForced{short}Mean", f3(sum(vals) / len(vals)), f"{rel}/cells.json:planning.forced_commit_accuracy[{a}]")
    # Experience and compute budget per seed (from the ledgers and per-cell counters).
    data = man["data"]
    rows = []
    for s in seeds:
        d = data[str(s)]
        for key, lab in (("pretrain_branch", "Pretraining, branching"), ("pretrain_nobranch", "Pretraining, no branching"),
                         ("head_train", "Readout pool, train"), ("head_dev", "Readout pool, dev")):
            led = d[key]["ledger"]
            rows.append(f"{s} & {lab} & {intc(led['transitions_by_kind']['main'])} & {intc(led['transitions_by_kind']['branch'])} & "
                        f"{intc(led['restores'])} & {intc(d[key]['stats']['episodes'])} \\\\")
        led = d["calibration_probe_ledger"]
        rows.append(f"{s} & Calibration (probe) & {intc(led['transitions_total'])} & 0 & {intc(led['restores'])} & {intc(led['resets'])} \\\\")
        rows.append("\\midrule" if s != seeds[-1] else "")
    (PAPER / "tables" / "study_experience.tex").write_text(
        "% Generated by paper/export_results.py from " + rel + "/manifest.json. Do not edit.\n"
        "\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\n"
        "Seed & Data & Main steps & Branch steps & Restores & Episodes \\\\\n\\midrule\n"
        + "\n".join(r for r in rows if r) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    rows = []
    for a in arms:
        s1 = [val(s, a, "stage1_cpu_s", None) for s in seeds]
        s1m = [val(s, a, "stage1_mac", None) for s in seeds]
        rd = [val(s, a, "readout_cpu_s", None) for s in seeds]
        te = [val(s, a, "test_transitions", None) for s in seeds]

        def rng(xs, fmt):
            xs = [x for x in xs if x is not None]
            if not xs:
                return "--"
            lo, hi = fmt(min(xs)), fmt(max(xs))
            return lo if lo == hi else f"{lo}--{hi}"

        rows.append(f"{label[a]} & {rng(s1, lambda x: f'{x:.0f}')} & {rng(s1m, lambda x: f'{x / 1e9:.2f}')} & "
                    f"{rng(rd, lambda x: f'{x:.0f}')} & {rng(te, intc)} \\\\")
    (PAPER / "tables" / "study_compute.tex").write_text(
        "% Generated by paper/export_results.py from " + rel + "/cells.json. Do not edit.\n"
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        "Arm & Pretrain CPU (s) & Pretrain GMAC & Readout CPU (s) & Test transitions \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    ov = {k: [data[str(s)]["split_checks"]["test_junction_prefix_obs_overlap"][k] for s in seeds]
          for k in ("pretrain_branch", "pretrain_nobranch", "head_train")}
    allov = [v for vs in ov.values() for v in vs]
    m.add("StudyPrefixOverlapMin", f2(min(allov)), f"{rel}/manifest.json:data.*.split_checks.test_junction_prefix_obs_overlap")
    m.add("StudyPrefixOverlapMax", f2(max(allov)), f"{rel}/manifest.json:data.*.split_checks.test_junction_prefix_obs_overlap")
    shared = [data[str(s)]["split_checks"]["pretrain_branch_nobranch_shared_root_episodes"] for s in seeds]
    m.add("StudySharedRootsMin", intc(min(shared)), f"{rel}/manifest.json:data.*.split_checks.pretrain_branch_nobranch_shared_root_episodes")
    m.add("StudySharedRootsMax", intc(max(shared)), f"{rel}/manifest.json:data.*.split_checks.pretrain_branch_nobranch_shared_root_episodes")
    seed_ov = [sum(v for v in data[str(s)]["split_checks"]["latent_seed_overlap_between_splits"].values()) for s in seeds]
    m.add("StudyLatentSeedOverlapTotal", str(sum(seed_ov)), f"{rel}/manifest.json:data.*.split_checks.latent_seed_overlap_between_splits")
    tr = man["env_transitions"]
    for k, name in (("pretrain_branch_restores", "PretrainRestores"), ("head_train_restores", "PoolTrainRestores"),
                    ("head_dev_restores", "PoolDevRestores"), ("pretrain_nobranch_restores", "NoBranchRestores"),
                    ("test_restores", "TestRestores"), ("test_eval", "TestTransitions"),
                    ("calibration_eval", "CalibTransitions"), ("split_check_test_prefix_eval", "PrefixCheckTransitions")):
        m.add(f"Study{name}", intc(tr[k]), f"{rel}/manifest.json:env_transitions.{k}")
    pool = tr["head_train_main"] + tr["head_dev_main"]
    m.add("StudyPoolTransitions", intc(pool), f"{rel}/manifest.json:env_transitions.head_train_main+head_dev_main")
    pre = tr["pretrain_branch_main"] + tr["pretrain_branch_branch"] + tr["pretrain_nobranch_main"]
    m.add("StudyPretrainTransitions", intc(pre), f"{rel}/manifest.json:env_transitions.pretrain_*")
    sel80 = sum(1 for c in cells if c.get("status") == "OK" and c.get("selected_at_max_candidate"))
    m.add("StudySelectedAtMax", str(sel80), f"{rel}/cells.json:selected_at_max_candidate")
    learned = ("acp_branch", "acp_nobranch", "msp_branch", "msp_nobranch")
    lf = [val(s, a, "forced_commit_accuracy") for s in seeds for a in learned]
    m.add("StudyForcedLearnedMin", f3(min(lf)), f"{rel}/cells.json:planning.forced_commit_accuracy[learned arms]")
    m.add("StudyForcedLearnedMax", f3(max(lf)), f"{rel}/cells.json:planning.forced_commit_accuracy[learned arms]")
    lp = [val(s, a, "test_accuracy", "probe") for s in seeds for a in learned]
    m.add("StudyProbeLearnedMin", f2(min(lp)), f"{rel}/cells.json:probe.test_accuracy[learned arms]")
    m.add("StudyProbeLearnedMax", f2(max(lp)), f"{rel}/cells.json:probe.test_accuracy[learned arms]")
    rp = [val(s, "random_recurrent", "test_accuracy", "probe") for s in seeds]
    m.add("StudyProbeRandMin", f2(min(rp)), f"{rel}/cells.json:probe.test_accuracy[random_recurrent]")
    m.add("StudyProbeRandMax", f2(max(rp)), f"{rel}/cells.json:probe.test_accuracy[random_recurrent]")
    tout = [val(s, a, "timeout_rate") for s in seeds for a in arms if a != "informative_control"]
    m.add("StudyTimeoutNonInfoMax", f3(max(tout)), f"{rel}/cells.json:planning.timeout_rate[non-informative arms]")
    # Pretraining prediction metrics (descriptive; ACP and MSP objectives are not like-for-like).
    rows = []
    for a in learned:
        first = [by[(s, a)]["stage1_train_loss_first_last"][0] for s in seeds]
        last = [by[(s, a)]["stage1_train_loss_first_last"][1] for s in seeds]
        nll = [by[(s, a)]["stage1_head_dev_prediction"]["obs_nll"] for s in seeds]
        rm = [by[(s, a)]["stage1_head_dev_prediction"]["reward_mse"] for s in seeds]
        fmt = lambda xs: " / ".join(f"{x:.3f}" for x in xs)
        rows.append(f"{label[a]} & {fmt(first)} & {fmt(last)} & {fmt(nll)} & {fmt(rm)} \\\\")
    (PAPER / "tables" / "study_pretrain.tex").write_text(
        "% Generated by paper/export_results.py from " + rel + "/cells.json. Do not edit.\n"
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        "Arm & Train loss, epoch 1 & Train loss, final & Readout-dev obs.\\ NLL & Readout-dev reward MSE \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    m.add("StudyCPUSeconds", intc(round(man["time"]["process_cpu_s"])), f"{rel}/manifest.json:time.process_cpu_s")
    m.add("StudyPeakRSSMB", f"{man['peak_rss_mb']:.1f}", f"{rel}/manifest.json:peak_rss_mb")
    m.add("StudyTotalGMAC", f"{man['total_mac'] / 1e9:.2f}", f"{rel}/manifest.json:total_mac")
    m.add("StudyEnvTransitions", intc(man["env_transitions"]["all"]), f"{rel}/manifest.json:env_transitions.all")
    m.add("StudyCommit", man["git"]["commit"][:7], f"{rel}/manifest.json:git.commit")


def export_config(path: Path, prefix: str, m: Macros, used: dict) -> dict:
    cfg = json.loads(path.read_text())
    used[str(path.relative_to(REPO))] = sha(path)
    rel = str(path.relative_to(REPO))
    mz = cfg["maze"]
    m.add(f"{prefix}CorridorMin", str(mz["corridor_min"]), f"{rel}:maze.corridor_min")
    m.add(f"{prefix}CorridorMax", str(mz["corridor_max"]), f"{rel}:maze.corridor_max")
    m.add(f"{prefix}Distractors", str(mz["n_distractors"]), f"{rel}:maze.n_distractors")
    m.add(f"{prefix}TimeoutSlack", str(mz["timeout_slack"]), f"{rel}:maze.timeout_slack")
    m.add(f"{prefix}Horizon", str(cfg["horizon"]), f"{rel}:horizon")
    m.add(f"{prefix}Hidden", str(cfg["hidden"]), f"{rel}:hidden")
    m.add(f"{prefix}BranchProb", str(cfg["branch_prob"]), f"{rel}:branch_prob")
    m.add(f"{prefix}ForwardProb", str(cfg["behavior_forward_prob"]), f"{rel}:behavior_forward_prob")
    ev = cfg["eval"]
    m.add(f"{prefix}NTest", str(ev["n_test"]), f"{rel}:eval.n_test")
    m.add(f"{prefix}NCalib", str(ev["n_calibration"]), f"{rel}:eval.n_calibration")
    m.add(f"{prefix}PlanHorizon", str(ev["plan_horizon"]), f"{rel}:eval.plan_horizon")
    m.add(f"{prefix}Gamma", str(ev["gamma"]), f"{rel}:eval.gamma")
    hd = cfg["head"]
    m.add(f"{prefix}HeadLR", str(hd["lr"]), f"{rel}:head.lr")
    m.add(f"{prefix}HeadEpochs", "/".join(str(e) for e in hd["epoch_candidates"]), f"{rel}:head.epoch_candidates")
    dec = cfg["decision"]
    m.add(f"{prefix}MIE", f"{dec['r1_mie'] if 'r1_mie' in dec else dec['mie']:.2f}", f"{rel}:decision.mie")
    m.add(f"{prefix}PosMin", f"{dec['positive_control_min']:.2f}", f"{rel}:decision.positive_control_min")
    caps = cfg["resource_caps"]
    m.add(f"{prefix}CPUCap", intc(caps["cpu_seconds_hard"]), f"{rel}:resource_caps.cpu_seconds_hard")
    return cfg


def export_study_config(path: Path, m: Macros, used: dict) -> None:
    cfg = json.loads(path.read_text())
    used[str(path.relative_to(REPO))] = sha(path)
    rel = str(path.relative_to(REPO))
    pt, hp, hd, ev, dec, caps = (cfg["pretraining"], cfg["readout_pool"], cfg["head"], cfg["eval"],
                                 cfg["decision"], cfg["resource_caps"])
    m.add("StudyCfgSeeds", ", ".join(str(s) for s in cfg["seeds"]), f"{rel}:seeds")
    m.add("StudyCfgPretrainBudget", intc(pt["budget"]), f"{rel}:pretraining.budget")
    m.add("StudyCfgBranchProb", str(pt["branch_prob"]), f"{rel}:pretraining.branch_prob")
    m.add("StudyCfgPretrainEpochs", str(pt["epochs"]), f"{rel}:pretraining.epochs")
    m.add("StudyCfgPretrainLR", str(pt["lr"]), f"{rel}:pretraining.lr")
    m.add("StudyCfgPoolTrain", intc(hp["train_budget"]), f"{rel}:readout_pool.train_budget")
    m.add("StudyCfgPoolDev", intc(hp["dev_budget"]), f"{rel}:readout_pool.dev_budget")
    m.add("StudyCfgHeadEpochs", "/".join(str(e) for e in hd["epoch_candidates"]), f"{rel}:head.epoch_candidates")
    m.add("StudyCfgNTest", str(ev["n_test"]), f"{rel}:eval.n_test")
    m.add("StudyCfgNCalib", str(ev["n_calibration"]), f"{rel}:eval.n_calibration")
    m.add("StudyCfgMIE", f"{dec['mie']:.2f}", f"{rel}:decision.mie")
    m.add("StudyCfgPosMin", f"{dec['positive_control_min']:.2f}", f"{rel}:decision.positive_control_min")
    m.add("StudyCfgSatMin", f"{dec['saturation_min']:.2f}", f"{rel}:decision.saturation_min")
    m.add("StudyCfgCPUCap", intc(caps["cpu_seconds_hard"]), f"{rel}:resource_caps.cpu_seconds_hard")
    m.add("StudyCfgStageCap", intc(caps["stage_cpu_seconds_total"]), f"{rel}:resource_caps.stage_cpu_seconds_total")


def main() -> int:
    sources = json.loads((PAPER / "result_sources.json").read_text())
    used: dict = {}
    m = Macros()
    m.add("ExportNote", "numbers exported from raw run files", "paper/export_results.py")
    dcfg = export_config(REPO / sources["diagnostic_config"], "DiagCfg", m, used)
    m.add("DiagCfgTrainBudget", intc(dcfg["train_budget"]), sources["diagnostic_config"] + ":train_budget")
    m.add("DiagCfgDevBudget", intc(dcfg["dev_budget"]), sources["diagnostic_config"] + ":dev_budget")
    m.add("DiagCfgNegDev", f"{dcfg['decision']['negative_control_max_abs_dev_from_half']:.2f}",
          sources["diagnostic_config"] + ":decision.negative_control_max_abs_dev_from_half")
    export_diag(REPO / sources["diagnostic"], m, used)
    if sources.get("study_config"):
        export_study_config(REPO / sources["study_config"], m, used)
    study = sources.get("study_2x2")
    if study:
        export_study(REPO / study, m, used)
        m.add("StudyAvailable", "yes", "paper/result_sources.json:study_2x2")
    else:
        m.add("StudyAvailable", "no", "paper/result_sources.json:study_2x2")
    (PAPER / "generated").mkdir(exist_ok=True)
    (PAPER / "generated" / "numbers.tex").write_text(
        "% Generated by paper/export_results.py. Do not edit by hand.\n" + "\n".join(m.lines) + "\n"
    )
    (PAPER / "generated" / "export_manifest.json").write_text(
        json.dumps({"sources": sources, "files_sha256": used, "n_macros": len(m.lines)}, indent=2, sort_keys=True) + "\n"
    )
    print(f"exported {len(m.lines)} macros from {len(used)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
