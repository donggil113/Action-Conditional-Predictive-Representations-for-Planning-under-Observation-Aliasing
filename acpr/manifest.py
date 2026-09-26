"""Run provenance: git state, hashes, environment, time and memory."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path


def _git(args, cwd) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_state(repo: Path) -> dict:
    commit = _git(["rev-parse", "HEAD"], repo) or "NO_COMMIT"
    porcelain = _git(["status", "--porcelain", "--untracked-files=no"], repo)
    return {"commit": commit, "tracked_files_dirty": bool(porcelain), "dirty_paths": porcelain.splitlines()}


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def source_hash(repo: Path) -> str:
    h = hashlib.sha256()
    for path in sorted((repo / "acpr").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def peak_rss_mb() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def environment_info() -> dict:
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "third_party_packages_used": [],
        "gpu_used": False,
    }


class Timer:
    def __init__(self):
        self.wall0 = time.perf_counter()
        self.cpu0 = time.process_time()

    def read(self) -> dict:
        return {
            "wall_s": time.perf_counter() - self.wall0,
            "process_cpu_s": time.process_time() - self.cpu0,
        }
