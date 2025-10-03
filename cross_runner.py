#!/usr/bin/env python3
"""
Batch launcher for run_folds.py

Runs the cross product of:
  DATASET IDs  ×  TRAINER classes
and for each pair calls run_folds.py.

Defaults:
- Process (dataset, trainer) pairs sequentially (one-by-one).
- For each pair, run *all folds in parallel* (passes --all --parallel).

You can override:
- --sequential      -> do NOT pass --parallel to run_folds.py
- --no-all          -> do NOT pass --all to run_folds.py (run_folds.py will use its default fold behavior)
- --devices / --batch -> passed through to run_folds.py when set
- --dry-run / --log-dir -> passed through to run_folds.py
- --val             -> passed through to run_folds.py to run validation-only

Examples
--------
# Run 3 datasets × 2 trainers (defaults to --all --parallel)
python run_all_folds.py -c 3d_fullres -d 003 004 005 -t MyTrainerA MyTrainerB --log-dir runs/aug20 --dry-run

# Same, but do folds sequentially (wrapper still processes pairs one-by-one)
python run_all_folds.py -c 3d_fullres -d 003 004 -t MyTrainer --sequential --devices 0 --dry-run

# Same, but keep parallel folds and cap concurrency in run_folds.py using --batch
python run_all_folds.py -c 3d_fullres -d 003 -t MyTrainer --batch 2 --devices 0,1 --log-dir runs/real

# Validation-only across all pairs
python run_all_folds.py -c 3d_fullres -d 003 004 -t MyTrainer --val --log-dir runs/val_only
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import shlex
import signal
import subprocess
import sys
import glob
import re
from pathlib import Path
from typing import List, Tuple

CHILD_PROCS: List[subprocess.Popen] = []


# ---------- nnU-Net dataset existence check ----------
def _env_paths(keys: List[str]) -> List[str]:
    """Return existing directories from env vars; supports PATH-like lists."""
    found: List[str] = []
    for k in keys:
        raw = os.environ.get(k, "")
        if not raw:
            continue
        for p in raw.split(os.pathsep):
            p = p.strip()
            if p and os.path.isdir(p):
                found.append(p)
    return found

def _normalize_dataset_id(ds: str) -> str:
    """
    Accepts '003' or 'Dataset003_Foo' or 'dataset003_bar' and returns '003'.
    """
    ds = ds.strip()
    m = re.search(r'(\d{3,})', ds)  # allow >=3 digits; pad to at least 3
    if not m:
        raise ValueError(f"Could not extract numeric ID from '{ds}'")
    n = int(m.group(1))
    return f"{n:03d}"

def find_dataset_dirs(ds: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Returns (normalized_id, hits). hits is a list of (kind, path) where kind in {'raw','preprocessed'}.
    Looks under nnUNet v2 and v1-style roots.
    """
    dsid = _normalize_dataset_id(ds)
    name_glob = f"Dataset{int(dsid):03d}_*"

    raw_roots = _env_paths(["nnUNet_raw", "nnUNet_raw_data_base"])
    pre_roots = _env_paths(["nnUNet_preprocessed"])

    hits: List[Tuple[str, str]] = []
    for root in raw_roots:
        for path in glob.glob(str(Path(root) / name_glob)):
            if os.path.isdir(path):
                hits.append(("raw", path))
    for root in pre_roots:
        for path in glob.glob(str(Path(root) / name_glob)):
            if os.path.isdir(path):
                hits.append(("preprocessed", path))
    return dsid, hits

def dataset_exists(ds: str) -> bool:
    _, hits = find_dataset_dirs(ds)
    return len(hits) > 0



def _ts() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, level: str = "INFO") -> None:
    print(f"[{_ts()}] [{level}] {msg}", flush=True)


def banner(title: str) -> None:
    line = "=" * max(60, len(title) + 10)
    log(line)
    log(f"= {title}")
    log(line)


def fmt_cmd(parts: List[str]) -> str:
    # Pretty-print a shell-safe version of the command
    return " ".join(shlex.quote(p) for p in parts)

def resolve_trainer_class(name: str):
    """
    Returns the trainer class object if it exists, otherwise None.
    Scans nnunetv2.training.nnUNetTrainer.* modules for a class named `name`
    that subclasses nnUNetTrainer.
    """
    try:
        import importlib
        import inspect
        import pkgutil

        base_mod = importlib.import_module("nnunetv2.training.nnUNetTrainer")
        base_cls = importlib.import_module(
            "nnunetv2.training.nnUNetTrainer.nnUNetTrainer"
        ).nnUNetTrainer  # type: ignore[attr-defined]

        for m in pkgutil.walk_packages(base_mod.__path__, base_mod.__name__ + "."):
            try:
                mod = importlib.import_module(m.name)
            except Exception:
                continue  # ignore broken/optional modules

            for _, obj in inspect.getmembers(mod, inspect.isclass):
                if obj.__name__ == name and issubclass(obj, base_cls):
                    return obj
    except ModuleNotFoundError:
        # nnUNetv2 not installed/visible; caller can decide to warn or skip check
        return None
    except Exception:
        return None
    return None


def trainer_exists(name: str) -> bool:
    return resolve_trainer_class(name) is not None


def setup_signal_handlers():
    def _handler(signum, frame):
        log(f"Received signal {signum}. Terminating {len(CHILD_PROCS)} child process(es)...", "WARN")
        for p in CHILD_PROCS:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                pass
        for p in CHILD_PROCS:
            try:
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        sys.exit(130)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def build_run_folds_cmd(
    run_folds_path: str,
    dataset: str,
    configuration: str,
    trainer: str,
    *,
    use_all: bool,
    use_parallel: bool,
    dry_run: bool,
    log_dir: str | None,
    devices: str | None,
    batch: int | None,
    val_only: bool,                 # <-- NEW
) -> Tuple[List[str], str | None]:
    """
    Construct the command to invoke run_folds.py for a single (dataset, trainer).
    Returns (argv_list, per_run_logdir).
    """
    argv: List[str] = [sys.executable, run_folds_path, "-d", dataset, "-c", configuration]
    if trainer:
        argv += ["-t", trainer]

    if use_all:
        argv.append("--all")
    if use_parallel:
        argv.append("--parallel")
    if dry_run:
        argv.append("--dry-run")
    if devices:
        argv += ["--devices", devices]
    if batch is not None:
        argv += ["--batch", str(batch)]
    if val_only:
        argv.append("--val")        # <-- NEW passthrough

    per_run_logdir = None
    if log_dir:
        per_run_logdir = os.path.join(log_dir, f"ds{dataset}__tr{trainer or 'default'}")
        argv += ["--log-dir", per_run_logdir]

    return argv, per_run_logdir


def main():
    ap = argparse.ArgumentParser(description="Cross-product launcher for run_folds.py")
    ap.add_argument(
        "-c", "--configuration", default="3d_fullres",
        help="nnUNetv2 configuration (default: 3d_fullres)"
    )
    ap.add_argument(
        "-d", "--datasets", nargs="+", required=True,
        help="One or more dataset IDs (e.g. 003 004 005)"
    )
    ap.add_argument(
        "-t", "--trainers", nargs="+", required=True,
        help="One or more trainer class names"
    )
    ap.add_argument(
        "--run-folds-path", default="run_folds.py",
        help="Path to run_folds.py (default: ./run_folds.py)"
    )
    # Defaults to 'all folds in parallel' per pair:
    mux_parallel = ap.add_mutually_exclusive_group()
    mux_parallel.add_argument("--parallel", dest="parallel", action="store_true", default=True,
                              help="Run folds in parallel for each (dataset, trainer) [default]")
    mux_parallel.add_argument("--sequential", dest="parallel", action="store_false",
                              help="Run folds sequentially for each (dataset, trainer)")

    mux_all = ap.add_mutually_exclusive_group()
    mux_all.add_argument("--all", dest="all_folds", action="store_true", default=True,
                         help="Run all 5 folds for each (dataset, trainer) [default]")
    mux_all.add_argument("--no-all", dest="all_folds", action="store_false",
                         help="Do not pass --all to run_folds.py (it will use its own default)")

    # Simple passthroughs + quality-of-life:
    ap.add_argument("--dry-run", action="store_true", help="Pass --dry-run through to run_folds.py")
    ap.add_argument("--log-dir", type=str, default=None,
                    help="Top-level log directory; subfolders are created per (dataset, trainer)")
    ap.add_argument("--devices", type=str, default=None,
                    help="Pass-through to run_folds.py, e.g. '0,1,2,3'")
    ap.add_argument("--batch", type=int, default=None,
                    help="Pass-through to run_folds.py (batched parallel waves).")
    ap.add_argument("--val", action="store_true",
                    help="Pass-through to run_folds.py to run validation-only (adds --val).")  # <-- NEW

    args = ap.parse_args()
    setup_signal_handlers()

    # Validate run_folds.py path
    run_folds_path = args.run_folds_path
    if not Path(run_folds_path).exists():
        log(f"run_folds.py not found at: {run_folds_path}", "ERR")
        sys.exit(2)

    # Validate trainers early
    unknown = [tr for tr in args.trainers if not trainer_exists(tr)]
    if unknown:
        log(f"Unknown trainer(s): {unknown}", "ERR")
        log("Make sure they are on PYTHONPATH or installed, "
            "and defined under nnunetv2.training.nnUNetTrainer.*", "ERR")
        sys.exit(2)

    # ---- Validate datasets exist on disk ----
    missing = []
    for ds in args.datasets:
        try:
            dsid, hits = find_dataset_dirs(ds)
        except ValueError as e:
            log(str(e), "ERR")
            missing.append(ds)
            continue

        if not hits:
            raw_roots = _env_paths(["nnUNet_raw", "nnUNet_raw_data_base"]) or ["(unset)"]
            pre_roots = _env_paths(["nnUNet_preprocessed"]) or ["(unset)"]
            log(f"No dataset folder found for ID {dsid}.", "ERR")
            log(f"Searched under nnUNet_raw / nnUNet_raw_data_base: {raw_roots}", "ERR")
            log(f"And nnUNet_preprocessed: {pre_roots}", "ERR")
            missing.append(ds)
        else:
            locs = ", ".join([f"{k}:{p}" for k, p in hits])
            log(f"Dataset {ds} (ID {dsid}) found at {locs}")

    if missing:
        log(f"Aborting: unknown dataset IDs: {missing}", "ERR")
        sys.exit(2)

    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    combos: List[Tuple[str, str]] = []
    for ds in args.datasets:
        for tr in args.trainers:
            combos.append((ds, tr))

    banner("run_all_folds – Plan")
    log(f"Configuration: {args.configuration}")
    log(f"Datasets:      {args.datasets}")
    log(f"Trainers:      {args.trainers}")
    log(f"Pairs:         {len(combos)}")
    log(f"Per-pair folds: {'ALL (0..4) in PARALLEL' if (args.all_folds and args.parallel) else ('ALL (0..4) SEQUENTIAL' if args.all_folds else ('Default folds' + (' PARALLEL' if args.parallel else ' SEQUENTIAL')))}")
    log(f"Validation-only: {args.val}")  # <-- NEW
    log(f"Dry-run:       {args.dry_run}")
    log(f"Top log dir:   {args.log_dir if args.log_dir else '(none)'}")
    log(f"Devices:       {args.devices if args.devices else '(inherit)'}")
    log(f"Batch:         {args.batch if args.batch is not None else '(none)'}")

    banner("Execution")
    failures: List[Tuple[str, str, int]] = []

    for i, (ds, tr) in enumerate(combos, 1):
        header = f"({i}/{len(combos)}) DATASET={ds}  TRAINER={tr}"
        banner(header)

        cmd, per_logdir = build_run_folds_cmd(
            run_folds_path=run_folds_path,
            dataset=ds,
            configuration=args.configuration,
            trainer=tr,
            use_all=args.all_folds,
            use_parallel=args.parallel,
            dry_run=args.dry_run,
            log_dir=args.log_dir,
            devices=args.devices,
            batch=args.batch,
            val_only=args.val,  # <-- NEW
        )

        # Ensure sub-logdir exists if requested
        if per_logdir:
            Path(per_logdir).mkdir(parents=True, exist_ok=True)

        # Show full command
        log("Command:")
        print("  " + fmt_cmd(cmd), flush=True)

        # Launch
        try:
            p = subprocess.Popen(cmd)
            CHILD_PROCS.append(p)
            rc = p.wait()
        except FileNotFoundError:
            log("Python executable or run_folds.py not found. Aborting.", "ERR")
            sys.exit(2)

        if rc != 0:
            log(f"FAILED with exit code {rc}", "ERR")
            failures.append((ds, tr, rc))
        else:
            log("SUCCESS")

    banner("Summary")
    if failures:
        log(f"{len(failures)} of {len(combos)} runs failed:", "ERR")
        for ds, tr, rc in failures:
            log(f"- dataset={ds}, trainer={tr}, rc={rc}", "ERR")
        sys.exit(1)
    else:
        log(f"All {len(combos)} runs completed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
