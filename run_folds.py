#!/usr/bin/env python3
"""
nnUNetv2 multi-fold launcher (verbose + strict device validation)

Key changes vs previous:
- Validates requested devices against *physical* GPU indices:
  - Uses `nvidia-smi --query-gpu=index` when available
  - Falls back to torch.cuda.device_count()
- In all-at-once parallel mode (no --batch), errors if valid devices < #folds
- Warns and drops invalid device indices; aborts if none remain
"""

from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import os
import shutil
import signal
import subprocess
import sys
import textwrap
from typing import List, Optional, Tuple

DEFAULT_FOLDS = [0, 1, 2, 3, 4]
CHILD_PROCS: List[subprocess.Popen] = []

# --------------------------- Logging helpers ---------------------------

def _ts() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str, level: str = "INFO") -> None:
    print(f"[{_ts()}] [{level}] {msg}", flush=True)

def banner(title: str) -> None:
    line = "=" * max(60, len(title) + 10)
    log(line)
    log(f"= {title}")
    log(line)

def fmt_cmd(cmd: List[str]) -> str:
    def q(x: str) -> str:
        if any(c.isspace() for c in x) or any(ch in x for ch in ['"', "'", "\\", "(", ")", "[", "]", "{", "}", ";", "&", "|", ">", "<"]):
            return '"' + x.replace('"', '\\"') + '"'
        return x
    return " ".join(q(c) for c in cmd)

# --------------------------- Device discovery & validation ---------------------------

def parse_devices_arg(dev_str: Optional[str]) -> Optional[List[int]]:
    if not dev_str:
        return None
    parts = [p.strip() for p in dev_str.replace(";", ",").split(",") if p.strip() != ""]
    try:
        devs = [int(p) for p in parts]
    except ValueError:
        raise SystemExit(f"Invalid --devices value: {dev_str!r}. Use a comma-separated list, e.g. '0,1,2'.")
    for d in devs:
        if d < 0:
            raise SystemExit("--devices must contain non-negative integers.")
    return devs

def physical_gpu_indices() -> List[int]:
    """
    Returns the list of *physical* GPU indices present on the system.
    Prefer nvidia-smi; fallback to torch count.
    """
    # nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                stderr=subprocess.STDOUT, text=True
            )
            devs = [int(x.strip()) for x in out.splitlines() if x.strip() != ""]
            if devs:
                return devs
        except Exception:
            pass
    # torch fallback
    try:
        import torch  # type: ignore
        n = torch.cuda.device_count()
        if n and n > 0:
            return list(range(n))
    except Exception:
        pass
    return []

def describe_devices(indices: List[int]) -> List[Tuple[int, Optional[str], Optional[str]]]:
    """
    Returns [(index, name, memory_total)] where possible via nvidia-smi.
    Unknowns are (None, None).
    """
    details: List[Tuple[int, Optional[str], Optional[str]]] = []
    if not indices:
        return details
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                stderr=subprocess.STDOUT, text=True
            )
            info = {}
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    idx = int(parts[0])
                    name = parts[1]
                    mem = parts[2]
                    info[idx] = (name, mem)
            for idx in indices:
                name, mem = info.get(idx, (None, None))
                details.append((idx, name, mem))
            return details
        except Exception:
            pass
    return [(i, None, None) for i in indices]

def validate_requested_devices(requested: List[int], physical: List[int]) -> Tuple[List[int], List[int]]:
    """
    Splits requested into (valid, invalid) according to physical list membership.
    """
    physical_set = set(physical)
    valid = [d for d in requested if d in physical_set]
    invalid = [d for d in requested if d not in physical_set]
    return valid, invalid

# --------------------------- Core logic ---------------------------

def build_cmd(dataset: str, configuration: str, fold: int, trainer: Optional[str], val_only: bool) -> List[str]:
    # nnUNetv2_train <dataset_id> <configuration> <fold> [-tr <trainer>] [--val]
    cmd = ["nnUNetv2_train", dataset, configuration, str(fold)]
    if trainer:
        cmd += ["-tr", trainer]
    if val_only:
        cmd += ["--val"]
    return cmd

def early_checks(args) -> None:
    banner("nnUNetv2 Launcher – Preflight")
    log(f"User: {getpass.getuser()}")
    log(f"Working directory: {os.getcwd()}")
    log(f"Python: {sys.version.split()[0]} (executable: {sys.executable})")

    exe = shutil.which("nnUNetv2_train")
    if exe:
        log(f"Found nnUNetv2_train at: {exe}")
    else:
        log("nnUNetv2_train not found in PATH. Execution will fail unless it's available.", "WARN")

    if args.all:
        folds = DEFAULT_FOLDS
    elif args.folds:
        folds = sorted(set(args.folds))
    else:
        folds = [0]
    if any(f not in DEFAULT_FOLDS for f in folds):
        raise SystemExit("Folds must be within 0..4 for nnUNetv2.")
    log(f"Planned folds: {folds}")

    env_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
    log(f"Parent CUDA_VISIBLE_DEVICES: {env_cvd}")

    arg_dump = textwrap.dedent(f"""
        Dataset:            {args.dataset}
        Configuration:      {args.configuration}
        Trainer:            {args.trainer if args.trainer else '(default)'}
        Mode:               {'Parallel' if (args.parallel or args.batch) else 'Sequential'}
        Batch size:         {args.batch if args.batch else '(none)'}
        Devices (arg):      {args.devices if args.devices else '(not provided)'}
        Single-device arg:  {args.single_device if args.single_device is not None else '(none)'}
        Validation-only:    {args.val}
        Log dir:            {args.log_dir if args.log_dir else '(none)'}
        Dry run:            {args.dry_run}
    """).strip()
    for line in arg_dump.splitlines():
        log(line)

def ensure_log_dir(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    os.makedirs(path, exist_ok=True)
    abs_path = os.path.abspath(path)
    log(f"Logging enabled. Directory: {abs_path}")
    return abs_path

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

def run_sequential(
    folds: List[int],
    dataset: str,
    configuration: str,
    trainer: Optional[str],
    device: Optional[int],
    dry_run: bool,
    log_dir: Optional[str],
    val_only: bool,
) -> int:
    banner("Sequential Execution")
    for f in folds:
        cmd = build_cmd(dataset, configuration, f, trainer, val_only)
        env = os.environ.copy()
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(device)
        log(f"Fold {f} command: {fmt_cmd(cmd)}")
        if device is not None:
            log(f"Fold {f} CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
        if dry_run:
            log(f"DRY-RUN: Skipping execution of fold {f}.")
            continue

        stdout_file = None
        stderr_file = None
        try:
            if log_dir:
                stdout_path = os.path.join(log_dir, f"fold{f}.out.log")
                stderr_path = os.path.join(log_dir, f"fold{f}.err.log")
                log(f"Fold {f} logs -> stdout: {stdout_path} | stderr: {stderr_path}")
                stdout_file = open(stdout_path, "w", buffering=1)
                stderr_file = open(stderr_path, "w", buffering=1)

            rc = subprocess.call(cmd, env=env, stdout=stdout_file or sys.stdout, stderr=stderr_file or sys.stderr)
        finally:
            if stdout_file:
                stdout_file.close()
            if stderr_file:
                stderr_file.close()

        if rc != 0:
            log(f"Fold {f} exited with code {rc}", "ERR")
            return rc
        log(f"Fold {f} completed successfully.")
    return 0

def run_parallel(
    folds: List[int],
    dataset: str,
    configuration: str,
    trainer: Optional[str],
    devices: List[int],
    batch: Optional[int],
    dry_run: bool,
    log_dir: Optional[str],
    val_only: bool,
) -> int:
    mode = "Parallel All-At-Once" if batch is None else f"Parallel Batched (concurrency={min(batch, len(devices))})"
    banner(mode)

    # All-at-once requires enough *valid* devices
    if batch is None:
        if len(devices) < len(folds):
            log(
                f"Not enough CUDA devices for parallel run-all: have {len(devices)}, need {len(folds)}.",
                "ERR",
            )
            log("Provide more devices, use --batch, or run sequentially.", "ERR")
            return 2
        concurrency = len(folds)
    else:
        if batch < 1:
            log("--batch must be >= 1", "ERR")
            return 2
        concurrency = min(batch, len(devices))
        if concurrency == 0:
            log("No valid CUDA devices available for batched execution.", "ERR")
            return 2

    # Device descriptions
    details = describe_devices(devices)
    if details:
        for idx, name, mem in details:
            s_name = name or "(unknown GPU)"
            s_mem = mem or "(unknown mem)"
            # Flag indices that aren't present in nvidia-smi's list
            if name is None and shutil.which("nvidia-smi"):
                log(f"Device {idx}: INVALID (not present per nvidia-smi)", "ERR")
            else:
                log(f"Device {idx}: {s_name} | Memory: {s_mem}")
    else:
        log("No detailed device info available; proceeding with indices only.")

    remaining = folds[:]
    wave_idx = 0
    while remaining:
        wave = remaining[:concurrency]
        remaining = remaining[concurrency:]
        wave_idx += 1

        log(f"Starting wave {wave_idx}: folds {wave} using up to {concurrency} device(s).")
        procs: List[Tuple[int, subprocess.Popen]] = []
        files_to_close = []

        for i, f in enumerate(wave):
            dev = devices[i % len(devices)]
            cmd = build_cmd(dataset, configuration, f, trainer, val_only)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(dev)

            log(f"[Wave {wave_idx}] Fold {f} -> device {dev}")
            log(f"[Wave {wave_idx}] Fold {f} command: {fmt_cmd(cmd)}")
            log(f"[Wave {wave_idx}] Fold {f} CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")

            if dry_run:
                log(f"DRY-RUN: Skipping execution of fold {f}.")
                continue

            stdout_stream = sys.stdout
            stderr_stream = sys.stderr
            if log_dir:
                stdout_path = os.path.join(log_dir, f"fold{f}.out.log")
                stderr_path = os.path.join(log_dir, f"fold{f}.err.log")
                log(f"[Wave {wave_idx}] Fold {f} logs -> stdout: {stdout_path} | stderr: {stderr_path}")
                sf = open(stdout_path, "w", buffering=1)
                ef = open(stderr_path, "w", buffering=1)
                stdout_stream, stderr_stream = sf, ef
                files_to_close.extend([sf, ef])

            p = subprocess.Popen(cmd, env=env, stdout=stdout_stream, stderr=stderr_stream)
            CHILD_PROCS.append(p)
            procs.append((f, p))

        if dry_run:
            continue

        failed = False
        for f, p in procs:
            rc = p.wait()
            if rc != 0:
                log(f"Fold {f} exited with code {rc}", "ERR")
                failed = True
            else:
                log(f"Fold {f} completed successfully.")
        for fh in files_to_close:
            try:
                fh.close()
            except Exception:
                pass
        if failed:
            return 1

    return 0

# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Launcher for nnUNetv2 folds with optional parallel GPU assignment (strict validation)."
    )
    ap.add_argument("-d", "--dataset", default="003", help="Dataset ID (e.g. 003)")
    ap.add_argument("-c", "--configuration", default="3d_fullres", help="nnUNetv2 configuration (e.g. 3d_fullres)")
    ap.add_argument("-t", "--trainer", default=None, help="Trainer class name for -tr")
    ap.add_argument("--folds", nargs="+", type=int, help="Specific folds to run, e.g. --folds 0 1 2")
    ap.add_argument("--all", action="store_true", help="Run all 5 folds (0..4)")
    ap.add_argument("--parallel", action="store_true", help="Run folds in parallel (requires enough GPUs)")
    ap.add_argument("--devices", type=str, help="Comma-separated GPU indices to use, e.g. '0,1,2,3'")
    ap.add_argument("--batch", type=int, default=None,
                    help="Optional: run folds in waves of size N across the given devices (implies parallel mode)")
    ap.add_argument("--single-device", dest="single_device", type=int, default=None,
                    help="Sequential mode only: force a specific device (sets CUDA_VISIBLE_DEVICES per run)")
    ap.add_argument("--log-dir", type=str, default=None, help="If set, write per-fold stdout/stderr logs here")
    ap.add_argument("--dry-run", action="store_true", help="Print what would run without executing")
    ap.add_argument("--val", action="store_true",
                    help="If set, append '--val' to nnUNetv2_train to run validation only (requires finished training).")

    args = ap.parse_args()

    # Treat --batch > 0 as parallel mode if not explicitly set
    if args.batch is not None and args.batch >= 1:
        args.parallel = True

    early_checks(args)
    setup_signal_handlers()
    log_dir = ensure_log_dir(args.log_dir)

    # Resolve folds
    if args.all:
        folds = DEFAULT_FOLDS
    elif args.folds:
        folds = sorted(set(args.folds))
        for f in folds:
            if f not in DEFAULT_FOLDS:
                raise SystemExit("Folds must be in 0..4 for nnUNetv2.")
    else:
        folds = [0]

    # Physical devices present
    physical = physical_gpu_indices()
    if not physical:
        raise SystemExit("No physical CUDA devices detected (via nvidia-smi or torch). Aborting.")
    log(f"Physical GPU indices detected: {physical}")

    # Determine devices to use
    requested_devices = parse_devices_arg(args.devices)
    if args.parallel:
        if requested_devices is None:
            # If none requested, default to *all physical* devices
            devices = list(physical)
            log(f"No --devices provided; using all physical GPUs: {devices}")
        else:
            valid, invalid = validate_requested_devices(requested_devices, physical)
            if invalid:
                log(f"These requested device indices are invalid/not present: {invalid}", "ERR")
            if not valid:
                raise SystemExit("None of the requested devices exist. Aborting.")
            if invalid:
                log(f"Proceeding with valid devices only: {valid}")
            devices = valid
    else:
        devices = []  # sequential path may still use --single-device

    # Execute
    if args.parallel:
        rc = run_parallel(
            folds=folds,
            dataset=args.dataset,
            configuration=args.configuration,
            trainer=args.trainer,
            devices=devices,
            batch=args.batch,
            dry_run=args.dry_run,
            log_dir=log_dir,
            val_only=args.val,
        )
    else:
        rc = run_sequential(
            folds=folds,
            dataset=args.dataset,
            configuration=args.configuration,
            trainer=args.trainer,
            device=args.single_device,
            dry_run=args.dry_run,
            log_dir=log_dir,
            val_only=args.val,
        )

    if args.dry_run:
        banner("Dry Run Summary")
        log("Dry run completed. No commands were executed.")
    else:
        banner("Execution Summary")
        if rc == 0:
            log("All requested folds completed successfully.")
        else:
            log(f"One or more folds failed. Exit code: {rc}", "ERR")

    sys.exit(rc)

if __name__ == "__main__":
    main()
