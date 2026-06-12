"""Collection: the `run` wrapper and node `snapshot`.

`nccl-doctor run --job-dir D -- <training command>`:
  1. injects the full diagnostic env profile (NCCL debug logs, topo/graph dumps,
     torch flight recorder) pointed into the job dir,
  2. snapshots node state + fabric counters before launch,
  3. execs the command,
  4. on exit, snapshots again and writes job.json — ready for `analyze`.

`nccl-doctor snapshot --job-dir D --phase start|end` is the same capture for
use from Slurm prolog/epilog on every node of a multi-node job (the wrapper
only captures the local node).

All external commands are best-effort: a missing binary degrades to an absent
artifact, never a failed job.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

DEBUG_ENV = {
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "INIT,ENV,GRAPH,NET,TUNING",
    # torch flight recorder — keystone artifact for true-origin attribution
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "20000",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
}


def _run(cmd: list[str], timeout: int = 30) -> Optional[str]:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else (r.stdout + r.stderr)
    except (subprocess.TimeoutExpired, OSError):
        return None


def _write(path: Path, content: Optional[str]) -> None:
    if content:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


# --------------------------------------------------------------- counters
def collect_ib_counters() -> dict[str, dict[str, float]]:
    """Per-HCA-port counters from sysfs (no perfquery dependency)."""
    ports: dict[str, dict[str, float]] = {}
    base = Path("/sys/class/infiniband")
    if not base.is_dir():
        return ports
    for dev in sorted(base.iterdir()):
        for portdir in sorted((dev / "ports").glob("*")):
            cdir = portdir / "counters"
            if not cdir.is_dir():
                continue
            key = f"{dev.name}:{portdir.name}"
            ports[key] = {}
            for cf in cdir.iterdir():
                try:
                    ports[key][_canon(cf.name)] = float(cf.read_text().strip())
                except (OSError, ValueError):
                    continue
    return ports


_CANON = {
    "symbol_error": "SymbolErrorCounter",
    "link_error_recovery": "LinkErrorRecoveryCounter",
    "link_downed": "LinkDownedCounter",
    "port_rcv_errors": "PortRcvErrors",
    "port_xmit_discards": "PortXmitDiscards",
    "port_xmit_wait": "PortXmitWait",
    "excessive_buffer_overrun_errors": "ExcessiveBufferOverrunErrors",
}


def _canon(name: str) -> str:
    return _CANON.get(name, name)


def collect_ethtool_counters() -> dict[str, dict[str, float]]:
    ports: dict[str, dict[str, float]] = {}
    out = _run(["sh", "-c", "ls /sys/class/net"])
    if not out:
        return ports
    for ifname in out.split():
        if ifname in ("lo",) or ifname.startswith(("docker", "veth")):
            continue
        stats = _run(["ethtool", "-S", ifname], timeout=10)
        if not stats:
            continue
        c: dict[str, float] = {}
        for line in stats.splitlines():
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            k = k.strip()
            if any(t in k for t in ("pause", "cnp", "out_of_buffer", "discards_phy")):
                try:
                    c[k] = float(v.strip())
                except ValueError:
                    pass
        if c:
            ports[ifname] = c
    return ports


def snapshot(job_dir: Path, phase: str) -> None:
    host = socket.gethostname().split(".")[0]
    hdir = job_dir / "host" / host
    _write(hdir / "dmesg.txt", _run(["dmesg", "-T"]) or _run(["dmesg"]))
    _write(hdir / "nvidia-smi-topo.txt", _run(["nvidia-smi", "topo", "-m"]))
    _write(hdir / "nvlink.txt", _run(["nvidia-smi", "nvlink", "--status"]))
    _write(hdir / "lspci.txt", _run(["lspci", "-vvv"], timeout=60))
    ports = {**collect_ib_counters(),
             **collect_ethtool_counters()}
    cpath = job_dir / "counters" / f"{host}.{phase}.json"
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(
        {"host": host, "timestamp": time.time(), "phase": phase, "ports": ports},
        indent=1))


# --------------------------------------------------------------- wrapper
def run_wrapped(job_dir: Path, command: list[str], job_name: Optional[str],
                fabric: str) -> int:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "logs").mkdir(exist_ok=True)
    (job_dir / "topo").mkdir(exist_ok=True)
    (job_dir / "flight_recorder").mkdir(exist_ok=True)

    env = dict(os.environ)
    env.update(DEBUG_ENV)
    env.setdefault("NCCL_DEBUG_FILE", str(job_dir / "logs" / "%h.%p.log"))
    env.setdefault("NCCL_TOPO_DUMP_FILE", str(job_dir / "topo" / "%h.xml"))
    env.setdefault("NCCL_GRAPH_DUMP_FILE", str(job_dir / "topo" / "graph.%h.xml"))
    env.setdefault("TORCH_NCCL_DEBUG_INFO_TEMP_FILE",
                   str(job_dir / "flight_recorder" / "torch_nccl_trace_"))

    (job_dir / "env.json").write_text(json.dumps(
        {k: v for k, v in env.items()
         if k.startswith(("NCCL_", "TORCH_", "CUDA_", "UCX_", "OMPI_", "SLURM_"))},
        indent=1, sort_keys=True))

    snapshot(job_dir, "start")
    t0 = time.time()
    proc = subprocess.run(command, env=env)
    rc = proc.returncode
    snapshot(job_dir, "end")

    outcome = "SUCCESS" if rc == 0 else "NCCL_ERROR"
    # heuristic: torch watchdog aborts exit nonzero and leave FR dumps / log lines
    if rc != 0 and any((job_dir / "logs").glob("*.log")):
        for p in (job_dir / "logs").glob("*.log"):
            if "Watchdog caught" in p.read_text(errors="replace"):
                outcome = "WATCHDOG_TIMEOUT"
                break

    meta = {
        "job_id": env.get("SLURM_JOB_ID", f"local-{int(t0)}"),
        "job_name": job_name or env.get("SLURM_JOB_NAME"),
        "cluster": env.get("SLURM_CLUSTER_NAME", "default"),
        "nodes": sorted({socket.gethostname().split('.')[0],
                         *env.get("SLURM_JOB_NODELIST", "").replace(",", " ").split()})
                 if env.get("SLURM_JOB_NODELIST") else
                 [socket.gethostname().split(".")[0]],
        "world_size": int(env.get("WORLD_SIZE", env.get("SLURM_NTASKS", 1) or 1)),
        "outcome": outcome,
        "fabric": fabric,
        "timestamp": t0,
    }
    (job_dir / "job.json").write_text(json.dumps(meta, indent=1))
    print(f"[nccl-doctor] artifacts in {job_dir} (outcome={outcome}); "
          f"run: nccl-doctor analyze {job_dir}", file=sys.stderr)
    return rc
