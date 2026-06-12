"""Generate synthetic job artifact directories that replicate real failure modes.

Scenario A ("fabric"): 4-node H100 job, watchdog timeout reported by rank 3,
true origin rank 17 on gpu-201 — IB retry exhaustion toward gpu-202 whose leaf
link is throwing symbol errors. A sibling run of the same job_name previously
PASSED on a node set avoiding gpu-202 (the user's "it works when I rerun it").

Scenario B ("config"): 2-node job where gpu-310 silently fell back to
NET/Socket and runs without GDRDMA, while gpu-311 logged a critical Xid 79.

Used by tests and by `make demo`.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

NOW = time.time()


def _w(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _wj(path: Path, obj) -> None:
    _w(path, json.dumps(obj, indent=1))


def nccl_log_common(host: str, pid: int, nchan: int = 16, gdr: bool = True,
                    net: str = "IB") -> str:
    p = f"{host}:{pid}:{pid+20} [0] NCCL INFO"
    gdrs = "/GDRDMA" if gdr else ""
    lines = [
        f"{p} Bootstrap : Using eth0:10.0.0.1<0>",
        f"{p} NCCL version 2.26.2+cuda12.4",
        f"{p} NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_4:1/IB ; OOB eth0:10.0.0.1<0>"
        if net == "IB" else
        f"{p} NET/Socket : Using [0]eth0:10.0.0.1<0>",
        f"{p} Using network {net}",
        f"{p} Channel 00/{nchan:02d} :    0   1   2   3   4   5   6   7",
        f"{p} Trees [0] 1/-1/-1->0->-1 [1] -1/-1/-1->0->1",
        f"{p} Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM",
        f"{p} Channel 00/0 : 4[4] -> 0[0] [receive] via NET/{net}/0{gdrs}",
        f"{p} Channel 01/0 : 0[0] -> 4[4] [send] via NET/{net}/1{gdrs}",
        f"{p} Connected all rings",
        f"{p} Connected all trees",
        f"{p} 8 coll channels, algorithm RING protocol LL128",
    ]
    return "\n".join(lines) + "\n"


def topo_xml(host: str, nics: int = 8) -> str:
    nic_xml = "\n".join(
        f'    <nic><net name="mlx5_{i}" dev="{i}" speed="400000"/></nic>'
        for i in range(nics))
    gpus = "\n".join(f'    <gpu dev="{i}" sm="90"/>' for i in range(8))
    return f'<system version="1">\n  <cpu numaid="0">\n{gpus}\n{nic_xml}\n  </cpu>\n</system>\n'


def counters(host: str, phase: str, symbol_end: int = 0, xmit_disc_end: int = 0) -> dict:
    base = {"SymbolErrorCounter": 12, "LinkErrorRecoveryCounter": 0,
            "LinkDownedCounter": 0, "PortRcvErrors": 3,
            "PortXmitDiscards": 7, "PortXmitWait": 19388}
    ports = {f"mlx5_{i}:1": dict(base) for i in (0, 4)}
    if phase == "end":
        for p in ports.values():
            p["PortXmitWait"] += 40_000
        ports["mlx5_4:1"]["SymbolErrorCounter"] += symbol_end
        ports["mlx5_4:1"]["PortXmitDiscards"] += xmit_disc_end
    return {"host": host, "timestamp": NOW, "phase": phase, "ports": ports}


def flight_recorder(rank: int, stuck: bool, frontier: int = 48211) -> dict:
    entries = [{"seq": s, "op": "ALLREDUCE", "state": "completed"}
               for s in range(frontier - 3, frontier)]
    if stuck:
        entries.append({"seq": frontier, "op": "ALLREDUCE", "state": "started"})
    else:
        entries.append({"seq": frontier, "op": "ALLREDUCE", "state": "completed"})
        entries.append({"seq": frontier + 1, "op": "ALLREDUCE", "state": "started"})
    return {"rank": rank, "entries": entries}


TOPO_MATRIX_OK = """\
        GPU0    GPU1    NIC0    NIC1    CPU Affinity    NUMA Affinity
GPU0     X      NV18    PIX     SYS     0-47            0
GPU1    NV18     X      SYS     PIX     48-95           1
NIC0    PIX     SYS      X      NODE
NIC1    SYS     PIX     NODE     X
"""

TOPO_MATRIX_BAD = TOPO_MATRIX_OK.replace(
    "GPU0     X      NV18    PIX     SYS", "GPU0     X      NV18    SYS     SYS")

NVLINK_OK = "GPU 0: NVIDIA H100\n" + "\n".join(
    f"\t Link {i}: 26.562 GB/s" for i in range(18)) + "\n"
NVLINK_DEGRADED = "GPU 0: NVIDIA H100\n" + "\n".join(
    f"\t Link {i}: 26.562 GB/s" for i in range(16)) + \
    "\n\t Link 16: <inactive>\n\t Link 17: <inactive>\n"

DMESG_CLEAN = "[Thu Jun 11 03:01:00 2026] systemd[1]: boot ok\n"
DMESG_XID79 = DMESG_CLEAN + (
    "[Thu Jun 11 03:13:58 2026] NVRM: Xid (PCI:0000:1b:00): 79, "
    "pid=9981, GPU has fallen off the bus.\n")


# ---------------------------------------------------------------- scenarios
def make_scenario_fabric(root: Path) -> Path:
    """Design-doc scenario: retry exhaustion + symbol errors + straggler."""
    job = root / "slurm-8841273"
    nodes = ["gpu-114", "gpu-115", "gpu-201", "gpu-202"]
    rank_to_host = {str(r): nodes[r // 8] for r in range(32)}
    _wj(job / "job.json", {
        "job_id": "slurm-8841273", "job_name": "llama-70b-pretrain",
        "cluster": "h100-prod-a", "nodes": nodes, "world_size": 32,
        "outcome": "WATCHDOG_TIMEOUT", "framework": "pytorch-2.7.0",
        "first_error_rank": 3, "fabric": "ib", "expected_nvlinks": 18,
        "timestamp": NOW, "rank_to_host": rank_to_host})
    _wj(job / "env.json", {"NCCL_DEBUG": "INFO", "NCCL_IB_TIMEOUT": "18"})

    for i, host in enumerate(nodes):
        pid = 9000 + i
        log = nccl_log_common(host, pid)
        if host == "gpu-114":
            log += (f"[E ProcessGroupNCCL.cpp:616] [Rank 3] Watchdog caught collective "
                    f"operation timeout: WorkNCCL(SeqNum=48211, OpType=ALLREDUCE, "
                    f"NumelIn=131072, Timeout(ms)=600000)\n")
        if host == "gpu-201":
            log += (f"gpu-201:{pid}:{pid+33} [2] NCCL WARN NET/IB : Got completion from "
                    f"peer 10.0.1.12<48133> with error 12, opcode 0, len 32768, vendor "
                    f"err 129 (Recv)\n")
        _w(job / "logs" / f"{host}.{pid}.log", log)
        _w(job / "topo" / f"{host}.xml", topo_xml(host))
        _w(job / "host" / host / "dmesg.txt", DMESG_CLEAN)
        _w(job / "host" / host / "nvidia-smi-topo.txt", TOPO_MATRIX_OK)
        _w(job / "host" / host / "nvlink.txt", NVLINK_OK)
        symbol = 1240 if host == "gpu-202" else 0
        disc = 312 if host == "gpu-201" else 0
        _wj(job / "counters" / f"{host}.start.json", counters(host, "start"))
        _wj(job / "counters" / f"{host}.end.json", counters(host, "end", symbol, disc))

    for r in range(32):
        _wj(job / "flight_recorder" / f"rank{r:02d}.json",
            flight_recorder(r, stuck=(16 <= r <= 23)))
    return job


def make_passing_sibling(root: Path) -> Path:
    """Earlier resubmission of the same job that PASSED on different nodes."""
    job = root / "slurm-8839912"
    nodes = ["gpu-114", "gpu-115", "gpu-117", "gpu-118"]  # avoided gpu-201/202
    _wj(job / "job.json", {
        "job_id": "slurm-8839912", "job_name": "llama-70b-pretrain",
        "cluster": "h100-prod-a", "nodes": nodes, "world_size": 32,
        "outcome": "SUCCESS", "framework": "pytorch-2.7.0", "fabric": "ib",
        "timestamp": NOW - 7200})
    for i, host in enumerate(nodes):
        _w(job / "logs" / f"{host}.{8000+i}.log", nccl_log_common(host, 8000 + i))
        _wj(job / "counters" / f"{host}.start.json", counters(host, "start"))
        _wj(job / "counters" / f"{host}.end.json", counters(host, "end"))
    return job


def make_scenario_config(root: Path) -> Path:
    """Socket fallback + missing GDR + degraded NVLink + Xid 79 + bad affinity."""
    job = root / "slurm-8851001"
    nodes = ["gpu-310", "gpu-311"]
    _wj(job / "job.json", {
        "job_id": "slurm-8851001", "job_name": "sdxl-finetune",
        "cluster": "h100-prod-a", "nodes": nodes, "world_size": 16,
        "outcome": "NCCL_ERROR", "framework": "pytorch-2.7.0",
        "fabric": "ib", "expected_nvlinks": 18, "timestamp": NOW})
    _w(job / "logs" / "gpu-310.7001.log",
       nccl_log_common("gpu-310", 7001, nchan=4, gdr=False, net="Socket"))
    _w(job / "logs" / "gpu-311.7002.log",
       nccl_log_common("gpu-311", 7002, nchan=16, gdr=False, net="IB"))
    _w(job / "topo" / "gpu-310.xml", topo_xml("gpu-310", nics=7))   # dead NIC
    _w(job / "topo" / "gpu-311.xml", topo_xml("gpu-311", nics=8))
    _w(job / "host" / "gpu-310" / "dmesg.txt", DMESG_CLEAN)
    _w(job / "host" / "gpu-311" / "dmesg.txt", DMESG_XID79)
    _w(job / "host" / "gpu-310" / "nvidia-smi-topo.txt", TOPO_MATRIX_BAD)
    _w(job / "host" / "gpu-311" / "nvidia-smi-topo.txt", TOPO_MATRIX_OK)
    _w(job / "host" / "gpu-310" / "nvlink.txt", NVLINK_DEGRADED)
    _w(job / "host" / "gpu-311" / "nvlink.txt", NVLINK_OK)
    return job


def main(root: Path = Path("demo_jobs")) -> None:
    if root.exists():
        shutil.rmtree(root)
    make_passing_sibling(root)
    make_scenario_fabric(root)
    make_scenario_config(root)
    print(f"fixtures written under {root}/")


if __name__ == "__main__":
    main()
