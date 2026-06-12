"""Parser for NCCL debug logs (NCCL_DEBUG=INFO, SUBSYS=INIT,ENV,GRAPH,NET,TUNING).

Tolerant by design: NCCL log formats drift between minor versions, so every
extraction is regex-based, line-oriented, and failure of one pattern never
aborts the parse. Facts we could not extract stay None/empty and downstream
rules simply don't fire.

Typical lines handled (formats observed across NCCL 2.18–2.27):

  gpu-114:12345:12361 [0] NCCL INFO NCCL version 2.26.2+cuda12.4
  gpu-114:12345:12361 [0] NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/RoCE ; OOB eth0:10.0.0.1<0>
  gpu-114:12345:12361 [0] NCCL INFO Using network IB
  gpu-114:12345:12361 [0] NCCL INFO Channel 00/16 :  0  1  2  3
  gpu-114:12345:12361 [0] NCCL INFO Channel 00/0 : 4[4] -> 0[0] [receive] via NET/IB/0/GDRDMA
  gpu-114:12345:12361 [0] NCCL INFO Channel 02/0 : 9[1] -> 1[1] [send] via NET/IB/1
  gpu-201:9981:10021  [2] NCCL WARN NET/IB : Got completion from peer 10.0.1.12<48133> with error 12, opcode 0, len 32768, vendor err 129 (Recv)
  [E ProcessGroupNCCL.cpp:616] [Rank 3] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=48211, OpType=ALLREDUCE, ...)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------- regexes
RE_VERSION = re.compile(r"NCCL version (\d+\.\d+(?:\.\d+)?)")
RE_NET_USING = re.compile(r"Using network (\w+)")
RE_IB_DEVS = re.compile(r"NET/IB\s*:\s*Using\s+(.+?)(?:;|$)")
RE_IB_DEV_ITEM = re.compile(r"\[\d+\](\w+):\d+/(\w+)")
RE_CHANNEL_TOTAL = re.compile(r"Channel (\d+)/(\d+)\s*:")
RE_NET_PATH = re.compile(
    r"Channel \d+/\d+ : \d+\[\d+\] -> \d+\[\d+\]\s*(?:\[(send|receive)\]\s*)?"
    r"via (NET/\w+/\d+(?:/\w+)*|P2P/\w+(?:/\w+)*|SHM(?:/\w+)*)")
RE_COMPLETION_ERR = re.compile(
    r"NET/IB\s*:?\s*Got completion (?:from peer ([\d\.]+)<\d+> )?with "
    r"(?:error|status[= ])\s*(\d+)(?:.*vendor err (\d+))?", re.IGNORECASE)
RE_WATCHDOG_RANK = re.compile(r"\[Rank (\d+)\]")
RE_WATCHDOG_SEQ = re.compile(r"SeqNum=(\d+)")
RE_WATCHDOG_OP = re.compile(r"OpType=(\w+)")
RE_ASYNC_ERR = re.compile(r"NCCL WARN (.+)")
RE_TREES = re.compile(r"NCCL INFO Trees \[\d+\]")
RE_RINGS_OK = re.compile(r"Connected all rings")
RE_TREES_OK = re.compile(r"Connected all trees")
RE_ALGOPROTO = re.compile(r"algorithm[ =:]+(\w+).*?protocol[ =:]+(\w+)", re.IGNORECASE)
RE_NVLS = re.compile(r"\bNVLS\b")
RE_LOG_PREFIX = re.compile(r"^([\w\-.]+):(\d+):(\d+)\s*\[(\d+)\]")

IBV_WC_STATUS = {
    "4": "IBV_WC_LOC_PROT_ERR (local protection error)",
    "5": "IBV_WC_WR_FLUSH_ERR (WQ flushed — usually secondary to another failure)",
    "10": "IBV_WC_REM_ACCESS_ERR (remote access error)",
    "11": "IBV_WC_REM_OP_ERR (remote operation error)",
    "12": "IBV_WC_RETRY_EXC_ERR (transport retry counter exceeded — packet loss "
          "or unreachable peer on the fabric path)",
    "13": "IBV_WC_RNR_RETRY_EXC_ERR (receiver-not-ready retries exhausted — "
          "remote side stalled, often a stuck/slow peer process)",
}


@dataclass
class CompletionError:
    host: str
    file: str
    lineno: int
    peer_ip: Optional[str]
    status: str
    vendor_err: Optional[str]
    line: str

    @property
    def status_name(self) -> str:
        return IBV_WC_STATUS.get(self.status, f"ibv_wc_status={self.status}")


@dataclass
class WatchdogEvent:
    rank: int
    seq: Optional[int]
    op: Optional[str]
    file: str
    lineno: int
    line: str


@dataclass
class LogFacts:
    """Aggregated facts extracted from all per-rank NCCL logs of one job."""
    nccl_versions: set[str] = field(default_factory=set)
    networks: dict[str, set[str]] = field(default_factory=dict)   # host -> {"IB","Socket"}
    hcas: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # host -> [(dev, linklayer)]
    channel_totals: dict[str, int] = field(default_factory=dict)  # host -> nChannels
    net_paths_gdr: dict[str, int] = field(default_factory=dict)   # host -> count via NET/*/GDRDMA
    net_paths_nogdr: dict[str, int] = field(default_factory=dict) # host -> count via NET/* w/o GDRDMA
    completion_errors: list[CompletionError] = field(default_factory=list)
    watchdog_events: list[WatchdogEvent] = field(default_factory=list)
    warns: list[tuple[str, str, int, str]] = field(default_factory=list)  # (host,file,lineno,msg)
    trees_built: dict[str, bool] = field(default_factory=dict)
    rings_connected: dict[str, bool] = field(default_factory=dict)
    trees_connected: dict[str, bool] = field(default_factory=dict)
    protocols_seen: set[str] = field(default_factory=set)
    algorithms_seen: set[str] = field(default_factory=set)
    nvls_seen: bool = False
    parsed_files: int = 0

    def hosts(self) -> set[str]:
        out: set[str] = set()
        for d in (self.networks, self.channel_totals, self.net_paths_gdr,
                  self.net_paths_nogdr, self.hcas):
            out.update(d.keys())
        return out


def _host_from_line_or_name(line: str, fallback: str) -> str:
    m = RE_LOG_PREFIX.match(line)
    return m.group(1) if m else fallback


def parse_log_file(path: Path, facts: LogFacts) -> None:
    fallback_host = path.name.split(".")[0]
    text = path.read_text(errors="replace")
    facts.parsed_files += 1
    for lineno, line in enumerate(text.splitlines(), 1):
        host = _host_from_line_or_name(line, fallback_host)

        if (m := RE_VERSION.search(line)):
            facts.nccl_versions.add(m.group(1))

        if (m := RE_NET_USING.search(line)):
            facts.networks.setdefault(host, set()).add(m.group(1))

        if (m := RE_IB_DEVS.search(line)):
            devs = RE_IB_DEV_ITEM.findall(m.group(1))
            if devs:
                facts.hcas.setdefault(host, [])
                for dev in devs:
                    if dev not in facts.hcas[host]:
                        facts.hcas[host].append(dev)

        if (m := RE_CHANNEL_TOTAL.search(line)):
            total = int(m.group(2))
            if total > 0:
                facts.channel_totals[host] = max(facts.channel_totals.get(host, 0), total)

        if (m := RE_NET_PATH.search(line)):
            via = m.group(2)
            if via.startswith("NET/"):
                if "GDRDMA" in via:
                    facts.net_paths_gdr[host] = facts.net_paths_gdr.get(host, 0) + 1
                else:
                    facts.net_paths_nogdr[host] = facts.net_paths_nogdr.get(host, 0) + 1

        if (m := RE_COMPLETION_ERR.search(line)):
            facts.completion_errors.append(CompletionError(
                host=host, file=path.name, lineno=lineno,
                peer_ip=m.group(1), status=m.group(2),
                vendor_err=m.group(3), line=line.strip()))

        if "Watchdog caught" in line or "Timeout at NCCL work" in line:
            mr = RE_WATCHDOG_RANK.search(line)
            if mr:
                ms = RE_WATCHDOG_SEQ.search(line)
                mo = RE_WATCHDOG_OP.search(line)
                facts.watchdog_events.append(WatchdogEvent(
                    rank=int(mr.group(1)),
                    seq=int(ms.group(1)) if ms else None,
                    op=mo.group(1) if mo else None,
                    file=path.name, lineno=lineno, line=line.strip()))

        if "NCCL WARN" in line:
            m = RE_ASYNC_ERR.search(line)
            if m:
                facts.warns.append((host, path.name, lineno, m.group(1).strip()))

        if RE_TREES.search(line):
            facts.trees_built[host] = True
        if RE_RINGS_OK.search(line):
            facts.rings_connected[host] = True
        if RE_TREES_OK.search(line):
            facts.trees_connected[host] = True

        if (m := RE_ALGOPROTO.search(line)):
            facts.algorithms_seen.add(m.group(1).upper())
            facts.protocols_seen.add(m.group(2).upper())
        if RE_NVLS.search(line):
            facts.nvls_seen = True


def parse_logs(log_dir: Path) -> LogFacts:
    facts = LogFacts()
    if not log_dir.is_dir():
        return facts
    for p in sorted(log_dir.glob("*.log")):
        try:
            parse_log_file(p, facts)
        except OSError:
            continue
    return facts
