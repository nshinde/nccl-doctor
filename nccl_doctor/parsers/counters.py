"""Fabric counter snapshots/deltas and PyTorch NCCL flight-recorder analysis.

Counter snapshot format (written by `nccl-doctor snapshot`, one per host):

  counters/<host>.start.json
  counters/<host>.end.json
  {"host": "gpu-201", "timestamp": 1718080000.0,
   "ports": {"mlx5_4": {"SymbolErrorCounter": 12, "PortXmitWait": 19388, ...}}}

Flight recorder format consumed here (one JSON per rank, exported from the
torch dump via `tools/fr_export.py` or written directly by the wrapper):

  flight_recorder/rank<NN>.json
  {"rank": 17, "entries": [
      {"seq": 48210, "op": "ALLREDUCE", "state": "completed"},
      {"seq": 48211, "op": "ALLREDUCE", "state": "started"}]}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------- counters
# IB counters: physical-layer health → HARDWARE track when nonzero/rising.
IB_PHYSICAL = ("SymbolErrorCounter", "LinkErrorRecoveryCounter", "LinkDownedCounter")
IB_LOSS = ("PortRcvErrors", "PortXmitDiscards", "ExcessiveBufferOverrunErrors")
IB_CONGESTION = ("PortXmitWait",)
# RoCE counters from ethtool -S (mlx5)
ROCE_PAUSE_PREFIXES = ("rx_prio", "tx_prio")           # *_pause / *_pause_duration
ROCE_CNP = ("np_cnp_sent", "rp_cnp_handled")
ROCE_DROP = ("out_of_buffer", "rx_discards_phy", "tx_discards_phy")


@dataclass
class CounterDelta:
    host: str
    port: str
    counter: str
    start: float
    end: float

    @property
    def delta(self) -> float:
        return self.end - self.start


def load_counter_deltas(counters_dir: Path) -> list[CounterDelta]:
    deltas: list[CounterDelta] = []
    if not counters_dir.is_dir():
        return deltas
    starts = {p.name[:-len(".start.json")]: p for p in counters_dir.glob("*.start.json")}
    for host, sp in sorted(starts.items()):
        ep = counters_dir / f"{host}.end.json"
        if not ep.exists():
            continue
        try:
            s = json.loads(sp.read_text())
            e = json.loads(ep.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for port, sc in s.get("ports", {}).items():
            ec = e.get("ports", {}).get(port, {})
            for counter, sval in sc.items():
                if counter in ec:
                    try:
                        deltas.append(CounterDelta(host, port, counter,
                                                   float(sval), float(ec[counter])))
                    except (TypeError, ValueError):
                        continue
    return deltas


def deltas_by(deltas: list[CounterDelta], counters: tuple[str, ...],
              min_delta: float = 1.0) -> list[CounterDelta]:
    return [d for d in deltas if d.counter in counters and d.delta >= min_delta]


def pause_storm_deltas(deltas: list[CounterDelta], min_delta: float) -> list[CounterDelta]:
    out = []
    for d in deltas:
        if d.counter.startswith(ROCE_PAUSE_PREFIXES) and "pause" in d.counter \
                and d.delta >= min_delta:
            out.append(d)
    return out


# ---------------------------------------------------------- flight recorder
@dataclass
class RankProgress:
    rank: int
    last_completed_seq: int = -1
    last_started_seq: int = -1
    last_op: Optional[str] = None


@dataclass
class StragglerAnalysis:
    per_rank: dict[int, RankProgress] = field(default_factory=dict)
    max_completed: int = -1
    stuck_ranks: list[int] = field(default_factory=list)      # behind the herd
    frontier_seq: Optional[int] = None                         # collective that desynced
    note: Optional[str] = None

    @property
    def conclusive(self) -> bool:
        return bool(self.stuck_ranks) and len(self.stuck_ranks) < len(self.per_rank)


def analyze_flight_recorder(fr_dir: Path) -> Optional[StragglerAnalysis]:
    if not fr_dir.is_dir():
        return None
    sa = StragglerAnalysis()
    for p in sorted(fr_dir.glob("rank*.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        rp = RankProgress(rank=int(d.get("rank", -1)))
        for e in d.get("entries", []):
            seq = int(e.get("seq", -1))
            state = e.get("state", "")
            if state == "completed" and seq > rp.last_completed_seq:
                rp.last_completed_seq = seq
                rp.last_op = e.get("op", rp.last_op)
            elif state in ("started", "scheduled") and seq > rp.last_started_seq:
                rp.last_started_seq = seq
                rp.last_op = e.get("op", rp.last_op)
        if rp.rank >= 0:
            sa.per_rank[rp.rank] = rp
    if not sa.per_rank:
        return None
    sa.max_completed = max(rp.last_completed_seq for rp in sa.per_rank.values())
    sa.stuck_ranks = sorted(r for r, rp in sa.per_rank.items()
                            if rp.last_completed_seq < sa.max_completed)
    if sa.stuck_ranks:
        sa.frontier_seq = min(sa.per_rank[r].last_completed_seq
                              for r in sa.stuck_ranks) + 1
        ops = {sa.per_rank[r].last_op for r in sa.stuck_ranks if sa.per_rank[r].last_op}
        sa.note = (f"{len(sa.stuck_ranks)}/{len(sa.per_rank)} ranks stuck at "
                   f"seq<{sa.max_completed} (desync at seq~{sa.frontier_seq}, "
                   f"op={','.join(sorted(ops)) or 'unknown'})")
    return sa
