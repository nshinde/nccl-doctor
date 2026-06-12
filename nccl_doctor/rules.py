"""Diagnostic rules engine.

Each rule is a pure function `rule(ctx) -> list[Finding]` registered under the
rule IDs from the design document. Rules consume the parsed EvidenceBundle and
never touch the filesystem, which makes them trivially unit-testable and lets
platform teams add signatures without touching parsers.

Thresholds live in THRESHOLDS so operators can override via
/etc/nccl-doctor/thresholds.json without code changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .models import Evidence, Finding, JobMeta, Severity, Track
from .parsers.nccl_log import LogFacts
from .parsers.host import HostFacts, TopoSignature
from .parsers.counters import (
    CounterDelta, StragglerAnalysis, deltas_by, pause_storm_deltas,
    IB_PHYSICAL, IB_LOSS, IB_CONGESTION, ROCE_CNP, ROCE_DROP,
)

THRESHOLDS = {
    "symbol_error_high": 100,        # delta during job window
    "symbol_error_low": 5,
    "link_downed": 1,
    "xmit_wait_high": 1_000_000,     # ticks; tune per fabric
    "rcv_errors": 50,
    "pause_storm": 100_000,          # pause frames or duration units
    "cnp_low_under_pause": 10,       # ECN suspiciously quiet while PFC is loud
    "nvlink_replay": 1,
}


def load_threshold_overrides(path: Path = Path("/etc/nccl-doctor/thresholds.json")) -> None:
    try:
        THRESHOLDS.update(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError):
        pass


@dataclass
class EvidenceBundle:
    job: JobMeta
    logs: LogFacts
    hosts: HostFacts
    topo_sigs: dict[str, TopoSignature] = field(default_factory=dict)
    counter_deltas: list[CounterDelta] = field(default_factory=list)
    straggler: Optional[StragglerAnalysis] = None
    env: dict[str, str] = field(default_factory=dict)


Rule = Callable[[EvidenceBundle], list[Finding]]
REGISTRY: dict[str, Rule] = {}


def rule(rule_id: str) -> Callable[[Rule], Rule]:
    def deco(fn: Rule) -> Rule:
        REGISTRY[rule_id] = fn
        return fn
    return deco


def run_all(ctx: EvidenceBundle) -> list[Finding]:
    findings: list[Finding] = []
    for rid, fn in sorted(REGISTRY.items()):
        try:
            findings.extend(fn(ctx))
        except Exception as e:  # a broken rule must never kill the analysis
            findings.append(Finding(
                rule=rid, severity=Severity.INFO, track=Track.APPLICATION,
                summary=f"rule {rid} raised {type(e).__name__}: {e} (rule skipped)",
                confidence=0.0))
    findings.sort(key=lambda f: (-f.severity.rank, -f.confidence))
    return findings


# ===================================================================== NET
@rule("NET-01")
def net_retry_exhaustion(ctx: EvidenceBundle) -> list[Finding]:
    """IB transport retry exhaustion / RNR exhaustion from completion errors."""
    out: list[Finding] = []
    by_pair: dict[tuple[str, str, str], list] = {}
    for ce in ctx.logs.completion_errors:
        by_pair.setdefault((ce.host, ce.peer_ip or "?", ce.status), []).append(ce)
    for (host, peer, status), errs in by_pair.items():
        first = errs[0]
        sev = Severity.HIGH if status in ("12", "13") else Severity.MEDIUM
        components = [host]
        if peer != "?":
            components.append(f"peer:{peer}")
        out.append(Finding(
            rule="NET-01", severity=sev, track=Track.HARDWARE,
            summary=(f"IB completion error on {host} -> peer {peer}: "
                     f"{first.status_name} ({len(errs)} occurrence(s))"),
            hosts=[host], components=components, confidence=0.85,
            evidence=[Evidence(f"nccl_log:{e.file}:L{e.lineno}", e.line, host=host)
                      for e in errs[:3]],
            data={"status": status, "count": len(errs), "peer": peer}))
    return out


@rule("NET-02")
def net_link_flap(ctx: EvidenceBundle) -> list[Finding]:
    out = []
    for d in deltas_by(ctx.counter_deltas,
                       ("LinkDownedCounter", "LinkErrorRecoveryCounter"),
                       THRESHOLDS["link_downed"]):
        out.append(Finding(
            rule="NET-02", severity=Severity.HIGH, track=Track.HARDWARE,
            summary=f"Link flap on {d.host}/{d.port}: {d.counter} +{d.delta:.0f} during job",
            hosts=[d.host], components=[f"{d.host}/{d.port}"], confidence=0.9,
            evidence=[Evidence(f"counters:{d.host}/{d.port}",
                               f"{d.counter}: {d.start:.0f} -> {d.end:.0f}", host=d.host)]))
    return out


@rule("NET-03")
def net_physical_degradation(ctx: EvidenceBundle) -> list[Finding]:
    out = []
    for d in deltas_by(ctx.counter_deltas, ("SymbolErrorCounter",),
                       THRESHOLDS["symbol_error_low"]):
        high = d.delta >= THRESHOLDS["symbol_error_high"]
        out.append(Finding(
            rule="NET-03",
            severity=Severity.HIGH if high else Severity.LOW,
            track=Track.HARDWARE,
            summary=(f"Symbol errors on {d.host}/{d.port}: +{d.delta:.0f} during job "
                     f"({'above' if high else 'below'} storm threshold "
                     f"{THRESHOLDS['symbol_error_high']}) — marginal cable/transceiver"),
            hosts=[d.host], components=[f"{d.host}/{d.port}"],
            confidence=0.9 if high else 0.5,
            evidence=[Evidence(f"counters:{d.host}/{d.port}",
                               f"SymbolErrorCounter: {d.start:.0f} -> {d.end:.0f}",
                               host=d.host)]))
    out.extend(Finding(
        rule="NET-03", severity=Severity.MEDIUM, track=Track.HARDWARE,
        summary=f"Receive/discard errors on {d.host}/{d.port}: {d.counter} +{d.delta:.0f}",
        hosts=[d.host], components=[f"{d.host}/{d.port}"], confidence=0.6,
        evidence=[Evidence(f"counters:{d.host}/{d.port}",
                           f"{d.counter}: {d.start:.0f} -> {d.end:.0f}", host=d.host)])
        for d in deltas_by(ctx.counter_deltas, IB_LOSS, THRESHOLDS["rcv_errors"]))
    return out


@rule("NET-04")
def net_congestion(ctx: EvidenceBundle) -> list[Finding]:
    return [Finding(
        rule="NET-04", severity=Severity.MEDIUM, track=Track.INFRASTRUCTURE,
        summary=(f"Fabric congestion at {d.host}/{d.port}: PortXmitWait "
                 f"+{d.delta:.3g} during job window"),
        hosts=[d.host], components=[f"{d.host}/{d.port}"], confidence=0.6,
        evidence=[Evidence(f"counters:{d.host}/{d.port}",
                           f"PortXmitWait: {d.start:.0f} -> {d.end:.0f}", host=d.host)])
        for d in deltas_by(ctx.counter_deltas, IB_CONGESTION,
                           THRESHOLDS["xmit_wait_high"])]


@rule("NET-05")
def roce_pfc_storm(ctx: EvidenceBundle) -> list[Finding]:
    if ctx.job.fabric != "roce":
        return []
    storms = pause_storm_deltas(ctx.counter_deltas, THRESHOLDS["pause_storm"])
    return [Finding(
        rule="NET-05", severity=Severity.HIGH, track=Track.INFRASTRUCTURE,
        summary=(f"PFC pause storm on {d.host}/{d.port}: {d.counter} +{d.delta:.3g} — "
                 f"head-of-line blocking; verify lossless class config"),
        hosts=[d.host], components=[f"{d.host}/{d.port}"], confidence=0.8,
        evidence=[Evidence(f"counters:{d.host}/{d.port}",
                           f"{d.counter}: {d.start:.0f} -> {d.end:.0f}", host=d.host)])
        for d in storms]


@rule("NET-06")
def roce_ecn_misconfig(ctx: EvidenceBundle) -> list[Finding]:
    if ctx.job.fabric != "roce":
        return []
    storms = pause_storm_deltas(ctx.counter_deltas, THRESHOLDS["pause_storm"])
    if not storms:
        return []
    cnp = sum(d.delta for d in deltas_by(ctx.counter_deltas, ROCE_CNP, 0.0))
    if cnp <= THRESHOLDS["cnp_low_under_pause"]:
        hosts = sorted({d.host for d in storms})
        return [Finding(
            rule="NET-06", severity=Severity.HIGH, track=Track.INFRASTRUCTURE,
            summary=("ECN/DCQCN appears inactive (CNP delta ~0) while PFC pause counters "
                     "explode — congestion is being handled by PFC alone. Verify switch "
                     "ECN marking thresholds and NIC DCQCN/trust-DSCP config."),
            hosts=hosts, components=hosts, confidence=0.75,
            evidence=[Evidence("counters:aggregate",
                               f"sum(np_cnp_sent,rp_cnp_handled) delta = {cnp:.0f} "
                               f"with {len(storms)} pause-storm port(s)")])]
    return []


# ==================================================================== TOPO
@rule("TOPO-01")
def topo_gpu_nic_affinity(ctx: EvidenceBundle) -> list[Finding]:
    out = []
    for host, tm in ctx.hosts.topo_matrices.items():
        for gpu, why in tm.gpus_without_local_nic():
            out.append(Finding(
                rule="TOPO-01", severity=Severity.MEDIUM, track=Track.INFRASTRUCTURE,
                summary=f"{host}/{gpu}: no PIX/PXB path to any NIC ({why}) — "
                        f"NIC traffic crosses the CPU complex",
                hosts=[host], components=[f"{host}/{gpu}"], confidence=0.7,
                evidence=[Evidence(f"nvidia-smi-topo:{host}", why, host=host)]))
    return out


@rule("TOPO-02")
def topo_gdr_disabled(ctx: EvidenceBundle) -> list[Finding]:
    out = []
    for host in sorted(ctx.logs.hosts()):
        nogdr = ctx.logs.net_paths_nogdr.get(host, 0)
        gdr = ctx.logs.net_paths_gdr.get(host, 0)
        if nogdr > 0 and gdr == 0:
            out.append(Finding(
                rule="TOPO-02", severity=Severity.MEDIUM, track=Track.INFRASTRUCTURE,
                summary=(f"GPUDirect RDMA not in use on {host}: {nogdr} NET path(s) "
                         f"staged through host memory. Check ACS on PCIe bridges, "
                         f"IOMMU passthrough, nvidia-peermem/DMABUF."),
                hosts=[host], components=[host], confidence=0.75,
                evidence=[Evidence(f"nccl_log:{host}",
                                   f"net paths via NET/* without GDRDMA: {nogdr}, "
                                   f"with GDRDMA: {gdr}", host=host)]))
    return out


@rule("TOPO-03")
def topo_nvlink_degraded(ctx: EvidenceBundle) -> list[Finding]:
    out = []
    expected = ctx.job.expected_nvlinks
    for host, gpus in ctx.hosts.nvlink.items():
        for gpu, (active, total) in gpus.items():
            exp = expected or total
            if active < exp:
                out.append(Finding(
                    rule="TOPO-03", severity=Severity.HIGH, track=Track.HARDWARE,
                    summary=f"{host}/{gpu}: {active}/{exp} NVLinks active",
                    hosts=[host], components=[f"{host}/{gpu}"], confidence=0.85,
                    evidence=[Evidence(f"nvlink:{host}",
                                       f"{gpu} active={active} expected={exp}",
                                       host=host)]))
    return out


@rule("TOPO-04")
def topo_pcie_downtrain(ctx: EvidenceBundle) -> list[Finding]:
    out = []
    for host, dts in ctx.hosts.downtrains.items():
        for dt in dts:
            out.append(Finding(
                rule="TOPO-04", severity=Severity.MEDIUM, track=Track.HARDWARE,
                summary=f"PCIe downtraining on {host}: {dt.device} negotiated "
                        f"{dt.sta} (capable {dt.cap})",
                hosts=[host], components=[f"{host}/{dt.device.split()[0]}"],
                confidence=0.8,
                evidence=[Evidence(f"lspci:{host}",
                                   f"{dt.device}: LnkSta {dt.sta} vs LnkCap {dt.cap}",
                                   host=host)]))
    return out


@rule("TOPO-05")
def topo_asymmetric_detection(ctx: EvidenceBundle) -> list[Finding]:
    if len(ctx.topo_sigs) < 2:
        return []
    keys: dict[tuple, list[str]] = {}
    for host, sig in ctx.topo_sigs.items():
        keys.setdefault(sig.key(), []).append(host)
    if len(keys) <= 1:
        return []
    # majority = largest group; on ties, the richer topology (more GPUs/NICs)
    # is the healthy reference — a dead device removes hardware, never adds it
    majority_key = max(keys, key=lambda k: (len(keys[k]), k[0], k[1]))
    outliers = [h for k, hs in keys.items() if k != majority_key for h in hs]
    maj = ctx.topo_sigs[keys[majority_key][0]]
    ev = []
    for h in outliers:
        s = ctx.topo_sigs[h]
        ev.append(Evidence(f"topo_xml:{h}",
                           f"{h}: gpus={s.gpu_count} nics={s.nic_count} "
                           f"vs majority gpus={maj.gpu_count} nics={maj.nic_count}",
                           host=h))
    return [Finding(
        rule="TOPO-05", severity=Severity.HIGH, track=Track.HARDWARE,
        summary=(f"Asymmetric topology detection: {sorted(outliers)} differ from the "
                 f"other {len(ctx.topo_sigs) - len(outliers)} node(s) "
                 f"(likely dead NIC / PCIe device) — every ring is dragged through "
                 f"the narrow node"),
        hosts=sorted(outliers), components=sorted(outliers),
        confidence=0.85, evidence=ev)]


# ==================================================================== ALGO
@rule("ALGO-01")
def algo_socket_fallback(ctx: EvidenceBundle) -> list[Finding]:
    out = []
    for host, nets in sorted(ctx.logs.networks.items()):
        if "Socket" in nets and "IB" not in nets:
            out.append(Finding(
                rule="ALGO-01", severity=Severity.HIGH, track=Track.NCCL_CONFIG,
                summary=(f"{host} fell back to NET/Socket on an RDMA cluster — IB plugin "
                         f"failed to load, NCCL_IB_HCA filtered everything, or device "
                         f"permissions broke. 10-50x slowdown that presents as a hang."),
                hosts=[host], components=[host], confidence=0.85,
                evidence=[Evidence(f"nccl_log:{host}", "Using network Socket", host=host)]))
    return out


@rule("ALGO-02")
def algo_ll128_implicated(ctx: EvidenceBundle) -> list[Finding]:
    if "LL128" not in ctx.logs.protocols_seen:
        return []
    if not (ctx.logs.watchdog_events or ctx.logs.completion_errors):
        return []
    return [Finding(
        rule="ALGO-02", severity=Severity.LOW, track=Track.NCCL_CONFIG,
        summary=("LL128 protocol active during a run that timed out. If failures "
                 "recur only with LL128 collectives, exclude it for this job class "
                 "(NCCL_PROTO=^LL128) as a bisection step."),
        confidence=0.35,
        evidence=[Evidence("nccl_log:tuning",
                           f"protocols seen: {sorted(ctx.logs.protocols_seen)}")])]


@rule("ALGO-03")
def algo_tree_construction_failed(ctx: EvidenceBundle) -> list[Finding]:
    hosts = ctx.logs.hosts()
    if not hosts:
        return []
    rings = {h for h in hosts if ctx.logs.rings_connected.get(h)}
    trees = {h for h in hosts if ctx.logs.trees_connected.get(h)
             or ctx.logs.trees_built.get(h)}
    if rings and not trees and len(hosts) > 1:
        return [Finding(
            rule="ALGO-03", severity=Severity.LOW, track=Track.NCCL_CONFIG,
            summary=("Rings connected but no tree construction observed on a multi-node "
                     "job — placement-sensitive graph search fallback. Compare graph "
                     "dumps across runs; consider pinning NCCL_TOPO_FILE."),
            confidence=0.4,
            evidence=[Evidence("nccl_log:graph",
                               f"rings on {len(rings)} host(s), trees on 0")])]
    return []


@rule("ALGO-04")
def algo_channel_anomaly(ctx: EvidenceBundle) -> list[Finding]:
    totals = ctx.logs.channel_totals
    if len(totals) < 2:
        return []
    values = sorted(set(totals.values()))
    if len(values) > 1:
        lo = min(totals.items(), key=lambda kv: kv[1])
        return [Finding(
            rule="ALGO-04", severity=Severity.MEDIUM, track=Track.NCCL_CONFIG,
            summary=(f"Channel-count mismatch across hosts: {dict(sorted(totals.items()))} "
                     f"— {lo[0]} built only {lo[1]} channels (path rejected during graph "
                     f"search). Fix the underlying path before forcing NCCL_MIN_NCHANNELS."),
            hosts=[lo[0]], components=[lo[0]], confidence=0.65,
            evidence=[Evidence("nccl_log:init", f"nChannels per host: {totals}")])]
    return []


# ==================================================================== HOST/XID
@rule("GPU-01")
def gpu_xid(ctx: EvidenceBundle) -> list[Finding]:
    out = []
    for host, events in ctx.hosts.xids.items():
        crit = [e for e in events if e.critical]
        for e in crit[:5]:
            out.append(Finding(
                rule="GPU-01", severity=Severity.CRITICAL, track=Track.HARDWARE,
                summary=f"Xid {e.xid} on {host} ({e.pci}): {e.meaning}",
                hosts=[host], components=[f"{host}/{e.pci}"], confidence=0.95,
                evidence=[Evidence(f"dmesg:{host}", f"Xid {e.xid}: {e.detail}",
                                   host=host)]))
    return out


# ==================================================================== STRAG
@rule("STRAG-01")
def straggler(ctx: EvidenceBundle) -> list[Finding]:
    sa = ctx.straggler
    if not sa or not sa.conclusive:
        return []
    hosts = sorted({h for h in (ctx.job.host_of_rank(r) for r in sa.stuck_ranks) if h})
    reporter = ctx.job.first_error_rank
    note = ""
    if reporter is not None and reporter not in sa.stuck_ranks:
        note = (f" Rank {reporter} raised the timeout but was NOT stuck — "
                f"it was the messenger, not the cause.")
    return [Finding(
        rule="STRAG-01", severity=Severity.HIGH, track=Track.SCHEDULER,
        summary=(f"Flight recorder: ranks {sa.stuck_ranks[:8]}"
                 f"{'...' if len(sa.stuck_ranks) > 8 else ''} desynced at "
                 f"seq~{sa.frontier_seq} ({sa.note}).{note}"),
        hosts=hosts, components=hosts, confidence=0.9,
        evidence=[Evidence("flight_recorder", sa.note or "")],
        data={"stuck_ranks": sa.stuck_ranks, "frontier_seq": sa.frontier_seq,
              "true_origin_hosts": hosts})]
