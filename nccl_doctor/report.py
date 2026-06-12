"""Analysis pipeline and report rendering.

`analyze_job(job_dir)` is the single entry point: it loads every artifact the
collectors produced, runs the rules engine, synthesizes a verdict, consults
the cross-run store, and returns the report dict (design doc §4.3 schema).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .models import Finding, JobMeta, Severity, Track, Verdict
from .parsers import nccl_log
from .parsers.host import parse_host_dir, parse_all_topo
from .parsers.counters import load_counter_deltas, analyze_flight_recorder
from .rules import EvidenceBundle, run_all, load_threshold_overrides
from .recommend import recommend
from .store import Store

SCHEMA_VERSION = "1.0"

# verdict precedence: first matching wins (hardware physics > fabric > config)
VERDICT_MAP: list[tuple[set, str]] = [
    ({"GPU-01"}, "GPU_HARDWARE_FAULT"),
    ({"TOPO-03", "TOPO-04", "TOPO-05"}, "NODE_HARDWARE_DEGRADATION"),
    ({"NET-02", "NET-03"}, "FABRIC_LINK_DEGRADATION"),
    ({"NET-01"}, "TRANSIENT_FABRIC_LOSS"),
    ({"NET-05", "NET-06"}, "ROCE_QOS_MISCONFIGURATION"),
    ({"NET-04"}, "FABRIC_CONGESTION"),
    ({"ALGO-01"}, "MISCONFIGURED_TRANSPORT"),
    ({"STRAG-01"}, "STRAGGLER_NODE"),
    ({"TOPO-01", "TOPO-02", "ALGO-04"}, "SUBOPTIMAL_TOPOLOGY_CONFIG"),
    ({"ALGO-02", "ALGO-03"}, "NCCL_TUNING_SUSPECT"),
]


def synthesize_verdict(findings: list[Finding], job: JobMeta,
                       straggler_data: Optional[dict]) -> Verdict:
    fired = {f.rule for f in findings if f.severity.rank >= Severity.MEDIUM.rank}
    fired_any = {f.rule for f in findings}
    verdict_name = "INCONCLUSIVE"
    for ruleset, name in VERDICT_MAP:
        if ruleset & fired:
            verdict_name = name
            break
    else:
        for ruleset, name in VERDICT_MAP:
            if ruleset & fired_any:
                verdict_name = name + "_SUSPECTED"
                break
    if job.outcome == "SUCCESS":
        verdict_name = "HEALTHY" if verdict_name == "INCONCLUSIVE" else \
            f"PASSED_WITH_LATENT_ISSUES({verdict_name})"

    relevant = [f for f in findings if f.severity.rank >= Severity.MEDIUM.rank]
    if relevant:
        top = max(f.confidence for f in relevant)
        corroboration = min(0.12 * (len({f.rule for f in relevant}) - 1), 0.12)
        confidence = min(0.97, top + corroboration)
    else:
        confidence = 0.3 if findings else 0.2

    origin: dict = {}
    if straggler_data:
        origin = {
            "true_origin_ranks": straggler_data.get("stuck_ranks", [])[:8],
            "origin_hosts": straggler_data.get("true_origin_hosts", []),
            "desync_seq": straggler_data.get("frontier_seq"),
        }
        if job.first_error_rank is not None and \
                job.first_error_rank not in straggler_data.get("stuck_ranks", []):
            origin["note"] = (f"Rank {job.first_error_rank} reported the timeout but "
                              f"was not stuck — true origin is "
                              f"{origin['origin_hosts'] or origin['true_origin_ranks']}.")
    elif job.first_error_rank is not None:
        origin = {"reporting_rank": job.first_error_rank,
                  "note": "No flight recorder available; reporting rank may not be "
                          "the origin."}
    return Verdict(verdict_name, confidence, origin)


def analyze_job(job_dir: Path, store: Optional[Store] = None) -> dict:
    load_threshold_overrides()
    job_dir = Path(job_dir)
    meta_path = job_dir / "job.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing — is this a nccl-doctor job dir?")
    job = JobMeta.from_dict(json.loads(meta_path.read_text()))

    env: dict[str, str] = {}
    if (p := job_dir / "env.json").exists():
        try:
            env = json.loads(p.read_text())
        except json.JSONDecodeError:
            pass

    logs = nccl_log.parse_logs(job_dir / "logs")
    if logs.nccl_versions and not job.nccl_version:
        job.nccl_version = sorted(logs.nccl_versions)[-1]

    straggler = analyze_flight_recorder(job_dir / "flight_recorder")

    ctx = EvidenceBundle(
        job=job, logs=logs,
        hosts=parse_host_dir(job_dir / "host"),
        topo_sigs=parse_all_topo(job_dir / "topo"),
        counter_deltas=load_counter_deltas(job_dir / "counters"),
        straggler=straggler, env=env)

    findings = run_all(ctx)

    strag_data = next((f.data for f in findings if f.rule == "STRAG-01"), None)
    verdict = synthesize_verdict(findings, job, strag_data)

    node_scores: dict[str, float] = {}
    fleet_median = 0.0
    retry_expl = None
    if store is not None:
        store.record_run(job, verdict.verdict, findings)
        node_scores = store.node_scores()
        fleet_median = store.fleet_median(job.nodes)
        implicated = {c.split("/")[0] for f in findings
                      if f.severity.rank >= Severity.MEDIUM.rank
                      for c in (f.components or f.hosts)
                      if not c.startswith("peer:")}
        retry_expl = store.retry_explanation(job, implicated)

    actions = recommend(findings, {n: node_scores.get(n, 0.0) for n in job.nodes},
                        fleet_median)

    topo_diag = _topology_diagnosis(ctx, findings)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "job": {
            "job_id": job.job_id, "job_name": job.job_name, "cluster": job.cluster,
            "nodes": job.nodes, "world_size": job.world_size,
            "nccl_version": job.nccl_version, "framework": job.framework,
            "fabric": job.fabric, "outcome": job.outcome,
            "first_error_rank": job.first_error_rank,
        },
        "error_analysis": {
            **verdict.to_dict(),
            "findings": [f.to_dict() for f in findings
                         if f.severity != Severity.INFO or f.confidence > 0],
        },
        "topology_diagnosis": topo_diag,
        "recommended_actions": [a.to_dict() for a in actions],
        "node_reliability": {
            "scores": {n: round(node_scores.get(n, 0.0), 3) for n in job.nodes},
            "fleet_median": round(fleet_median, 3),
        },
        "parser_coverage": {
            "nccl_logs_parsed": logs.parsed_files,
            "topo_xmls": len(ctx.topo_sigs),
            "hosts_with_artifacts": len(set(ctx.hosts.xids) | set(ctx.hosts.nvlink)
                                        | set(ctx.hosts.topo_matrices)
                                        | set(ctx.hosts.downtrains)),
            "counter_deltas": len(ctx.counter_deltas),
            "flight_recorder_ranks": len(straggler.per_rank) if straggler else 0,
        },
    }
    if retry_expl:
        report["retry_explanation"] = retry_expl
    return report


def _topology_diagnosis(ctx: EvidenceBundle, findings: list[Finding]) -> dict:
    hosts = sorted(ctx.logs.hosts() | set(ctx.hosts.nvlink))
    gdr_hosts = [h for h in hosts if ctx.logs.net_paths_gdr.get(h, 0) > 0]
    nogdr_only = [h for h in hosts if ctx.logs.net_paths_nogdr.get(h, 0) > 0
                  and ctx.logs.net_paths_gdr.get(h, 0) == 0]
    asym = [f.summary for f in findings if f.rule == "TOPO-05"]
    nvlink_issues = [f.summary for f in findings if f.rule == "TOPO-03"]
    diag = {
        "asymmetries": asym,
        "gdr_enabled": bool(gdr_hosts) and not nogdr_only,
        "gdr_missing_on": nogdr_only,
        "nvlink_status": nvlink_issues or "no degraded NVLinks observed",
        "channels_per_host": ctx.logs.channel_totals,
        "networks_per_host": {h: sorted(v) for h, v in ctx.logs.networks.items()},
    }
    if not any([asym, nvlink_issues, nogdr_only]):
        diag["note"] = "Topology healthy; failure (if any) localizes elsewhere."
    return diag


# ------------------------------------------------------------- text render
def render_text(report: dict) -> str:
    L: list[str] = []
    j = report["job"]
    ea = report["error_analysis"]
    L.append("=" * 72)
    L.append(f" nccl-doctor report — job {j['job_id']} ({j['outcome']})")
    L.append("=" * 72)
    L.append(f" Nodes: {', '.join(j['nodes'])}   world_size={j['world_size']}   "
             f"NCCL {j.get('nccl_version') or '?'}")
    L.append("")
    L.append(f" VERDICT: {ea['verdict']}   (confidence {ea['confidence']:.2f})")
    origin = ea.get("origin") or {}
    if origin.get("note"):
        L.append(f"   {origin['note']}")
    if origin.get("origin_hosts"):
        L.append(f"   origin host(s): {', '.join(origin['origin_hosts'])}  "
                 f"desync at seq~{origin.get('desync_seq')}")
    L.append("")
    L.append(" FINDINGS")
    if not ea["findings"]:
        L.append("   (none)")
    for f in ea["findings"]:
        L.append(f"   [{f['severity'].upper():8s}] {f['rule']:8s} {f['summary']}")
        for e in f.get("evidence", [])[:2]:
            L.append(f"              ↳ {e['source']}: {e['detail'][:110]}")
    L.append("")
    L.append(" RECOMMENDED ACTIONS")
    for a in report["recommended_actions"]:
        L.append(f"   P{a['priority']} [{a['type']}] {a['action']}")
        if a.get("command"):
            L.append(f"        $ {a['command']}")
        if a.get("caveat"):
            L.append(f"        caveat: {a['caveat']}")
    if report.get("retry_explanation"):
        L.append("")
        L.append(" WHY DID THE RETRY PASS?")
        L.append(f"   {report['retry_explanation']}")
    nr = report.get("node_reliability", {})
    if nr.get("scores"):
        L.append("")
        L.append(f" NODE RELIABILITY (fleet median {nr['fleet_median']})")
        for n, s in sorted(nr["scores"].items(), key=lambda kv: -kv[1]):
            flag = "  <-- elevated" if nr["fleet_median"] and s > 5 * nr["fleet_median"] \
                   and s > 0.2 else ""
            L.append(f"   {n:16s} {s:.3f}{flag}")
    L.append("")
    return "\n".join(L)
