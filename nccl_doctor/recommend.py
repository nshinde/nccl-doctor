"""Mapping of failure profiles (fired rules) to recommended actions.

Implements design doc §3.1. Recommendation text templates can reference
finding fields. Ordering: HARDWARE/INFRA fixes always outrank env-var
mitigations — config tuning over a broken fabric just hides faults.
"""
from __future__ import annotations

from .models import Action, Finding, Severity, Track

_PRIORITY = {Track.HARDWARE: 1, Track.INFRASTRUCTURE: 1, Track.SCHEDULER: 2,
             Track.NCCL_CONFIG: 3, Track.APPLICATION: 3}


def recommend(findings: list[Finding], node_scores: dict[str, float],
              fleet_median: float, drain_threshold_ratio: float = 5.0) -> list[Action]:
    actions: list[Action] = []
    fired = {f.rule for f in findings}

    def add(a: Action) -> None:
        if not any(x.action == a.action for x in actions):
            actions.append(a)

    for f in findings:
        if f.severity in (Severity.INFO,):
            continue
        comp = ", ".join(f.components) or ", ".join(f.hosts) or "affected path"

        if f.rule == "NET-01":
            status = f.data.get("status")
            if status == "12":
                add(Action(Track.HARDWARE, 1, f"Inspect fabric path for {comp}: "
                    f"transport retries exhausted indicates packet loss or an "
                    f"unreachable peer. Correlate with NET-02/NET-03 counters; check "
                    f"cable/transceiver and switch port.", "NET-01", owner="network-ops"))
                add(Action(Track.NCCL_CONFIG, 3,
                    "Interim mitigation for this job class: NCCL_IB_TIMEOUT=20 "
                    "(default 18; exponential, 4.096us x 2^n) and NCCL_IB_RETRY_CNT=10 "
                    "(default 7).", "NET-01",
                    caveat="Masks brief loss; remove after the fabric fix."))
            elif status == "13":
                add(Action(Track.SCHEDULER, 2, f"RNR retries exhausted toward {comp}: "
                    f"the remote process stalled. Check the peer host for straggler "
                    f"causes (throttling, Xid, CPU contention).", "NET-01"))

        elif f.rule == "NET-02":
            add(Action(Track.HARDWARE, 1, f"Link flapped on {comp} during the job — "
                f"reseat/replace cable or transceiver; check switch port logs.",
                "NET-02", owner="network-ops"))

        elif f.rule == "NET-03" and f.severity == Severity.HIGH:
            add(Action(Track.HARDWARE, 1, f"Symbol error storm on {comp}: marginal "
                f"physical link. Replace cable/transceiver; re-run mlxlink BER check "
                f"after repair.", "NET-03", owner="network-ops"))

        elif f.rule == "NET-04":
            add(Action(Track.INFRASTRUCTURE, 2, f"Congestion at {comp}: verify adaptive "
                f"routing is enabled fabric-wide; check for oversubscribed uplinks on "
                f"the implicated leaf.", "NET-04", owner="network-ops"))
            add(Action(Track.NCCL_CONFIG, 3, "If congestion is structural at this scale, "
                "trial NCCL_IB_QPS_PER_CONNECTION=2 (spreads flows across AR paths).",
                "NET-04", caveat="Measure with nccl-tests before/after; not free."))

        elif f.rule == "NET-05":
            add(Action(Track.INFRASTRUCTURE, 1, "Fix RoCE lossless class config: PFC "
                "enabled exactly on the RDMA priority end-to-end (NIC + every switch "
                "hop); verify trust-DSCP and that NCCL traffic lands in that class "
                "(NCCL_IB_TC).", "NET-05", owner="network-ops"))

        elif f.rule == "NET-06":
            add(Action(Track.INFRASTRUCTURE, 1, "Enable/verify ECN marking on switches "
                "and DCQCN on NICs — congestion currently handled by PFC alone, which "
                "produces pause storms and intermittent timeouts.", "NET-06",
                owner="network-ops"))

        elif f.rule == "TOPO-01":
            add(Action(Track.INFRASTRUCTURE, 2, f"Fix GPU/NIC placement or launcher "
                f"binding for {comp} (Slurm --gpu-bind/--cpu-bind, topology-aware pod "
                f"spec). NIC traffic should not cross the CPU complex.", "TOPO-01"))

        elif f.rule == "TOPO-02":
            add(Action(Track.INFRASTRUCTURE, 1, f"Enable GPUDirect RDMA on {comp}: "
                f"disable ACS on PCIe bridges (BIOS/setpci), IOMMU passthrough, load "
                f"nvidia-peermem or enable DMABUF.", "TOPO-02",
                caveat="Do NOT force GDR over a path with ACS enabled — corruption risk."))

        elif f.rule == "TOPO-03":
            add(Action(Track.HARDWARE, 1, f"Degraded NVLink on {comp}: run dcgmi diag "
                f"-r 3; if links remain down, ticket the baseboard.", "TOPO-03",
                owner="dc-ops"))

        elif f.rule == "TOPO-04":
            add(Action(Track.HARDWARE, 1, f"PCIe downtraining on {comp}: reseat the "
                f"card/riser; check for AER errors; verify slot bifurcation/BIOS.",
                "TOPO-04", owner="dc-ops"))

        elif f.rule == "TOPO-05":
            add(Action(Track.HARDWARE, 1, f"Topology asymmetry on {comp}: enumerate "
                f"PCIe devices vs. the SKU golden config (likely dead NIC). Drain "
                f"until it matches.", "TOPO-05", owner="dc-ops"))
            add(Action(Track.NCCL_CONFIG, 3, "Pin NCCL_TOPO_FILE to a golden per-SKU "
                "topology XML to remove per-run discovery variance (also turns silent "
                "asymmetry into a loud init failure).", "TOPO-05"))

        elif f.rule == "ALGO-01":
            add(Action(Track.NCCL_CONFIG, 2, f"Restore RDMA transport on {comp}: set an "
                f"explicit allowlist NCCL_IB_HCA==mlx5_0,mlx5_1,... and "
                f"NCCL_SOCKET_IFNAME=^lo,docker0; verify the net plugin .so loads and "
                f"/dev/infiniband perms inside the container.", "ALGO-01"))

        elif f.rule == "ALGO-02":
            add(Action(Track.NCCL_CONFIG, 3, "Bisection step: NCCL_PROTO=^LL128 for "
                "this job class only.", "ALGO-02",
                caveat="Do not pin NCCL_PROTO=Simple globally; it costs latency everywhere."))

        elif f.rule == "ALGO-04":
            add(Action(Track.NCCL_CONFIG, 3, f"Channel deficit on {comp}: fix the "
                f"rejected path first (see TOPO findings); only then consider "
                f"NCCL_MIN_NCHANNELS.", "ALGO-04"))

        elif f.rule == "GPU-01":
            add(Action(Track.HARDWARE, 1, f"GPU fault on {comp} ({f.summary}). Drain "
                f"the node and run full diagnostics (dcgmi diag -r 4).", "GPU-01",
                owner="dc-ops"))

        elif f.rule == "STRAG-01":
            for h in f.hosts:
                add(Action(Track.SCHEDULER, 2, f"Straggler origin host {h}: check "
                    f"throttle reasons, Xid, CPU contention, dataloader stalls before "
                    f"resubmitting onto it.", "STRAG-01"))

    # reliability-score-driven drains (design doc §3.2 step 5)
    for node, score in sorted(node_scores.items(), key=lambda kv: -kv[1]):
        if fleet_median > 0 and score >= drain_threshold_ratio * fleet_median and score > 0.2:
            add(Action(Track.SCHEDULER, 2,
                f"{node} reliability score {score:.2f} vs fleet median "
                f"{fleet_median:.2f} ({score/fleet_median:.1f}x) — recommend drain "
                f"pending hardware review.", "SCORE",
                command=f"scontrol update nodename={node} state=drain "
                        f"reason='nccl-doctor reliability {score:.2f}'"))

    # generic fallback when nothing fired but the job timed out
    if not actions and not fired:
        add(Action(Track.APPLICATION, 3,
            "No NCCL/fabric signature found. Verify the timeout wasn't a legitimately "
            "long collective (raise framework timeout for first-step/graph-capture "
            "phases) and enable the full collection profile "
            "(NCCL_DEBUG=INFO, flight recorder, counter snapshots) for the next run.",
            "NONE"))

    actions.sort(key=lambda a: (a.priority, a.type.value))
    return actions
