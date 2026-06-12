# Technical Design Document: NCCL Diagnostic & Tuning Tool ("nccl-doctor")

**Status:** Draft v1.0
**Audience:** AI Infrastructure / HPC Platform Engineering
**Problem Statement:** Distributed training jobs intermittently fail with NCCL timeouts or "surprise failures," then pass on resubmission. This pattern indicates transient fabric issues, placement-dependent topology problems, or fragile NCCL configuration — not application bugs. We need an automated tool that ingests failure telemetry, identifies root cause, and emits actionable configuration or hardware remediation.

---

## 0. Design Principles

1. **Diagnosis before tuning.** "Fails once, passes on retry" is almost never an algorithm-selection problem. The tool must first answer *which rank/node/link stalled and why* before suggesting env vars. A tuner bolted onto an unhealthy fabric just hides hardware faults.
2. **Evidence-based verdicts.** Every recommendation must cite the specific log line, counter delta, or topology asymmetry that triggered it. No "try these 12 env vars" cargo-cult output.
3. **Cross-run memory.** Single failures are often inconclusive. The tool fingerprints every run (node set, topology, config, counters, outcome) so flaky hardware reveals itself statistically over days, even when each individual failure looks random.
4. **Cheap by default, deep on failure.** Always-on collection must add near-zero overhead. Expensive captures (stack dumps, flight-recorder flushes, full counter sweeps) trigger only on hang detection or job failure.
5. **Build on NCCL's native hooks.** Use the profiler plugin API, RAS subsystem, tuner plugin API, and topo/graph dump files rather than reinventing instrumentation.

---

## 1. Data Ingestion & Telemetry

### 1.1 Job-Level Collection (always on, captured by launch wrapper)

| Category | What to collect | How |
|---|---|---|
| NCCL identity | NCCL version, CUDA version, driver version, network plugin version (e.g., `nccl-rdma-sharp-plugins`, AWS OFI), PyTorch/framework version | `ldd`, `python -c "import torch; ..."`, parse NCCL INFO banner |
| Full environment | Every `NCCL_*`, `UCX_*`, `OMPI_*`, `TORCH_NCCL_*` var, plus `CUDA_VISIBLE_DEVICES`, CPU affinity/binding flags | Wrapper dumps `env` at launch, per rank |
| NCCL debug logs | Per-rank log files | `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,NET,TUNING,COLL`, `NCCL_DEBUG_FILE=/var/log/nccl/%h.%p.log` (one file per host+PID — never interleave to stdout at scale) |
| Topology as NCCL saw it | Detected topology + computed graphs | `NCCL_TOPO_DUMP_FILE=/var/log/nccl/topo.%h.xml`, `NCCL_GRAPH_DUMP_FILE=/var/log/nccl/graph.%h.xml` |
| Job metadata | Job ID, node list, rank→host→GPU mapping, world size, start/end time, exit codes per rank, which rank raised the first error | Slurm (`scontrol show job`, epilog env) or K8s pod annotations |
| Framework-side trace | PyTorch NCCL flight recorder dump (per-collective enqueue/start/complete state for the last N collectives on every rank at timeout) | `TORCH_NCCL_TRACE_BUFFER_SIZE=20000`, `TORCH_NCCL_DUMP_ON_TIMEOUT=1`, `TORCH_NCCL_DEBUG_INFO_TEMP_FILE=...` |

**Why the flight recorder matters:** the watchdog kills the job on whichever rank *notices* the desync first — frequently not the rank that caused it. The flight recorder lets the analyzer find the collective with mismatched sequence numbers across ranks and identify the true straggler/stuck rank. This single artifact resolves the most common misattribution in user bug reports.

### 1.2 Host-Level Collection (node agent, snapshot at job start + job end/failure)

| Category | Data points | Source |
|---|---|---|
| GPU health | Xid events, ECC counts (volatile + aggregate), retired pages / row-remap, throttle reasons (thermal, power, HW slowdown), clocks | `dmesg -T \| grep -i xid`, `nvidia-smi -q`, DCGM (`dcgmi diag -r 1`, field group for Xid/thermal) |
| GPU interconnect | NVLink state per link, NVLink CRC/replay/recovery error counters, expected vs. actual active link count for the GPU SKU | `nvidia-smi nvlink --status`, `nvidia-smi nvlink -e`, DCGM NVLink fields |
| PCIe | Negotiated gen/width vs. capability (`LnkSta` vs `LnkCap`), AER errors, ACS enablement per bridge, IOMMU mode | `lspci -vvv`, `dmesg \| grep -i aer`, `lspci -vvv \| grep -i acsctl` |
| Host topology | GPU↔NIC↔CPU affinity matrix, NUMA layout | `nvidia-smi topo -m`, `nvidia-smi topo -mp`, `numactl --hardware` |
| System | OOM events, CPU steal/contention, IRQ distribution, hugepage/pinned-memory limits (`ulimit -l`), kernel version | `journalctl -k`, `/proc/interrupts`, `dmesg` |

### 1.3 Fabric-Level Collection (counter deltas: snapshot at start, on hang, at end)

**InfiniBand (per HCA port):**
- `SymbolErrorCounter`, `LinkErrorRecoveryCounter`, `LinkDownedCounter` — physical-layer health and flaps
- `PortRcvErrors`, `PortXmitDiscards`, `ExcessiveBufferOverrunErrors`
- `PortXmitWait` — the canonical congestion signal (time spent with data ready but no credits)
- `VL15Dropped`, `PortRcvSwitchRelayErrors` — SM/routing problems
- Sources: `perfquery -x`, `ibqueryerrors`, periodic `ibdiagnet` sweeps; switch-side via UFM/telemetry if available

**RoCE (per mlx5 port):**
- `ethtool -S`: `rx_prio*_pause`, `tx_prio*_pause` (PFC activity), `rx_pause_duration`, `np_cnp_sent`, `rp_cnp_handled` (ECN/DCQCN activity), `out_of_buffer`, `rx_discards_phy`, `tx_discards_phy`
- QoS config: `mlnx_qos -i <if>` (PFC enabled on the lossless priority? trust mode DSCP?), DCQCN enable state, `cma_roce_tos`
- Link flaps: `ip -s link`, NIC firmware event log (`mlxlink`, `mstdump` on demand)

**Cross-node consistency:** NIC firmware versions, MTU, link speed — collected from all nodes and *diffed* (a single NIC at FW N-1 or MTU 1500 in a 4096-MTU fabric causes exactly this class of intermittent failure).

### 1.4 Capturing Transient State Right Before the Timeout

This is the hard part: by the time the job is dead, the congestion event or stalled QP is gone. Four mechanisms, layered:

1. **Ring-buffer counter sampler (node agent).** Sample the fabric and NVLink counters above every 2–5 s into an in-memory ring buffer (last ~10 min). On job failure, flush the buffer to the report. This converts "counters at start vs. end" into a *time series across the failure window* — you can see the pause-frame storm or symbol-error burst at T-30s.
2. **Hang detection before the watchdog fires.** The PyTorch watchdog default is 10 min (`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` / op timeout). Configure the collector to treat "no collective completed in 60–120 s" (via profiler plugin events or framework heartbeat) as a *pre-failure trigger* that fires while the job is still alive:
   - Query **NCCL RAS** (`ncclras`, NCCL ≥ 2.24) to get live per-rank collective progress and identify stuck vs. lagging ranks.
   - Run `py-spy dump` / `gdb -p` against every rank's process to capture Python + native stacks (is the stuck rank in `ibv_poll_cq`, in a dataloader, in a host sync?).
   - Take an immediate fabric counter snapshot.
3. **NCCL profiler plugin (NCCL ≥ 2.23).** Ship a small profiler `.so` that records per-collective and per-proxy-op start/stop events to a per-rank mmap'd buffer. On timeout, the buffer shows exactly which proxy op (channel, peer, network transfer) never completed — i.e., which *link*, not just which node.
4. **Flight recorder dump on timeout** (Section 1.1) — last-resort post-mortem that always works even if the agent missed the hang.

---

## 2. Diagnostic Heuristics — The "What Went Wrong" Engine

The analyzer is a rules engine: each rule consumes parsed evidence (log events, counter deltas, topology facts), emits a *finding* with a confidence score, and findings combine into a verdict. Rules are grouped into three families.

### 2.1 Topology Bottlenecks

| Rule ID | Detection logic | Evidence source |
|---|---|---|
| `TOPO-01` GPU↔NIC affinity crosses CPU interconnect | In `nvidia-smi topo -m`, the GPU↔NIC cell used for that GPU's traffic is `NODE`, `PHB`, or `SYS` instead of `PIX`/`PXB`; or NCCL log shows a NIC selected whose PCIe path crosses the CPU root complex | topo matrix + NCCL `NET` log (`NCCL INFO NET/IB : Using [0]mlx5_3 ...` cross-referenced with GPU placement) |
| `TOPO-02` GPUDirect RDMA silently disabled | NCCL log lacks `GDRDMA` on net transports / shows `NET/IB: ... via SHM or host buffers`; root cause check: ACS enabled on PCIe bridges, IOMMU in non-passthrough mode, or `nvidia-peermem`/DMABUF missing | NCCL `NET` log + `lspci ACSCtl` + `lsmod` |
| `TOPO-03` Missing/degraded NVLink | Active NVLink count < expected for SKU (e.g., H100 SXM: 18 links); or NVLink replay/CRC counters increment during the failure window | `nvidia-smi nvlink --status/-e`, ring-buffer deltas |
| `TOPO-04` PCIe downtraining | `LnkSta` gen/width < `LnkCap` (e.g., x16 Gen5 device negotiated x8 or Gen3) on a GPU or NIC | `lspci -vvv` |
| `TOPO-05` Asymmetric topology detection | `NCCL_TOPO_DUMP_FILE` XMLs differ across nodes of the same hardware SKU (NIC count, speeds, PCIe layout) — a node with a dead NIC silently detects 7/8 HCAs and drags every ring through a narrow path | XML diff across all nodes in the job |
| `TOPO-06` Rail misalignment | In rail-optimized fabrics, NCCL channel↔NIC assignment sends traffic across rails (leaf-spine-leaf instead of staying on-rail); detected by comparing graph dump channel→NIC mapping with the cabling/rail map | graph XML + fabric inventory |
| `TOPO-07` NUMA/affinity misbinding | Rank's CPU affinity is on the far socket from its GPU/NIC; proxy thread starvation symptoms (high proxy op latency in profiler with idle fabric) | wrapper-captured affinity + topo matrix |

### 2.2 Network & Fabric Issues (the usual culprits for retry-passes failures)

| Rule ID | Detection logic | Evidence source |
|---|---|---|
| `NET-01` Transport retry exhaustion | NCCL log contains IB async/completion errors with `IBV_WC_RETRY_EXC_ERR` (vendor code 0x81 / "Got completion ... error 12") — the QP retried `NCCL_IB_RETRY_CNT` times over `NCCL_IB_TIMEOUT` and gave up. Verdict: packet loss between the two endpoints, not NCCL | per-rank NCCL log; identifies *both* endpoints of the bad path |
| `NET-02` Link flap | `LinkDownedCounter` or `LinkErrorRecoveryCounter` delta > 0 during job window; `ip -s link` carrier transitions | ring-buffer counter series |
| `NET-03` Physical-layer degradation | `SymbolErrorCounter` rate above threshold (e.g., >10/min sustained), or rising BER from `mlxlink` — classic marginal cable/transceiver; explains "different node set on retry passes" | counter series, per port |
| `NET-04` Congestion (IB) | `PortXmitWait` delta large relative to transfer volume on specific switch/HCA ports; collective bandwidth from profiler far below baseline only at specific scales | counter series + profiler |
| `NET-05` PFC misconfiguration / storm (RoCE) | Pause counters on the wrong priority (RDMA traffic not in the lossless class → silent drops → retry storms), or pause duration exploding (PFC storm / head-of-line blocking propagating through the fabric) | `ethtool -S`, `mlnx_qos` config lint |
| `NET-06` ECN/DCQCN misconfiguration (RoCE) | Near-zero `np_cnp_sent` under heavy load (ECN marking not configured on switches → congestion handled by PFC alone → storms), or wildly asymmetric CNP counts across nodes; DSCP/trust mode mismatches across NICs | NIC counters + config lint |
| `NET-07` Heterogeneous fabric config | MTU, FW version, link speed, or QoS trust mode differs across the job's NICs | cross-node config diff |
| `NET-08` SM/routing instability (IB) | `VL15Dropped`, `PortRcvSwitchRelayErrors` deltas; subnet manager failover events in the window | counters + SM logs |

### 2.3 NCCL Algorithm, Protocol & Discovery Issues

| Rule ID | Detection logic | Evidence source |
|---|---|---|
| `ALGO-01` Socket fallback on an RDMA cluster | `NCCL INFO NET/Socket : Using ...` on a machine with healthy HCAs — IB plugin failed to load, `NCCL_IB_HCA` filter excluded everything, or RDMA device perms broke. Causes 10–50× slowdown that *looks* like a hang and trips timeouts | NCCL `INIT/NET` log |
| `ALGO-02` Suspicious protocol selection | `TUNING` log shows LL/LL128 selected at message sizes where Simple should win, or LL128 active on a platform/path where it isn't validated (LL128 requires specific platform guarantees; on non-qualified paths it can cause corruption or hangs). Heuristic: failures correlate with LL128 collectives in the flight recorder | `NCCL_DEBUG_SUBSYS=TUNING` log + flight recorder |
| `ALGO-03` Ring vs. Tree pathology | At large scale, Tree latency should win for small/medium messages; if `GRAPH` dump shows tree construction failed (fell back to ring-only) on some runs but not others, topology detection is placement-sensitive | graph XML diff across runs |
| `ALGO-04` Channel count anomaly | Channels created ≠ expected for the platform (e.g., 2 instead of 16 because one ring path was rejected); throughput collapses, long collectives, watchdog fires under load spikes | `INIT` log (`NCCL INFO ... nChannels`) |
| `ALGO-05` NVLS/SHARP not engaged | On NVSwitch systems (H100/B200) NVLS absent from algo list, or SHARP plugin failed init — not a failure cause per se, but a large perf delta users misread as "slow = about to time out" | `INIT`/`TUNING` log |
| `ALGO-06` Bandwidth regression vs. golden baseline | Periodic `nccl-tests` sweep results per (node-count, collective, size) stored as baselines; failed job's profiler-measured busbw < 70% of baseline → something environmental changed even if no error fired | baseline DB + profiler |

### 2.4 The Straggler Rule (cuts across all families)

`STRAG-01`: From the flight recorder, compute per-rank lag on the last completed collective. If one rank is consistently last by a large margin across the trailing window, it is the straggler; cross-reference its host's throttle reasons (thermal/power), Xid events, CPU contention, and dataloader stalls (Python stack from py-spy). This rule resolves the majority of "watchdog timeout on rank 0" reports where rank 0 was merely the messenger.

---

## 3. Configuration & Tuning Recommendations

### 3.1 Failure Profile → Recommendation Map

| Failure profile (rules fired) | Recommended change | Rationale & caveats |
|---|---|---|
| `NET-01` retry exhaustion, *first occurrence on this path* | `NCCL_IB_TIMEOUT=20–22` (default 18; value is exponential, 4.096 µs × 2^n), `NCCL_IB_RETRY_CNT=10` (default 7) | Buys tolerance to brief loss/congestion. **Mitigation, not fix** — open a fabric ticket; if `NET-03` fires on the same port, this is hardware |
| `NET-01` + `NET-04` congestion at scale | `NCCL_IB_QPS_PER_CONNECTION=2–4`, verify adaptive routing enabled on fabric; consider `NCCL_IB_SPLIT_DATA_ON_QPS=1` | Spreads flows across paths; primary fix is fabric-side (AR, ECN) |
| `NET-05`/`NET-06` RoCE QoS broken | Fix switch/NIC config first (PFC on lossless class, ECN marking thresholds, trust DSCP); set `NCCL_IB_TC` to the DSCP-mapped traffic class NCCL should use | Env vars cannot fix a lossy RoCE fabric — flag as **infrastructure fix**, severity high |
| `ALGO-01` socket fallback | `NCCL_IB_HCA==mlx5_0,mlx5_1,...` (explicit allowlist, `=` prefix for exact match), `NCCL_SOCKET_IFNAME=^lo,docker0`; fix plugin path / device cgroup perms | Pin interfaces explicitly in containerized environments — interface enumeration order is not stable |
| `ALGO-02` LL128 implicated | `NCCL_PROTO=^LL128` as a targeted exclusion for affected job class | Don't globally pin `NCCL_PROTO=Simple` — that throws away latency performance everywhere |
| `ALGO-03`/`TOPO-05` placement-sensitive topology detection | Generate and pin `NCCL_TOPO_FILE=/etc/nccl/topo-<sku>.xml` from a known-good node of each hardware SKU | Removes per-run discovery variance entirely; re-generate on hardware/driver change |
| `ALGO-04` low channel count | `NCCL_MIN_NCHANNELS=<expected>` only after fixing the underlying path rejection (often `TOPO-02`/`TOPO-04`) | Forcing channels over a broken path makes things worse |
| `TOPO-01`/`TOPO-07` affinity problems | Fix launcher binding (`--gpu-bind`, `--cpu-bind` in Slurm; topology-aware pod spec in K8s); `NCCL_IGNORE_CPU_AFFINITY=1` only as diagnostic | Prefer fixing the binding over overriding NCCL |
| `TOPO-02` GDR disabled | Disable ACS on PCIe bridges (BIOS or `setpci`), IOMMU passthrough, install/load `nvidia-peermem` or use DMABUF; verify with `NCCL_NET_GDR_LEVEL` semantics rather than forcing it | Forcing GDR over a path with ACS on → data corruption risk; this is a node-config fix |
| Multi-NIC small-scale jobs crossing rails | `NCCL_PXN_DISABLE=0` (ensure PXN active), verify `NCCL_CROSS_NIC` setting matches fabric design (rail-optimized → keep default 0/2 semantics) | Rail-aware settings depend on cabling; encode fabric design into the rule |
| Watchdog fires under known-long collectives (huge allgathers, first-step graph capture) | Raise framework timeout for that job class (`init_process_group(timeout=...)`), not NCCL IB timeouts | Distinguish "collective legitimately slow" from "collective stuck" via profiler progress events |
| `STRAG-01` + thermal/Xid evidence on one host | **No env var.** Cordon node, file hardware ticket | See 3.2 |

### 3.2 Config Tuning vs. Faulty Hardware — the Differentiation Logic

This is the most valuable judgment the tool makes. Decision procedure:

1. **Locality test.** Does the failure evidence localize to specific hardware (one port's symbol errors, one GPU's Xid, one node always the straggler)? Localized → hardware track. Distributed across nodes/links → config/systemic track.
2. **Recurrence correlation (cross-run store).** For every failure, record the node set and implicated components. Maintain per-component reliability scores (e.g., EWMA of failure involvement, weighted by how strongly rules implicated it). A node whose involvement-in-failure rate is 5× the fleet median is hardware-suspect *even if no single run was conclusive*. This is exactly how "retry passes" failures get caught: the retry ran on different nodes.
3. **Counter physics.** Symbol errors, CRC/replay errors, Xid events, ECC growth, link flaps are *physical* signals — they are never fixed by env vars. Any rule citing them routes to the hardware track regardless of other findings.
4. **Reproducibility test (active).** For config hypotheses, the tool can queue a canary: re-run `nccl-tests` on the same node set with and without the proposed env change. Config issues reproduce deterministically under the same placement; transient hardware issues don't.
5. **Action thresholds.** Hardware-suspect score > T_warn → flag in report and notify operators; > T_cordon → emit Slurm drain (`scontrol update nodename=... state=drain reason="nccl-doctor: <rule>"`) or K8s taint, pending human confirmation (auto-cordon optional, off by default).

---

## 4. Tool Architecture

### 4.1 Components & Tech Stack (lightweight, no new databases to operate)

```
┌─────────────────────────────────────────────────────────────────┐
│                      JOB LIFECYCLE                              │
│                                                                 │
│  Slurm prolog / K8s initContainer                               │
│    └─ node snapshot (counters, topo, health)  ──┐               │
│                                                 │               │
│  Launch wrapper (`nccl-doctor run -- <cmd>`)    │               │
│    └─ injects NCCL_DEBUG/_FILE/_DUMP vars,      │               │
│       TORCH_NCCL_* flight-recorder flags,       │               │
│       profiler plugin .so                       │               │
│                                                 ▼               │
│  Node agent (systemd unit, ~0 overhead)    artifact dir         │
│    ├─ 2–5s counter ring buffer             /var/log/nccl/<job>  │
│    └─ hang trigger: RAS query + py-spy                          │
│                                                                 │
│  Slurm epilog / K8s sidecar on failure                          │
│    └─ final snapshot, gather rank logs → object store / NFS     │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
        Analyzer (Python): parsers → rules engine → verdict
                            │
        Run-fingerprint store (SQLite/DuckDB) ←─ reliability scores
                            │
                            ▼
        Outputs: CLI report (`nccl-doctor analyze <job>`),
                 JSON artifact, Slurm drain / K8s taint hooks,
                 optional Prometheus metrics + Grafana
```

- **Language:** Python 3.11+ (`click` CLI, `pydantic` models for findings/reports, `lxml` for topo XML).
- **Storage:** per-job artifact directory (logs, XMLs, counter series as Parquet) on shared FS or S3; **SQLite/DuckDB** for the fingerprint store — single file, trivially operable, fine for thousands of jobs/day.
- **Profiler plugin:** small C shim implementing the NCCL profiler plugin interface, writing events to an mmap ring per rank.
- **Deployment modes:**
  - *User CLI:* `nccl-doctor analyze --job-id 12345` — post-mortem on demand; works with whatever artifacts exist.
  - *Cluster mode (recommended):* prolog/epilog (Slurm) or operator + DaemonSet agent (K8s) make collection automatic; analysis runs in the epilog and posts the report to the job's stdout/annotation, so the user gets the verdict in the same place they saw the failure.
- **Rules as data:** rules live in versioned YAML (pattern, evidence query, severity, recommendation template) so platform engineers add signatures without code changes.

### 4.2 Workflow on a Failure

1. Epilog detects nonzero exit / watchdog kill → gathers per-rank NCCL logs, flight-recorder dumps, topo/graph XMLs, agent ring buffers, start/end counter snapshots.
2. Analyzer parses everything into typed evidence objects.
3. Rules engine evaluates all rules; findings scored and deduplicated; straggler analysis pinpoints origin rank.
4. Cross-run store updated; reliability scores recomputed.
5. Report rendered (human text + JSON below); hardware-track findings optionally trigger drain/taint workflow.

### 4.3 Sample JSON Output

```json
{
  "schema_version": "1.0",
  "job": {
    "job_id": "slurm-8841273",
    "cluster": "h100-prod-a",
    "nodes": ["gpu-114", "gpu-115", "gpu-201", "gpu-202"],
    "world_size": 32,
    "nccl_version": "2.26.2",
    "framework": "pytorch-2.7.0",
    "outcome": "WATCHDOG_TIMEOUT",
    "first_error_rank": 3,
    "wallclock_at_failure": "2026-06-11T03:14:22Z"
  },
  "error_analysis": {
    "verdict": "TRANSIENT_FABRIC_LOSS",
    "confidence": 0.86,
    "origin": {
      "true_origin_rank": 17,
      "origin_host": "gpu-201",
      "note": "Rank 3 raised the timeout but flight recorder shows ranks 16-23 stuck on AllReduce seq=48211 while others completed seq=48212; proxy op on rank 17 channel 5 to peer gpu-202/mlx5_4 never completed."
    },
    "findings": [
      {
        "rule": "NET-01",
        "severity": "high",
        "summary": "IB transport retry exhaustion on gpu-201:mlx5_4 <-> gpu-202:mlx5_4",
        "evidence": [
          "gpu-201 rank17 nccl log 03:14:19: 'NET/IB: Got completion ... status 12 (transport retry counter exceeded)'",
          "perfquery delta gpu-201/mlx5_4: PortXmitDiscards +312 in failure window"
        ]
      },
      {
        "rule": "NET-03",
        "severity": "high",
        "summary": "Symbol errors rising on leaf switch port swl-12/0/7 (gpu-202 mlx5_4 uplink): +1,240 over 90s pre-failure",
        "evidence": ["agent ring buffer gpu-202 03:12:50-03:14:20"]
      }
    ]
  },
  "topology_diagnosis": {
    "asymmetries": [],
    "gdr_enabled": true,
    "nvlink_status": "all links active on all nodes",
    "rails_aligned": true,
    "note": "Topology healthy; failure localizes to one fabric path."
  },
  "recommended_actions": [
    {
      "type": "INFRASTRUCTURE",
      "priority": 1,
      "action": "Inspect/replace cable or transceiver on swl-12/0/7 <-> gpu-202:mlx5_4; symbol error rate indicates marginal physical link.",
      "owner": "network-ops"
    },
    {
      "type": "NCCL_CONFIG",
      "priority": 2,
      "action": "Interim mitigation for this job class: NCCL_IB_TIMEOUT=20, NCCL_IB_RETRY_CNT=10",
      "caveat": "Masks brief loss; remove after link repair."
    },
    {
      "type": "SCHEDULER",
      "priority": 3,
      "action": "gpu-202 reliability score 0.41 (fleet median 0.06, 3rd implication in 7 days) — recommend drain pending link repair.",
      "command": "scontrol update nodename=gpu-202 state=drain reason='nccl-doctor NET-03 swl-12/0/7'"
    }
  ],
  "retry_explanation": "Previous resubmission (slurm-8839912) succeeded because the scheduler placed it on gpu-117/118 instead of gpu-202, avoiding the degraded link. The failure is placement-correlated, not random.",
  "artifacts": {
    "report_dir": "s3://nccl-doctor/h100-prod-a/slurm-8841273/",
    "flight_recorder": "ranks 0-31 captured",
    "counter_series": "parquet, 03:04-03:14, 2s resolution"
  }
}
```

The `retry_explanation` field is deliberate: it directly answers the user's "but it passed when I reran it!" — which is the single biggest source of distrust in the platform.

---

## 5. Phased Implementation Plan

| Phase | Scope | Exit criterion |
|---|---|---|
| **P0 (2–3 wks)** | Launch wrapper + log/env collection; NCCL log parser; rules `NET-01`, `ALGO-01`, `TOPO-02/04`, dmesg Xid; CLI report | Tool produces a correct verdict on ≥5 historical failures |
| **P1 (3–4 wks)** | Node agent with counter ring buffer; flight-recorder ingestion + straggler analysis; full NET/TOPO rule set; JSON schema | True-origin rank identified automatically; counter time series in every failure report |
| **P2 (3 wks)** | Fingerprint store, reliability scoring, drain/taint integration; `retry_explanation` generation | First flaky node caught by statistics before users escalate |
| **P3 (ongoing)** | RAS + profiler plugin live hang capture; nccl-tests baseline sweeps + `ALGO-06`; tuner plugin from measured optima; canary re-runs | Mean-time-to-root-cause < 10 min from job failure |

## 6. Risks & Open Questions

- **Log volume at scale:** `NCCL_DEBUG=INFO` on 4k ranks is heavy; mitigate with per-rank files, log rotation, and optionally INFO only on ranks 0..N + local-rank-0s, escalating to all ranks on repeat failures.
- **RoCE vs. IB rule sets diverge significantly** — confirm fabric type(s) to prioritize (Section 2.2 covers both, but switch-side telemetry integration differs: UFM vs. vendor sFlow/gNMI).
- **Privilege boundaries:** counter collection and `setpci`/drain actions need elevated agent privileges; keep the analyzer unprivileged and the agent minimal.
- **NCCL version drift:** log formats and plugin ABIs change between minor versions; parsers must be version-keyed (the INFO banner gives the version).
