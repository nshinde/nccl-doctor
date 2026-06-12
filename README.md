# NCCL Doctor

Evidence-based NCCL failure diagnosis, topology linting, and tuning advice for GPU clusters.

NCCL Doctor is built for the painful class of distributed training failures where a job times out, someone resubmits it, and the retry passes. Those failures are usually not model bugs. They are often placement-sensitive fabric problems, topology drift, hardware degradation, or fragile NCCL configuration.

This tool collects the right artifacts around a job, analyzes them with a rules engine, identifies the likely origin of the hang, and produces a report that separates hardware fixes from temporary NCCL mitigations.

## Why It Exists

When a PyTorch or NCCL watchdog fires, the rank that reports the timeout is often just the messenger. NCCL Doctor is designed to answer the questions platform teams actually need answered:

- Which rank, host, NIC, GPU, or link most likely caused the failure?
- Was this a hardware/fabric issue or a configuration issue?
- Why did the same workload pass after a retry?
- Which action should be tried first, and which "fixes" only mask the issue?
- Is one node becoming statistically unreliable across runs?

## Highlights

- **Zero runtime dependencies:** stdlib-only Python 3.10+ so it can run on locked-down HPC login and compute nodes.
- **Evidence-first reports:** every finding points to a log line, counter delta, topology fact, or flight recorder signal.
- **True-origin analysis:** PyTorch flight recorder data is used to distinguish stuck ranks from reporting ranks.
- **Hardware vs. config routing:** physical signals such as Xid events, symbol errors, link flaps, and NVLink degradation are routed to hardware or infrastructure actions instead of env-var folklore.
- **Cross-run memory:** a SQLite store tracks component involvement over time and explains retry-pass behavior when placement changes.
- **Cluster friendly:** includes Slurm prolog, epilog, and sbatch examples.

## Current Status

NCCL Doctor is an early but working prototype. It includes:

- A CLI wrapper that injects NCCL debug/topology/flight-recorder collection settings.
- Snapshot helpers for node/fabric counters and host health.
- Parsers for NCCL logs, counter snapshots, flight recorder JSON, dmesg Xid events, `nvidia-smi topo`, NVLink status, and PCIe link state.
- A 17-rule diagnostic engine covering network, topology, algorithm/config, GPU health, and straggler attribution.
- Human-readable and JSON report output.
- Synthetic end-to-end fixtures and unit tests.

The companion design document describes the broader production roadmap, including a node agent, NCCL profiler-plugin shim, RAS queries, ring-buffer counter sampling, and active canary validation.

## Installation

From a checkout:

```bash
python3 -m pip install .
```

For quick local development without installing:

```bash
python3 -m nccl_doctor.cli --help
```

## Quickstart

Wrap a distributed training launch:

```bash
nccl-doctor run \
  --job-dir /shared/nccl-doctor/$SLURM_JOB_ID \
  --job-name llama-70b-pretrain \
  --fabric ib \
  -- \
  torchrun --nnodes 4 --nproc-per-node 8 train.py
```

Analyze after a failure:

```bash
nccl-doctor analyze /shared/nccl-doctor/$SLURM_JOB_ID
nccl-doctor analyze /shared/nccl-doctor/$SLURM_JOB_ID --json
nccl-doctor analyze /shared/nccl-doctor/$SLURM_JOB_ID --out report.json
```

Show fleet reliability scores from the local store:

```bash
nccl-doctor scores
```

Try the synthetic scenarios:

```bash
python3 tests/make_fixtures.py

nccl-doctor analyze demo_jobs/slurm-8839912 --store /tmp/nccl-doctor-store.db
nccl-doctor analyze demo_jobs/slurm-8841273 --store /tmp/nccl-doctor-store.db
nccl-doctor analyze demo_jobs/slurm-8851001 --store /tmp/nccl-doctor-store.db
```

The fabric scenario demonstrates a retry-pass failure: rank 3 reports the timeout, but flight-recorder evidence shows ranks 16-23 on `gpu-201` were the stuck ranks. The report ties the failure to an InfiniBand retry exhaustion and symbol-error storm on a specific fabric path.

## Example Report

A sample human report is available at [examples/sample-report.txt](examples/sample-report.txt).

```text
VERDICT: FABRIC_LINK_DEGRADATION   (confidence 0.97)
  Rank 3 reported the timeout but was not stuck - true origin is ['gpu-201'].

FINDINGS
  [HIGH] NET-03   Symbol errors on gpu-202/mlx5_4:1: +1240 during job
  [HIGH] STRAG-01 Flight recorder: ranks [16, 17, 18, 19, 20, 21, 22, 23] desynced
  [HIGH] NET-01   IB completion error on gpu-201 -> peer 10.0.1.12

RECOMMENDED ACTIONS
  P1 [HARDWARE] Replace cable/transceiver; re-run mlxlink BER check after repair.
  P3 [NCCL_CONFIG] Interim mitigation: NCCL_IB_TIMEOUT=20 and NCCL_IB_RETRY_CNT=10.
```

## Artifact Layout

NCCL Doctor expects a job artifact directory that can be produced by the wrapper, Slurm integrations, or your own collection pipeline:

```text
<job-dir>/
  job.json
  env.json
  logs/<host>.<pid>.log
  topo/<host>.xml
  host/<host>/dmesg.txt
  host/<host>/nvidia-smi-topo.txt
  host/<host>/nvlink.txt
  host/<host>/lspci.txt
  counters/<host>.start.json
  counters/<host>.end.json
  flight_recorder/rank<NN>.json
  report.json
  report.txt
```

The analyzer is intentionally artifact-driven. Slurm, Kubernetes, SSH collection, or a future daemon can all feed the same layout.

## Diagnostic Rules

| Family | Rules | Detects |
| --- | --- | --- |
| Network | `NET-01`..`NET-06` | IB retry exhaustion, link flaps, symbol-error storms, congestion, RoCE PFC storms, ECN/DCQCN issues |
| Topology | `TOPO-01`..`TOPO-05` | Bad GPU/NIC affinity, GDR disabled, degraded NVLink, PCIe downtraining, asymmetric topology |
| Algorithm/config | `ALGO-01`..`ALGO-04` | Socket fallback, suspicious protocol selection, graph fallback, channel-count anomalies |
| Host | `GPU-01` | Critical NVIDIA Xid events |
| Straggler | `STRAG-01` | Flight-recorder desync and true stuck-rank attribution |

Findings are scored and mapped to prioritized actions. Hardware and infrastructure fixes outrank NCCL env-var mitigations.

## Slurm Integration

The [integrations](integrations) directory contains:

- `slurm-prolog.sh` for start snapshots.
- `slurm-epilog.sh` for end snapshots and analysis.
- `sbatch-example.sh` for user-driven wrapping.

Multi-node jobs should write artifacts to a shared filesystem path visible from all participating nodes.

## Repository Layout

```text
nccl_doctor/             Python package and CLI
nccl_doctor/parsers/     Log, counter, flight-recorder, and host parsers
integrations/            Slurm integration scripts
tools/                   Utility scripts
tests/                   Unit tests and synthetic fixtures
docs/design.md           Production design document
examples/sample-report.txt
```

## Development

Run the test suite:

```bash
python3 -m unittest discover -s tests -v
```

Expected today: 12 tests, including two end-to-end synthetic scenarios.

Run the CLI from the source tree:

```bash
python3 -m nccl_doctor.cli analyze demo_jobs/slurm-8841273
```

## Design Notes

The design target is a low-overhead diagnostic stack:

1. Collect cheap artifacts on every job.
2. Trigger deeper capture only on hang or failure.
3. Preserve per-job evidence in a portable artifact directory.
4. Update a cross-run reliability store.
5. Emit a report that platform engineers and training users can both act on.

Read the full design at [docs/design.md](docs/design.md).

## Roadmap

- NCCL profiler-plugin shim for per-proxy-op hang localization.
- Ring-buffer node agent for counter time series before timeout.
- NCCL RAS integration for live stuck-rank queries.
- `nccl-tests` golden-baseline regression checks.
- Kubernetes collection examples.
- Optional Prometheus metrics and Grafana dashboards.
- Tuner plugin generated from measured cluster optima.

## License

No license has been selected yet. Add one before publishing this as an open-source project.
