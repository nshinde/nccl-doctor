"""Cross-run fingerprint store (design doc §3.2).

SQLite single-file DB: trivially operable, fine for thousands of jobs/day.
Implements:
  * run fingerprints (job, nodes, outcome, verdict)
  * component implications (which node/port each finding blamed, with weight)
  * exponentially-decayed reliability scores per component
  * retry explanation: why did the resubmission pass?
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Optional

from .models import Finding, JobMeta, Severity

SEV_WEIGHT = {Severity.INFO: 0.0, Severity.LOW: 0.15, Severity.MEDIUM: 0.4,
              Severity.HIGH: 0.8, Severity.CRITICAL: 1.0}
HALF_LIFE_DAYS = 7.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  job_id TEXT PRIMARY KEY,
  job_name TEXT,
  cluster TEXT,
  ts REAL,
  nodes TEXT,           -- json list
  world_size INTEGER,
  outcome TEXT,
  verdict TEXT
);
CREATE TABLE IF NOT EXISTS implications (
  job_id TEXT,
  ts REAL,
  component TEXT,       -- "gpu-201" or "gpu-201/mlx5_4"
  rule TEXT,
  weight REAL
);
CREATE INDEX IF NOT EXISTS idx_impl_component ON implications(component);
CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(job_name, ts);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    # ------------------------------------------------------------ writes
    def record_run(self, job: JobMeta, verdict: str, findings: list[Finding]) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?)",
            (job.job_id, job.job_name, job.cluster, job.timestamp,
             json.dumps(sorted(job.nodes)), job.world_size, job.outcome, verdict))
        self.db.execute("DELETE FROM implications WHERE job_id=?", (job.job_id,))
        for f in findings:
            w = SEV_WEIGHT.get(f.severity, 0.0) * max(f.confidence, 0.1)
            if w <= 0:
                continue
            for comp in (f.components or f.hosts):
                if comp.startswith("peer:"):
                    continue
                node = comp.split("/")[0]
                self.db.execute(
                    "INSERT INTO implications VALUES (?,?,?,?,?)",
                    (job.job_id, job.timestamp, node, f.rule, w))
                if "/" in comp:
                    self.db.execute(
                        "INSERT INTO implications VALUES (?,?,?,?,?)",
                        (job.job_id, job.timestamp, comp, f.rule, w))
        self.db.commit()

    # ------------------------------------------------------------ scores
    def node_scores(self, now: Optional[float] = None) -> dict[str, float]:
        """Decayed sum of implication weights per node (not per port)."""
        now = now or time.time()
        scores: dict[str, float] = {}
        for comp, ts, w in self.db.execute(
                "SELECT component, ts, weight FROM implications"):
            if "/" in comp:
                continue
            age_days = max(0.0, (now - ts) / 86400.0)
            scores[comp] = scores.get(comp, 0.0) + w * math.pow(0.5, age_days / HALF_LIFE_DAYS)
        return scores

    def fleet_median(self, all_nodes: list[str]) -> float:
        scores = self.node_scores()
        vals = sorted(scores.get(n, 0.0) for n in all_nodes) or [0.0]
        return vals[len(vals) // 2]

    # --------------------------------------------------- retry explanation
    def retry_explanation(self, job: JobMeta,
                          implicated_nodes: set[str]) -> Optional[str]:
        """If a sibling run of the same job_name succeeded on a different node
        set, explain the pass/fail difference in terms of implicated nodes."""
        if not job.job_name:
            return None
        rows = list(self.db.execute(
            "SELECT job_id, ts, nodes, outcome FROM runs "
            "WHERE job_name=? AND job_id<>? ORDER BY ts DESC LIMIT 10",
            (job.job_name, job.job_id)))
        for sib_id, _ts, nodes_json, outcome in rows:
            if outcome != "SUCCESS":
                continue
            sib_nodes = set(json.loads(nodes_json))
            avoided = implicated_nodes - sib_nodes
            if avoided:
                return (f"Sibling run {sib_id} of the same job succeeded on a node set "
                        f"that did not include {sorted(avoided)} — the implicated "
                        f"hardware. The failure is placement-correlated, not random.")
            if sib_nodes != set(job.nodes):
                return (f"Sibling run {sib_id} succeeded on a different node set "
                        f"({sorted(sib_nodes ^ set(job.nodes))} differ); failure is "
                        f"likely placement- or transient-fabric-correlated.")
            return (f"Sibling run {sib_id} succeeded on the SAME node set — points at "
                    f"a transient fabric event or congestion rather than a fixed "
                    f"hardware fault.")
        return None

    def close(self) -> None:
        self.db.close()
