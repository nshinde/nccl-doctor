"""Core data models for nccl-doctor.

Stdlib-only (dataclasses) by design: this tool must run on air-gapped HPC
login/compute nodes with nothing but a system Python.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return ["info", "low", "medium", "high", "critical"].index(self.value)


class Track(str, Enum):
    """Which remediation track a finding routes to (design doc §3.2)."""
    HARDWARE = "HARDWARE"          # physical signal — never fixable by env vars
    INFRASTRUCTURE = "INFRASTRUCTURE"  # fabric/switch/host config (QoS, ACS, BIOS)
    NCCL_CONFIG = "NCCL_CONFIG"    # env var / launcher tuning
    SCHEDULER = "SCHEDULER"        # cordon / drain / placement
    APPLICATION = "APPLICATION"    # framework-side (timeouts, dataloader)


@dataclass
class Evidence:
    source: str          # e.g. "nccl_log:gpu-201.9981.log:L482"
    detail: str          # the actual line / counter delta / fact
    host: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Finding:
    rule: str                    # e.g. "NET-01"
    severity: Severity
    track: Track
    summary: str
    evidence: list[Evidence] = field(default_factory=list)
    hosts: list[str] = field(default_factory=list)       # implicated hosts
    components: list[str] = field(default_factory=list)  # e.g. "gpu-201/mlx5_4"
    confidence: float = 0.6
    data: dict[str, Any] = field(default_factory=dict)   # rule-specific extras

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "severity": self.severity.value,
            "track": self.track.value,
            "summary": self.summary,
            "confidence": round(self.confidence, 2),
            "hosts": self.hosts,
            "components": self.components,
            "evidence": [e.to_dict() for e in self.evidence],
            **({"data": self.data} if self.data else {}),
        }


@dataclass
class Action:
    type: Track
    priority: int
    action: str
    rule: str
    owner: Optional[str] = None
    command: Optional[str] = None
    caveat: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"type": self.type.value, "priority": self.priority,
             "action": self.action, "rule": self.rule}
        for k in ("owner", "command", "caveat"):
            v = getattr(self, k)
            if v:
                d[k] = v
        return d


@dataclass
class JobMeta:
    job_id: str
    nodes: list[str]
    world_size: int
    outcome: str = "UNKNOWN"            # SUCCESS | WATCHDOG_TIMEOUT | NCCL_ERROR | CRASH
    cluster: str = "default"
    job_name: Optional[str] = None      # used to correlate retries of "the same job"
    framework: Optional[str] = None
    nccl_version: Optional[str] = None
    first_error_rank: Optional[int] = None
    expected_nvlinks: Optional[int] = None   # per-GPU active links for this SKU
    fabric: str = "ib"                  # "ib" | "roce"
    timestamp: float = field(default_factory=time.time)
    rank_to_host: dict[str, str] = field(default_factory=dict)  # "17" -> "gpu-201"

    @classmethod
    def from_dict(cls, d: dict) -> "JobMeta":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def host_of_rank(self, rank: int) -> Optional[str]:
        return self.rank_to_host.get(str(rank))


@dataclass
class Verdict:
    verdict: str
    confidence: float
    origin: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "confidence": round(self.confidence, 2),
                "origin": self.origin}
