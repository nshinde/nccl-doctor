"""Parsers for NCCL topo XML dumps and host-level artifacts.

Host artifact directory layout (produced by `nccl-doctor snapshot`):

  host/<hostname>/dmesg.txt            # `dmesg -T`
  host/<hostname>/nvidia-smi-topo.txt  # `nvidia-smi topo -m`
  host/<hostname>/nvlink.txt           # `nvidia-smi nvlink --status`
  host/<hostname>/lspci.txt            # `lspci -vvv`
  topo/<hostname>.xml                  # NCCL_TOPO_DUMP_FILE
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ------------------------------------------------------------ topo XML
@dataclass
class TopoSignature:
    host: str
    gpu_count: int = 0
    nic_count: int = 0
    nics: list[str] = field(default_factory=list)        # dev names
    nic_speeds: list[str] = field(default_factory=list)  # speed attrs if present
    pci_link_attrs: list[str] = field(default_factory=list)

    def key(self) -> tuple:
        """Comparable signature: same-SKU hosts must match on this."""
        return (self.gpu_count, self.nic_count, tuple(sorted(self.nic_speeds)))


def parse_topo_xml(path: Path) -> Optional[TopoSignature]:
    try:
        root = ET.fromstring(path.read_text(errors="replace"))
    except (ET.ParseError, OSError):
        return None
    sig = TopoSignature(host=path.stem)
    for el in root.iter():
        tag = el.tag.lower()
        if tag == "gpu":
            sig.gpu_count += 1
        elif tag in ("nic", "net"):
            if tag == "net" or not list(el.iter("net")):
                sig.nic_count += 1
                name = el.get("name") or el.get("dev") or ""
                if name:
                    sig.nics.append(name)
                speed = el.get("speed") or el.get("latency") or ""
                if speed:
                    sig.nic_speeds.append(speed)
        elif tag == "pcilink":
            sig.pci_link_attrs.append(f"{el.get('width','?')}x@{el.get('speed','?')}")
    return sig


def parse_all_topo(topo_dir: Path) -> dict[str, TopoSignature]:
    out: dict[str, TopoSignature] = {}
    if not topo_dir.is_dir():
        return out
    for p in sorted(topo_dir.glob("*.xml")):
        sig = parse_topo_xml(p)
        if sig:
            out[sig.host] = sig
    return out


# ------------------------------------------------------ nvidia-smi topo -m
GOOD_GPU_NIC = {"PIX", "PXB"}
BAD_GPU_NIC = {"PHB", "NODE", "SYS"}


@dataclass
class TopoMatrix:
    host: str
    # gpu -> {nic_name: link_type}
    gpu_nic_links: dict[str, dict[str, str]] = field(default_factory=dict)

    def gpus_without_local_nic(self) -> list[tuple[str, str]]:
        """GPUs whose *best* NIC link still crosses the CPU complex."""
        bad = []
        for gpu, links in self.gpu_nic_links.items():
            if not links:
                continue
            if not any(v in GOOD_GPU_NIC for v in links.values()):
                best = min(links.items(), key=lambda kv: _link_badness(kv[1]))
                bad.append((gpu, f"best NIC link is {best[1]} via {best[0]}"))
        return bad


def _link_badness(link: str) -> int:
    order = ["NV", "PIX", "PXB", "PHB", "NODE", "SYS"]
    for i, p in enumerate(order):
        if link.startswith(p):
            return i
    return len(order)


def parse_topo_matrix(path: Path, host: str) -> Optional[TopoMatrix]:
    try:
        lines = [l for l in path.read_text(errors="replace").splitlines() if l.strip()]
    except OSError:
        return None
    header = None
    for l in lines:
        toks = l.split()
        if "GPU0" in toks and any(t.startswith(("NIC", "mlx")) for t in toks):
            header = toks
            break
    if not header:
        return None
    nic_cols = [(i, name) for i, name in enumerate(header) if name.startswith(("NIC", "mlx"))]
    tm = TopoMatrix(host=host)
    for l in lines:
        toks = l.split()
        if not toks or not toks[0].startswith("GPU"):
            continue
        gpu, vals = toks[0], toks[1:]
        tm.gpu_nic_links[gpu] = {}
        for i, name in nic_cols:
            if i < len(vals):
                tm.gpu_nic_links[gpu][name] = vals[i]
    return tm


# ------------------------------------------------------------ dmesg Xid
RE_XID = re.compile(r"NVRM: Xid \((PCI:[0-9a-fA-F:.]+)\):\s*(\d+),?\s*(.*)")

CRITICAL_XIDS = {
    "48": "Double-bit ECC error",
    "63": "ECC page retirement / row remap (pending)",
    "64": "ECC page retirement / row remap failure",
    "74": "NVLink error",
    "79": "GPU has fallen off the bus",
    "92": "High single-bit ECC error rate",
    "94": "Contained ECC error",
    "95": "Uncontained ECC error",
    "119": "GSP RPC timeout",
    "120": "GSP error",
}


@dataclass
class XidEvent:
    host: str
    pci: str
    xid: str
    detail: str

    @property
    def meaning(self) -> str:
        return CRITICAL_XIDS.get(self.xid, f"Xid {self.xid}")

    @property
    def critical(self) -> bool:
        return self.xid in CRITICAL_XIDS


def parse_dmesg_xids(path: Path, host: str) -> list[XidEvent]:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    return [XidEvent(host=host, pci=m.group(1), xid=m.group(2), detail=m.group(3).strip())
            for m in RE_XID.finditer(text)]


# ------------------------------------------------------------ nvlink
RE_NVLINK = re.compile(r"Link (\d+):\s*(.+)")


def parse_nvlink_active_counts(path: Path) -> dict[str, tuple[int, int]]:
    """Return {gpu_label: (active, total)} from `nvidia-smi nvlink --status`."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return {}
    out: dict[str, tuple[int, int]] = {}
    cur = None
    for line in text.splitlines():
        if line.startswith("GPU"):
            cur = line.split(":")[0].strip()
            out[cur] = (0, 0)
        elif cur and (m := RE_NVLINK.search(line)):
            a, t = out[cur]
            t += 1
            if "inactive" not in m.group(2).lower() and "<" not in m.group(2):
                a += 1
            out[cur] = (a, t)
    return out


# ------------------------------------------------------------ lspci downtraining
RE_LNKCAP = re.compile(r"LnkCap:.*?Speed (\S+),\s*Width x(\d+)")
RE_LNKSTA = re.compile(r"LnkSta:.*?Speed (\S+)(?:\s*\(downgraded\))?,\s*Width x(\d+)")
RE_DEVHDR = re.compile(r"^([0-9a-fA-F:.]+)\s+(.+)$")


@dataclass
class PcieDowntrain:
    host: str
    device: str
    cap: str
    sta: str


def parse_lspci_downtraining(path: Path, host: str) -> list[PcieDowntrain]:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    results: list[PcieDowntrain] = []
    dev, cap = None, None
    for line in text.splitlines():
        if not line.startswith((" ", "\t")):
            m = RE_DEVHDR.match(line.strip())
            dev, cap = (f"{m.group(1)} {m.group(2)[:60]}" if m else line.strip()[:70]), None
            continue
        if (m := RE_LNKCAP.search(line)):
            cap = (m.group(1), int(m.group(2)))
        elif (m := RE_LNKSTA.search(line)) and cap and dev:
            sta = (m.group(1), int(m.group(2)))
            cap_gt = _gts(cap[0])
            sta_gt = _gts(sta[0])
            relevant = any(k in dev.lower() for k in ("nvidia", "mellanox", "infiniband",
                                                      "ethernet controller", "3d controller"))
            if relevant and (sta[1] < cap[1] or (cap_gt and sta_gt and sta_gt < cap_gt)):
                results.append(PcieDowntrain(
                    host=host, device=dev,
                    cap=f"{cap[0]} x{cap[1]}", sta=f"{sta[0]} x{sta[1]}"))
            cap = None
    return results


def _gts(s: str) -> Optional[float]:
    m = re.match(r"([\d.]+)GT/s", s)
    return float(m.group(1)) if m else None


# ------------------------------------------------------------ bundle
@dataclass
class HostFacts:
    xids: dict[str, list[XidEvent]] = field(default_factory=dict)
    topo_matrices: dict[str, TopoMatrix] = field(default_factory=dict)
    nvlink: dict[str, dict[str, tuple[int, int]]] = field(default_factory=dict)
    downtrains: dict[str, list[PcieDowntrain]] = field(default_factory=dict)


def parse_host_dir(host_root: Path) -> HostFacts:
    hf = HostFacts()
    if not host_root.is_dir():
        return hf
    for hdir in sorted(p for p in host_root.iterdir() if p.is_dir()):
        host = hdir.name
        if (p := hdir / "dmesg.txt").exists():
            ev = parse_dmesg_xids(p, host)
            if ev:
                hf.xids[host] = ev
        if (p := hdir / "nvidia-smi-topo.txt").exists():
            tm = parse_topo_matrix(p, host)
            if tm:
                hf.topo_matrices[host] = tm
        if (p := hdir / "nvlink.txt").exists():
            nl = parse_nvlink_active_counts(p)
            if nl:
                hf.nvlink[host] = nl
        if (p := hdir / "lspci.txt").exists():
            dt = parse_lspci_downtraining(p, host)
            if dt:
                hf.downtrains[host] = dt
    return hf
