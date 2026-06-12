#!/usr/bin/env python3
"""Convert PyTorch NCCL flight-recorder dumps (pickle) into nccl-doctor JSON.

torch writes one dump per rank at TORCH_NCCL_DEBUG_INFO_TEMP_FILE + <rank>
when TORCH_NCCL_DUMP_ON_TIMEOUT=1 fires. Format drifts across torch versions,
so extraction is defensive: we only need (seq, op, state) per entry.

Usage: fr_export.py <dump_file_or_glob>... --out flight_recorder/
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path


def convert(path: Path) -> dict | None:
    try:
        with path.open("rb") as f:
            d = pickle.load(f)
    except Exception as e:
        print(f"skip {path}: {e}", file=sys.stderr)
        return None
    m = re.search(r"(\d+)$", path.name)
    rank = int(m.group(1)) if m else -1
    entries = []
    for e in d.get("entries", []) if isinstance(d, dict) else []:
        seq = e.get("seq_id", e.get("seq", -1))
        op = e.get("profiling_name", e.get("op", "")) or ""
        op = op.split(":")[-1].split("(")[0].upper().replace("NCCL_", "") or "UNKNOWN"
        state = str(e.get("state", "")).lower()
        if "complete" in state:
            state = "completed"
        elif "start" in state or "progress" in state:
            state = "started"
        else:
            state = "scheduled"
        entries.append({"seq": int(seq), "op": op, "state": state})
    return {"rank": rank, "entries": entries}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in args.dumps:
        d = convert(p)
        if d and d["rank"] >= 0:
            (args.out / f"rank{d['rank']:02d}.json").write_text(json.dumps(d))
            n += 1
    print(f"exported {n} rank dump(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
