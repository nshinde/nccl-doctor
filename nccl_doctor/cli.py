"""nccl-doctor CLI.

  nccl-doctor run --job-dir DIR [--job-name N] [--fabric ib|roce] -- CMD...
  nccl-doctor snapshot --job-dir DIR --phase start|end
  nccl-doctor analyze JOB_DIR [--json] [--store PATH] [--no-store]
  nccl-doctor scores [--store PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import collect
from .report import analyze_job, render_text
from .store import Store

DEFAULT_STORE = Path.home() / ".nccl_doctor" / "store.db"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # split `run ... -- cmd` manually so argparse never eats the command
    cmd_tail: list[str] = []
    if argv and argv[0] == "run" and "--" in argv:
        i = argv.index("--")
        argv, cmd_tail = argv[:i], argv[i + 1:]

    ap = argparse.ArgumentParser(prog="nccl-doctor",
                                 description="NCCL failure diagnosis & tuning")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="wrap a training launch with full collection")
    p_run.add_argument("--job-dir", required=True, type=Path)
    p_run.add_argument("--job-name", default=None)
    p_run.add_argument("--fabric", choices=["ib", "roce"], default="ib")

    p_snap = sub.add_parser("snapshot", help="capture node state (prolog/epilog)")
    p_snap.add_argument("--job-dir", required=True, type=Path)
    p_snap.add_argument("--phase", choices=["start", "end"], required=True)

    p_an = sub.add_parser("analyze", help="analyze a job artifact directory")
    p_an.add_argument("job_dir", type=Path)
    p_an.add_argument("--json", action="store_true", help="emit JSON instead of text")
    p_an.add_argument("--out", type=Path, default=None, help="also write JSON here")
    p_an.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p_an.add_argument("--no-store", action="store_true",
                      help="skip the cross-run fingerprint store")

    p_sc = sub.add_parser("scores", help="show node reliability scores")
    p_sc.add_argument("--store", type=Path, default=DEFAULT_STORE)

    args = ap.parse_args(argv)

    if args.cmd == "run":
        if not cmd_tail:
            ap.error("run requires `-- <command>`")
        return collect.run_wrapped(args.job_dir, cmd_tail, args.job_name, args.fabric)

    if args.cmd == "snapshot":
        collect.snapshot(args.job_dir, args.phase)
        return 0

    if args.cmd == "analyze":
        store = None if args.no_store else Store(args.store)
        try:
            report = analyze_job(args.job_dir, store)
        finally:
            if store:
                store.close()
        payload = json.dumps(report, indent=2)
        if args.out:
            args.out.write_text(payload)
        print(payload if args.json else render_text(report))
        return 0

    if args.cmd == "scores":
        store = Store(args.store)
        scores = store.node_scores()
        store.close()
        if not scores:
            print("no implications recorded yet")
            return 0
        for n, s in sorted(scores.items(), key=lambda kv: -kv[1]):
            print(f"{n:20s} {s:.3f}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
