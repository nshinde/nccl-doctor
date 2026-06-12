"""Unit tests (stdlib unittest — no pytest dependency on cluster nodes).

Run: python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nccl_doctor.models import JobMeta
from nccl_doctor.parsers.nccl_log import LogFacts, parse_log_file
from nccl_doctor.parsers.host import (parse_dmesg_xids, parse_topo_matrix,
                                      parse_lspci_downtraining,
                                      parse_nvlink_active_counts)
from nccl_doctor.parsers.counters import analyze_flight_recorder
from nccl_doctor.rules import EvidenceBundle, REGISTRY
from nccl_doctor.parsers.host import HostFacts
from nccl_doctor.store import Store
from nccl_doctor.report import analyze_job

from make_fixtures import (make_scenario_fabric, make_passing_sibling,
                           make_scenario_config, TOPO_MATRIX_BAD, DMESG_XID79,
                           NVLINK_DEGRADED)


def _tmp(text: str) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
    f.write(text)
    f.close()
    return Path(f.name)


class TestNcclLogParser(unittest.TestCase):
    def test_completion_error_and_version(self):
        facts = LogFacts()
        parse_log_file(_tmp(
            "gpu-201:9981:10021 [2] NCCL INFO NCCL version 2.26.2+cuda12.4\n"
            "gpu-201:9981:10021 [2] NCCL WARN NET/IB : Got completion from peer "
            "10.0.1.12<48133> with error 12, opcode 0, len 32768, vendor err 129 (Recv)\n"
        ), facts)
        self.assertIn("2.26.2", facts.nccl_versions)
        self.assertEqual(len(facts.completion_errors), 1)
        ce = facts.completion_errors[0]
        self.assertEqual((ce.host, ce.peer_ip, ce.status, ce.vendor_err),
                         ("gpu-201", "10.0.1.12", "12", "129"))
        self.assertIn("retry counter exceeded", ce.status_name)

    def test_gdr_vs_nogdr_paths_and_socket(self):
        facts = LogFacts()
        parse_log_file(_tmp(
            "h1:1:2 [0] NCCL INFO Using network Socket\n"
            "h1:1:2 [0] NCCL INFO Channel 00/0 : 4[4] -> 0[0] [receive] via NET/IB/0\n"
            "h1:1:2 [0] NCCL INFO Channel 01/0 : 0[0] -> 4[4] [send] via NET/IB/1/GDRDMA\n"
        ), facts)
        self.assertEqual(facts.networks["h1"], {"Socket"})
        self.assertEqual(facts.net_paths_nogdr["h1"], 1)
        self.assertEqual(facts.net_paths_gdr["h1"], 1)

    def test_watchdog(self):
        facts = LogFacts()
        parse_log_file(_tmp(
            "[E ProcessGroupNCCL.cpp:616] [Rank 3] Watchdog caught collective "
            "operation timeout: WorkNCCL(SeqNum=48211, OpType=ALLREDUCE)\n"), facts)
        self.assertEqual(facts.watchdog_events[0].rank, 3)
        self.assertEqual(facts.watchdog_events[0].seq, 48211)


class TestHostParsers(unittest.TestCase):
    def test_xid(self):
        ev = parse_dmesg_xids(_tmp(DMESG_XID79), "gpu-311")
        self.assertEqual(ev[0].xid, "79")
        self.assertTrue(ev[0].critical)

    def test_topo_matrix_bad_affinity(self):
        tm = parse_topo_matrix(_tmp(TOPO_MATRIX_BAD), "gpu-310")
        self.assertIsNotNone(tm)
        bad = tm.gpus_without_local_nic()
        self.assertEqual(bad[0][0], "GPU0")

    def test_nvlink_counts(self):
        nl = parse_nvlink_active_counts(_tmp(NVLINK_DEGRADED))
        self.assertEqual(nl["GPU 0"], (16, 18))

    def test_lspci_downtrain(self):
        text = ("1b:00.0 3D controller: NVIDIA Corporation GH100\n"
                "\tLnkCap:\tPort #0, Speed 32GT/s, Width x16\n"
                "\tLnkSta:\tSpeed 16GT/s (downgraded), Width x8\n")
        dt = parse_lspci_downtraining(_tmp(text), "h1")
        self.assertEqual(len(dt), 1)
        self.assertIn("x8", dt[0].sta)


class TestStraggler(unittest.TestCase):
    def test_stuck_ranks(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for r in range(4):
                stuck = r in (2, 3)
                entries = [{"seq": 9, "op": "ALLREDUCE", "state": "completed"}]
                if not stuck:
                    entries.append({"seq": 10, "op": "ALLREDUCE", "state": "completed"})
                (d / f"rank{r:02d}.json").write_text(
                    json.dumps({"rank": r, "entries": entries}))
            sa = analyze_flight_recorder(d)
            self.assertEqual(sa.stuck_ranks, [2, 3])
            self.assertEqual(sa.frontier_seq, 10)
            self.assertTrue(sa.conclusive)


class TestRulesRegistry(unittest.TestCase):
    def test_expected_rules_registered(self):
        for rid in ("NET-01", "NET-02", "NET-03", "NET-04", "NET-05", "NET-06",
                    "TOPO-01", "TOPO-02", "TOPO-03", "TOPO-04", "TOPO-05",
                    "ALGO-01", "ALGO-02", "ALGO-03", "ALGO-04",
                    "GPU-01", "STRAG-01"):
            self.assertIn(rid, REGISTRY)

    def test_rule_crash_is_contained(self):
        from nccl_doctor import rules as R
        def boom(ctx):
            raise RuntimeError("boom")
        R.REGISTRY["ZZZ-99"] = boom
        try:
            ctx = EvidenceBundle(job=JobMeta("j", [], 1), logs=LogFacts(),
                                 hosts=HostFacts())
            findings = R.run_all(ctx)
            self.assertTrue(any(f.rule == "ZZZ-99" for f in findings))
        finally:
            del R.REGISTRY["ZZZ-99"]


class TestEndToEnd(unittest.TestCase):
    def test_fabric_scenario_verdict_and_retry_explanation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_passing_sibling(root)
            fail = make_scenario_fabric(root)
            store = Store(root / "store.db")
            analyze_job(root / "slurm-8839912", store)
            report = analyze_job(fail, store)
            store.close()
            self.assertEqual(report["error_analysis"]["verdict"],
                             "FABRIC_LINK_DEGRADATION")
            rules = {f["rule"] for f in report["error_analysis"]["findings"]}
            self.assertLessEqual({"NET-01", "NET-03", "STRAG-01"}, rules)
            origin = report["error_analysis"]["origin"]
            self.assertIn("gpu-201", origin["origin_hosts"])
            self.assertIn("placement-correlated", report["retry_explanation"])
            # the reporting rank must be identified as messenger, not cause
            self.assertNotIn(3, origin["true_origin_ranks"])

    def test_config_scenario(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            job = make_scenario_config(root)
            report = analyze_job(job, None)
            rules = {f["rule"] for f in report["error_analysis"]["findings"]}
            self.assertLessEqual({"GPU-01", "ALGO-01", "TOPO-02", "TOPO-03",
                                  "TOPO-05", "ALGO-04", "TOPO-01"}, rules)
            self.assertEqual(report["error_analysis"]["verdict"],
                             "GPU_HARDWARE_FAULT")
            # outlier must be the node with FEWER nics
            t5 = next(f for f in report["error_analysis"]["findings"]
                      if f["rule"] == "TOPO-05")
            self.assertIn("gpu-310", t5["hosts"])


if __name__ == "__main__":
    unittest.main()
