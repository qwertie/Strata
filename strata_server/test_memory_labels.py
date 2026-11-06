"""Checks allocation labels across the engine protocol and the server memory view."""
from unittest import mock

from strata_common.engine_client import StrataEngine
from strata_server.test_parallel import FakeEngineCase


def test_arbitrary_labels_survive_the_engine_protocol():
    engine = StrataEngine("missing-executable", [], lazy=True)
    lines = [["MEMORY", "gpu0", "total=1000", "free=100"],
             ["MEMORY_BLOCK", "gpu0", "250", "New%20category", "Pool%20%2F%20one", "Drafter%20%26%20buffers"],
             ["MEMORY_BLOCK", "gpu0", "150", "New%20category", "Pool%20two", "%E5%B1%A4%201"],
             ["SETTINGS", "rows=8"]]
    with mock.patch.object(engine, "_ask", return_value=iter(lines)) as ask:
        report = engine.memory()
    memory = report["memories"][0]
    assert memory["uses"] == {"New category": 400}
    assert memory["blocks"] == [
        {"category": "New category", "pool": "Pool / one", "label": "Drafter & buffers", "bytes": 250},
        {"category": "New category", "pool": "Pool two", "label": "\u5c64 1", "bytes": 150}]
    assert "MEMORY_BLOCK" in ask.call_args.kwargs["also"]


class MemoryLabels(FakeEngineCase):
    def test_runtime_bytes_belong_to_strata_and_other_apps_stay_separate(self):
        self.start(2, more=["--settings"])
        source = self.engine.memory()
        gpu = source["memories"][0]
        gpu.update(total=1000, free=100, unified=True, name="Test GPU", uses={"other": 600},
                   blocks=[{"category": "other", "pool": "Drafter", "label": "Weights", "bytes": 600}])
        split = {"share": 1000, "share_used": 900, "in_share": 700, "borrowed": 0,
                 "borrow_limit": 0, "borrowed_by": "engine"}
        with mock.patch("strata_common.telemetry.read_shared_gpu_memory", return_value=split):
            first = self.svc.memory_view(source)["memories"][0]
            second = self.svc.memory_view(source)["memories"][0]
        assert first["uses"]["other"] == 700
        assert first["other_apps"] == 200
        assert first["blocks"][0]["pool"] == "Drafter"
        assert first == second
        assert gpu["uses"]["other"] == 600
        assert len(gpu["blocks"]) == 1
