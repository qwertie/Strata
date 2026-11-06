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


def test_logical_members_survive_the_engine_protocol():
    engine = StrataEngine("missing-executable", [], lazy=True)
    lines = [["MEMORY", "ram"],
             ["MEMORY_BLOCK", "ram", "200", "experts", "Pool", "Layer%202", "expert%2F2", "7,8"],
             ["SETTINGS"]]
    with mock.patch.object(engine, "_ask", return_value=iter(lines)):
        block = engine.memory()["memories"][0]["blocks"][0]
    assert block["identity_group"] == "expert/2"
    assert block["ids"] == [7, 8]


def test_live_token_limits_update_request_validation_only_for_engines_that_advertise_them():
    engine = StrataEngine("missing-executable", [], lazy=True)
    engine.max_context = 204800
    engine.info.update(context_capacity=204800, tokens_max=401000)
    lines = [["SETTINGS", "context=180000", "context_capacity=204800", "tokens_max=350000"]]
    with mock.patch.object(engine, "_ask", return_value=iter(lines)):
        report = engine.change_settings({"context": 180000, "tokens_max": 350000})
    assert report["settings"]["context_capacity"] == 204800
    assert engine.max_context == 180000
    assert engine.info["tokens_max"] == 350000
    engine.info.clear()
    with mock.patch.object(engine, "_ask", return_value=iter([["SETTINGS", "context=4096", "tokens_max=8192"]])):
        engine.memory()
    assert engine.max_context == 180000
    assert "tokens_max" not in engine.info


class MemoryLabels(FakeEngineCase):
    def test_packed_model_heading_follows_the_manifest_to_its_first_gguf(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        self.start(2)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            pack = root / "pack"
            pack.mkdir()
            (pack / "ds4.txt").write_text("model_dir models\n", encoding="utf-8")
            (pack / "tensors.txt").write_text("file 0 first shard.gguf\nfile 1 second.gguf\n", encoding="utf-8")
            self.engine.spawn = ("engine", [], folder, None, None, False, None)
            self.engine.model_path = "pack"
            with mock.patch("strata_common.gguf_reader.GGUFFile") as read:
                read.return_value.metadata = {"general.name": "DeepSeek"}
                details = self.svc.model_details
            read.assert_called_once_with(root / "models" / "first shard.gguf")
            assert details["name"] == "DeepSeek"
            assert details["model_path"] == str(root / "models" / "first shard.gguf")

    def test_model_heading_uses_the_header_and_resolved_source_paths(self):
        from pathlib import Path

        self.start(2)
        self.engine.spawn = ("engine/strata-q38.exe", [], str(Path.cwd()), None, None, False, None)
        self.engine.model_path = "models/first-shard.gguf"
        self.svc.config_path = "model.json"
        with mock.patch("strata_common.gguf_reader.GGUFFile") as read:
            read.return_value.metadata = {"general.name": "Qwen3.8 Flash Next"}
            details = self.svc.model_details
            assert self.svc.model_details == details
        read.assert_called_once_with(Path("models/first-shard.gguf").resolve())
        assert details == {"name": "Qwen3.8 Flash Next", "engine": "strata-q38.exe",
                           "config_path": str(Path("model.json").resolve()),
                           "model_path": str(Path("models/first-shard.gguf").resolve())}

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


def test_conversation_activity_and_pause_folder_survive_the_protocol():
    from strata_common.engine_client import parse_conversation_line

    line = "CONVERSATION parked1 disk parked 3 3 200 200 last_tokens_at=1700000000 1,2,3".split()
    conversation = parse_conversation_line(line)
    assert conversation["last_tokens_at"] == 1700000000
    assert conversation["preview"] == [1, 2, 3]
    engine = StrataEngine("missing-executable", [], lazy=True)
    lines = [["MEMORY", "ssd"], line, ["PAUSE_DIRECTORY", "C:/pause", "files"], ["SETTINGS"]]
    with mock.patch.object(engine, "_ask", return_value=iter(lines)):
        report = engine.memory()
    assert report["pause_directory"] == "C:/pause files"
    assert report["conversations"][0]["last_tokens_at"] == 1700000000
