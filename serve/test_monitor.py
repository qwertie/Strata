"""Request inspection through HTTP; no native model or GPU is required."""
import json
from array import array
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, serve


class ReloadableEngine(MockEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running, self.unloaded = True, False

    def alive(self):
        return self.running

    def unload(self):
        self.running, self.unloaded = False, True

    def restart(self):
        self.running, self.unloaded = True, False


class MonitorOff(unittest.TestCase):
    """#332 is opt-in: by default no prompt or answer is kept, and the monitor's endpoints do not exist."""

    def test_off_by_default(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "Hello.", max_context=4096), tok,
                      ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"messages": [{"role": "user", "content": "secret"}]}).encode())
            with urllib.request.urlopen(req, timeout=10) as r:
                self.assertEqual(r.status, 200)
            self.assertEqual(len(svc.api_requests), 0)
            with urllib.request.urlopen(base + "/health", timeout=10) as r:
                self.assertNotIn("api_monitor", json.loads(r.read()))   # the default answer is unchanged
            for path in ("/api/requests", "/api-monitor", "/api/context"):
                with self.assertRaises(urllib.error.HTTPError) as e:
                    urllib.request.urlopen(base + path, timeout=10)
                self.assertEqual(e.exception.code, 404, path)
                e.exception.close()
        finally:
            httpd.shutdown()
            httpd.server_close()


class Monitor(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = ReloadableEngine(self.tok, "Thought.</think>\nHello <script>alert(1)</script>.", max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.api_monitor = True                       # opt-in ("api_monitor": true / --api-monitor)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read().decode()
            return response.status, json.loads(raw) if "application/json" in response.headers.get("Content-Type", "") else raw

    def chat(self, api="openai", **kwargs):
        # Anthropic thinks only when asked (#278): "thinking" there, reasoning_effort for OpenAI
        think = {"thinking": {"type": "enabled", "budget_tokens": 1024}} if api == "anthropic" else             {"reasoning_effort": "low"}
        return self.request("/v1/messages" if api == "anthropic" else "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "Say hello."}],
                             "max_tokens": 256, **think, **kwargs})

    def latest(self):
        summary = self.request("/api/requests")[1]["requests"][0]
        return self.request("/api/requests?id=" + summary["id"])[1]

    def test_output_and_reasoning_for_both_dialects_and_stream_modes(self):
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    code, reply = self.chat(api, stream=stream)
                    self.assertEqual(code, 200)
                    record = self.latest()
                    self.assertEqual(record["state"], "completed")
                    self.assertEqual(record["http_status"], 200)
                    self.assertIn("Say hello.", record["input"])
                    self.assertIn("<script>alert(1)</script>", record["output"])
                    self.assertIn("Thought.", record["reasoning"])
                    self.assertGreater(record["usage"].get("completion_tokens", record["usage"].get("output_tokens", 0)), 0)
                    self.assertGreaterEqual(record["wallclock_s"], record["first_token_s"])
                    self.assertNotIn("_clock", record)
                    if not stream:
                        self.assertEqual(json.loads(record["response"]), reply)
                    else:
                        self.assertNotIn("response", record)

    def test_error_and_engine_death_are_not_successes(self):
        # a request the server refuses is an error record: the strict mode (fit_max_tokens off) refuses a max_tokens
        # that overshoots the context, the default shortens it (serve/test_server.py MaxTokens)
        self.svc.fit_max_tokens = False
        self.assertEqual(self.chat(max_tokens=20000)[0], 400)
        self.svc.fit_max_tokens = True
        self.assertEqual(self.latest()["state"], "error")
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream), mock.patch.object(self.engine, "generate", side_effect=EngineDied("stopped")):
                    self.chat(api, stream=stream)
                    record = self.latest()
                    self.assertEqual(record["state"], "error")
                    self.assertIn("stopped", record["error"]["message"])
                    self.assertEqual(record["output"], "")
                    self.assertNotIn("timings", record)

    def test_history_is_authorized_and_does_not_capture_headers(self):
        self.svc.api_key = "secret"
        auth = {"Authorization": "Bearer secret"}
        self.assertEqual(self.request("/api/requests")[0], 401)
        self.assertEqual(self.request("/api/requests?id=missing")[0], 401)
        self.assertEqual(len(self.svc.api_requests), 0)
        self.assertEqual(self.request("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, auth)[0], 200)
        summary = self.request("/api/requests", headers=auth)[1]["requests"][0]
        self.assertNotIn("input", summary)
        self.assertNotIn("output", summary)
        self.assertEqual(self.request("/api/requests?id=" + summary["id"])[0], 401)
        detail = self.request("/api/requests?id=" + summary["id"], headers=auth)[1]
        self.assertNotIn("secret", json.dumps(detail))

    def test_capture_is_bounded_and_eviction_returns_404(self):
        huge = "x" * 262200
        first = self.svc.begin_request("/v1/chat/completions", {"messages": [{"content": huge}]})
        self.assertEqual(len(first["input"]), 262144)
        self.assertTrue(first["input_truncated"])
        for _ in range(100):
            self.svc.begin_request("/v1/chat/completions", {"messages": []})
        self.assertEqual(len(self.request("/api/requests")[1]["requests"]), 100)
        self.assertEqual(self.request("/api/requests?id=" + first["id"])[0], 404)
        # Exercise output capture without making a mock engine emit 262K tokens.
        handler = object.__new__(self.httpd.RequestHandlerClass)
        handler.record = self.svc.api_requests[-1]
        chunk = {"choices": [{"delta": {"content": huge, "reasoning_content": huge}}]}
        list(handler._capture((item for item in [chunk]), "openai"))
        for field in ("output", "reasoning"):
            self.assertEqual(len(handler.record[field]), 262144)
            self.assertTrue(handler.record[field + "_truncated"])

    def test_queue_wait_and_loading_are_attributed_separately(self):
        self.engine.unload()
        entered, release = threading.Event(), threading.Event()
        original = self.engine.restart
        def restart():
            entered.set()
            self.assertTrue(release.wait(5))
            original()
        with mock.patch.object(self.engine, "restart", side_effect=restart), ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.chat)
            self.assertTrue(entered.wait(5))
            second = pool.submit(self.chat)
            deadline = time.monotonic() + 5
            while len(self.svc.api_requests) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(self.svc.api_requests), 2)
            history = self.request("/api/requests")[1]
            self.assertEqual({r["state"] for r in history["requests"]}, {"queued", "loading"})
            self.assertFalse(history["loaded"])
            self.assertTrue(history["auto_load"])
            time.sleep(0.03)
            release.set()
            self.assertEqual(first.result()[0], 200)
            self.assertEqual(second.result()[0], 200)
        records = list(self.svc.api_requests)
        self.assertGreater(records[0]["load_s"], 0.02)
        self.assertGreater(records[1]["queue_s"], 0.02)
        self.assertLess(records[1]["load_s"], records[0]["load_s"])
        self.assertTrue(all(r["state"] == "completed" for r in records))

    def test_context_of_an_engine_that_cannot_list_it(self):
        code, body = self.request("/api/context")
        self.assertEqual(code, 200)
        self.assertEqual((body["available"], body["reason"]), (False, "engine"))   # an engine before "lists"
        self.assertTrue(self.request("/health")[1]["api_monitor"])

    def lists(self, held):
        """The engine answers LIST_SESSION_TOKENS / LIST_PARKED_CONVERSATIONS with `held` (StrataEngine.list_held)."""
        self.engine.can_list = True
        self.engine.list_held = lambda: {k: [{**e, "tokens": array("i", e["tokens"])} for e in v]
                                         for k, v in held.items()}

    def test_context_lists_then_gives_an_entry_with_its_text(self):
        # "é" is two bytes, one token each: the first token's piece is empty, the second completes it
        ids = [256, *"hi é".encode(), 257]
        self.lists({"sessions": [{"name": "main", "state": "holding", "tokens": ids}],
                    "parked": [{"where": "ram", "bytes": 1536, "tokens": [1, 2]}]})
        code, body = self.request("/api/context")
        self.assertEqual(code, 200)
        self.assertEqual((body["available"], body["busy"], body["max_context"]), (True, False, self.svc.reported_ctx()))
        session, parked = body["entries"]
        self.assertEqual((session["name"], session["state"], session["count"]), ("Session", "holding", len(ids)))
        self.assertNotIn("tokens", session)              # the list carries no tokens: an entry is read on its own
        self.assertEqual({k: parked[k] for k in ("name", "kind", "where", "bytes", "count")},
                         {"name": "Parked conversation 1 of 1", "kind": "parked", "where": "ram", "bytes": 1536,
                          "count": 2})
        code, got = self.request(f"/api/context?list={body['list']}&entry=0")
        self.assertEqual(code, 200)
        self.assertEqual(got["tokens"], ids)
        self.assertEqual(got["pieces"], ["<|im_start|>", "h", "i", " ", "", "é", "<|im_end|>"])
        self.assertEqual(self.request(f"/api/context?list={body['list']}&entry=1")[1]["tokens"], [1, 2])
        # an entry of a list that is no longer the last one read, or not in it: the page reads the list again
        self.request("/api/context")
        for query in (f"list={body['list']}&entry=0", f"list={body['list'] + 1}&entry=2", "list=x&entry=0"):
            self.assertEqual(self.request("/api/context?" + query)[0], 404, query)

    def test_context_while_a_request_has_the_engine(self):
        self.lists({"sessions": [{"name": "main", "state": "holding", "tokens": [5, 6]}], "parked": []})
        self.svc.CONTEXT_WAIT_S = 0.05
        with self.svc.fifo:                              # a request is running: nothing read yet
            body = self.request("/api/context")[1]
        self.assertEqual((body["busy"], body["list"], body["entries"]), (True, None, []))
        first = self.request("/api/context")[1]
        with self.svc.fifo:                              # ... then the list read last, marked busy
            body = self.request("/api/context")[1]
        self.assertEqual((body["busy"], body["list"], body["at"]), (True, first["list"], first["at"]))
        self.assertEqual(body["entries"], first["entries"])

    def test_context_needs_the_key(self):
        self.svc.api_key = "k"
        self.assertEqual(self.request("/api/context")[0], 401)
        self.assertEqual(self.request("/api/context", headers={"Authorization": "Bearer k"})[0], 200)

    def test_page_and_relative_assets_are_served(self):
        code, html = self.request("/api-monitor")
        self.assertEqual(code, 200)
        self.assertIn('src="/web/monitor.js"', html)
        code, script = self.request("/web/monitor.js")
        self.assertEqual(code, 200)
        self.assertIn("textContent", script)
        self.assertNotIn("innerHTML", script)
        for el in ("ctx-view", "ctx-refresh", "ctx-save", "ctx-binary", "ctx-slot", "ctx-sum", "ctx-note", "ctx-pager",
                   "ctx-prev", "ctx-next", "ctx-range"):
            self.assertIn(f'id="{el}"', html)
            self.assertIn(f'"{el}"', script)

    def test_the_main_page_links_to_the_monitor(self):
        web = Path(__file__).parent / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("api-monitor-link", "req-monitor"):
            self.assertIn(f'id="{el}"', html)
            self.assertIn(f'"{el}"', js)
        self.assertIn('href="api-monitor"', html)
        self.assertEqual(self.request("/web/../server.py")[0], 404)


class ListHeld(unittest.TestCase):
    """StrataEngine.list_held: the engine's LIST_SESSION_TOKENS and LIST_PARKED_CONVERSATIONS answers."""

    @staticmethod
    def bare(lines):
        """A StrataEngine without a process: its output lines are queued up front."""
        import io
        import queue
        from types import SimpleNamespace
        from serve.server import StrataEngine
        engine = StrataEngine.__new__(StrataEngine)
        engine.proc = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
        engine.lines, engine.wlock, engine.ended = queue.Queue(), threading.Lock(), False
        for line in lines:
            engine.lines.put(line + "\n")
        return engine

    def test_sessions_and_parked_conversations(self):
        engine = self.bare(["SESSION_TOKENS main holding 3 248045,0,7", "NOTE sessions: something else",
                            "SESSION_TOKENS slot0 decoding 1 9", "SESSION_TOKENS slot1 empty 0", "END_SESSION_TOKENS",
                            "PARKED_CONVERSATION 4 ram 1535867580 2 1,2", "PARKED_CONVERSATION 9 disk 17 0",
                            "END_PARKED_CONVERSATIONS"])
        got = engine.list_held()
        self.assertEqual(engine.proc.stdin.getvalue(), "LIST_SESSION_TOKENS\nLIST_PARKED_CONVERSATIONS\n")
        self.assertEqual([(s["name"], s["state"], s["tokens"].tolist()) for s in got["sessions"]],
                         [("main", "holding", [248045, 0, 7]), ("slot0", "decoding", [9]), ("slot1", "empty", [])])
        self.assertEqual([(p["where"], p["bytes"], p["tokens"].tolist()) for p in got["parked"]],
                         [("ram", 1535867580, [1, 2]), ("disk", 17, [])])

    def test_an_engine_that_refuses_or_ends(self):
        from serve.server import EngineDied
        with self.assertRaises(EngineDied):
            self.bare(["ERR unknown command"]).list_held()
        engine = self.bare(["SESSION_TOKENS main empty 0"])
        engine.lines.put(None)                            # its output closed
        with self.assertRaises(EngineDied):
            engine.list_held()


class ConversationCacheCard(unittest.TestCase):
    """#596: /metrics' "conversation_cache": the parked conversations from the engine's log lines, the budget and
    slots from its INFO line, and how much of the prompts the cache gave back."""

    PARK = ("strata serve: conversation cache: parked 5000 tokens in 12.0 ms; parked=1 bytes=104857600 evictions=0 "
            "snapshot_bytes=104857600 reused_kv_bytes=0\n")
    RESTORE = "strata serve: conversation cache: restored 4000 tokens (exact) in 30.0 ms; parked=2 bytes=209715200\n"

    def test_the_log(self):
        from serve.server import ConvCacheLog
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text("an earlier run\n" + self.PARK, encoding="utf-8")
            start = log.stat().st_size                 # this run starts here: the earlier run's park is not counted
            c = ConvCacheLog()
            self.assertEqual(c.poll(str(log), start)["parked"], 0)
            with open(log, "a", encoding="utf-8") as f:
                f.write("strata serve: something else\n" + self.PARK + self.RESTORE + "strata serve: conversation ca")
            st = c.poll(str(log), start)
            self.assertEqual((st["parked"], st["bytes"], st["parks"], st["restores"]), (2, 209715200, 1, 1))
            self.assertEqual((st["last_event"], st["last_tokens"]), ("restored", 4000))
            with open(log, "a", encoding="utf-8") as f:   # the cut line is read once it is whole
                f.write("che: dropped 1 superseded copy of this conversation; parked=1\n")
            self.assertEqual(c.poll(str(log), start)["parked"], 1)
            self.assertEqual(c.poll(str(log), log.stat().st_size)["parked"], 0)   # the engine started again
            self.assertEqual(c.poll(None, None)["parked"], 0)

    def test_metrics(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "Thought.</think>\nHello.", max_context=4096)
        engine.info = {"conversation_cache_mib": 8192, "conversation_cache_slots": 4}
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                c = json.loads(r.read())["conversation_cache"]
            self.assertEqual((c["enabled"], c["budget_mib"], c["slots"], c["parked"], c["requests"]),
                             (True, 8192, 4, 0, 0))
            req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode())
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                c = json.loads(r.read())["conversation_cache"]
            self.assertEqual(c["requests"], 1)
            self.assertGreater(c["last_prompt"], 0)
            engine.info = {}
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                self.assertFalse(json.loads(r.read())["conversation_cache"]["enabled"])   # off: the reuse lines only
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_the_card_is_on_the_page(self):
        web = Path(__file__).parent / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("cc-card", "cc-slots-text", "cc-mem-text", "cc-facts", "cc-note"):
            self.assertIn(f'id="{el}"', html)
            self.assertIn(f'"{el}"', js)


if __name__ == "__main__":
    unittest.main()
