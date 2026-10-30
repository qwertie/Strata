"""Runs one check of the NVIDIA test session (run.ps1) on one engine build and writes what it printed.

    drive.py <scenario> --exe ENGINE --out FILE --data STRATA_DATA [--cards 1,0] [--args "..."] [--env K=V ...]
             [--deepseek GGUF_DIR] [--other SAVED_FILE]

Scenarios (the engine's --serve protocol unless said otherwise):
  q38       Qwen3.8: the protocol's commands and bad lines, two batch slots, slots that continue a conversation
  q38state  Qwen3.8: prefix reuse, parking, SAVE / RESTORE, and a RESTORE of --other (the build before's file)
  q36       Qwen3.6: the same for its engine (strata-q36)
  multi     Qwen3.8: a 2,000-token prompt, a 128-token answer, a second turn; the startup lines appended
  ds        DeepSeek V4.1: greedy and seeded answers, a 128-token one for the speed
  server    the HTTP server on port 8091 with this engine (Qwen3.8): two greedy chats and their speed
FILE gets the engine's stdout (or the server's answers), FILE.stderr its stderr; FILE ends with "exit <code>".
--cards are the cards the engine sees, numbered as nvidia-smi numbers them. --env adds the check's settings.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from dsv41.smoke import tokenizer  # noqa: E402
from report import startup_lines  # noqa: E402

#: Never 8080, where the owner's own server runs
PORT = 8091
DS_PACK = "packs/deepseek-v4.1-flash-ist-das-2.00bpw"
QWEN_USER, QWEN_ANSWER = "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["q38", "q38state", "q36", "multi", "ds", "server"])
    ap.add_argument("--exe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cards", default="")
    ap.add_argument("--args", default="")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--deepseek")
    ap.add_argument("--other")
    a = ap.parse_args()
    env = dict(e.split("=", 1) for e in a.env)
    if a.scenario == "server":
        return serve(a, env)
    if a.cards:
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=a.cards)
    e = Engine([a.exe, "--serve", *model_args(a), *a.args.split()], env, a.out)
    try:
        e.until("READY")
        SCENARIOS[a.scenario](e, a)
        e.send("QUIT")
    except (SystemExit, queue.Empty, OSError) as x:
        e.note(f"STOPPED: {x or 'no output for 15 minutes'}")
    return e.finish(a.scenario == "multi")


class Engine:
    """Starts an engine with --serve, sends it command lines and copies its stdout to a file."""

    def __init__(self, cmd: list[str], env: dict[str, str], out_path: str):
        self.err_path = out_path + ".stderr"
        env = {**os.environ, **env, "PATH": str(Path(cmd[0]).parent) + os.pathsep + os.environ["PATH"]}
        with open(self.err_path, "w", encoding="utf-8") as err:
            self.p = subprocess.Popen(cmd, cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                      text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
        self.lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self.pump, daemon=True).start()
        self.out = open(out_path, "w", encoding="utf-8", newline="\n")
        self.note(f">>> start: {' '.join(map(str, cmd[1:]))}")

    def pump(self):
        for line in self.p.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def note(self, s: str):
        self.out.write(s + "\n")
        self.out.flush()

    def until(self, *ends: str) -> str:
        """Copies stdout lines to the file until one starts with one of `ends`, and returns that one."""
        while True:
            line = self.lines.get(timeout=900)
            if line is None:
                raise SystemExit("the engine ended early")
            self.out.write(line)
            if line.startswith(ends):
                self.out.flush()
                return line

    def send(self, cmd: str, *ends: str):
        self.note(f">>> {cmd}")
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()
        if ends:
            self.until(*ends)

    def gen(self, head: str, ids: list[int]) -> list[int]:
        """Sends `GEN <head> <ids>` and returns the prompt followed by the answer's tokens."""
        self.note(f">>> GEN {head} <{len(ids)} ids>")
        self.p.stdin.write(f"GEN {head} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        got = []
        while (line := self.until("T ", "DONE", "ERR")).startswith("T "):
            got.append(int(line.split()[1]))
        return ids + got

    def bgen(self, slot: int, max_new: int, ids: list[int]) -> list[int]:
        """Sends a BGEN, waits for its slot to end, and returns the prompt followed by the answer's tokens."""
        self.note(f">>> BGEN {slot} {max_new} <{len(ids)} ids>")
        self.p.stdin.write(f"BGEN {slot} {max_new} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        got = []
        while True:
            f = self.until("T ", "BT ", "BDONE", "BADM", "ERR").split()
            if f[0] == "T" or f[0] == "BT" and f[1] == str(slot):
                got.append(int(f[1] if f[0] == "T" else f[2]))
            elif f[0] == "ERR" or f[0] == "BDONE" and f[1] == str(slot) or f[0] == "BADM" and f[2] == "0":
                return ids + got

    def admit(self, cmds: list[str]):
        """Sends BGEN lines together and copies the output until each of their slots has ended."""
        for c in cmds:
            self.send(c)
        left = len(cmds)
        while left:
            f = self.until("BDONE", "BADM", "ERR").split()
            if f[0] != "BADM" or f[2] == "0":
                left -= 1

    def finish(self, with_startup: bool) -> int:
        """Waits for the engine to end, writes the rest of its output (and the startup lines) and "exit <code>"."""
        self.p.stdin.close()
        try:
            while (rest := self.lines.get(timeout=300)) is not None:
                self.out.write(rest)
            code = self.p.wait(timeout=300)
        except (queue.Empty, subprocess.TimeoutExpired):
            self.p.kill()
            code = "killed"
        if with_startup:
            self.note(">>> startup lines")
            self.out.write(startup_lines(Path(self.err_path).read_text(encoding="utf-8", errors="replace")))
        self.note(f"exit {code}")
        self.out.close()
        return 0 if code == 0 else 1


def model_args(a) -> list[str]:
    """Gets the engine's model arguments for a scenario, from the files in --data (README.md lists them)."""
    d = Path(a.data)
    if a.scenario == "q36":
        gguf = next((d / "models/qwen36-UD-Q4_K_XL").glob("*.gguf"))
        return ["--native", str(gguf), "--max-context", "32768", "--prefill", "auto", "--mtp", "2", "--batch", "2"]
    if a.scenario == "ds":
        return ["--pack", str(deepseek_pack(a)), "--max-context", "8192", "--eos", "1", "--dspark-conf", "0.65"]
    more = {"q38": ["--batch", "2"], "q38state": ["--conversation-cache-mib", "4096"]}
    return qwen38_args(d) + more.get(a.scenario, [])


def qwen38_args(d: Path) -> list[str]:
    """Gets Qwen3.8 IQ3_S's arguments as setup writes them, at a 32K context; the drafter when mtp/rt is there."""
    shards = sorted((d / "models/IQ3_S").glob("*IQ3_S-0000?-of-00002.gguf"))
    args = ["--pack", d / "packs/iq3_s", "--native", shards[0], "--ple-gguf", shards[1], "--expert-profile",
            ROOT / "data/expert-profile.bin", "--expert-cache", "auto", "--prefill", "auto", "--max-context", "32768",
            "--kv", "int8"]
    if (d / "mtp/rt").is_dir():
        args += ["--spec", "4", "--spec-min-p", "0.5", "--mtp", d / "mtp/rt"]
    return [str(x) for x in args]


def deepseek_pack(a) -> Path:
    """Copies the DeepSeek pack beside the output, pointed at the GGUF folder of this PC and with the shipped
    hot-expert profile (each engine start rewrites the pack's profile)."""
    dst = Path(a.out).parent / "dspack"
    shutil.copytree(Path(a.data) / DS_PACK, dst, dirs_exist_ok=True)
    info = (dst / "ds4.txt").read_bytes()
    (dst / "ds4.txt").write_bytes(re.sub(rb"(?m)^model_dir .*?(\r?)$", lambda m: b"model_dir " +
                                         str(Path(a.deepseek)).encode() + m[1], info))
    shutil.copyfile(ROOT / "data/dsv41/profile-2.00.bin", dst / "expert-profile.bin")
    return dst


def q38(e: Engine, a):
    p = list(range(1000, 1040))
    e.send("LIST_CONVERSATIONS", "END_CONVERSATIONS")
    e.gen("24", p)
    e.gen("16 temperature=0 seed=3 cvec=1", p + [1, 2, 3])
    e.send("GET_CONTEXT_WINDOWS main", "END_CONTEXT_WINDOWS")
    for bad in ("GEN x 1,2", "GEN 4", "FOO bar", "BGEN 7 4 1,2", "BGEN 0 0 1,2", "GENI 4 C:/nonexistent.bin 1,2"):
        e.send(bad, "ERR")
    e.gen("8 logit_bias=bad", p)
    e.gen("8 top_k=1 penalty_repeat=1.0 pin=4 ckpt=0 unknown_key=7", p)
    for c in ("BSTOP 0", "BYIELD 1", "STOP"):
        e.send(c)
    e.admit([f"BGEN 0 12 {ids(p + [5])}"])
    e.admit([f"BGEN 0 20 {ids(p + [7])}", f"BGEN 1 20 {ids(p + [8, 9])}"])
    look = ["LIST_CONVERSATIONS"] + [f"GET_CONTEXT_WINDOWS {w}" for w in ("main", "slot0", "slot1")]
    x, y = e.bgen(0, 12, list(range(1000, 1200))), e.bgen(1, 12, list(range(2000, 2150)))
    looks(e, look)
    x = e.gen("8", x + [5])                    # continues slot 0's conversation: copied back from the slot
    x, y = e.bgen(0, 10, x + [6]), e.bgen(1, 10, y + [7])
    looks(e, look)
    e.gen("8", y + [8, 9])                     # slot 1's back into the main session
    looks(e, look)


def q38state(e: Engine, a):
    look = ["LIST_CONVERSATIONS", "GET_CONTEXT_WINDOWS main"]
    saved = a.out + ".ses"
    x = e.gen("16", list(range(1000, 1300)))
    x = e.gen("12", x + [5, 6, 7])             # prefix reuse
    y = e.gen("12", list(range(2000, 2260)))   # another conversation: the first is parked
    x = e.gen("12", x + [8, 9])                # and comes back from the cache
    looks(e, look)
    e.send(f"SAVE {saved}", "SAVED", "SERR", "ERR")
    e.gen("12", y + [1])
    for f, more in ((saved, [10]), (a.other or saved, [11])):
        e.send(f"RESTORE {f}", "RESTORED", "SERR", "ERR", "FATAL")
        looks(e, look)
        e.gen("12", x + more)
    e.send("RESTORE C:/nonexistent.ses", "RESTORED", "SERR", "ERR", "FATAL")


def q36(e: Engine, a):
    p = list(range(1000, 1040))
    e.send("LIST_CONVERSATIONS", "END_CONVERSATIONS")
    e.gen("24", p)
    e.gen("16 temperature=0 seed=3 top_k=5 img_keys=zz unknown=1", p + [1, 2, 3])
    e.gen("16 temperature=0.8 top_p=0.9 top_k=30 min_p=0.05 seed=7", p + [4])
    e.gen("16 penalty_last_n=64 penalty_repeat=1.2 penalty_freq=0.1 penalty_present=0.2", p + [4])
    e.send("GET_CONTEXT_WINDOWS slot0", "END_CONTEXT_WINDOWS")
    for bad in ("GET_CONTEXT_WINDOWS nosuch", "GEN x 1,2", "GEN 4", "FOO bar", "GENX 4 1,2", "BSTOP", "BGEN 7 4 1,2",
                "BGEN 0 0 1,2", "GEN 4 1,999999999", "GENI 4 C:/nonexistent.bin 1,2", "GENI 4 onlyfile",
                f"GEN 40000 {ids(p)}"):
        e.send(bad, "ERR", "END_CONTEXT_WINDOWS")
    for c in ("", "BSTOP 0", "BYIELD 1", "STOP"):
        e.send(c)
    e.gen("8", p + [9])
    e.admit([f"BGEN 0 12 {ids(p + [5])}"])
    e.admit([f"BGEN 1 12 temperature=0.7 seed=11 {ids(p + [6])}"])
    e.admit([f"BGEN 0 20 {ids(p + [7])}", f"BGEN 1 20 {ids(p + [8, 9])}"])
    e.send("LIST_CONVERSATIONS", "END_CONVERSATIONS")


def multi(e: Engine, a):
    tok = tokenizer(Path(a.data) / "packs/iq3_s")
    chat = lambda text: tok.encode(text, parse_special=True)  # noqa: E731
    doc = tok.encode((ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8"))[:1950]
    x = e.gen("128", chat(QWEN_USER + "Summarize this document in five bullet points.\n\n") + doc + chat(QWEN_ANSWER))
    e.gen("128", x + chat("<|im_end|>\n" + QWEN_USER + "Now name three open questions it leaves." + QWEN_ANSWER))


def ds(e: Engine, a):
    tok = tokenizer(Path(a.data) / DS_PACK)
    chat = lambda text: tok.encode(  # noqa: E731
        f"<｜begin▁of▁sentence｜><｜User｜>{text}<｜Assistant｜></think>", parse_special=True)
    code = chat("Write a Python function that returns the n-th Fibonacci number iteratively. Code only.")
    e.gen("48", code)
    e.gen("32 temperature=0.8 top_p=0.9 top_k=30 min_p=0.05 seed=7",
          chat("Tell me a two-sentence story about a lighthouse."))
    e.gen("64", chat("Count from 1 to 40, separated by commas."))
    e.gen("128", code)


SCENARIOS = {"q38": q38, "q38state": q38state, "q36": q36, "multi": multi, "ds": ds}


def looks(e: Engine, cmds: list[str]):
    for c in cmds:
        e.send(c, "END_CONVERSATIONS" if c == "LIST_CONVERSATIONS" else "END_CONTEXT_WINDOWS", "ERR")


def ids(v: list[int]) -> str:
    return ",".join(map(str, v))


def serve(a, env: dict[str, str]) -> int:
    """Starts the server with this engine (its run config beside the output), asks it two greedy questions, writes
    the answers and their speed, and ends the server and its engine."""
    import psutil
    out = Path(a.out)
    cfg = {"exe": a.exe, "args": qwen38_args(Path(a.data)) + a.args.split(), "cwd": str(ROOT), "port": PORT,
           "tokenizer": str(Path(a.data) / "packs/iq3_s/tokenizer"), "model_name": "qwen3.8-flash-next-iq3_s",
           "log": str(out) + ".engine.log", "lib_dirs": [str(Path(a.exe).parent)], "backend": "cuda", "env": env,
           "gpu": [int(c) for c in a.cards.split(",") if c]}
    out.with_suffix(".json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    doc = (ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")[:8500]
    prompts = {"short": "What is the capital of Australia? Answer in one sentence.",
               "long": "Read the following document and summarize its main rules in five bullet points.\n\n" + doc}
    lines, code, p = [], 1, None
    try:
        if is_port_taken():
            raise OSError(f"port {PORT} is taken")
        with open(str(out) + ".stderr", "w", encoding="utf-8") as err:
            p = subprocess.Popen([sys.executable, str(ROOT / "strata_server/server.py"), "--engine", "strata",
                                  "--config", str(out.with_suffix(".json")), "--port", str(PORT)], cwd=str(ROOT),
                                 stdout=err, stderr=err)
        wait_healthy(p, 1800)
        for name, text in prompts.items():
            body = {"model": "x", "messages": [{"role": "user", "content": text}], "temperature": 0,
                    "presence_penalty": 0, "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                got = json.load(r)
            t = got.get("timings") or {}
            speed = f"decode={t.get('predicted_per_second') or 0} prompt={t.get('prompt_per_second') or 0}"
            lines += [f">>> {name}", got["choices"][0]["message"].get("content") or ""]
            if name == "long":                 # the short answer is too short to time
                lines.append(f"SPEED {name} {speed}")
        code = 0
    except (OSError, ValueError, KeyError) as x:
        lines.append(f"STOPPED: {x}")
    finally:
        try:
            for proc in psutil.Process(p.pid).children(recursive=True) + [psutil.Process(p.pid)] if p else []:
                proc.kill()
        except psutil.Error:
            pass
    out.write_text("\n".join(lines + [f"exit {code}"]) + "\n", encoding="utf-8", newline="\n")
    return code


def is_port_taken() -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def wait_healthy(p: subprocess.Popen, timeout_s: int):
    """Waits until the server answers /health with 200 (it has loaded the engine); TimeoutError otherwise."""
    end = time.time() + timeout_s
    while time.time() < end and p.poll() is None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(5)
    raise TimeoutError(f"the server was not ready (exit code {p.poll()})")


if __name__ == "__main__":
    sys.exit(main())
