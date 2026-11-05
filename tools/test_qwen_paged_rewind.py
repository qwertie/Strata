"""Checks paged Qwen tool continuations, retries and checkpoint persistence on a real model.

Run with a private engine config and a new output directory:
  python tools/test_qwen_paged_rewind.py CONFIG OUTPUT
The config controls GPU/RAM budgets, drafting and layer splits. No server is stopped or changed.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from dsv41.smoke import tokenizer
from strata_common.engine_client import StrataEngine, child_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--restore-from", type=Path, help="Replays another run's saved rewind on this layer split")
    parser.add_argument("--system-repeats", type=int, default=16, help="Sets the size of the shared prompt prefix")
    options = parser.parse_args()
    if options.system_repeats < 1:
        parser.error("--system-repeats must be positive")
    cfg = json.loads(options.config.read_text())
    work = options.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    tok = tokenizer(Path(cfg["tokenizer"]).parent)
    encode = partial(tok.encode, parse_special=True)
    header = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    effort = "<|im_start|>system\nReasoning effort is set to medium.<|im_end|>\n"
    system = "<|im_start|>system\nAnswer questions about the following records.\n" + (
        "amber birch cedar delta ember fjord grove harbor iris juniper.\n" * options.system_repeats
    ) + "<|im_end|>\n"
    history = encode(system + "<|im_start|>user\nName three colors.<|im_end|>\n")
    records = []
    args = cfg["args"] + ["--pause-dir", str(work / "pause"),
                          "--tail-role-token", str(encode("system")[0])]
    eng = StrataEngine(cfg["exe"], args, cwd=str(ROOT), log=str(work / "engine.log"), env=child_env(cfg))
    try:
        if options.restore_from:
            replay = json.loads((options.restore_from / "replay.json").read_text())
            eng.session_file("restore", str((options.restore_from / "session.bin").resolve()))
            answer, reused = generate(eng, replay["prompt"])
            assert answer == replay["answer"] and reused >= replay["boundary"], ("cross-cut rewind", reused)
            records.append({"case": "cross_cut_rewind", "reused": reused, "tokens_equal": True})
            print(json.dumps(records, indent=2), flush=True)
            (work / "results.json").write_text(json.dumps(records, indent=2))
            return
        for turn in range(4):
            prompt = history + encode(effort + header)
            answer, reused = generate(eng, prompt)
            assert answer, "The model returned no tokens"
            if turn:
                assert reused >= previous_boundary, ("tool continuation reread history", reused, previous_boundary)
                assert not paused(eng), "A tool continuation created a paused backup"
            records.append({"case": "tool", "turn": turn, "reused": reused, "prompt": len(prompt)})
            previous_boundary = len(history)
            history += encode(header) + answer + encode(
                "<|im_end|>\n<|im_start|>user\nTool result: blue, green, red. Name another color.<|im_end|>\n"
            )
        other_chat = encode(system + "<|im_start|>user\nName one tree.<|im_end|>\n" + header)
        generate(eng, other_chat)
        assert paused(eng), "A different chat sharing the system prompt was discarded"
        generate(eng, prompt)
        save = work / "session.bin"
        eng.session_file("save", str(save))
        expected, reused = generate(eng, prompt)
        assert reused >= previous_boundary, ("retry missed its checkpoint", reused, previous_boundary)
        (work / "replay.json").write_text(json.dumps({
            "prompt": prompt, "answer": expected, "boundary": previous_boundary,
        }))
        retried, _ = generate(eng, prompt)
        assert retried == expected, "A repeated prompt changed its greedy tokens"
        eng.session_file("restore", str(save))
        restored, reused = generate(eng, prompt)
        assert restored == expected and reused >= previous_boundary, ("SAVE/RESTORE rewind mismatch", reused)
        records.append({"case": "saved_rewind", "reused": reused, "tokens_equal": True})
        unrelated = encode("<|im_start|>system\nIndependent conversation.<|im_end|>\n"
                           "<|im_start|>user\nName one animal.<|im_end|>\n" + header)
        ssd_budget = eng.memory()["settings"]["ssd_pause_mib"]
        for place in ("ram", "disk"):
            if place == "disk":
                eng.change_settings({"ssd_pause_mib": ssd_budget})
                eng.change_settings({"tokens_min": 0, "tokens_max": 0})
            else:
                # Equal-prefix SSD copies can otherwise win over the freshly paused RAM copy.
                eng.change_settings({"ssd_pause_mib": 0})
            generate(eng, unrelated)
            result, reused = generate_paused(eng, prompt, place, work / "engine.log")
            assert result == expected and reused >= previous_boundary, ("paused rewind mismatch", place, reused)
            records.append({"case": place + "_rewind", "reused": reused, "tokens_equal": True})
        print(json.dumps(records, indent=2), flush=True)
        (work / "results.json").write_text(json.dumps(records, indent=2))
    finally:
        eng.close()


def generate(eng, tokens):
    """Gets a short greedy answer and the engine's reported reused prefix."""
    answer = [t for t in eng.generate(tokens, 12, {"temperature": 0}, threading.Event()) if t is not None]
    return answer, int(eng.last.get("reused") or eng.reused or 0)


def generate_paused(eng, tokens, place, log):
    """Checks the saved prompt's tier and the transfer actually used to resume it."""
    held = paused(eng)
    windows = eng.context_windows([p["id"] for p in held])
    matches = [p for p in held if windows.get(p["id"]) is not None and
               list(windows[p["id"]][:len(tokens)]) == tokens]
    assert matches and all(p["place"] == place for p in matches), ("paused prompt in wrong tier", place, matches)
    offset = log.stat().st_size
    answer, reused = generate(eng, tokens)
    with log.open("rb") as stream:
        stream.seek(offset)
        recent = stream.read().decode("utf-8", errors="replace")
    transfer = "back from RAM" if place == "ram" else "back from the SSD"
    assert transfer in recent, ("missing restore transfer", place, recent)
    return answer, reused


def paused(eng):
    """Lists only conversations held outside the GPU pool."""
    return [p for p in eng.list_conversations([], 0) if p["state"] == "parked"]


if __name__ == "__main__":
    main()
