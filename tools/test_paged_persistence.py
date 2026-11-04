#!/usr/bin/env python3
"""Checks paged SAVE/RESTORE and orphan adoption with drafting enabled (Windows GPU test).

Run with a private config whose args enable MTP/DSpark and suffix drafting, and an empty output directory:
  python tools/test_paged_persistence.py CONFIG OUTPUT [--cuts 20 30]
Qwen cuts use two stages on GPU 0. All engines are sequential; no server or network port is used.
"""
import argparse
import ctypes
import json
import re
from pathlib import Path
import shutil
import sys
import threading

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from dsv41.smoke import tokenizer
from strata_common.engine_client import StrataEngine, child_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--foreign', type=Path, help='A saved session of another model; must stay unclaimed')
    parser.add_argument('--cuts', type=int, nargs='*', default=[])
    options = parser.parse_args()
    cfg = json.loads(options.config.read_text())
    work = options.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    args = cfg['args']
    assert '--suffix-draft' in args and int(args[args.index('--suffix-draft') + 1]) > 0
    ds = '--dspark' in args
    assert (ds and args[args.index('--dspark') + 1] != 'none') or '--mtp' in args
    tok = tokenizer(Path(cfg['tokenizer']).parent)
    if ds:
        prompt = '<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>'
        end = '<\uff5cAssistant\uff5c></think>'
    else:
        prompt = '<|im_start|>user\n'
        end = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    repeat = 'amber birch cedar delta ember fjord grove harbor iris juniper. '
    ids = tok.encode(prompt + repeat * 12 + '\nRepeat the list ten times.' + end, parse_special=True)
    followup = ('<\uff5cUser\uff5c>' if ds else '<|im_end|>\n<|im_start|>user\n')
    tail = tok.encode(followup + 'Repeat the list ten times without explanation.' + end, parse_special=True)
    save = work / 'session.bin'
    results = []
    source = None
    expected = None
    stages = [None, *options.cuts, None, None]
    for index, cut in enumerate(stages):
        pause = work / 'pause'
        pause.mkdir(exist_ok=True)
        run_args = args + ['--pause-dir', str(pause)]
        if cut is not None:
            run_args += ['--split-device', '0', '--layer-split', str(cut)]
        log = work / (str(index) + '.log')
        adopt = index > 0 and index % 2 == 0
        if adopt:
            shutil.copyfile(save, pause / 'paused-2147483000-1-1.bin')
            (pause / 'paused-2147483000-1-2.bin').write_bytes(b'damaged')
            if options.foreign:
                shutil.copyfile(options.foreign, pause / 'paused-2147483000-1-3.bin')
        with GuardedEngine(cfg, run_args, log) as eng:
            if source is None:
                answer, _ = generate(eng, ids, 32)
                source = ids + answer + tail
            else:
                if not adopt:
                    eng.session_file('restore', str(save))
                continued, reused = generate(eng, source, 32)
                assert len(continued) >= 16, 'Continuation ended before exercising draft windows'
                assert reused > 0 and continued == expected, ('persistence parity', cut, adopt, reused)
                if adopt:
                    assert (pause / 'paused-2147483000-1-2.bin').exists(), 'damaged orphan was deleted'
                    if options.foreign:
                        assert (pause / 'paused-2147483000-1-3.bin').exists(), 'foreign session was claimed'
                source += continued + tail
            if index + 1 < len(stages):
                eng.session_file('save', str(save))
                expected, _ = generate(eng, source, 32)
            results.append({'cut': cut, 'adopt': adopt, 'passed': True})
            print('PASS', results[-1], flush=True)
    assert any(re.search(r'suffix drafts: \d+ of [1-9]\d*', p.read_text())
               for p in work.glob('*.log')), 'Suffix drafting offered no tokens'
    (work / 'results.json').write_text(json.dumps(results, indent=2))


def generate(eng, tokens, count):
    """Collects a greedy continuation and the prefix reuse reported by the engine."""
    out = [t for t in eng.generate(tokens, count, {'temperature': 0}, threading.Event()) if t is not None]
    return out, int(eng.last.get('reused') or eng.reused or 0)


class GuardedEngine:
    """Owns one test engine and a PID-scoped memory watchdog, including during startup."""

    def __init__(self, cfg, args, log):
        self.cfg, self.args, self.log = cfg, args, log
        self.stop = threading.Event()
        self.eng = None
        self.tripped = False

    def __enter__(self):
        for proc in psutil.process_iter(['name']):
            name = (proc.info['name'] or '').lower().removesuffix('.exe')
            assert name not in {'docker desktop', 'com.docker.backend', 'vmmem', 'vmmemwsl'}, name
            assert not name.startswith('strata'), 'Another engine is running: ' + name
        commit, physical = get_available_gib()
        print('PRESTART free commit/physical GiB', commit, physical, flush=True)
        pools = sum(int(n) for n in self.args[self.args.index('--gpu-mib') + 1].split(','))
        stages = 2 if '--layer-split' in self.args else 1
        assert pools * stages <= 60 * 1024, 'GPU budget exceeds the approved 60 GiB'
        ram = int(self.args[self.args.index('--ram-mib') + 1])
        reserve = 24 if '--dspark' in self.args else 12
        assert commit > (pools * stages + ram) / 1024 + reserve and physical > 12, (
            'Insufficient free memory for this correctness test')
        self.thread = threading.Thread(target=self.guard, daemon=True)
        self.thread.start()
        try:
            env = child_env(self.cfg)
            env['STRATA_KEEP_LEFTOVERS'] = '1'
            self.eng = StrataEngine(self.cfg['exe'], self.args, cwd=str(ROOT), log=str(self.log), env=env)
            return self.eng
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        try:
            if self.eng is not None:
                self.eng.close()
        finally:
            self.stop.set()
            self.thread.join()
        assert not self.tripped, 'Memory watchdog tripped'

    def guard(self):
        while not self.stop.wait(0.5):
            commit, physical = get_available_gib()
            if commit < 3 or physical < 4:
                self.tripped = True
                for proc in psutil.Process().children(recursive=True):
                    if proc.name().lower().startswith('strata'):
                        proc.kill()
                return


def get_available_gib():
    """Reads Windows' available commit and physical memory in GiB."""
    class Status(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ('total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'extended')]
    status = Status()
    status.length = ctypes.sizeof(status)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.avail_page / 2**30, status.avail_phys / 2**30


if __name__ == '__main__':
    main()
