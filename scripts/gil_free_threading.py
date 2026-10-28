"""GIL vs free-threaded Python (PEP 703) on the same pure-Python code. Standard library only, so it
runs on any interpreter - e.g. the free-threaded build `python3.14t` (install: `uv python install 3.14t`).

    python scripts/gil_free_threading.py <subset.json>           # prints one JSON line

Part A: the naive search (hpss.naive, stdlib only) on 1..16 threads.
Part B: a shared counter incremented by 8 threads - without a lock, with a lock per increment, and
        "accumulate locally, lock once" (week 3). Without the GIL the unlocked version loses updates.
"""

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hpss.naive import search_naive  # noqa: E402  (stdlib-only module)


def part_a(queries, corpus, k=10, workers=(1, 2, 4, 8, 16), runs=5):
    def run(w):
        spans = [(i * len(queries) // w, (i + 1) * len(queries) // w) for i in range(w)]
        with ThreadPoolExecutor(w) as ex:
            list(ex.map(lambda s: search_naive(queries[s[0]:s[1]], corpus, k), spans))

    out = {}
    for w in workers:
        run(w)                                              # warm-up
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            run(w)
            ts.append(time.perf_counter() - t0)
        out[w] = sorted(ts)[len(ts) // 2]
    return out


def part_b(threads=8, n=200_000, runs=5):
    """Each version: 1 warm-up + `runs` timed runs; reports the median time and the WORST count seen."""
    counter = [0]
    lock = threading.Lock()

    def no_lock():
        for _ in range(n):
            counter[0] += 1                                 # read-modify-write: not atomic

    def lock_each():
        for _ in range(n):
            with lock:
                counter[0] += 1

    def local_then_lock():
        local = 0
        for _ in range(n):
            local += 1
        with lock:
            counter[0] += local

    def once(fn):
        counter[0] = 0
        ts = [threading.Thread(target=fn) for _ in range(threads)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return counter[0], time.perf_counter() - t0

    res = {}
    for name, fn in [("no lock", no_lock), ("lock per increment", lock_each), ("local sum, lock once", local_then_lock)]:
        once(fn)                                            # warm-up
        out = [once(fn) for _ in range(runs)]
        secs = sorted(s for _, s in out)
        res[name] = {"count": min(c for c, _ in out), "expected": threads * n, "seconds": secs[len(secs) // 2],
                     "counts": [c for c, _ in out]}
    return res


def main():
    data = json.loads(Path(sys.argv[1]).read_text())
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(json.dumps({"python": sys.version.split()[0], "gil_enabled": gil,
                      "search": part_a(data["queries"], data["corpus"]), "counter": part_b()}))


if __name__ == "__main__":
    main()
