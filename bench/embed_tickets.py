"""Embed synthetic ticket texts for the load test's --vectors mode.

Writes DIR/vectors.npy (the first --n texts) and DIR/queries.npy (the last --queries texts) as float32,
using the ticket generator in bench/make_corpus.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from governed_rag.models import STEmbedder

MAX_MISSES = 1000  # consecutive draws that repeat a text before the generator counts as exhausted


def _load_make_corpus():
    path = Path(__file__).resolve().parent / "make_corpus.py"
    spec = importlib.util.spec_from_file_location("make_corpus", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def distinct_texts(total: int, seed: int, corpus) -> list[str]:
    rng = random.Random(seed)
    texts: list[str] = []
    seen: set[str] = set()
    draws = misses = 0
    while len(texts) < total:
        text = corpus._ticket_text(rng, corpus.ISSUES[draws % len(corpus.ISSUES)])
        draws += 1
        if text in seen:
            misses += 1
            if misses > MAX_MISSES:
                raise RuntimeError(f"only {len(texts)} distinct ticket texts can be generated, need {total}")
            continue
        misses = 0
        seen.add(text)
        texts.append(text)
    return texts


def main(argv=None, embedder=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--queries", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args(argv)
    if args.n < 1 or args.queries < 1 or args.batch < 1:
        print("invalid arguments: need --n >= 1, --queries >= 1, --batch >= 1", file=sys.stderr)
        return 2

    total = args.n + args.queries
    try:
        texts = distinct_texts(total, args.seed, _load_make_corpus())
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    if embedder is None:
        embedder = STEmbedder()
    start = time.perf_counter()
    rows: list = []
    for lo in range(0, total, args.batch):
        rows += embedder.embed(texts[lo:lo + args.batch])
    matrix = np.asarray(rows, dtype=np.float32)
    seconds = time.perf_counter() - start

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "vectors.npy", matrix[:args.n])
    np.save(args.out / "queries.npy", matrix[args.n:])
    print(json.dumps({"vectors": args.n, "queries": args.queries, "distinct_texts": len(set(texts)),
                      "seconds": round(seconds, 3)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
