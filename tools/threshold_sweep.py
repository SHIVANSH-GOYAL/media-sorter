"""Offline clustering-threshold sweep. Reads embeddings only; never writes to the DB.

Usage:
    python tools/threshold_sweep.py --min 0.35 --max 0.60 --step 0.05
"""

import argparse
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config


def greedy(embs, media_ids, threshold):
    n, dim = embs.shape
    centroids = np.zeros((n, dim), dtype=np.float32)
    counts = np.zeros(n, dtype=np.int64)
    assignments = np.zeros(n, dtype=np.int64)
    k = 0
    for i, emb in enumerate(embs):
        if k:
            sims = centroids[:k] @ emb
            best = int(np.argmax(sims))
            if 1.0 - float(sims[best]) <= threshold:
                counts[best] += 1
                merged = centroids[best] * (counts[best] - 1) + emb
                norm = np.linalg.norm(merged)
                centroids[best] = merged / norm if norm else merged
                assignments[i] = best
                continue
        centroids[k] = emb
        counts[k] = 1
        assignments[i] = k
        k += 1
    return assignments[:], centroids[:k], counts[:k]


def same_photo_collisions(assignments, media_ids):
    photos = set()
    pairs = 0
    buckets = {}
    for a, m in zip(assignments.tolist(), media_ids.tolist()):
        buckets.setdefault(m, Counter())[a] += 1
    for media_id, counts in buckets.items():
        for c in counts.values():
            if c > 1:
                photos.add(media_id)
                pairs += c * (c - 1) // 2
    return len(photos), pairs


def ari(a, b):
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)
    comb = lambda x: x * (x - 1) / 2
    sum_ij = comb(table).sum()
    sum_a = comb(table.sum(axis=1)).sum()
    sum_b = comb(table.sum(axis=0)).sum()
    total = comb(len(a))
    expected = sum_a * sum_b / total if total else 0
    maximum = (sum_a + sum_b) / 2
    return (sum_ij - expected) / (maximum - expected) if maximum != expected else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(config.DB_PATH))
    ap.add_argument("--min", type=float, default=0.35)
    ap.add_argument("--max", type=float, default=0.60)
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, media_id, embedding, cluster_id FROM faces ORDER BY id").fetchall()
    if not rows:
        sys.exit("no faces in database")
    embs = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.where(norms > 0, norms, 1)
    media_ids = np.array([r["media_id"] for r in rows])
    current = np.array([r["cluster_id"] if r["cluster_id"] is not None else -1 for r in rows])

    print(f"faces={len(rows)} media={len(set(media_ids.tolist()))} db={args.db}")
    print(f"{'thr':>5} {'people':>7} {'clusters':>9} {'singles':>8} {'biggest':>8} "
          f"{'intra':>6} {'photo-clash':>12} {'clash-pairs':>12} {'ARI@db':>7} {'sec':>5}")
    thresholds = np.round(np.arange(args.min, args.max + 1e-9, args.step), 3)
    for thr in thresholds:
        t0 = time.time()
        assignments, centroids, counts = greedy(embs, media_ids, float(thr))
        elapsed = time.time() - t0
        people = int((counts >= 2).sum())
        singles = int((counts == 1).sum())
        biggest = int(counts.max()) if len(counts) else 0
        intra = float(np.mean(np.sum(embs * centroids[assignments], axis=1)))
        photos, pairs = same_photo_collisions(assignments, media_ids)
        score = ari(current, assignments)
        print(f"{thr:5.2f} {people:7d} {len(counts):9d} {singles:8d} {biggest:8d} "
              f"{intra:6.3f} {photos:12d} {pairs:12d} {score:7.3f} {elapsed:5.1f}")


if __name__ == "__main__":
    main()
