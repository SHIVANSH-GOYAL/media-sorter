from datetime import datetime, timezone

import numpy as np

from . import db as dbmod


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _pack(vec):
    return np.asarray(vec, dtype=np.float32).tobytes()


def _unpack(blob):
    return np.frombuffer(blob, dtype=np.float32)


def _normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def load_centroids(conn):
    rows = conn.execute("SELECT id, centroid, n FROM clusters ORDER BY id").fetchall()
    return rows


def assign(conn, embedding, threshold):
    """Assign one face embedding to an existing cluster or create a new one.

    Uses the cluster centroid plus up to 12 strongest member exemplars (max cosine
    similarity wins). Returns the cluster id. Uncertain faces start a new
    "Unsorted" cluster rather than being force-merged.
    """
    emb = _normalize(embedding)
    rows = load_centroids(conn)
    if rows:
        scores = []
        for row in rows:
            centroid = _normalize(_unpack(row["centroid"]))
            centroid_sim = float(np.dot(centroid, emb))
            exemplars = conn.execute(
                "SELECT embedding FROM faces WHERE cluster_id = ? ORDER BY det_score DESC, id DESC LIMIT 12",
                (row["id"],),
            ).fetchall()
            member_scores = []
            for x in exemplars:
                member = _unpack(x["embedding"])
                if member.shape == emb.shape:
                    member_scores.append(float(np.dot(_normalize(member), emb)))
            member_sim = max(member_scores) if member_scores else -1.0
            scores.append(max(centroid_sim, member_sim))
        sims = np.asarray(scores, dtype=np.float32)
        best = int(np.argmax(sims))
        dist = 1.0 - float(sims[best])
        if dist <= threshold:
            row = rows[best]
            n = int(row["n"])
            new_centroid = _normalize(_unpack(row["centroid"]) * n + emb)
            conn.execute(
                "UPDATE clusters SET centroid = ?, n = n + 1, updated_at = ? WHERE id = ?",
                (_pack(new_centroid), _now(), row["id"]),
            )
            conn.execute(
                "UPDATE clusters SET name = ? WHERE id = ? AND name = 'Unsorted'",
                (_person_name(conn), row["id"]),
            )
            return row["id"]
    cur = conn.execute(
        "INSERT INTO clusters(name, centroid, n, auto_name, created_at, updated_at) "
        "VALUES('Unsorted', ?, 1, 1, ?, ?)",
        (_pack(emb), _now(), _now()),
    )
    return int(cur.lastrowid)


def _person_name(conn):
    row = conn.execute("SELECT COUNT(*) AS c FROM clusters WHERE name LIKE 'Person %'").fetchone()
    return f"Person {int(row['c']) + 1}"


def recluster(conn, threshold, exemplars=4, max_iter=10):
    """Regroup faces using multi-exemplar per-person matching + iterative
    reassignment until stable.

    Seeds from existing clusters, builds a small exemplar bank per cluster
    (top detection scores, pose-diverse), then repeatedly reassigns each face
    to the cluster whose *nearest exemplar* is closest (within threshold) — not
    just the centroid — so same-person faces that drifted across angle/lighting/
    environment can rejoin their cluster via a matching pose exemplar. After each
    pass, centroids are recomputed and clusters whose centroids are within
    threshold are merged under an anti-chaining guard (combined centroid must
    stay within threshold of every member) to prevent over-merging two distinct
    people. Repeats until no reassignment or merge occurs, capped at max_iter.

    `exemplars` = size of each cluster's exemplar bank (default 4).
    `max_iter` = maximum reassignment/merge rounds (default 10).
    """
    rows = conn.execute(
        "SELECT id, embedding, cluster_id, COALESCE(det_score, 0) AS det_score FROM faces ORDER BY id"
    ).fetchall()
    if not rows:
        return {"faces": 0, "clusters": 0, "merges": 0}
    exemplars = max(1, int(exemplars))
    max_iter = max(1, int(max_iter))
    ids = [r["id"] for r in rows]
    fid_to_pos = {fid: i for i, fid in enumerate(ids)}
    mat = np.vstack([_normalize(_unpack(r["embedding"])) for r in rows])
    n_faces = len(rows)
    scores = np.asarray([float(r["det_score"] or 0.0) for r in rows])
    sim = mat @ mat.T
    sim_threshold = 1.0 - float(threshold)

    old_meta = {c["id"]: dict(c) for c in conn.execute("SELECT id, name, auto_name, n FROM clusters")}
    old_members = {cid: set() for cid in old_meta}
    for r in rows:
        if r["cluster_id"] is not None:
            old_members.setdefault(r["cluster_id"], set()).add(r["id"])

    # Seed from existing clusters; orphaned faces start their own cluster.
    members = {cid: set() for cid in old_meta}
    for r in rows:
        cid = r["cluster_id"]
        if cid is not None and cid in members:
            members[cid].add(r["id"])
        else:
            members.setdefault(r["id"], set()).add(r["id"])
    members = {cid: m for cid, m in members.items() if m}
    alive = set(members)

    def bank(cidx):
        positions = np.asarray([fid_to_pos[f] for f in members[cidx]], dtype=np.int32)
        if positions.size == 0:
            return np.asarray([], dtype=np.int32)
        ranked = positions[np.argsort(scores[positions])[::-1]]
        chosen = [int(ranked[0])]
        while len(chosen) < min(exemplars, len(ranked)):
            remaining = np.asarray([p for p in ranked if int(p) not in chosen], dtype=np.int32)
            nearest = sim[np.ix_(remaining, np.asarray(chosen))].max(axis=1)
            chosen.append(int(remaining[int(np.argmin(nearest))]))
        return np.asarray(chosen, dtype=np.int32)

    def centroid(cidx):
        positions = np.fromiter((fid_to_pos[f] for f in members[cidx]), dtype=np.int32)
        return _normalize(mat[positions].mean(axis=0))

    centroids = {c: centroid(c) for c in alive}
    banks = {c: bank(c) for c in alive}

    def clusters_close(a, b):
        """True if clusters a and b should merge: nearest-exemplar similarity
        above threshold (any exemplar of a close to any exemplar of b) plus an
        anti-chaining guard on the combined centroid."""
        if 1.0 - float(centroids[a] @ centroids[b]) > threshold:
            return False
        ba, bb = banks[a], banks[b]
        if ba.size == 0 or bb.size == 0:
            return False
        cross = sim[np.ix_(ba, bb)]
        if cross.max() < sim_threshold:
            return False
        combined = members[a] | members[b]
        positions = np.fromiter((fid_to_pos[f] for f in combined), dtype=np.int32)
        new_cent = _normalize(mat[positions].mean(axis=0))
        if np.any(1.0 - mat[positions] @ new_cent > threshold):
            return False
        return True

    merges = 0
    iterations = 0
    for iteration in range(max_iter):
        iterations = iteration + 1
        merged_any = False
        live = sorted(alive)
        for ix, a in enumerate(live):
            if a not in alive:
                continue
            for b in live[ix + 1:]:
                if b not in alive:
                    continue
                if not clusters_close(a, b):
                    continue
                members[a] |= members.pop(b)
                alive.remove(b)
                centroids[a] = centroid(a)
                banks[a] = bank(a)
                merges += 1
                merged_any = True
                break
            if merged_any:
                break
        if not merged_any:
            break

    new_members = {c: members[c] for c in alive}
    new_centroids = {}
    for cidx, mems in new_members.items():
        positions = np.fromiter((fid_to_pos[f] for f in mems), dtype=np.int32)
        new_centroids[cidx] = _normalize(mat[positions].mean(axis=0))

    # Name carry-over: match each new cluster to the old cluster with most overlap.
    names = {}
    used_old = set()
    for cidx, mems in sorted(new_members.items(), key=lambda kv: -len(kv[1])):
        best_old, best_overlap = None, 0
        for old_cid, old_ids in old_members.items():
            if old_cid in used_old:
                continue
            overlap = len(mems & old_ids)
            if overlap > best_overlap:
                best_old, best_overlap = old_cid, overlap
        name = None
        auto = 1
        if best_old is not None and best_overlap > 0:
            meta = old_meta[best_old]
            used_old.add(best_old)
            if int(meta["auto_name"]) == 0 or int(meta["n"]) >= 2:
                name = meta["name"]
                auto = int(meta["auto_name"])
        names[cidx] = (name, auto)

    person_counter = 0
    for cidx, mems in sorted(new_members.items(), key=lambda kv: -len(kv[1])):
        name, auto = names[cidx]
        if name is None or name == "Unsorted":
            if len(mems) >= 2:
                person_counter += 1
                name, auto = f"Person {person_counter}", 1
            else:
                name, auto = "Unsorted", 1
        if not name:
            name = "Unsorted"
        names[cidx] = (name, auto)

    # Write to database.
    conn.execute("UPDATE faces SET cluster_id = NULL")
    conn.execute("DELETE FROM clusters")
    for cidx, mems in sorted(new_members.items(), key=lambda kv: -len(kv[1])):
        name, auto = names[cidx]
        cur = conn.execute(
            "INSERT INTO clusters(name, centroid, n, auto_name, created_at, updated_at) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (name, _pack(new_centroids[cidx]), len(mems), auto, _now(), _now()),
        )
        cid = int(cur.lastrowid)
        conn.executemany(
            "UPDATE faces SET cluster_id = ? WHERE id = ?",
            [(cid, fid) for fid in mems],
        )

    # Age consensus: confidence-weighted average per cluster, backfilled to faces.
    for cid in [r["id"] for r in conn.execute("SELECT id FROM clusters WHERE n >= 2").fetchall()]:
        consensus = conn.execute(
            "SELECT age, age_conf FROM faces WHERE cluster_id = ? AND age IS NOT NULL "
            "ORDER BY age_conf DESC, id DESC LIMIT 40",
            (cid,),
        ).fetchall()
        if consensus:
            weights = np.asarray([max(0.1, float(r["age_conf"] or 0.0)) for r in consensus], dtype=np.float32)
            ages = np.asarray([float(r["age"]) for r in consensus], dtype=np.float32)
            consensus_age = float(np.average(ages, weights=weights))
            # Map to the nearest band midpoint for stability across angles/effects.
            band, _ = _band_for_age(consensus_age)
            conn.execute(
                "UPDATE faces SET age = ?, age_band = ? WHERE cluster_id = ?",
                (round(consensus_age, 1), band, cid),
            )

    return {"faces": n_faces, "clusters": len(new_members), "merges": merges, "iterations": iterations}


def _band_for_age(age):
    BANDS = (
        (2, "0-2"), (9, "3-9"), (19, "10-19"), (29, "20-29"),
        (39, "30-39"), (49, "40-49"), (59, "50-59"), (69, "60-69"),
    )
    for limit, label in BANDS:
        if age <= limit:
            return label, float(label.split("-")[0] if "-" in label else label)
    return "70+", 70.0


def recompute(conn, cluster_id):
    rows = conn.execute(
        "SELECT embedding FROM faces WHERE cluster_id = ?", (cluster_id,)
    ).fetchall()
    if not rows:
        conn.execute("DELETE FROM clusters WHERE id = ?", (cluster_id,))
        return None
    mat = np.vstack([_normalize(_unpack(r["embedding"])) for r in rows])
    centroid = mat.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    conn.execute(
        "UPDATE clusters SET centroid = ?, n = ?, updated_at = ? WHERE id = ?",
        (_pack(centroid), len(rows), _now(), cluster_id),
    )
    return len(rows)


def merge(conn, source_id, target_id):
    if source_id == target_id:
        raise ValueError("choose two different people")
    src = conn.execute("SELECT id, name, auto_name FROM clusters WHERE id = ?", (source_id,)).fetchone()
    dst = conn.execute("SELECT id, name, auto_name FROM clusters WHERE id = ?", (target_id,)).fetchone()
    if not src or not dst:
        raise ValueError("person not found")
    name, auto = dst["name"], int(dst["auto_name"])
    if auto and not int(src["auto_name"]) and src["name"] not in ("Unsorted",) and not src["name"].startswith("Person "):
        name, auto = src["name"], 0
    conn.execute("UPDATE faces SET cluster_id = ? WHERE cluster_id = ?", (target_id, source_id))
    conn.execute("DELETE FROM clusters WHERE id = ?", (source_id,))
    conn.execute(
        "UPDATE clusters SET name = ?, auto_name = ?, updated_at = ? WHERE id = ?",
        (name, auto, _now(), target_id),
    )
    recompute(conn, target_id)
    return {"id": target_id, "name": name}


def merge_same_name(conn, name: str):
    """Manually merge all clusters that share an exact name into one.

    Chooses the largest cluster as the target and merges every other same-named
    cluster into it. Returns the target cluster id and how many merges happened.
    Safe because the user explicitly names the group to merge.
    """
    rows = conn.execute(
        "SELECT id, n FROM clusters WHERE name = ? ORDER BY n DESC, id ASC", (name,)
    ).fetchall()
    if len(rows) < 2:
        return {"id": rows[0]["id"] if rows else None, "merges": 0, "name": name}
    target_id = int(rows[0]["id"])
    merges = 0
    for row in rows[1:]:
        merge(conn, row["id"], target_id)
        merges += 1
    return {"id": target_id, "merges": merges, "name": name}


def detach(conn, face_id):
    face = conn.execute(
        "SELECT id, embedding, cluster_id FROM faces WHERE id = ?", (face_id,)
    ).fetchone()
    if not face:
        raise ValueError("face not found")
    emb = _normalize(_unpack(face["embedding"]))
    cur = conn.execute(
        "INSERT INTO clusters(name, centroid, n, auto_name, created_at, updated_at) "
        "VALUES('Unsorted', ?, 1, 1, ?, ?)",
        (_pack(emb), _now(), _now()),
    )
    new_id = int(cur.lastrowid)
    old_id = face["cluster_id"]
    conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?", (new_id, face_id))
    if old_id is not None:
        recompute(conn, old_id)
    return new_id
