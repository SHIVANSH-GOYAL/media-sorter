import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import clustering, config, db as dbmod, scanner
from app.faceengine import age_band, _fairface_classes
from app import dupes, nlq


@pytest.fixture()
def library(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(config, "DATA", data)
    monkeypatch.setattr(config, "DB_PATH", data / "library.db")
    monkeypatch.setattr(config, "THUMBS", data / "thumbs")
    monkeypatch.setattr(config, "FACES", data / "faces")
    config.ensure_dirs()
    dbmod.init_db()
    yield tmp_path


def _make_photo(path, color=(120, 40, 30)):
    arr = np.zeros((240, 320, 3), dtype=np.uint8)
    arr[:, :] = color
    arr[40:200, 80:240] = (200, 180, 160)
    Image.fromarray(arr).save(path, "JPEG", quality=90)


def test_scan_is_incremental_and_dedupes(library):
    folder = library / "photos"
    folder.mkdir()
    _make_photo(folder / "a.jpg")
    _make_photo(folder / "b.jpg", color=(10, 120, 200))
    shutil.copy(folder / "a.jpg", folder / "a_copy.jpg")

    first = scanner.scan(folder, engine=None, progress=False)
    assert first["processed"] == 2
    assert first["duplicates"] == 1
    assert first["errors"] == 0
    assert first["faces"] == 0

    with dbmod.db() as conn:
        assert conn.execute("SELECT COUNT(*) AS c FROM media").fetchone()["c"] == 2
        rows = conn.execute("SELECT id, thumb, taken_at FROM media").fetchall()
        assert all(r["thumb"] and (config.DATA / r["thumb"]).exists() for r in rows)
        assert all(r["taken_at"] for r in rows)

    second = scanner.scan(folder, engine=None, progress=False)
    assert second["processed"] == 0
    assert second["skipped"] == 2


def test_clustering_assigns_and_creates_clusters(library):
    rng = np.random.default_rng(3)
    base = rng.normal(size=512).astype(np.float32)
    base /= np.linalg.norm(base)
    near = base + rng.normal(scale=0.02, size=512).astype(np.float32)
    near /= np.linalg.norm(near)
    other = rng.normal(size=512).astype(np.float32)
    other /= np.linalg.norm(other)

    with dbmod.db() as conn:
        a = clustering.assign(conn, base, 0.45)
        b = clustering.assign(conn, near, 0.45)
        c = clustering.assign(conn, other, 0.45)
        assert a == b
        assert c != a
        rows = {r["id"]: r for r in conn.execute("SELECT id, n, centroid FROM clusters")}
        assert rows[a]["n"] == 2
        assert abs(float(np.linalg.norm(np.frombuffer(rows[a]["centroid"], dtype=np.float32))) - 1.0) < 1e-4


def test_recluster_keeps_custom_names(library):
    with dbmod.db() as conn:
        v = np.ones(512, dtype=np.float32) / np.sqrt(512)
        w = -v
        cur = conn.execute(
            "INSERT INTO media(path, size, mtime, indexed_at) VALUES('x.jpg', 1, 1, '2026-01-01')"
        )
        mid = int(cur.lastrowid)
        fids = []
        for emb in (v, v, w):
            cur = conn.execute(
                "INSERT INTO faces(media_id, embedding) VALUES(?, ?)", (mid, emb.tobytes())
            )
            fids.append(int(cur.lastrowid))
        cid = clustering.assign(conn, v, 0.45)
        conn.execute("UPDATE faces SET cluster_id = ? WHERE id IN (?, ?)", (cid, fids[0], fids[1]))
        conn.execute("UPDATE clusters SET name = 'Alice', auto_name = 0 WHERE id = ?", (cid,))
        cid2 = clustering.assign(conn, w, 0.45)
        conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?", (cid2, fids[2]))
        result = clustering.recluster(conn, 0.45)
        assert result["faces"] == 3
        names = {r["name"] for r in conn.execute("SELECT name FROM clusters")}
        assert "Alice" in names


def test_age_band_mapping():
    assert age_band(1) == "0-2"
    assert age_band(24) == "20-29"
    assert age_band(34.5) == "30-39"
    assert age_band(75) == "70+"


def test_fairface_class_id_mapping():
    assert _fairface_classes(np.array([0]), np.array([1]), np.array([4])) == (34.5, "30-39", "female", 0.5)
    assert _fairface_classes(np.array([6]), np.array([0]), np.array([0])) == (1, "0-2", "male", 0.5)


def test_recluster_merges_duplicate_clusters_and_keeps_distinct_person(library):
    rng = np.random.default_rng(44)
    base = rng.normal(size=512).astype(np.float32)
    base /= np.linalg.norm(base)
    other = rng.normal(size=512).astype(np.float32)
    other /= np.linalg.norm(other)
    vectors = [
        base,
        _normalize_test(base + rng.normal(scale=0.04, size=512).astype(np.float32)),
        _normalize_test(base + rng.normal(scale=0.04, size=512).astype(np.float32)),
        _normalize_test(base + rng.normal(scale=0.04, size=512).astype(np.float32)),
        other,
        _normalize_test(other + rng.normal(scale=0.03, size=512).astype(np.float32)),
    ]
    with dbmod.db() as conn:
        mid = conn.execute(
            "INSERT INTO media(path, size, mtime, indexed_at) VALUES('faces.jpg', 1, 1, 'now')"
        ).lastrowid
        face_ids = []
        for emb in vectors:
            face_ids.append(conn.execute(
                "INSERT INTO faces(media_id, embedding, det_score) VALUES(?, ?, 0.9)",
                (mid, emb.tobytes()),
            ).lastrowid)
        c1 = conn.execute("INSERT INTO clusters(name, centroid, n) VALUES('Person 1', ?, 2)", (base.tobytes(),)).lastrowid
        c2 = conn.execute("INSERT INTO clusters(name, centroid, n) VALUES('Person 2', ?, 2)", (base.tobytes(),)).lastrowid
        c3 = conn.execute("INSERT INTO clusters(name, centroid, n) VALUES('Person 3', ?, 2)", (other.tobytes(),)).lastrowid
        for fid, cid in zip(face_ids, (c1, c1, c2, c2, c3, c3)):
            conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?", (cid, fid))
        result = clustering.recluster(conn, 0.45, exemplars=4, max_iter=10)
        assert result["clusters"] == 2
        assert conn.execute("SELECT COUNT(*) FROM clusters WHERE n >= 2").fetchone()[0] == 2


def _normalize_test(vec):
    return vec / np.linalg.norm(vec)


def test_gallery_and_thumb(library):
    folder = library / "photos"
    folder.mkdir()
    _make_photo(folder / "a.jpg")
    scanner.scan(folder, engine=None, progress=False)

    from app.web import app

    client = TestClient(app)
    home = client.get("/")
    assert home.status_code == 200
    assert "Media Sorter" in home.text
    assert "/thumb/1" in home.text
    thumb = client.get("/thumb/1")
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/jpeg"
    assert client.get("/api/stats").json()["media"] == 1
    assert client.get("/healthz").json() == {"ok": True}

    with dbmod.db() as conn:
        conn.execute(
            "INSERT INTO clusters(name, centroid, n, auto_name) VALUES('Alice', ?, 2, 0)",
            (np.zeros(512, dtype=np.float32).tobytes(),),
        )
        cid = conn.execute("SELECT MAX(id) AS id FROM clusters").fetchone()["id"]

    people = client.get("/people")
    assert people.status_code == 200
    assert "Alice" in people.text
    assert client.get("/people?q=nobody").status_code == 200

    renamed = client.post(
        f"/api/people/{cid}/rename",
        data={"name": "Alicia", "next": "/people?q=Ali"},
        follow_redirects=False,
    )
    assert renamed.status_code == 303
    assert renamed.headers["location"] == "/people?q=Ali"
    with dbmod.db() as conn:
        assert conn.execute("SELECT name FROM clusters WHERE id = ?", (cid,)).fetchone()["name"] == "Alicia"


def test_v2_rules_sql_and_dupes(library):
    people = [{"id": 1, "name": "Alice", "n": 4}, {"id": 2, "name": "Bob", "n": 3}]
    places = [{"place": "College / Sports meet", "n": 4}]
    intent = nlq.parse_rules("Alice and Bob at Sports meet", people, places)
    assert intent["mode"] == "all"
    where, params, order = nlq.build_sql(intent)
    assert "HAVING COUNT(DISTINCT cluster_id) = 2" in where
    assert params[:2] == [1, 2]
    assert order == "DESC"
    groups = dupes.group_rows([
        {"id": 1, "dhash": "0000000000000000"},
        {"id": 2, "dhash": "0000000000000001"},
        {"id": 3, "dhash": "ffffffffffffffff"},
    ], max_distance=1)
    assert [len(g) for g in groups] == [2]


def test_v2_pages_and_favorite(library):
    folder = library / "photos"
    folder.mkdir()
    _make_photo(folder / "a.jpg")
    scanner.scan(folder, engine=None, progress=False)
    from app.web import app
    client = TestClient(app)
    assert client.get("/search?q=photos").status_code == 200
    response = client.post("/api/media/1/favorite", data={"next": "/"}, follow_redirects=False)
    assert response.status_code == 303
    with dbmod.db() as conn:
        assert conn.execute("SELECT favorite FROM media WHERE id=1").fetchone()["favorite"] == 1


def test_gallery_media_type_and_single_person_filters(library):
    from app.web import app
    client = TestClient(app)
    with dbmod.db() as conn:
        conn.execute("INSERT INTO media(path, size, mtime, kind, indexed_at) VALUES('a.jpg', 1, 1, 'image', 'now')")
        conn.execute("INSERT INTO media(path, size, mtime, kind, indexed_at) VALUES('b.mp4', 1, 1, 'video', 'now')")
        cid = conn.execute(
            "INSERT INTO clusters(name, centroid, n, auto_name) VALUES('Alice', ?, 2, 0)",
            (np.zeros(512, dtype=np.float32).tobytes(),),
        ).lastrowid
        cid_b = conn.execute(
            "INSERT INTO clusters(name, centroid, n, auto_name) VALUES('Bob', ?, 2, 0)",
            (np.zeros(512, dtype=np.float32).tobytes(),),
        ).lastrowid
        emb = np.zeros(512, dtype=np.float32).tobytes()
        conn.execute("INSERT INTO faces(media_id, embedding, cluster_id) VALUES(1, ?, ?)", (emb, cid))
        # media 3: Alice + Bob (two different people) -> excluded by single-person filter
        conn.execute(
            "INSERT INTO media(path, size, mtime, kind, indexed_at) VALUES('c.jpg', 1, 1, 'image', 'now')"
        )
        conn.execute("INSERT INTO faces(media_id, embedding, cluster_id) VALUES(3, ?, ?)", (emb, cid))
        conn.execute("INSERT INTO faces(media_id, embedding, cluster_id) VALUES(3, ?, ?)", (emb, cid_b))
    assert "2 match" in client.get("/?kind=image").text
    assert "1 match" in client.get("/?kind=video").text
    # media 1 has only Alice -> included by single-person filter; media 3 has two people -> excluded
    assert "1 match" in client.get("/?person=1&only=1").text
    # without only, both media 1 and 3 show for Alice
    assert "2 match" in client.get("/?person=1").text
