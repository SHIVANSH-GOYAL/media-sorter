from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from . import clustering, config, db as dbmod, dupes, nlq

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
PAGE_SIZE = 72
PEOPLE_PAGE = 60
FACE_PAGE = 60
BAND_ORDER = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

app = FastAPI(title="Media Sorter")


def _filters(request: Request):
    q = request.query_params
    person = q.get("person", "").strip()
    band = q.get("band", "").strip()
    dfrom = q.get("dfrom", "").strip()
    dto = q.get("dto", "").strip()
    text = q.get("text", "").strip()
    fav = q.get("fav", "") in ("1", "on", "true")
    kind = q.get("kind", "").strip().lower()
    if kind not in ("image", "video"):
        kind = ""
    only = q.get("only", "") in ("1", "on", "true")
    try:
        page = max(1, int(q.get("page", "1")))
    except ValueError:
        page = 1
    return person, band, dfrom, dto, text, fav, kind, only, page


def _where(person, band, dfrom, dto, text="", fav=False, kind="", only=False):
    clauses = []
    params = []
    if band:
        clauses.append("EXISTS (SELECT 1 FROM faces f WHERE f.media_id = m.id AND f.age_band = ?)")
        params.append(band)
    if person == "unsorted":
        clauses.append(
            "EXISTS (SELECT 1 FROM faces f WHERE f.media_id = m.id AND f.cluster_id IN "
            "(SELECT id FROM clusters WHERE n = 1))"
        )
    elif person.isdigit():
        clauses.append("EXISTS (SELECT 1 FROM faces f WHERE f.media_id = m.id AND f.cluster_id = ?)")
        params.append(int(person))
    if dfrom:
        clauses.append("m.taken_at >= ?")
        params.append(dfrom + " 00:00:00")
    if dto:
        clauses.append("m.taken_at <= ?")
        params.append(dto + " 23:59:59")
    if text:
        term = nlq._like(text)
        clauses.append("(m.place LIKE ? ESCAPE '\\' OR m.path LIKE ? ESCAPE '\\' OR m.camera LIKE ? ESCAPE '\\')")
        params.extend([term, term, term])
    if fav:
        clauses.append("m.favorite = 1")
    if kind:
        clauses.append("m.kind = ?")
        params.append(kind)
    if only and person.isdigit():
        clauses.append(
            "(SELECT COUNT(DISTINCT f2.cluster_id) FROM faces f2 "
            "WHERE f2.media_id = m.id AND f2.cluster_id IS NOT NULL) = 1 "
            "AND EXISTS (SELECT 1 FROM faces f3 "
            "WHERE f3.media_id = m.id AND f3.cluster_id = ?)"
        )
        params.append(int(person))
    sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return sql, params


def _chips(conn, media_ids, seek_people=None):
    chips = {}
    if not media_ids:
        return chips, {}
    qmarks = ",".join("?" * len(media_ids))
    for fr in conn.execute(
        f"SELECT f.media_id, f.age_band, f.cluster_id, c.name, c.n FROM faces f "
        f"LEFT JOIN clusters c ON c.id = f.cluster_id WHERE f.media_id IN ({qmarks})",
        media_ids,
    ).fetchall():
        entry = chips.setdefault(fr["media_id"], {"names": [], "bands": set()})
        if fr["name"] and (fr["n"] or 0) >= 2:
            if fr["name"] not in entry["names"]:
                entry["names"].append(fr["name"])
        if fr["age_band"]:
            entry["bands"].add(fr["age_band"])
    chips = {
        k: {"names": v["names"], "bands": sorted(v["bands"], key=BAND_ORDER.index)}
        for k, v in chips.items()
    }
    seeks = nlq.seek_times(conn, media_ids, seek_people or [])
    return chips, seeks


def _people_all(conn):
    return conn.execute(
        "SELECT c.id, c.name, c.n, "
        "(SELECT f.id FROM faces f WHERE f.cluster_id = c.id "
        " ORDER BY f.det_score DESC LIMIT 1) AS face_id "
        "FROM clusters c WHERE c.n >= 2 ORDER BY c.n DESC, c.name LIMIT 2000"
    ).fetchall()


def _stats(conn):
    return {
        "media": conn.execute("SELECT COUNT(*) AS c FROM media").fetchone()["c"],
        "faces": conn.execute("SELECT COUNT(*) AS c FROM faces").fetchone()["c"],
        "people": conn.execute("SELECT COUNT(*) AS c FROM clusters WHERE n >= 2").fetchone()["c"],
        "unsorted": conn.execute(
            "SELECT COUNT(*) AS c FROM faces f JOIN clusters c ON c.id = f.cluster_id WHERE c.n = 1"
        ).fetchone()["c"],
        "videos": conn.execute("SELECT COUNT(*) AS c FROM media WHERE kind = 'video'").fetchone()["c"],
        "favorites": conn.execute("SELECT COUNT(*) AS c FROM media WHERE favorite = 1").fetchone()["c"],
        "first": conn.execute("SELECT MIN(taken_at) AS d FROM media").fetchone()["d"],
        "last": conn.execute("SELECT MAX(taken_at) AS d FROM media").fetchone()["d"],
    }


def _here(request):
    return str(request.url.path) + (("?" + request.url.query) if request.url.query else "")


def _safe_next(value):
    return value if value.startswith("/") and not value.startswith("//") else "/"


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
def gallery(request: Request):
    person, band, dfrom, dto, text, fav, kind, only, page = _filters(request)
    where_sql, params = _where(person, band, dfrom, dto, text, fav, kind, only)
    with dbmod.db() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS c FROM media m{where_sql}", params).fetchone()["c"]
        offset = (page - 1) * PAGE_SIZE
        rows = conn.execute(
            f"SELECT m.*, (SELECT COUNT(*) FROM faces f WHERE f.media_id = m.id) AS face_count "
            f"FROM media m{where_sql} ORDER BY m.taken_at DESC, m.id DESC LIMIT ? OFFSET ?",
            params + [PAGE_SIZE, offset],
        ).fetchall()

        people_all = _people_all(conn)
        stats = _stats(conn)
        bands = [
            r["age_band"]
            for r in conn.execute(
                "SELECT age_band, COUNT(*) AS c FROM faces WHERE age_band IS NOT NULL "
                "GROUP BY age_band ORDER BY c DESC"
            ).fetchall()
        ]
        media_ids = [r["id"] for r in rows]
        seek_people = [int(person)] if person.isdigit() else []
        chips, seeks = _chips(conn, media_ids, seek_people)

    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    base = {"person": person, "band": band, "dfrom": dfrom, "dto": dto, "text": text}
    if fav:
        base["fav"] = "1"
    if kind:
        base["kind"] = kind
    if only:
        base["only"] = "1"
    prev_qs = urlencode({**base, "page": page - 1})
    next_qs = urlencode({**base, "page": page + 1})

    return TEMPLATES.TemplateResponse(
        request,
        "gallery.html",
        {
            "rows": rows,
            "chips": chips,
            "seeks": seeks,
            "people": people_all[:60],
            "people_all": people_all,
            "unsorted_faces": stats["unsorted"],
            "here": _here(request),
            "stats": stats,
            "bands": bands,
            "band_order": BAND_ORDER,
            "sel": {"person": person, "band": band, "dfrom": dfrom, "dto": dto, "text": text, "fav": fav, "kind": kind, "only": only},
            "page": page,
            "pages": pages,
            "total": total,
            "prev_qs": prev_qs,
            "next_qs": next_qs,
        },
    )


@app.get("/search")
def search(request: Request):
    q = request.query_params.get("q", "").strip()
    quick = request.query_params.get("quick", "") == "1"
    try:
        page = max(1, int(request.query_params.get("page", "1")))
    except ValueError:
        page = 1

    with dbmod.db() as conn:
        saved = conn.execute(
            "SELECT id, name, query FROM saved_searches ORDER BY id DESC LIMIT 30"
        ).fetchall()
        suggestions = []
        if not q:
            return TEMPLATES.TemplateResponse(
                request,
                "search.html",
                {
                    "q": "",
                    "intent": None,
                    "rows": [],
                    "chips": {},
                    "seeks": {},
                    "total": 0,
                    "page": 1,
                    "pages": 1,
                    "here": _here(request),
                    "saved": saved,
                    "stats": _stats(conn),
                    "quick": quick,
                    "ai": nlq.llm_available(),
                    "models": nlq.llm_settings()[1],
                },
            )

        intent = nlq.understand(conn, q, use_llm=not quick)
        total, rows = nlq.search_media(conn, intent, page, PAGE_SIZE)
        chips, seeks = _chips(conn, [r["id"] for r in rows], [p["id"] for p in intent["people"]])
        people = nlq.load_people(conn)
        for name in intent["unknown"][:5]:
            import difflib

            names = {p["name"]: p for p in people}
            matches = difflib.get_close_matches(name, list(names), n=3, cutoff=0.45)
            suggestions.extend(matches[:2])
        stats_data = _stats(conn)

    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    base = {"q": q}
    if quick:
        base["quick"] = "1"
    return TEMPLATES.TemplateResponse(
        request,
        "search.html",
        {
            "q": q,
            "intent": intent,
            "rows": rows,
            "chips": chips,
            "seeks": seeks,
            "total": total,
            "page": page,
            "pages": pages,
            "prev_qs": urlencode({**base, "page": page - 1}),
            "next_qs": urlencode({**base, "page": page + 1}),
            "here": _here(request),
            "saved": saved,
            "stats": stats_data,
            "quick": quick,
            "ai": nlq.llm_available(),
            "models": nlq.llm_settings()[1],
            "suggestions": list(dict.fromkeys(suggestions)),
        },
    )


@app.get("/people")
def people_page(request: Request):
    q = request.query_params.get("q", "").strip()
    try:
        page = max(1, int(request.query_params.get("page", "1")))
    except ValueError:
        page = 1
    where = "WHERE c.n >= 2"
    params = []
    if q:
        where += " AND c.name LIKE ?"
        params.append(f"%{q}%")
    with dbmod.db() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS c FROM clusters c {where}", params).fetchone()["c"]
        rows = conn.execute(
            "SELECT c.id, c.name, c.n, c.auto_name, "
            "(SELECT f.id FROM faces f WHERE f.cluster_id = c.id "
            " ORDER BY f.det_score DESC LIMIT 1) AS face_id "
            f"FROM clusters c {where} ORDER BY c.n DESC, c.name LIMIT ? OFFSET ?",
            params + [PEOPLE_PAGE, (page - 1) * PEOPLE_PAGE],
        ).fetchall()
        dup_names = {r["name"] for r in conn.execute(
            "SELECT name FROM clusters WHERE n >= 2 GROUP BY name HAVING COUNT(*) > 1"
        ).fetchall()}
        pages = max(1, (total + PEOPLE_PAGE - 1) // PEOPLE_PAGE)
        base = {"q": q} if q else {}
        return TEMPLATES.TemplateResponse(
            request,
            "people.html",
            {
                "rows": rows,
                "q": q,
                "page": page,
                "pages": pages,
                "total": total,
                "prev_qs": urlencode({**base, "page": page - 1}),
                "next_qs": urlencode({**base, "page": page + 1}),
                "here": _here(request),
                "dup_names": dup_names,
                "stats": _stats(conn),
            },
        )


@app.get("/person/{cluster_id}")
def person_page(request: Request, cluster_id: int):
    try:
        page = max(1, int(request.query_params.get("page", "1")))
    except ValueError:
        page = 1
    with dbmod.db() as conn:
        person = conn.execute(
            "SELECT id, name, n, auto_name FROM clusters WHERE id = ?", (cluster_id,)
        ).fetchone()
        if not person:
            raise HTTPException(404)
        total = conn.execute(
            "SELECT COUNT(*) AS c FROM faces WHERE cluster_id = ?", (cluster_id,)
        ).fetchone()["c"]
        faces = conn.execute(
            "SELECT f.id, f.age_band, f.age, f.age_conf, f.det_score, f.frame_time, "
            "f.media_id, f.thumb, m.taken_at, m.kind, m.path, m.place "
            "FROM faces f JOIN media m ON m.id = f.media_id WHERE f.cluster_id = ? "
            "ORDER BY m.taken_at DESC, f.id DESC LIMIT ? OFFSET ?",
            (cluster_id, FACE_PAGE, (page - 1) * FACE_PAGE),
        ).fetchall()
        targets = conn.execute(
            "SELECT id, name, n FROM clusters WHERE n >= 2 AND id != ? ORDER BY n DESC LIMIT 500",
            (cluster_id,),
        ).fetchall()
    pages = max(1, (total + FACE_PAGE - 1) // FACE_PAGE)
    return TEMPLATES.TemplateResponse(
        request,
        "person.html",
        {
            "person": person,
            "faces": faces,
            "total": total,
            "targets": targets,
            "page": page,
            "pages": pages,
            "here": _here(request),
            "prev_qs": urlencode({"page": page - 1}),
            "next_qs": urlencode({"page": page + 1}),
        },
    )


@app.get("/duplicates")
def duplicates_page(request: Request):
    with dbmod.db() as conn:
        rows = conn.execute(
            "SELECT id, dhash, size, taken_at, path, thumb, favorite FROM media "
            "WHERE kind = 'image' AND dhash IS NOT NULL ORDER BY id"
        ).fetchall()
    groups = dupes.group_rows(rows, max_distance=6)
    total_groups = len(groups)
    shown = groups[:200]
    wasted = sum(sum(m["size"] or 0 for m in g[1:]) for g in groups)
    return TEMPLATES.TemplateResponse(
        request,
        "duplicates.html",
        {
            "groups": shown,
            "total_groups": total_groups,
            "wasted_mb": round(wasted / (1024 * 1024), 1),
            "here": _here(request),
        },
    )


@app.get("/thumb/{media_id}")
def thumb(media_id: int):
    with dbmod.db() as conn:
        row = conn.execute("SELECT thumb FROM media WHERE id = ?", (media_id,)).fetchone()
    if not row or not row["thumb"]:
        raise HTTPException(404)
    p = config.DATA / row["thumb"]
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/face/{face_id}")
def face_thumb(face_id: int):
    with dbmod.db() as conn:
        row = conn.execute("SELECT thumb FROM faces WHERE id = ?", (face_id,)).fetchone()
    if not row or not row["thumb"]:
        raise HTTPException(404)
    p = config.DATA / row["thumb"]
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/media/{media_id}")
def media_file(media_id: int):
    with dbmod.db() as conn:
        row = conn.execute("SELECT path FROM media WHERE id = ?", (media_id,)).fetchone()
    if not row:
        raise HTTPException(404)
    p = Path(row["path"])
    if not p.exists():
        raise HTTPException(404, detail="Original file is missing")
    return FileResponse(p)


@app.post("/api/people/{cluster_id}/rename")
def rename(cluster_id: int, name: str = Form(""), next_url: str = Form("/", alias="next")):
    name = " ".join(name.split())[:60]
    with dbmod.db() as conn:
        row = conn.execute("SELECT id FROM clusters WHERE id = ?", (cluster_id,)).fetchone()
        if not row:
            raise HTTPException(404)
        if name:
            conn.execute(
                "UPDATE clusters SET name = ?, auto_name = 0 WHERE id = ?", (name, cluster_id)
            )
    return RedirectResponse(_safe_next(next_url), status_code=303)


@app.post("/api/people/{cluster_id}/merge")
def merge_people(cluster_id: int, target_id: int = Form(...), next_url: str = Form("/", alias="next")):
    with dbmod.db() as conn:
        try:
            result = clustering.merge(conn, cluster_id, target_id)
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc))
    target = f"/person/{result['id']}?merged=1"
    return RedirectResponse(target, status_code=303)


@app.post("/api/faces/{face_id}/detach")
def detach_face(face_id: int, next_url: str = Form("/", alias="next")):
    with dbmod.db() as conn:
        try:
            clustering.detach(conn, face_id)
        except ValueError as exc:
            raise HTTPException(404, detail=str(exc))
    return RedirectResponse(_safe_next(next_url), status_code=303)


@app.post("/api/faces/{face_id}/move")
def move_face(face_id: int, target_id: int = Form(...), next_url: str = Form("/", alias="next")):
    with dbmod.db() as conn:
        face = conn.execute("SELECT id, cluster_id FROM faces WHERE id = ?", (face_id,)).fetchone()
        if not face:
            raise HTTPException(404, detail="face not found")
        target = conn.execute("SELECT id FROM clusters WHERE id = ?", (target_id,)).fetchone()
        if not target:
            raise HTTPException(404, detail="target cluster not found")
        if face["cluster_id"] == target_id:
            return RedirectResponse(_safe_next(next_url), status_code=303)
        conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?", (target_id, face_id))
        clustering.recompute(conn, face["cluster_id"]) if face["cluster_id"] else None
        clustering.recompute(conn, target_id)
    return RedirectResponse(_safe_next(next_url), status_code=303)


@app.post("/api/faces/{face_id}/age")
def set_face_age(face_id: int, age_band: str = Form(...), next_url: str = Form("/", alias="next")):
    valid = {"0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"}
    if age_band not in valid:
        raise HTTPException(400, detail="invalid age band")
    midpoints = {"0-2":1,"3-9":6,"10-19":15,"20-29":24.5,"30-39":34.5,"40-49":44.5,"50-59":54.5,"60-69":64.5,"70+":75}
    with dbmod.db() as conn:
        face = conn.execute("SELECT id, cluster_id FROM faces WHERE id = ?", (face_id,)).fetchone()
        if not face:
            raise HTTPException(404, detail="face not found")
        conn.execute(
            "UPDATE faces SET age = ?, age_band = ?, age_conf = 1.0 WHERE id = ?",
            (midpoints[age_band], age_band, face_id),
        )
        clustering.recompute(conn, face["cluster_id"]) if face["cluster_id"] else None
    return RedirectResponse(_safe_next(next_url), status_code=303)


@app.post("/api/people/{cluster_id}/age")
def set_cluster_age(cluster_id: int, age_band: str = Form(...), next_url: str = Form("/", alias="next")):
    valid = {"0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"}
    if age_band not in valid:
        raise HTTPException(400, detail="invalid age band")
    midpoints = {"0-2":1,"3-9":6,"10-19":15,"20-29":24.5,"30-39":34.5,"40-49":44.5,"50-59":54.5,"60-69":64.5,"70+":75}
    with dbmod.db() as conn:
        row = conn.execute("SELECT id FROM clusters WHERE id = ?", (cluster_id,)).fetchone()
        if not row:
            raise HTTPException(404, detail="person not found")
        conn.execute("UPDATE faces SET age = ?, age_band = ?, age_conf = 1.0 WHERE cluster_id = ?",
                     (midpoints[age_band], age_band, cluster_id))
    return RedirectResponse(_safe_next(next_url), status_code=303)


@app.post("/api/media/{media_id}/favorite")
def toggle_favorite(media_id: int, next_url: str = Form("/", alias="next")):
    with dbmod.db() as conn:
        row = conn.execute("SELECT favorite FROM media WHERE id = ?", (media_id,)).fetchone()
        if not row:
            raise HTTPException(404)
        conn.execute("UPDATE media SET favorite = ? WHERE id = ?", (0 if row["favorite"] else 1, media_id))
    return RedirectResponse(_safe_next(next_url), status_code=303)


@app.post("/api/media/{media_id}/age")
def set_media_age(media_id: int, age_band: str = Form(...), next_url: str = Form("/", alias="next")):
    valid = {"0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"}
    if age_band not in valid:
        raise HTTPException(400, detail="invalid age band")
    midpoints = {"0-2":1,"3-9":6,"10-19":15,"20-29":24.5,"30-39":34.5,"40-49":44.5,"50-59":54.5,"60-69":64.5,"70+":75}
    with dbmod.db() as conn:
        row = conn.execute("SELECT id FROM media WHERE id = ?", (media_id,)).fetchone()
        if not row:
            raise HTTPException(404, detail="media not found")
        # Update all faces of this media item across all their clusters.
        conn.execute(
            "UPDATE faces SET age = ?, age_band = ?, age_conf = 1.0 WHERE media_id = ?",
            (midpoints[age_band], age_band, media_id),
        )
    return RedirectResponse(_safe_next(next_url), status_code=303)


@app.post("/api/people/merge-same-name")
def merge_same_name(name: str = Form(...), next_url: str = Form("/people", alias="next")):
    with dbmod.db() as conn:
        result = clustering.merge_same_name(conn, name)
    return RedirectResponse(_safe_next(next_url), status_code=303)


@app.post("/api/searches/save")
def save_search(name: str = Form(""), q: str = Form(""), next_url: str = Form("/search", alias="next")):
    name = " ".join(name.split())[:60] or q[:60]
    q = " ".join(q.split())[:300]
    if q:
        with dbmod.db() as conn:
            conn.execute(
                "INSERT INTO saved_searches(name, query, created_at) VALUES(?, ?, datetime('now'))",
                (name, q),
            )
    return RedirectResponse(_safe_next(next_url), status_code=303)


@app.post("/api/searches/{search_id}/delete")
def delete_search(search_id: int, next_url: str = Form("/search", alias="next")):
    with dbmod.db() as conn:
        conn.execute("DELETE FROM saved_searches WHERE id = ?", (search_id,))
    return RedirectResponse(_safe_next(next_url), status_code=303)


@app.post("/api/recluster")
def recluster():
    with dbmod.db() as conn:
        result = clustering.recluster(conn, dbmod.threshold(conn))
    return RedirectResponse(f"/?reclustered={result['clusters']}", status_code=303)


@app.get("/api/ready")
def ready():
    from pathlib import Path
    models_dir = config.DATA.parent / "models"
    required = ["det_10g.onnx", "w600k_r50.onnx", "fairface.onnx"]
    missing = [m for m in required if not (models_dir / m).exists()]
    venv_python = Path(".venv/Scripts/python.exe")
    db_exists = config.DB_PATH.exists()
    with dbmod.db() as conn:
        media_count = conn.execute("SELECT COUNT(*) AS c FROM media").fetchone()["c"]
    return {
        "python_venv": venv_python.exists(),
        "models": {m: (models_dir / m).exists() for m in required},
        "models_missing": missing,
        "database": db_exists,
        "media_count": media_count,
        "ready": not missing and venv_python.exists() and db_exists,
    }


@app.get("/api/stats")
def stats():
    with dbmod.db() as conn:
        data = _stats(conn)
        data["clusters"] = conn.execute("SELECT COUNT(*) AS c FROM clusters").fetchone()["c"]
        data["threshold"] = dbmod.threshold(conn)
        return data
