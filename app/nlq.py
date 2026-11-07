import json
import os
import re
from datetime import date, datetime, timedelta

import httpx

from . import db as dbmod

BANDS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
PLACE_STOP = {
    "the", "a", "an", "at", "in", "on", "of", "and", "or", "with", "together",
    "photo", "photos", "picture", "pictures", "pic", "pics", "image", "images",
    "video", "videos", "clip", "clips", "movie", "movies", "show", "find", "me",
    "all", "my", "our", "from", "to", "for", "between", "during", "where",
    "this", "that", "these", "those", "last", "next", "year", "month", "week",
    "day", "yesterday", "today", "same", "both", "either", "not", "without",
    "except", "only", "was", "were", "is", "are", "taken", "clicked", "shot",
}
DATE_WORDS = set(MONTHS) | {
    "year", "month", "week", "day", "yesterday", "today", "last", "this",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}


def _like(term):
    safe = str(term).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{safe}%"


def load_people(conn):
    return [
        dict(r)
        for r in conn.execute(
            "SELECT id, name, n, auto_name FROM clusters WHERE n >= 2 ORDER BY n DESC, name"
        ).fetchall()
    ]


def load_places(conn, limit=None):
    sql = "SELECT place, COUNT(*) AS n FROM media WHERE place IS NOT NULL GROUP BY place ORDER BY n DESC"
    params = []
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def resolve_person(value, people):
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.isdigit():
        for p in people:
            if p["id"] == int(raw):
                return p
    target = _norm(raw)
    for p in people:
        if _norm(p["name"]) == target:
            return p
    for p in people:
        if target and target in _norm(p["name"]):
            return p
    try:
        import difflib

        names = {_norm(p["name"]): p for p in people}
        match = difflib.get_close_matches(target, list(names), n=1, cutoff=0.72)
        if match:
            return names[match[0]]
    except Exception:
        pass
    return None


def _find_people_in_text(text, people):
    """Return people in order of appearance plus their spans, skipping negated ones."""
    found = []
    working = text
    occupied = []
    for p in sorted(people, key=lambda x: -len(x["name"])):
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(p["name"]) + r"(?![A-Za-z0-9])", re.I)
        for m in pattern.finditer(working):
            if any(not (m.end() <= s or m.start() >= e) for s, e in occupied):
                continue
            occupied.append((m.start(), m.end()))
            before = working[max(0, m.start() - 24): m.start()].lower()
            negated = bool(re.search(r"(without|except|not|no)\s+(?:the\s+)?$", before))
            found.append((m.start(), p, negated))
    found.sort(key=lambda item: item[0])
    return found, occupied


def _spans_removed(text, spans):
    chars = list(text)
    for s, e in spans:
        for i in range(s, min(e, len(chars))):
            chars[i] = " "
    return "".join(chars)


def _dates_from_rules(text, today):
    low = text.lower()
    dfrom = dto = None

    m = re.search(
        r"\b(?:from\s+)?((?:19|20)\d{2})\s*(?:to|until|through|till|-|–)\s*((?:19|20)\d{2})\b",
        low,
    )
    if m:
        a, b = sorted((int(m.group(1)), int(m.group(2))))
        dfrom, dto = date(a, 1, 1), date(b, 12, 31)

    m = re.search(r"\b(19|20)\d{2}\b", low)
    if dfrom is None and m:
        year = int(m.group(0))
        dfrom, dto = date(year, 1, 1), date(year, 12, 31)

    m = re.search(r"\b(" + "|".join(MONTHS) + r")\s+((?:19|20)\d{2})\b", low)
    if m:
        month = MONTHS[m.group(1)]
        year = int(m.group(2))
        start = date(year, month, 1)
        end = date(year + (month == 12), 1 if month == 12 else month + 1, 1) - timedelta(days=1)
        dfrom, dto = start, end

    for kw, delta_from, delta_to in (
        ("today", 0, 0),
        ("yesterday", 1, 1),
        ("last week", 7, 7),
        ("this week", None, None),
        ("last month", None, None),
        ("this month", None, None),
        ("last year", None, None),
        ("this year", None, None),
    ):
        if kw not in low:
            continue
        if kw == "today":
            dfrom = dto = today
        elif kw == "yesterday":
            d = today - timedelta(days=1)
            dfrom = dto = d
        elif kw == "last week":
            start = today - timedelta(days=today.weekday() + 7)
            end = start + timedelta(days=6)
            dfrom, dto = start, end
        elif kw == "this week":
            start = today - timedelta(days=today.weekday())
            dfrom, dto = start, today
        elif kw == "last month":
            first = today.replace(day=1)
            end = first - timedelta(days=1)
            dfrom, dto = end.replace(day=1), end
        elif kw == "this month":
            dfrom, dto = today.replace(day=1), today
        elif kw == "last year":
            dfrom, dto = date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)
        elif kw == "this year":
            dfrom, dto = date(today.year, 1, 1), today
        break

    m = re.search(r"\bbetween\s+((?:19|20)\d{2})\s+and\s+((?:19|20)\d{2})\b", low)
    if m:
        a, b = sorted((int(m.group(1)), int(m.group(2))))
        dfrom, dto = date(a, 1, 1), date(b, 12, 31)

    if dfrom is None:
        for token in ("spring", "summer", "autumn", "winter", "monsoon", "diwali", "holi", "christmas"):
            m = re.search(token + r"\s+((?:19|20)\d{2})", low)
            if m:
                year = int(m.group(1))
                if token in ("christmas", "diwali", "holi"):
                    dfrom, dto = date(year, 10, 1), date(year, 12, 31)
                elif token in ("winter",):
                    dfrom, dto = date(year, 11, 1), date(year + 1, 2, 28)
                else:
                    dfrom, dto = date(year, 1, 1), date(year, 12, 31)
                break

    return (dfrom.isoformat() if dfrom else None, dto.isoformat() if dto else None)


def _bands_from_text(low):
    bands = []
    groups = (
        (("baby", "babies", "toddler", "toddlers", "infant"), ["0-2"]),
        (("child", "children", "kid", "kids"), ["3-9", "10-19"]),
        (("teen", "teens", "teenager", "teenagers"), ["10-19"]),
        (("twenties", "in their 20s"), ["20-29"]),
        (("thirties", "in their 30s"), ["30-39"]),
        (("forties", "in their 40s"), ["40-49"]),
        (("fifties", "in their 50s"), ["50-59"]),
        (("elderly", "old people", "senior", "seniors"), ["60-69", "70+"]),
    )
    for words, values in groups:
        if any(re.search(r"\b" + re.escape(w) + r"\b", low) for w in words):
            bands.extend(values)
    return list(dict.fromkeys(bands))


def _place_from_text(text, places):
    tokens = set(_norm(text).split())
    tokens = {t for t in tokens if len(t) > 2 and t not in PLACE_STOP and not t.isdigit()}
    best = None
    best_score = 0.0
    for place in places:
        ptokens = [t for t in _norm(place["place"]).split() if len(t) > 2 and t not in PLACE_STOP]
        if not ptokens:
            continue
        hits = len([t for t in ptokens if t in tokens])
        if not hits:
            continue
        score = hits / len(ptokens) + 0.001 * hits
        if score > best_score:
            best, best_score = place["place"], score
    return best


def parse_rules(query, people, places, today=None):
    today = today or date.today()
    low = query.lower()
    found, spans = _find_people_in_text(query, people)
    include, exclude = [], []
    for _, p, negated in found:
        bucket = exclude if negated else include
        if not any(x["id"] == p["id"] for x in bucket):
            bucket.append({"id": p["id"], "name": p["name"], "n": p["n"]})

    mode = "all"
    if len(include) <= 1:
        mode = "any"
    elif re.search(r"\beither\b|\bor\b", low):
        mode = "any"

    rest = _spans_removed(query, spans)
    dfrom, dto = _dates_from_rules(rest, today)
    bands = _bands_from_text(low)
    place = _place_from_text(rest, places)

    unknown = []
    for m in re.finditer(r"\bperson\s+([0-9]+|[A-Za-z][\w'-]*)", rest, re.I):
        if resolve_person(m.group(1), people) is None:
            label = " ".join(m.group(0).split()).title()
            if label not in unknown:
                unknown.append(label)

    kind = None
    if re.search(r"\b(video|videos|clip|clips|movie|movies)\b", low) and not re.search(r"\bphoto|picture|image", low):
        kind = "video"
    elif re.search(r"\b(photo|photos|picture|pictures|image|images|pic|pics)\b", low):
        kind = "image"

    sort = "oldest" if re.search(r"\b(oldest|earliest)\b", low) else "newest"

    bits = []
    if include:
        bits.append(" + ".join(p["name"] for p in include) + (" together" if mode == "all" and len(include) > 1 else ""))
    if exclude:
        bits.append("without " + ", ".join(p["name"] for p in exclude))
    if place:
        bits.append(f"at {place}")
    if dfrom:
        bits.append(f"from {dfrom} to {dto}" if dto and dto != dfrom else f"on {dfrom}")
    if bands:
        bits.append("ages " + ", ".join(bands))
    if kind:
        bits.append(kind + "s")
    if unknown:
        bits.append("could not find " + ", ".join(unknown))
    explanation = " · ".join(bits) if bits else "everything, newest first"

    return {
        "query": query,
        "people": include,
        "mode": mode,
        "exclude": exclude,
        "unknown": unknown,
        "place": place,
        "from": dfrom,
        "to": dto,
        "bands": bands,
        "kind": kind,
        "sort": sort,
        "explanation": explanation,
        "source": "quick",
        "model": None,
        "notes": [],
    }


def llm_settings():
    return (
        os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"),
        os.environ.get("MEDIA_SORTER_LLM", "qwen2.5:3b-instruct"),
    )


def llm_available():
    if os.environ.get("MEDIA_SORTER_LLM", "").lower() in ("off", "0", "none"):
        return False
    base, model = llm_settings()
    try:
        r = httpx.get(base + "/api/tags", timeout=3.0)
        if r.status_code != 200:
            return False
        names = [m.get("name", "") for m in r.json().get("models", [])]
        return any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)
    except Exception:
        return False


SYSTEM_PROMPT = """You are the search planner for a private, offline photo library.
Turn the person's sentence into one JSON search plan. You do not answer with photos; you only plan.

Today is {today} ({weekday}).

People in the library (name: number of photos):
{people}

Folders, which act as places (folder: number of files):
{places}

Rules:
- Use person names only from the list. "person 1" means exactly "Person 1". Put names you cannot find in "unknown_people".
- "A and B", "A with B", "A + B", "both A and B", "A and B together" -> mode "all" (same photo).
- "A or B", "either A or B", "any of A and B" -> mode "any".
- "without A", "not A", "except A" -> exclude_people.
- Place is a folder keyword. Prefer a folder from the list; if the person names a place not listed, keep their words so the path can be searched.
- Dates must be YYYY-MM-DD. Use today's date for relative words: last month, this year, last week, yesterday.
- "photos" -> kind "image"; "videos" or "clips" -> "video"; unstated -> "any".
- Age words: baby/toddler -> ["0-2"]; kids/children -> ["3-9","10-19"]; teenager -> ["10-19"]; elderly -> ["60-69","70+"].
- Never invent people, dates, or places. If a field is not requested, use null, [] or "any".
- explanation: one short sentence in plain words, at most 20 words.

Return only this JSON object:
{{"people": [], "mode": "all", "exclude_people": [], "unknown_people": [], "place": null,
"from": null, "to": null, "age_bands": [], "kind": "any", "sort": "newest", "explanation": ""}}"""


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


def llm_plan(query, people, places, today=None):
    today = today or date.today()
    base, model = llm_settings()
    people_lines = "\n".join(f"{p['name']}: {p['n']}" for p in people)
    place_lines = "\n".join(f"{p['place']}: {p['n']}" for p in places[:90]) or "(none)"
    system = SYSTEM_PROMPT.format(
        today=today.isoformat(),
        weekday=today.strftime("%A"),
        people=people_lines,
        places=place_lines,
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 320, "num_ctx": 8192},
        "keep_alive": "30m",
    }
    r = httpx.post(base + "/api/chat", json=payload, timeout=60.0)
    r.raise_for_status()
    data = r.json()
    content = (data.get("message") or {}).get("content", "")
    return _extract_json(content)


def intent_from_plan(plan, query, people, today=None):
    today = today or date.today()
    include, unknown = [], []
    for value in plan.get("people") or []:
        p = resolve_person(value, people)
        if p:
            if not any(x["id"] == p["id"] for x in include):
                include.append({"id": p["id"], "name": p["name"], "n": p["n"]})
        else:
            unknown.append(str(value))

    exclude = []
    for value in plan.get("exclude_people") or []:
        p = resolve_person(value, people)
        if p and not any(x["id"] == p["id"] for x in exclude):
            exclude.append({"id": p["id"], "name": p["name"], "n": p["n"]})

    for value in plan.get("unknown_people") or []:
        if str(value) not in unknown:
            unknown.append(str(value))

    mode = "all" if str(plan.get("mode", "all")).lower() == "all" else "any"
    if len(include) <= 1:
        mode = "any"

    def _valid_date(value):
        if not value:
            return None
        try:
            return datetime.strptime(str(value)[:10], "%Y-%m-%d").date().isoformat()
        except ValueError:
            return None

    dfrom, dto = _valid_date(plan.get("from")), _valid_date(plan.get("to"))
    if dfrom and dto and dfrom > dto:
        dfrom, dto = dto, dfrom

    bands = [b for b in (plan.get("age_bands") or []) if b in BANDS]
    kind = str(plan.get("kind") or "any").lower()
    kind = kind if kind in ("image", "video") else None
    place = str(plan.get("place")).strip()[:80] if plan.get("place") else None
    sort = "oldest" if str(plan.get("sort", "newest")).lower() == "oldest" else "newest"

    explanation = str(plan.get("explanation") or "").strip()[:160]
    if not explanation:
        explanation = _explain(include, exclude, place, dfrom, dto, bands, kind, mode)

    return {
        "query": query,
        "people": include,
        "mode": mode,
        "exclude": exclude,
        "unknown": unknown,
        "place": place,
        "from": dfrom,
        "to": dto,
        "bands": bands,
        "kind": kind,
        "sort": sort,
        "explanation": explanation,
        "source": "ai",
        "model": llm_settings()[1],
        "notes": [],
    }


def _explain(include, exclude, place, dfrom, dto, bands, kind, mode):
    bits = []
    if include:
        bits.append(" + ".join(p["name"] for p in include) + (" together" if mode == "all" and len(include) > 1 else ""))
    if exclude:
        bits.append("without " + ", ".join(p["name"] for p in exclude))
    if place:
        bits.append(f"at {place}")
    if dfrom:
        bits.append(f"{dfrom} to {dto}" if dto and dto != dfrom else dfrom)
    if bands:
        bits.append("ages " + ", ".join(bands))
    if kind:
        bits.append(kind + "s")
    return " · ".join(bits) if bits else "everything, newest first"


def _has_constraints(intent):
    return bool(
        intent["people"] or intent["exclude"] or intent["place"] or intent["from"]
        or intent["to"] or intent["bands"] or intent["kind"]
    )


def understand(conn, query, use_llm=True, today=None):
    people, places = load_people(conn), load_places(conn)
    quick = parse_rules(query, people, places, today)
    if not use_llm or not llm_available():
        quick["notes"].append("Local AI is offline, so quick matching was used.")
        return quick
    try:
        plan = llm_plan(query, people, places, today)
        intent = intent_from_plan(plan, query, people, today)
    except Exception as exc:
        quick["notes"].append(f"Local AI could not answer ({type(exc).__name__}); quick matching was used.")
        return quick
    if not _has_constraints(intent) and _has_constraints(quick):
        quick["notes"].append("The AI found no filters, so quick matching was used.")
        return quick
    return intent


def build_sql(intent):
    clauses, params = [], []
    ids = [p["id"] for p in intent["people"]]
    if ids:
        marks = ",".join("?" * len(ids))
        if intent["mode"] == "all" and len(ids) > 1:
            clauses.append(
                f"m.id IN (SELECT media_id FROM faces WHERE cluster_id IN ({marks}) "
                f"GROUP BY media_id HAVING COUNT(DISTINCT cluster_id) = {len(ids)})"
            )
        else:
            clauses.append(f"m.id IN (SELECT media_id FROM faces WHERE cluster_id IN ({marks}))")
        params.extend(ids)
    if intent["exclude"]:
        marks = ",".join("?" * len(intent["exclude"]))
        clauses.append(f"m.id NOT IN (SELECT media_id FROM faces WHERE cluster_id IN ({marks}))")
        params.extend([p["id"] for p in intent["exclude"]])
    if intent["place"]:
        clauses.append("(m.place LIKE ? ESCAPE '\\' OR m.path LIKE ? ESCAPE '\\')")
        term = _like(intent["place"])
        params.extend([term, term])
    if intent["from"]:
        clauses.append("m.taken_at >= ?")
        params.append(intent["from"] + " 00:00:00")
    if intent["to"]:
        clauses.append("m.taken_at <= ?")
        params.append(intent["to"] + " 23:59:59")
    if intent["bands"]:
        marks = ",".join("?" * len(intent["bands"]))
        clauses.append(
            f"EXISTS (SELECT 1 FROM faces f WHERE f.media_id = m.id AND f.age_band IN ({marks}))"
        )
        params.extend(intent["bands"])
    if intent["kind"]:
        clauses.append("m.kind = ?")
        params.append(intent["kind"])
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    order = "ASC" if intent["sort"] == "oldest" else "DESC"
    return where, params, order


def search_media(conn, intent, page=1, page_size=48):
    where, params, order = build_sql(intent)
    total = conn.execute(f"SELECT COUNT(*) AS c FROM media m{where}", params).fetchone()["c"]
    rows = conn.execute(
        f"SELECT m.*, (SELECT COUNT(*) FROM faces f WHERE f.media_id = m.id) AS face_count "
        f"FROM media m{where} ORDER BY m.taken_at {order}, m.id {order} LIMIT ? OFFSET ?",
        params + [page_size, (page - 1) * page_size],
    ).fetchall()
    return total, rows


def seek_times(conn, media_ids, cluster_ids):
    if not media_ids or not cluster_ids:
        return {}
    marks = ",".join("?" * len(cluster_ids))
    qmarks = ",".join("?" * len(media_ids))
    rows = conn.execute(
        f"SELECT media_id, MIN(frame_time) AS t FROM faces WHERE frame_time IS NOT NULL "
        f"AND cluster_id IN ({marks}) AND media_id IN ({qmarks}) GROUP BY media_id",
        list(cluster_ids) + list(media_ids),
    ).fetchall()
    return {r["media_id"]: r["t"] for r in rows}
