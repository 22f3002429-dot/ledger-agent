"""Acme ledger agent: POST {"question": "..."} -> {"answer": ...}

The LLM only turns the question into a JSON spec; all numbers are computed here.
Env vars: EXPORT_URL (signed ./export link), GEMINI_API_KEY, optional GEMINI_MODEL.
"""
import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

USD_PER_UNIT = {"USD": 1, "EUR": 1.11, "INR": 0.01245}
IST = timezone(timedelta(hours=5, minutes=30))
TTL = 300

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_cache = {"at": 0.0, "rows": None, "cat": None}
_lock = asyncio.Lock()


# ---------- engine ----------
def r2(x):
    return float(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def parse_orders(text):
    latest = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not r.get("id"):
            continue
        upd = parse_ts(r["updated_at"])
        prev = latest.get(r["id"])
        if prev is None or upd >= prev["_upd"]:
            c = parse_ts(r["created_at"]).astimezone(IST).date()
            u = upd.astimezone(IST).date()
            r["_upd"] = upd
            r["created_day"] = c.isoformat()
            r["updated_day"] = u.isoformat()
            r["month"] = c.isoformat()[:7]
            r["usd"] = float(r["amount"]) * USD_PER_UNIT.get(r["currency"], float("nan"))
            latest[r["id"]] = r
    return list(latest.values())


def catalog(rows):
    u = lambda k: sorted({r[k] for r in rows})
    return {"regions": u("region"), "products": u("product"), "currencies": u("currency")}


def _lst(v):
    if v is None:
        return None
    return [str(x).strip().lower() for x in (v if isinstance(v, list) else [v])]


def apply_filters(rows, spec, ignore_status=False):
    status = None if ignore_status else _lst(spec.get("status"))
    region, product = _lst(spec.get("region")), _lst(spec.get("product"))
    customer, currency = _lst(spec.get("customer")), _lst(spec.get("currency"))
    field = "updated_day" if spec.get("date_field") == "updated_at" else "created_day"
    lo, hi = spec.get("date_from"), spec.get("date_to")
    out = []
    for r in rows:
        if status and r["status"].lower() not in status:
            continue
        if region and r["region"].lower() not in region:
            continue
        if product and r["product"].lower() not in product:
            continue
        if customer and r["customer"].lower() not in customer:
            continue
        if currency and r["currency"].lower() not in currency:
            continue
        if lo and r[field] < lo:
            continue
        if hi and r[field] > hi:
            continue
        out.append(r)
    return out


def aggregate(rows, spec):
    a = spec.get("aggregate") or "sum_usd"
    if a == "count":
        return len(rows)
    if a == "sum_qty":
        return sum(r["qty"] for r in rows)
    if a == "distinct_customers":
        return len({r["customer"] for r in rows})
    if a == "distinct_products":
        return len({r["product"] for r in rows})
    if a == "sum_usd":
        return r2(sum(r["usd"] for r in rows))
    if a == "avg_usd":
        return r2(sum(r["usd"] for r in rows) / len(rows)) if rows else 0
    if a == "max_usd":
        return r2(max(r["usd"] for r in rows)) if rows else 0
    if a == "min_usd":
        return r2(min(r["usd"] for r in rows)) if rows else 0
    raise ValueError("unknown aggregate " + str(a))


def run_spec(all_rows, spec):
    if spec.get("aggregate") == "refund_rate":
        base = apply_filters(all_rows, spec, ignore_status=True)
        if not base:
            return 0
        return r2(100 * sum(r["status"].lower() == "refunded" for r in base) / len(base))
    rows = apply_filters(all_rows, spec)
    gb = spec.get("group_by")
    if not gb:
        return aggregate(rows, spec)
    if gb not in ("region", "product", "customer", "month", "currency", "status"):
        raise ValueError("unknown group_by " + str(gb))
    groups = {}
    for r in rows:
        groups.setdefault(r[gb], []).append(r)
    pairs = [(k, aggregate(v, spec)) for k, v in groups.items()]
    desc = spec.get("sort") != "asc"
    pairs.sort(key=lambda p: p[0])                       # deterministic tie-break
    pairs.sort(key=lambda p: p[1], reverse=desc)         # stable sort keeps key order on ties
    if spec.get("limit"):
        pairs = pairs[: int(spec["limit"])]
    ret = spec.get("return")
    if ret == "key":
        return pairs[0][0] if pairs else None
    if ret == "keys":
        return [p[0] for p in pairs]
    if ret == "value":
        return pairs[0][1] if pairs else 0
    return dict(pairs)


def execute(rows, spec):
    if isinstance(spec.get("steps"), list):
        vals = [run_spec(rows, s) for s in spec["steps"]]
        a, b = (vals + [None, None])[:2]
        c = spec.get("combine")
        if c == "diff":
            return r2(a - b)
        if c == "sum":
            return r2(sum(vals))
        if c == "ratio":
            return r2(a / b) if b else None
        if c == "pct_of":
            return r2(a / b * 100) if b else None
        if c == "pct_change":
            return r2((a - b) / b * 100) if b else None
        return vals
    return run_spec(rows, spec)


# ---------- LLM: question -> spec ----------
def system_prompt(cat, today):
    return f"""You convert questions about an order ledger into a JSON query spec. Output ONLY JSON, no prose.
Today is {today} (Asia/Kolkata). Business dates are Asia/Kolkata dates.
Data: orders with fields id, customer (like "C1026"), region, product, qty, currency, status, created_at.
Regions: {json.dumps(cat["regions"])}
Products (use these exact spellings, map plurals/synonyms): {json.dumps(cat["products"])}
Statuses: paid, refunded, void.

Spec fields (omit what is not needed):
{{
 "aggregate": "sum_usd" | "count" | "sum_qty" | "distinct_customers" | "avg_usd" | "max_usd" | "min_usd" | "refund_rate",
 "status": ["paid"],
 "region": ["North"], "product": ["Air Fryer"], "customer": ["C1026"], "currency": ["EUR"],
 "date_field": "created_at",
 "date_from": "YYYY-MM-DD", "date_to": "YYYY-MM-DD",
 "group_by": "region" | "product" | "customer" | "month" | "currency" | "status",
 "sort": "desc" | "asc", "limit": 1,
 "return": "value" | "key" | "keys" | "pairs"
}}
Rules:
- date_field is always created_at unless the question explicitly says otherwise. Dates are inclusive; resolve months, quarters, halves, years, "last month", "this year" etc. from today.
- Revenue / sales / income / money earned => aggregate sum_usd with status ["paid"].
- Refund amount / money refunded => sum_usd with status ["refunded"]. Number of refunds / refunded orders => count with status ["refunded"].
- "Orders" count with no status mentioned => count with status ["paid"]. If the question says all orders / any status, omit status.
- Average order value => avg_usd with status ["paid"].
- Units / items sold => sum_qty with status ["paid"].
- Refund rate / percentage refunded => refund_rate (ignores status; percent 0-100).
- "Which/what is the top/best/highest X" => group_by X, sort desc, limit 1, return "key". "Lowest/least" => sort asc. "Top N" => limit N, return "keys". "How much did the top X bring in" => return "value".
- Breakdown "by region" etc. => group_by with return "pairs".
- Month-only dates without a year mean 2026.
- Comparisons between two things: {{"steps":[spec1,spec2],"combine":"diff"|"ratio"|"sum"|"pct_of"|"pct_change"}} (diff = first minus second; pct_change = (first-second)/second).
Return the JSON object only."""


async def question_to_spec(question, cat):
    today = (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).date().isoformat()
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    async with httpx.AsyncClient(timeout=9.0) as client:
        res = await client.post(
            url,
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "content-type": "application/json"},
            json={
                "systemInstruction": {"parts": [{"text": system_prompt(cat, today)}]},
                "contents": [{"role": "user", "parts": [{"text": question}]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
            },
        )
    if res.status_code != 200:
        raise RuntimeError(f"LLM {res.status_code}: {res.text[:200]}")
    text = "".join(p.get("text", "") for p in res.json()["candidates"][0]["content"]["parts"])
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < 0:
        raise RuntimeError("no JSON in LLM output")
    return json.loads(text[a : b + 1])


# ---------- data loading ----------
async def get_rows():
    if _cache["rows"] is not None and time.time() - _cache["at"] < TTL:
        return _cache["rows"], _cache["cat"]
    async with _lock:
        if _cache["rows"] is not None and time.time() - _cache["at"] < TTL:
            return _cache["rows"], _cache["cat"]
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
            res = await client.get(os.environ["EXPORT_URL"])
        res.raise_for_status()
        rows = parse_orders(res.text)
        _cache.update(at=time.time(), rows=rows, cat=catalog(rows))
    return _cache["rows"], _cache["cat"]


@app.on_event("startup")
async def warm():
    try:
        await get_rows()
    except Exception as e:  # first request will retry
        print("warmup failed:", e)


@app.get("/")
async def health():
    return {"ok": True, "usage": 'POST {"question": "..."}'}


@app.post("/")
@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.json()
        rows, cat = await get_rows()
        # {"spec": {...}} bypasses the LLM, handy for testing
        spec = body.get("spec") or await question_to_spec(str(body.get("question", "")), cat)
        out = {"answer": execute(rows, spec)}
        if body.get("debug"):
            out["spec"] = spec
        return out
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
