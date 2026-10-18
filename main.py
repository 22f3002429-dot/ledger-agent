"""Acme ledger agent: POST {"question": "..."} -> {"answer": ...}

The LLM only turns the question into a JSON spec; all numbers are computed here.
Env vars: EXPORT_URL (signed .../export link). Optional: GROQ_API_KEY (free LLM backup for odd wordings).
"""
import asyncio
import calendar
import json
import os
import re
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
        if c == "absdiff":
            return r2(abs(a - b))
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


MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
          "sep": 9, "oct": 10, "nov": 11, "dec": 12}
MON_RE = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DEFAULT_YEAR = 2026


def _today():
    return (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).date()


def _iso(y, m, d):
    return f"{y:04d}-{m:02d}-{d:02d}"


def _mrange(y, m):
    return _iso(y, m, 1), _iso(y, m, calendar.monthrange(y, m)[1])


def _mnum(s):
    return MONTHS[s[:3]]


def parse_dates(q):
    """Return (date_from, date_to) or (None, None)."""
    today = _today()
    ym = re.search(r"\b(20\d\d)\b", q)
    year = int(ym.group(1)) if ym else DEFAULT_YEAR

    # explicit ISO dates
    iso = re.findall(r"\b(20\d\d)-(\d\d)-(\d\d)\b", q)
    if iso:
        ds = sorted(_iso(*map(int, t)) for t in iso)
        return ds[0], ds[-1]
    # "march 5, 2026" / "5 march 2026" / "5th of march"
    days = []
    for m in re.finditer(MON_RE + r"\s+(\d{1,2})(?:st|nd|rd|th)?(?!\d)(?:,?\s*(20\d\d))?", q):
        days.append((int(m.group(3)) if m.group(3) else year, _mnum(m.group(1)), int(m.group(2))))
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + MON_RE + r"(?:,?\s*(20\d\d))?", q):
        days.append((int(m.group(3)) if m.group(3) else year, _mnum(m.group(2)), int(m.group(1))))
    days = [d for d in days if 1 <= d[2] <= 31]
    if days:
        ds = sorted(_iso(*d) for d in days)
        return ds[0], ds[-1]

    # relative
    m = re.search(r"(?:last|past|previous)\s+(\d+)\s+days", q)
    if m:
        return (today - timedelta(days=int(m.group(1)))).isoformat(), today.isoformat()
    if re.search(r"\blast month\b|\bprevious month\b", q):
        y, mo = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
        return _mrange(y, mo)
    if re.search(r"\bthis month\b|\bcurrent month\b", q):
        return _mrange(today.year, today.month)
    if re.search(r"\blast year\b|\bprevious year\b", q):
        return _iso(today.year - 1, 1, 1), _iso(today.year - 1, 12, 31)
    if re.search(r"\bthis year\b|\bcurrent year\b|\bytd\b|\byear to date\b", q):
        return _iso(today.year, 1, 1), (today.isoformat() if re.search(r"ytd|year to date", q) else _iso(today.year, 12, 31))
    qm = re.search(r"\bq([1-4])\b|\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\b", q)
    if qm:
        n = int(qm.group(1)) if qm.group(1) else {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}[qm.group(2)]
        return _iso(year, 3 * n - 2, 1), _mrange(year, 3 * n)[1]
    if re.search(r"\blast quarter\b|\bprevious quarter\b", q):
        cq = (today.month - 1) // 3 + 1
        y, n = (today.year, cq - 1) if cq > 1 else (today.year - 1, 4)
        return _iso(y, 3 * n - 2, 1), _mrange(y, 3 * n)[1]
    if re.search(r"\bthis quarter\b|\bcurrent quarter\b", q):
        n = (today.month - 1) // 3 + 1
        return _iso(today.year, 3 * n - 2, 1), _mrange(today.year, 3 * n)[1]
    hm = re.search(r"\bh([12])\b|\b(first|second|1st|2nd)\s+half\b", q)
    if hm:
        n = int(hm.group(1)) if hm.group(1) else (1 if hm.group(2) in ("first", "1st") else 2)
        return (_iso(year, 1, 1), _iso(year, 6, 30)) if n == 1 else (_iso(year, 7, 1), _iso(year, 12, 31))

    # month names ("may" only when it clearly means the month)
    found = []
    for m in re.finditer(r"\b" + MON_RE + r"\b", q):
        name = m.group(1)
        if name == "may" and not re.search(r"(?:in|of|during|for|since|from|to|through|until|and|between|by)\s+may\b|\bmay\s+(?:20\d\d|\d)", q):
            continue
        found.append(_mnum(name))
    if found:
        lo, hi = min(found), max(found)
        return _iso(year, lo, 1), _mrange(year, hi)[1]
    if ym:
        return _iso(year, 1, 1), _iso(year, 12, 31)
    return None, None


def _find_entities(q, cat):
    regions = [r for r in cat["regions"] if re.search(r"\b" + r.lower() + r"\b", q)]
    products = []
    for p in cat["products"]:
        pat = re.escape(p.lower()).replace(r"\ ", r"[\s\-]?") + r"(?:s|es)?\b"
        if re.search(r"\b" + pat, q):
            products.append(p)
    customers = [c.upper() for c in re.findall(r"\bc\d{3,5}\b", q)]
    return regions, products, customers


DIMS = {
    "product": r"\bproducts?\b|\bitems?\b|\bappliances?\b|\bmodels?\b",
    "region": r"\bregions?\b|\bareas?\b|\bterritor(?:y|ies)\b|\bzones?\b",
    "customer": r"\bcustomers?\b|\bbuyers?\b|\bclients?\b|\bshoppers?\b",
    "month": r"\bmonths?\b|\bmonthly\b",
}
SUP_HI = r"\b(top|best|highest|most|largest|biggest|leading|greatest|maximum|max|best[- ]selling|top[- ]selling|bottom)\b"
SUP_LO = r"\b(lowest|least|worst|smallest|fewest|minimum|min|bottom)\b"
COMPLEX = r"\b(compare|comparison|compared|difference|differ|versus|vs|ratio|growth|grew|change|changed|trend|than|relative|increase|decrease|drop)\b"


def rule_spec(question, cat):
    """Best-effort rule-based parse. Returns (spec, confident)."""
    q = " " + re.sub(r"\s+", " ", question.lower().replace("?", " ").replace("!", " ")) + " "
    q = q.replace("$", " usd ").replace("%", " percent ")
    spec = {}
    regions, products, customers = _find_entities(q, cat)
    if regions:
        spec["region"] = regions
    if products:
        spec["product"] = products
    if customers:
        spec["customer"] = customers
    lo, hi = parse_dates(q)
    if lo:
        spec["date_from"], spec["date_to"] = lo, hi
    spec["date_field"] = "created_at"

    count_words = bool(re.search(r"how many|number of|count of|\bcount\b|\bno\. of\b|\btotal number\b|\bhow often\b", q))
    money_words = bool(re.search(r"amount|value|how much|\btotal\b|\busd\b|dollar|worth|money|\bsum\b|revenue|sales", q))
    explicit = False

    # metric
    agg, status = "sum_usd", ["paid"]
    if re.search(r"refund rate|rate of refund|(percent|percentage|share|proportion|fraction|ratio).{0,40}refund|refund.{0,40}(percent|rate|percentage)", q):
        agg, status, explicit = "refund_rate", None, True
    elif "refund" in q or "returned" in q:
        explicit = True
        status = ["refunded"]
        if money_words:
            agg = "sum_usd"
        elif count_words or re.search(r"\b(most|fewest|least)\b", q):
            agg = "count"
    elif re.search(r"\bvoid(?:ed)?\b|cancel", q):
        explicit = True
        status = ["void"]
        agg = "count" if count_words and not money_words else "sum_usd"
    elif re.search(r"(unique|distinct|different)\s+customers|number of customers|how many customers|customer count|customers (?:placed|bought|ordered|purchased)", q):
        agg, explicit = "distinct_customers", True
    elif re.search(r"\b(average|avg|mean)\b", q):
        agg, explicit = "avg_usd", True
    elif re.search(r"\bunits?\b|\bquantity\b|\bqty\b|\bpieces\b|items (?:sold|ordered)|how many (?:items|units|pieces)", q):
        agg, explicit = "sum_qty", True
    elif re.search(r"(largest|biggest|highest|maximum|max|smallest|lowest|minimum|min)\s+(?:single\s+)?(?:order|sale|purchase|transaction)", q):
        agg, explicit = ("min_usd" if re.search(r"smallest|lowest|minimum|min\b", q) else "max_usd"), True
    elif count_words and re.search(r"\borders?\b|\bsales\b|\bpurchases\b|\btransactions\b", q) and not re.search(r"revenue|how much", q):
        agg, explicit = "count", True
    elif re.search(r"most orders|fewest orders|least orders|order count", q):
        agg, explicit = "count", True
    elif re.search(r"revenue|sales|income|earn|spent|spend|bring|brought|generat|made|\bhow much\b|\btotal\b|\bgross\b|turnover", q):
        explicit = True
    if status:
        spec["status"] = status
    spec["aggregate"] = agg

    # currency filter (only when it describes the orders, not the output unit)
    cur_map = {"usd": "USD", "dollar": "USD", "dollars": "USD", "eur": "EUR", "euro": "EUR", "euros": "EUR",
               "inr": "INR", "rupee": "INR", "rupees": "INR"}
    cm = re.search(r"\b(usd|eur|inr|euros?|rupees?|dollars?)\b[- ]+(?:orders|transactions|sales|purchases|denominated|payments)", q) \
        or re.search(r"(?:placed|paid|billed|priced|denominated|charged|ordered)\s+in\s+(usd|eur|inr|euros?|rupees?|dollars?)\b", q) \
        or re.search(r"\bin\s+(euros?|rupees?)\b", q)
    if cm:
        spec["currency"] = [cur_map[cm.group(1)]]

    # comparisons between exactly two regions / products / customers
    if re.search(COMPLEX, q):
        for kind, names in (("region", regions), ("product", products), ("customer", customers)):
            if len(names) == 2:
                pos = sorted(names, key=lambda nm: q.find(nm.lower()))
                if re.search(r"\b(compare|comparison|versus|vs)\b", q) and not re.search(r"differ|gap|how much more|than", q):
                    spec[kind] = pos
                    spec["group_by"], spec["return"], spec["sort"] = kind, "pairs", "desc"
                    return spec, explicit
                one = {k: v for k, v in spec.items() if k != kind}
                a_, b_ = dict(one, **{kind: [pos[0]]}), dict(one, **{kind: [pos[1]]})
                if re.search(r"\bratio\b|how many times", q):
                    comb = "ratio"
                elif re.search(r"differ|gap", q):
                    comb = "absdiff"
                elif re.search(r"more|higher|greater|exceed|ahead|than", q):
                    comb = "diff"
                else:
                    return spec, False
                return {"steps": [a_, b_], "combine": comb}, explicit

    # ranking / grouping
    dim = next((d for d, pat in DIMS.items() if re.search(pat, q)), None)
    sup_hi, sup_lo = re.search(SUP_HI, q), re.search(SUP_LO, q)
    top_n = re.search(r"\b(?:top|bottom|best|worst)\s+(\d+)\b|\b(\d+)\s+(?:best|top|worst|highest|lowest)\b", q)
    n = None
    if top_n:
        n = int(top_n.group(1) or top_n.group(2))
    if agg in ("distinct_customers",) and not (sup_hi or sup_lo):
        dim = None
    if (sup_hi or sup_lo) and dim and not re.search(r"how many " + dim, q):
        spec["group_by"] = dim
        spec["sort"] = "asc" if (sup_lo and not sup_hi) or (sup_lo and sup_hi and sup_hi.group(1) == "bottom") else "desc"
        if re.search(r"\bbottom\b|\bworst\b", q):
            spec["sort"] = "asc"
        spec["limit"] = n or 1
        spec["return"] = "keys" if (n and n > 1) else "key"
        if spec["return"] == "key" and re.search(r"how much|what (?:was|is) the (?:revenue|total|amount)|how many .*(?:did|does)", q) and not re.search(r"\bwhich\b|\bwho\b|\bwhat (?:product|region|customer|month)\b", q):
            spec["return"] = "value"
        if dim == "customer":  # drop an entity filter that was just the grouping word
            spec.pop("customer", None) if not customers else None
    elif dim and re.search(r"\b(?:by|per|each|every|breakdown|break down|split|across|wise)\b|monthly", q) and not count_words:
        spec["group_by"] = dim
        spec["return"] = "pairs"
        spec["sort"] = "desc"
    elif dim == "month" and re.search(r"monthly", q):
        spec["group_by"] = "month"
        spec["return"] = "pairs"

    confident = explicit and not re.search(COMPLEX, q)
    if re.search(r"\bwhich\b|\bwho\b", q) and "group_by" not in spec:
        confident = False
    return spec, confident


async def llm_spec(question, cat):
    """Optional free backup (Groq). Only used if GROQ_API_KEY is set."""
    today = _today().isoformat()
    async with httpx.AsyncClient(timeout=8.0) as client:
        res = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]},
            json={
                "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt(cat, today)},
                    {"role": "user", "content": question},
                ],
            },
        )
    if res.status_code != 200:
        raise RuntimeError(f"LLM {res.status_code}: {res.text[:200]}")
    text = res.json()["choices"][0]["message"]["content"]
    a, b = text.find("{"), text.rfind("}")
    return json.loads(text[a : b + 1])


async def question_to_spec(question, cat):
    spec, confident = rule_spec(question, cat)
    if confident or not os.environ.get("GROQ_API_KEY"):
        return spec
    try:
        return await llm_spec(question, cat)
    except Exception as e:
        print("LLM backup failed, using rules:", e)
        return spec


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
