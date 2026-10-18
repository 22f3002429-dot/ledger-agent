from fastapi import FastAPI, Request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
import re

START_URL = "https://exam.sanand.workers.dev/questionData?email=22f3002429%40ds.study.iitm.ac.in&quizSign=1yxJW5ZTG5lxxecP044bcmNgM9N6kggTmcjXms14SFjt6wRBOpZHC0wnnD0LAQEbvNRuPy0AdTEgDZHxWkeSajfUKn0Ob6p7D6V7691jj9QRlIRZtYnkhWxhDsWo%2BikqrZDKZCkSCAbyto9W4pPXcdPhbAHsPaWJKUbim0WBuW84opfM2TYCw9vtOfs2Xx5j7edCRP2csI%2Fpaz%2BQ5FserZFImPE0nQNz4i8NvsUGZ2S22MFH5dkDJ0MeBWLCdo5THXDxnwrfShejv9KoGpRPYpPARWxztDYNvzJFT8kp5a2OKKwKKApZch2w2PvI4Tv5vTtRZfgYQbLd0jfMoUI8JA%3D%3D&questionId=q-ledger-agent-server&path=/"

app = FastAPI()

orders = []
rates = {}


def load_data():
    global orders, rates

    root = requests.get(START_URL, timeout=15).json()
    links = root["links"]

    # Load exchange rates.
    rates = requests.get(links["rates"], timeout=15).json()

    # If rates are nested, inspect and adjust, e.g. rates["rates"].
    if "rates" in rates and isinstance(rates["rates"], dict):
        rates = rates["rates"]

    # Load every orders page.
    all_rows = []
    url = links["orders"]

    while url:
        page = requests.get(url, timeout=15).json()

        # Adjust if the real key is "data", "results", etc.
        rows = page.get("orders", [])
        all_rows.extend(rows)

        # Common pagination patterns; inspect page 1 and keep the right one.
        url = (
            page.get("next")
            or page.get("next_url")
            or page.get("links", {}).get("next")
        )

    # Keep only the latest version of each order.
    latest = {}
    for row in all_rows:
        order_id = row["order_id"]
        updated = row["updated_at"]

        if order_id not in latest or updated > latest[order_id]["updated_at"]:
            latest[order_id] = row

    # Keep only paid orders.
    orders = [
        row for row in latest.values()
        if row.get("status") == "paid"
    ]


def to_utc(date_text):
    # Handles ISO dates such as 2026-03-14T18:30:00+05:30 or Z.
    return datetime.fromisoformat(date_text.replace("Z", "+00:00"))


def month_bounds(month_name, year):
    months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    month_number = months.index(month_name.lower()) + 1

    start = datetime(year, month_number, 1, tzinfo=ZoneInfo("Asia/Kolkata"))

    if month_number == 12:
        end = datetime(year + 1, 1, 1, tzinfo=ZoneInfo("Asia/Kolkata"))
    else:
        end = datetime(year, month_number + 1, 1, tzinfo=ZoneInfo("Asia/Kolkata"))

    return start, end


def amount_in_usd(row):
    currency = row.get("currency", "USD").upper()
    amount = float(row["amount"])

    if currency == "USD":
        return amount

    # Adjust this line after seeing the exact /rates format.
    rate = float(rates[currency])
    return amount / rate


@app.on_event("startup")
def startup():
    load_data()


@app.post("/")
async def answer(request: Request):
    body = await request.json()
    q = body["question"].lower()

    matching = orders

    # Date filter.
    year_match = re.search(r"\b(20\d{2})\b", q)
    month_match = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        q
    )

    if year_match and month_match:
        year = int(year_match.group(1))
        month = month_match.group(1)
        start, end = month_bounds(month, year)

        matching = [
            row for row in matching
            if start <= to_utc(row["created_at"]) < end
        ]

    # Region filter. Add more regions if your data has them.
    for region in ["north", "south", "east", "west"]:
        if region in q:
            matching = [
                row for row in matching
                if row.get("region", "").lower() == region
            ]

    # Product filter: compare against exact product names from the data.
    product_names = sorted(
        {row["product"] for row in orders if row.get("product")},
        key=len,
        reverse=True
    )

    product = next(
        (p for p in product_names if p.lower() in q),
        None
    )

    if product:
        matching = [
            row for row in matching
            if row.get("product") == product
        ]

    # Refunds: adjust after inspecting whether refunds are negative amounts
    # or a separate type/status field.
    if "refund" in q:
        matching = [
            row for row in matching
            if float(row["amount"]) < 0
        ]

    total_usd = sum(amount_in_usd(row) for row in matching)

    return {"answer": round(total_usd, 2)}
