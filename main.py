from fastapi import FastAPI, Request
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
import re

START_URL = "https://exam.sanand.workers.dev/questionData?email=22f3002429%40ds.study.iitm.ac.in&quizSign=1yxJW5ZTG5lxxecP044bcmNgM9N6kggTmcjXms14SFjt6wRBOpZHC0wnnD0LAQEbvNRuPy0AdTEgDZHxWkeSajfUKn0Ob6p7D6V7691jj9QRlIRZtYnkhWxhDsWo%2BikqrZDKZCkSCAbyto9W4pPXcdPhbAHsPaWJKUbim0WBuW84opfM2TYCw9vtOfs2Xx5j7edCRP2csI%2Fpaz%2BQ5FserZFImPE0nQNz4i8NvsUGZ2S22MFH5dkDJ0MeBWLCdo5THXDxnwrfShejv9KoGpRPYpPARWxztDYNvzJFT8kp5a2OKKwKKApZch2w2PvI4Tv5vTtRZfgYQbLd0jfMoUI8JA%3D%3D&questionId=q-ledger-agent-server&path=/"

app = FastAPI()

orders = []
rates = {}
sample_order = {}


def first_value(row, *names):
    for name in names:
        if name in row:
            return row[name]
    return None


def find_list(page):
    for key in ("orders", "data", "results", "items", "transactions"):
        if isinstance(page.get(key), list):
            return page[key]
    return None


def find_next(page):
    return (
        page.get("next")
        or page.get("next_url")
        or page.get("links", {}).get("next")
    )


def load_data():
    global orders, rates, sample_order

    root = requests.get(START_URL, timeout=20).json()
    links = root["links"]

    rates_response = requests.get(links["rates"], timeout=20).json()
    rates = rates_response.get("rates", rates_response)

    all_rows = []
    url = links["orders"]

    while url:
        page = requests.get(url, timeout=20).json()

        rows = find_list(page)

        if rows is None:
            raise ValueError(
                f"Could not find orders list. Page keys: {list(page.keys())}"
            )

        if not sample_order and rows:
            sample_order = rows[0]
            print("SAMPLE ORDER:", sample_order)

        all_rows.extend(rows)
        url = find_next(page)

    latest = {}

    for row in all_rows:
        order_id = first_value(row, "order_id", "orderId", "id", "order")
        updated_at = first_value(row, "updated_at", "updatedAt", "updated")

        if order_id is None or updated_at is None:
            raise ValueError(f"Missing ID or updated_at in row: {row}")

        if (
            order_id not in latest
            or str(updated_at) > str(latest[order_id]["_updated"])
        ):
            latest[order_id] = {**row, "_updated": updated_at}

    # Keep every current order: paid, void, refunded, etc.
    orders = list(latest.values())

    print(f"Loaded {len(orders)} current orders.")


def get_order_date(row):
    value = first_value(
        row,
        "created_at",
        "createdAt",
        "order_date",
        "orderDate",
        "date",
        "business_date",
        "timestamp"
    )

    if value is None:
        raise ValueError(f"No date field found in row: {row}")

    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def get_amount(row):
    value = first_value(row, "amount", "total", "value", "revenue")

    if value is None:
        raise ValueError(f"No amount field found in row: {row}")

    return float(value)


def get_currency(row):
    value = first_value(row, "currency", "currency_code", "curr")
    return str(value or "USD").upper()


def amount_in_usd(row):
    amount = get_amount(row)
    currency = get_currency(row)

    if currency == "USD":
        return amount

    if currency not in rates:
        raise ValueError(f"No exchange rate for {currency}. Rates: {rates}")

    return amount / float(rates[currency])


def is_refunded(row):
    status = str(row.get("status", "")).lower()
    return status in ("refund", "refunded") or get_amount(row) < 0


def month_bounds(month_name, year):
    months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]

    month_number = months.index(month_name.lower()) + 1
    tz = ZoneInfo("Asia/Kolkata")

    start = datetime(year, month_number, 1, tzinfo=tz)

    if month_number == 12:
        end = datetime(year + 1, 1, 1, tzinfo=tz)
    else:
        end = datetime(year, month_number + 1, 1, tzinfo=tz)

    return start, end


@app.on_event("startup")
def startup():
    load_data()


@app.get("/debug")
def debug():
    return {
        "orders_loaded": len(orders),
        "rates": rates,
        "sample_order": sample_order,
        "statuses": sorted({str(row.get("status")) for row in orders}),
    }


@app.post("/")
async def answer(request: Request):
    body = await request.json()
    q = body["question"].lower()

    matching = orders

    # Date filter
    year_match = re.search(r"\b(20\d{2})\b", q)
    month_match = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        q
    )

    if year_match and month_match:
        year = int(year_match.group(1))
        start, end = month_bounds(month_match.group(1), year)

        matching = [
            row for row in matching
            if start <= get_order_date(row) < end
        ]

    # Region filter
    for region in ["north", "south", "east", "west"]:
        if region in q:
            matching = [
                row for row in matching
                if str(first_value(row, "region", "region_name", "zone") or "").lower() == region
            ]

    # Product filter
    product_names = sorted(
        {
            str(first_value(row, "product", "product_name", "item"))
            for row in orders
            if first_value(row, "product", "product_name", "item")
        },
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
            if str(first_value(row, "product", "product_name", "item")) == product
        ]

    # Customer filter, e.g. "customer C1026"
    customer_match = re.search(r"\bc\d+\b", q)

    if customer_match:
        customer_id = customer_match.group(0).upper()
        matching = [
            row for row in matching
            if str(first_value(row, "customer", "customer_id", "customerid") or "").upper() == customer_id
        ]

    # Refund questions
    if "refund" in q:
        refunded = [row for row in matching if is_refunded(row)]

        if any(word in q for word in ("count", "how many", "number of")):
            return {"answer": len(refunded)}

        return {
            "answer": round(
                sum(amount_in_usd(row) for row in refunded),
                2
            )
        }

    # Revenue questions: only paid orders count as revenue
    if any(word in q for word in ("revenue", "sales", "total", "amount", "spend", "spent")):
        matching = [
            row for row in matching
            if str(row.get("status", "")).lower() == "paid"
        ]

        return {
            "answer": round(
                sum(amount_in_usd(row) for row in matching),
                2
            )
        }

    # Count questions without the word "refund"
    if any(word in q for word in ("count", "how many", "number of")):
        return {"answer": len(matching)}

    # Default: count matching orders
    return {"answer": len(matching)}
