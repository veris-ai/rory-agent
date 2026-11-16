"""Disposable HTTP fixtures, not Veris twins or a Stripe/Fineract emulator.

Only the routes used by live_smoke.py exist. Unknown operations fail closed.
All records and writes are synthetic, held in memory, and lost on exit.
"""

from copy import deepcopy
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class Backends:
    def __init__(self):
        self.app = FastAPI()
        self.requests = []
        self.payments = []
        self.customer = {
            "id": "cus_fixture", "object": "customer", "name": "Jane Smith",
            "email": "jane@example.invalid", "phone": "+12025550123",
            "address": {"line1": "123 Example Street", "postal_code": "60614"},
            "metadata": {
                "account_number": "4417-88231", "fineract_client_id": "1",
                "paperless": "false", "supplier": "Acme Energy",
                "supplier_type": "default", "income_fpl_pct": "300",
            },
        }
        self.invoice = {
            "id": "in_fixture", "object": "invoice", "customer": "cus_fixture",
            "number": "FIXTURE-001", "status": "open", "total": 4217,
            "amount_due": 4217, "amount_remaining": 4217,
            "due_date": 1793491200, "created": 1790812800,
            "period_start": 1788220800, "period_end": 1790812800,
        }
        self.lines = [
            {"id": "il_supply", "object": "line_item", "description": "Electric supply",
             "amount": 3000, "metadata": {"charge_type": "supply", "kwh": "300", "rate_cents_per_kwh": "10"}},
            {"id": "il_delivery", "object": "line_item", "description": "Delivery",
             "amount": 1000, "metadata": {"charge_type": "delivery", "kwh": "300"}},
            {"id": "il_tax", "object": "line_item", "description": "Tax",
             "amount": 217, "metadata": {"charge_type": "tax"}},
        ]
        self.app.add_api_route("/{path:path}", self.handle, methods=["GET", "POST", "DELETE", "PUT"])

    @staticmethod
    def listing(data, search=False):
        return {"object": "search_result" if search else "list", "data": data, "has_more": False, "url": "/v1/fixture"}

    async def handle(self, request: Request, path: str):
        self.requests.append({"method": request.method, "path": path})
        if request.method == "GET":
            if path == "fineract-provider/api/v1/clients":
                rows = []
                if request.query_params.get("externalId") == "4417-88231":
                    rows = [{"id": 1, "externalId": "4417-88231", "displayName": "Jane Smith"}]
                return {"pageItems": rows}
            if path == "v1/customers/search":
                rows = [self.customer] if "'4417-88231'" in request.query_params.get("query", "") else []
                return self.listing(rows, search=True)
            if path == "v1/customers/cus_fixture":
                return deepcopy(self.customer)
            if path == "v1/payment_methods":
                return self.listing([{"id": "pm_fixture", "object": "payment_method", "card": {
                    "brand": "visa", "last4": "4242", "exp_month": 12, "exp_year": 2030,
                }}])
            if path == "v1/subscriptions":
                return self.listing([])
            if path == "v1/invoices":
                return self.listing([deepcopy(self.invoice)])
            if path == "v1/invoices/in_fixture":
                return deepcopy(self.invoice)
            if path == "v1/invoices/in_fixture/lines":
                return self.listing(self.lines)
        if request.method == "POST":
            form = parse_qs((await request.body()).decode())
            if path == "v1/customers/cus_fixture":
                for key, values in form.items():
                    if key.startswith("metadata[") and key.endswith("]"):
                        self.customer["metadata"][key[9:-1]] = values[0]
                return deepcopy(self.customer)
            if path == "v1/invoices/in_fixture/pay":
                assert form.get("payment_method") == ["pm_fixture"]
                self.payments.append("declined" if not self.payments else "paid")
                if len(self.payments) == 1:
                    return JSONResponse(status_code=402, content={"error": {
                        "type": "card_error", "code": "card_declined",
                        "message": "The fixture card was declined. No payment was made.",
                    }})
                self.invoice.update(status="paid", amount_remaining=0)
                return deepcopy(self.invoice)
        return JSONResponse(status_code=501, content={"error": "Unsupported disposable fixture operation"})
