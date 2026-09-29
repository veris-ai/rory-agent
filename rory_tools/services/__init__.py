"""Vendor clients for the two services Rory calls.

Each client speaks to the vendor's REAL hostname with the vendor's real
credentials and, for Stripe, the vendor's real SDK. The one concession to a
bench is a base-URL override per twin (`STRIPE_API_BASE`, `FINERACT_API_BASE`),
injected by the platform (Fineract is self-hosted, so its base URL is always required), so the code path exercised
in a benchmark is otherwise the one that ships.

    stripe     billing       api.stripe.com     accounts, bills, usage, payments
    fineract   arrangements  FINERACT_API_BASE  payment plans, instalments

Why the split, and why it is not a compromise: real utilities genuinely run
billing and payment-arrangement/collections in separate subsystems, and keeping
the two consistent is the cross-system reasoning this agent exists to test. The
seam is real, and it is deliberately not hidden — but it IS enforced, in
`impl.py`, where opening an arrangement reads the arrears from Stripe and
opens the Fineract loan at exactly that principal. Neither vendor can guarantee
that on its own.
"""

from .fineract_client import FineractClient
from .stripe_client import StripeClient

__all__ = ["FineractClient", "StripeClient"]
