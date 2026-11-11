"""Deterministic synthetic support-ticket generator with planted ground truth.

The generated data has a known ``intent`` per ticket, a planted duplicate-burst
incident (a crash after release 4.2.0) and a planted volume spike (declined payments
in the final days), so retrieval, clustering and incident detection can all be
measured against truth.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

INTENTS: dict[str, tuple[str, list[str]]] = {
    "billing_double_charge": ("Payments", [
        "I was charged twice for my {plan} plan this month, please refund the extra charge",
        "Double charge on my card for order {order}, same amount billed two times",
        "Why are there two identical payments of {amount} on my statement",
        "billed 2x for the same subscription again {email}",
        "my bank shows a duplicate transaction from you for {amount}",
    ]),
    "refund_delay": ("Payments", [
        "Returned my item {days} days ago and still no refund for order {order}",
        "where is my refund? you said 5 business days and it has been {days}",
        "refund for {amount} has not reached my account yet",
        "I cancelled order {order} but the money never came back",
        "still waiting on my reimbursement, it has been weeks",
    ]),
    "payment_declined": ("Payments", [
        "My card keeps getting declined at checkout even though it works everywhere else",
        "payment failed three times with error code {code}, card is valid",
        "Checkout says transaction declined for my {plan} upgrade",
        "cannot pay, every card I try gets rejected on your site",
        "payment not going through, says try another payment method",
    ]),
    "coupon_invalid": ("Ecommerce", [
        "Promo code {promo} says invalid but it was in your newsletter",
        "discount code not applying at checkout, says expired",
        "your coupon {promo} doesn't work, cart total is unchanged",
        "voucher rejected even though the offer ends next week",
        "tried the welcome discount code and it gives an error",
    ]),
    "delivery_late": ("Ecommerce", [
        "Order {order} was supposed to arrive {days} days ago, tracking hasn't moved",
        "my package is late and the courier has no update",
        "delivery date passed and still no parcel, order {order}",
        "tracking stuck on 'in transit' for a week now",
        "when will my shipment actually arrive? it is overdue",
    ]),
    "wrong_item": ("Ecommerce", [
        "I received the wrong product, ordered a {product} and got something else",
        "got someone else's order in my box, order {order}",
        "you sent me the wrong size and wrong colour",
        "package contained a completely different item than what I paid for",
        "incorrect item delivered for order {order}, need an exchange",
    ]),
    "missing_item": ("Ecommerce", [
        "box arrived but one of the items is missing from order {order}",
        "only got 2 of the 3 products I ordered",
        "my parcel was missing the {product}, invoice lists it",
        "half my order is not in the package",
        "items missing from delivery, the bag was sealed",
    ]),
    "login_failure": ("Account", [
        "I can't log in, it says incorrect password but I just reset it",
        "login page keeps spinning and never signs me in on {device}",
        "unable to sign in to my account since yesterday",
        "password is right but the app says invalid credentials",
        "keeps logging me out and won't let me back in",
    ]),
    "otp_not_received": ("Account", [
        "Not receiving the verification code by SMS on {phone}",
        "OTP never arrives so I can't finish signing in",
        "two factor code is not coming to my phone",
        "waited 20 minutes for the one time password, nothing",
        "the 2FA text message never shows up",
    ]),
    "account_locked": ("Account", [
        "My account got locked after a few failed attempts, please unlock it",
        "account suspended for no reason, {email}",
        "it says my account is temporarily blocked",
        "locked out of my profile, security lock won't clear",
        "why was my account disabled? I did nothing wrong",
    ]),
    "app_crash": ("Mobile App", [
        "The app crashes when I open the settings page on {device}",
        "app keeps force closing in the middle of checkout",
        "random crashes every few minutes on {device}",
        "app freezes then shuts down when I upload a photo",
        "it closes by itself whenever I switch tabs",
    ]),
    "crash_after_update": ("Mobile App", [
        "App crashes immediately on launch after updating to version 4.2.0",
        "after the 4.2.0 update the app crashes on launch",
        "since updating to 4.2.0 the app crashes immediately on launch",
        "version 4.2.0 update: app crashes on launch every time",
    ]),
    "slow_performance": ("Mobile App", [
        "the app is extremely slow, pages take forever to load",
        "everything lags on {device}, scrolling is choppy",
        "loading spinner for 30 seconds before anything shows",
        "the website is so sluggish today",
        "search results take ages to load",
    ]),
    "battery_drain": ("Mobile App", [
        "your app is draining my battery even in the background",
        "battery drops 30 percent an hour with the app open on {device}",
        "phone gets hot and battery dies fast because of this app",
        "huge battery usage from the app in settings",
        "background activity is killing my battery life",
    ]),
    "notification_spam": ("Mobile App", [
        "getting way too many push notifications, it's spam",
        "I turned off alerts but still get notifications every hour",
        "stop sending me promotional notifications",
        "notification settings are ignored, constant pings",
        "the app spams me with alerts at night",
    ]),
    "internet_outage": ("Telecom", [
        "No internet at home since this morning in {city}",
        "broadband is down again, router lights are red",
        "wifi connection keeps dropping all day",
        "there's a network outage in {city}, when will it be fixed",
        "my fiber connection has no signal at all",
    ]),
    "roaming_charges": ("Telecom", [
        "I got a huge bill for roaming in {country}, my plan includes it",
        "charged for international data usage on my trip",
        "roaming fees of {amount} are wrong, roaming pack was active",
        "why am I paying extra for using data abroad",
        "unexpected overseas data charges on my last bill",
    ]),
    "flight_cancelled": ("Travel", [
        "My flight to {city} was cancelled and no one rebooked me",
        "flight cancelled last minute, need a refund or new booking",
        "you cancelled my flight and the app shows no alternatives",
        "cancelled booking, stuck at the airport with no help",
        "my trip got cancelled, how do I get rebooked",
    ]),
    "baggage_lost": ("Travel", [
        "My luggage didn't arrive in {city}, bag tag {code}",
        "the airline lost my suitcase on the connecting flight",
        "checked bag missing for {days} days now",
        "baggage claim says my bag is untraceable",
        "where is my luggage? nobody at the desk can tell me",
    ]),
    "cancel_subscription": ("Account", [
        "How do I cancel my {plan} subscription",
        "I want to end my membership and stop auto renewal",
        "please cancel my plan, I can't find the option",
        "unsubscribe me from the monthly plan",
        "stop my subscription before the next billing date",
    ]),
    "feature_request_dark_mode": ("Product", [
        "Please add a dark mode, the white screen hurts at night",
        "any plans for a dark theme on {device}",
        "would love a night mode option",
        "dark mode please! everyone else has it",
        "can you add a darker colour scheme",
    ]),
    "food_cold": ("Food Delivery", [
        "My food arrived cold and soggy",
        "the delivery took {days}0 minutes and the meal was cold",
        "pizza showed up stone cold, want a refund",
        "order {order} was lukewarm and the fries were soggy",
        "food was cold by the time the driver got here",
    ]),
}  # fmt: skip

GOLDEN_QUERIES: list[dict[str, Any]] = [
    {"query": "charged two times for the same purchase", "relevant_labels": ["billing_double_charge"]},
    {"query": "still waiting for my money back", "relevant_labels": ["refund_delay"]},
    {"query": "refund not processed after return", "relevant_labels": ["refund_delay"]},
    {"query": "card rejected when paying", "relevant_labels": ["payment_declined"]},
    {"query": "discount code is not accepted", "relevant_labels": ["coupon_invalid"]},
    {"query": "package has not arrived after the expected date", "relevant_labels": ["delivery_late"]},
    {"query": "received a different product than ordered", "relevant_labels": ["wrong_item"]},
    {"query": "products missing from my parcel", "relevant_labels": ["missing_item"]},
    {"query": "cannot sign in with my password", "relevant_labels": ["login_failure"]},
    {"query": "verification sms never comes", "relevant_labels": ["otp_not_received"]},
    {"query": "OTP code not delivered", "relevant_labels": ["otp_not_received"]},
    {"query": "account blocked after failed attempts", "relevant_labels": ["account_locked"]},
    {"query": "app closes by itself randomly", "relevant_labels": ["app_crash", "crash_after_update"]},
    {"query": "app crashing since the latest update", "relevant_labels": ["crash_after_update", "app_crash"]},
    {"query": "app is laggy and slow to load", "relevant_labels": ["slow_performance"]},
    {"query": "phone battery dies fast", "relevant_labels": ["battery_drain"]},
    {"query": "too many push alerts", "relevant_labels": ["notification_spam"]},
    {"query": "no internet connection at home", "relevant_labels": ["internet_outage"]},
    {"query": "network down in my area", "relevant_labels": ["internet_outage"]},
    {"query": "big bill for using data abroad", "relevant_labels": ["roaming_charges"]},
    {"query": "flight got cancelled need rebooking", "relevant_labels": ["flight_cancelled"]},
    {"query": "airline lost my suitcase", "relevant_labels": ["baggage_lost"]},
    {"query": "how do I stop my membership", "relevant_labels": ["cancel_subscription"]},
    {"query": "please add a dark theme", "relevant_labels": ["feature_request_dark_mode"]},
    {"query": "meal delivered cold", "relevant_labels": ["food_cold"]},
]  # fmt: skip

_PREFIX = ["", "", "", "hi,", "hello team,", "@SupportCo", "URGENT:", "ugh.", "hey"]
_SUFFIX = ["", "", "", "thanks", "pls help", "this is ridiculous", "any update?", "dm me", "!!"]
_FILL: dict[str, list[str]] = {
    "plan": ["Pro", "Basic", "Family", "Premium"],
    "device": ["iPhone 14", "Pixel 8", "Galaxy S23", "iPad", "OnePlus 12"],
    "product": ["blue hoodie", "phone case", "kettle", "headphones", "desk lamp"],
    "city": ["Pune", "Austin", "Leeds", "Toronto", "Bhopal", "Lyon"],
    "country": ["Spain", "Japan", "Dubai", "Canada"],
    "promo": ["SAVE20", "WELCOME10", "FLASH50"],
}


_SYNONYMS: dict[str, list[str]] = {
    "app": ["app", "application"],
    "crashes": ["crashes", "quits", "dies"],
    "refund": ["refund", "reimbursement"],
    "cancelled": ["cancelled", "canceled", "called off"],
    "order": ["order", "purchase"],
    "package": ["package", "parcel", "shipment"],
    "account": ["account", "profile"],
    "password": ["password", "passcode"],
    "slow": ["slow", "laggy"],
    "notifications": ["notifications", "alerts"],
    "luggage": ["luggage", "baggage", "suitcase"],
    "card": ["card", "credit card"],
    "internet": ["internet", "wifi", "broadband"],
    "cold": ["cold", "lukewarm"],
    "subscription": ["subscription", "membership"],
    "charged": ["charged", "billed"],
    "flight": ["flight", "plane"],
    "declined": ["declined", "rejected", "refused"],
    "missing": ["missing", "not included"],
    "code": ["code", "pin"],
}


def _perturb(text: str, rng: random.Random) -> str:
    """Synonym swaps and occasional typos so near-duplicates aren't byte-identical."""
    words = []
    for word in text.split():
        key = word.lower().strip(",.!?")
        if key in _SYNONYMS and rng.random() < 0.5:
            word = word.lower().replace(key, rng.choice(_SYNONYMS[key]))
        if len(word) > 5 and rng.random() < 0.04:
            i = rng.randrange(len(word) - 1)
            word = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        words.append(word)
    return " ".join(words)


def _fill(template: str, rng: random.Random) -> str:
    values = {k: rng.choice(v) for k, v in _FILL.items()}
    values.update(
        order=f"#A{rng.randint(10000, 99999)}",
        amount=f"${rng.randint(5, 199)}.{rng.choice(['00', '99', '49'])}",
        days=str(rng.randint(3, 21)),
        code=f"E{rng.randint(100, 999)}",
        email=f"user{rng.randint(1, 9999)}@example.com",
        phone=f"+1 415-555-{rng.randint(1000, 9999)}",
    )
    text = _perturb(template.format(**values), rng)
    text = " ".join(p for p in (rng.choice(_PREFIX), text, rng.choice(_SUFFIX)) if p)
    return text.lower() if rng.random() < 0.3 else text


def generate_tickets(
    n: int = 3000, seed: int = 7, days: int = 60, start: datetime | None = None
) -> pd.DataFrame:
    """Generate ``n`` tickets over ``days`` days with a planted incident and spike."""
    rng = random.Random(seed)
    start = start or datetime(2026, 8, 1)
    background = [i for i in INTENTS if i != "crash_after_update"]
    n_incident = max(8, n // 40)
    n_spike = max(10, n // 30)
    n_base = n - n_incident - n_spike
    incident_start = start + timedelta(days=int(days * 0.7))
    rows: list[dict[str, Any]] = []

    def add(intent: str, ts: datetime) -> None:
        product, templates = INTENTS[intent]
        rows.append(
            {
                "intent": intent,
                "product": product,
                "channel": rng.choice(["email", "chat", "twitter", "app"]),
                "body": _fill(rng.choice(templates), rng),
                "created_at": ts.isoformat(timespec="minutes"),
            }
        )

    for _ in range(n_base):
        add(rng.choice(background), start + timedelta(minutes=rng.randint(0, days * 1440 - 1)))
    for _ in range(n_incident):
        add("crash_after_update", incident_start + timedelta(minutes=rng.randint(0, 30 * 60)))
    spike_from = start + timedelta(days=days - 3)
    for _ in range(n_spike):
        add("payment_declined", spike_from + timedelta(minutes=rng.randint(0, 3 * 1440 - 1)))

    df = pd.DataFrame(rows).sort_values("created_at", kind="stable").reset_index(drop=True)
    df.insert(0, "ticket_id", [f"TCK-{i:05d}" for i in range(len(df))])
    return df
