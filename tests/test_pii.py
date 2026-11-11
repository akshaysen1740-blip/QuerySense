import pytest

from signalsift.pii import redact


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("mail me at jane.doe@example.com please", "EMAIL"),
        ("call +1 415-555-0132 now", "PHONE"),
        ("card 4111 1111 1111 1111 was charged", "CARD"),
        ("see https://example.com/orders/1", "URL"),
        ("server 192.168.0.12 is down", "IP"),
        ("@JohnSmith please reply", "HANDLE"),
    ],
)
def test_redacts_each_kind(text: str, label: str) -> None:
    res = redact(text)
    assert f"[{label}]" in res.text
    assert res.counts[label] == 1


def test_email_is_not_double_counted_as_handle() -> None:
    res = redact("contact bob@corp.io")
    assert res.counts == {"EMAIL": 1}


def test_order_numbers_survive() -> None:
    assert redact("order #A12345 is late").text == "order #A12345 is late"


def test_clean_text_unchanged() -> None:
    res = redact("the app crashes on launch")
    assert res.text == "the app crashes on launch" and res.counts == {}
