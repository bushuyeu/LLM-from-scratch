import re


_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

_PHONE_RE = re.compile(
    r"\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}"
)

_IP_RE = re.compile(
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
)


def mask_emails(text: str) -> tuple[str, int]:
    matches = _EMAIL_RE.findall(text)
    masked = _EMAIL_RE.sub("|||EMAIL_ADDRESS|||", text)
    return masked, len(matches)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    matches = _PHONE_RE.findall(text)
    masked = _PHONE_RE.sub("|||PHONE_NUMBER|||", text)
    return masked, len(matches)


def mask_ips(text: str) -> tuple[str, int]:
    matches = _IP_RE.findall(text)
    masked = _IP_RE.sub("|||IP_ADDRESS|||", text)
    return masked, len(matches)
