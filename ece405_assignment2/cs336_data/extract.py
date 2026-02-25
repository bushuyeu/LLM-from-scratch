from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """Extract plain text from raw HTML bytes.

    Decodes bytes to string (detecting encoding if UTF-8 fails),
    then extracts visible text using Resiliparse.
    """
    try:
        html_str = html_bytes.decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        encoding = detect_encoding(html_bytes)
        if encoding:
            html_str = html_bytes.decode(encoding, errors="replace")
        else:
            html_str = html_bytes.decode("utf-8", errors="replace")

    return extract_plain_text(html_str)
