"""
Plain text (.txt, .md) — with one twist: a .txt file that's actually a
WhatsApp chat export gets detected and reformatted into something
retrievable, instead of being indexed as one wall of undifferentiated
timestamps.

WhatsApp exports look like (Android):
    12/08/23, 9:41 PM - John Doe: Hey, how's it going?
or (iOS):
    [12/08/23, 9:41:02 PM] John Doe: Hey, how's it going?

Multi-line messages (no leading timestamp) are continuations of the
previous message, not new ones — the parser folds those back in rather
than treating every newline as a new chunk-worthy unit.
"""
from __future__ import annotations

import re
from pathlib import Path

TEXT_EXTENSIONS = {".txt", ".md"}

_WHATSAPP_LINE_RE = re.compile(
    r"^\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s+"          # date
    r"(\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?)\]?"  # time
    r"\s*-?\s*([^:]{1,64}):\s(.*)$"                 # sender: message
)


def _looks_like_whatsapp_export(text: str, sample_lines: int = 15) -> bool:
    lines = [l for l in text.splitlines() if l.strip()][:sample_lines]
    if len(lines) < 3:
        return False
    matches = sum(1 for l in lines if _WHATSAPP_LINE_RE.match(l))
    return (matches / len(lines)) >= 0.5


def _parse_whatsapp_export(text: str) -> tuple[str, dict]:
    messages: list[dict] = []
    for line in text.splitlines():
        m = _WHATSAPP_LINE_RE.match(line)
        if m:
            date, time, sender, body = m.groups()
            messages.append({"date": date, "time": time, "sender": sender.strip(), "text": body.strip()})
        elif messages and line.strip():
            # continuation of the previous message (no timestamp prefix)
            messages[-1]["text"] += "\n" + line.strip()

    formatted = "\n".join(
        f"{m['sender']} ({m['date']} {m['time']}): {m['text']}" for m in messages
    )
    participants = sorted({m["sender"] for m in messages})
    metadata = {
        "format": "whatsapp_export",
        "message_count": len(messages),
        "participants": participants,
    }
    return formatted, metadata


def load_text_or_whatsapp(path: Path) -> tuple[str, dict]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    metadata = {"source_path": str(path), "filename": path.name}

    if path.suffix.lower() == ".txt" and _looks_like_whatsapp_export(raw):
        text, wa_meta = _parse_whatsapp_export(raw)
        metadata.update(wa_meta)
        return text, metadata

    metadata["format"] = "plain_text"
    return raw, metadata
