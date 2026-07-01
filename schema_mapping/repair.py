"""Automatic repair of malformed JSON from AI output.

Attempts common fixes in order: strip fences, fix unquoted keys,
fix trailing commas, fix single quotes, fix Python literals,
fix truncated JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any


def try_parse_json(raw: str) -> dict[str, Any] | None:
    """Attempt to parse *raw* as JSON, applying repairs if needed.

    Returns the parsed dict on success, or ``None`` if all attempts fail.
    """
    # Attempt 1: direct parse
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: repaired
    repaired = repair_json(text)
    if repaired != text:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # Attempt 3: try extracting JSON from markdown code block
    extracted = _extract_code_block(text)
    if extracted and extracted != text:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    return None


def repair_json(raw: str) -> str:
    """Apply all known repairs to *raw*, returning the best attempt.

    Repairs are idempotent — calling twice on the same string is safe.
    """
    text = raw.strip()

    text = _strip_fences(text)
    text = _fix_python_literals(text)
    text = _fix_single_quotes(text)       # ' → "  before quoting keys
    text = _fix_unquoted_keys(text)
    text = _fix_trailing_commas(text)
    text = _fix_truncated_json(text)

    return text.strip()


# ── Individual repair helpers ──────────────────────────────────────────


def _strip_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ``` or ``` ... ```)."""
    # ```json ... ```
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def _fix_unquoted_keys(text: str) -> str:
    """Quote unquoted Python-style keys (e.g. ``name: "value"``).

    Avoids re-quoting keys already wrapped in double quotes.
    """
    return re.sub(
        r'(?<!")(\b[a-zA-Z_][a-zA-Z0-9_]*\b)(?=\s*:)',
        r'"\1"',
        text,
    )


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before ``}`` or ``]``."""
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    return text


def _fix_single_quotes(text: str) -> str:
    """Replace single quotes with double quotes for JSON compliance.

    Handles nested quotes and escaped quotes.
    """
    # Replace single quotes that are likely JSON string delimiters
    result = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"' and (i == 0 or text[i - 1] != "\\"):
            in_string = not in_string
            result.append(ch)
        elif ch == "'" and not in_string:
            # Check if this is a JSON string boundary
            result.append('"')
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def _fix_python_literals(text: str) -> str:
    """Replace Python ``None``, ``True``, ``False`` with JSON equivalents."""
    text = re.sub(r'\bNone\b', 'null', text)
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    return text


def _fix_truncated_json(text: str) -> str:
    """Attempt to close an unclosed JSON object or array.

    If we detect an unclosed object ``{...`` or array ``[...``,
    add the missing closing bracket(s).
    """
    # Count opening/closing braces and brackets
    open_braces = text.count("{")
    close_braces = text.count("}")
    open_brackets = text.count("[")
    close_brackets = text.count("]")

    result = text
    if open_braces > close_braces:
        result += "}" * (open_braces - close_braces)
    if open_brackets > close_brackets:
        result += "]" * (open_brackets - close_brackets)

    return result


def _extract_code_block(text: str) -> str:
    """Extract JSON from a markdown code block (crude fallback)."""
    # Try to find the first { ... } block
    start = text.find("{")
    if start == -1:
        return ""
    # Count braces to find matching close
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # Unclosed — return from start to end
    return text[start:]
