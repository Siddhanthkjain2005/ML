"""Deterministic record preparation; every transform leaves the input untouched."""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

_SPACE = re.compile(r"\s+")
_NUMBERS = re.compile(r"\d+")
_LEGAL_ALIASES = {
    "corporation": "corp", "limited": "ltd", "private": "pvt",
    "incorporated": "inc", "company": "co",
}
_LEGAL_SUFFIXES = frozenset({"corp", "ltd", "pvt", "inc", "co", "llc", "llp", "plc", "lp"})
_ACRONYM_STOP = _LEGAL_SUFFIXES | frozenset({"and", "the", "of"})


def text_value(value: Any) -> str:
    """Coerce scalar text without reinterpreting any literal string sentinel."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return value if isinstance(value, str) else str(value)


def normalize_text(value: Any) -> str:
    """NFKC + casefold, ampersand expansion, Unicode punctuation separation.

    Letters, numbers and combining marks from every script are retained. Accent
    removal is deliberately a separate alternate representation.
    """
    text = unicodedata.normalize("NFKC", text_value(value)).casefold().replace("&", " and ")
    text = "".join(char if unicodedata.category(char)[0] in "LNM" else " " for char in text)
    return _SPACE.sub(" ", text).strip()


def fold_diacritics(value: Any) -> str:
    """Accent-fold an alternate view; this is not transliteration."""
    text = normalize_text(value)
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def legal_name_alternates(value: Any) -> tuple[str, str]:
    """Return canonical legal suffixes and a conservative suffix-free alternate."""
    tokens = [_LEGAL_ALIASES.get(t, t) for t in normalize_text(value).split()]
    canonical = " ".join(tokens)
    core = list(tokens)
    while core and core[-1] in _LEGAL_SUFFIXES:
        core.pop()
    return canonical, " ".join(core) if core else canonical


def normalize_legal_name(value: Any) -> str:
    """Canonical name with common legal suffix spellings made consistent."""
    return legal_name_alternates(value)[0]


def number_tokens(value: Any) -> frozenset[str]:
    """Normalize Unicode decimal digits and leading zeros, preserving all runs."""
    result = set()
    for run in _NUMBERS.findall(normalize_text(value)):
        digits = "".join(str(unicodedata.decimal(c)) for c in run)
        result.add(digits.lstrip("0") or "0")
    return frozenset(result)


def acronym(value: Any) -> str:
    tokens = legal_name_alternates(value)[0].split()
    return "".join(t[0] for t in tokens if t not in _ACRONYM_STOP and t[0].isalpha())


@dataclass(frozen=True, slots=True)
class PreparedText:
    normalized: str
    folded: str
    tokens: frozenset[str]
    sorted_tokens: str
    numbers: frozenset[str]
    acronym: str
    legal: str
    core: str


@dataclass(frozen=True, slots=True)
class PreparedRecord:
    entity_id: str
    source: str
    country: str
    name: PreparedText
    address: PreparedText


def prepare_text(value: Any, *, business_name: bool = False) -> PreparedText:
    normalized = normalize_text(value)
    tokens = normalized.split()
    legal_tokens = [_LEGAL_ALIASES.get(token, token) for token in tokens] if business_name else tokens
    legal = " ".join(legal_tokens)
    core_tokens = list(legal_tokens)
    if business_name:
        while core_tokens and core_tokens[-1] in _LEGAL_SUFFIXES:
            core_tokens.pop()
    core = " ".join(core_tokens) if core_tokens else legal
    numbers = frozenset(
        "".join(str(unicodedata.decimal(c)) for c in run).lstrip("0") or "0"
        for run in _NUMBERS.findall(normalized)
    )
    return PreparedText(
        normalized=normalized,
        folded="".join(c for c in unicodedata.normalize("NFD", normalized) if unicodedata.category(c) != "Mn"),
        tokens=frozenset(tokens), sorted_tokens=" ".join(sorted(tokens)),
        numbers=numbers,
        acronym="".join(t[0] for t in legal_tokens if t not in _ACRONYM_STOP and t[0].isalpha()) if business_name else "",
        legal=legal, core=core,
    )


def prepare_record(record: Mapping[str, Any] | PreparedRecord) -> PreparedRecord:
    """Prepare once per entity and reuse this object across candidate pairs."""
    if isinstance(record, PreparedRecord):
        return record
    entity_id = text_value(record.get("entity_id", ""))
    source = text_value(record.get("source", "")).upper()
    if not source:
        source = entity_id.partition("-")[0].upper() if "-" in entity_id else ""
    return PreparedRecord(
        entity_id=entity_id, source=source, country=normalize_text(record.get("country")),
        name=prepare_text(record.get("business_name"), business_name=True),
        address=prepare_text(record.get("business_address")),
    )


def prepare_records(records: Iterable[Mapping[str, Any] | PreparedRecord]) -> list[PreparedRecord]:
    return [prepare_record(record) for record in records]
