"""
normalization.py
=================
Conservative, auditable normalization for business names and addresses.

Design principles (Requirements Analysis §5):
  * Keep raw fields untouched; produce *additional* normalized/tokenized
    representations rather than destroying the original.
  * Be conservative on names: aggressive fuzzy correction risks merging two
    distinct businesses (a false positive we can never fully undo). Legal
    suffix / abbreviation normalization is deterministic and dictionary
    driven, not learned fuzzy matching.
  * Addresses may be partial, reordered, landmark-based, or missing
    components entirely. A missing component is never treated as a
    mismatch -- that is a decision handled downstream in feature
    engineering (see features.py), not here.
  * No geocoding. No external lookups. Every normalization rule is a plain
    string transformation over the record's own fields.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Dictionaries (deterministic, conservative, dataset-agnostic)
# ---------------------------------------------------------------------------

# Legal-entity suffixes to strip once the name has been tokenized. Ordered by
# phrase length (longest first) so multi-word suffixes match before their
# single-word substrings do.
LEGAL_SUFFIXES = [
    "private limited", "pvt limited", "pvt ltd", "private ltd",
    "limited liability company", "limited liability partnership",
    "public limited company", "joint stock company",
    "corporation", "incorporated", "company limited",
    "limited", "corp", "inc", "llc", "llp", "plc", "ltd", "co",
    "gmbh", "sarl", "sa", "ag", "bv", "nv", "oy", "ab", "as", "spa", "srl",
]

# General business-word abbreviation expansions applied at the token level.
# Kept intentionally small and unambiguous to avoid over-normalization.
ABBREVIATION_MAP = {
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "intl": "international",
    "natl": "national",
    "assoc": "associates",
    "assn": "association",
    "mfg": "manufacturing",
    "svcs": "services",
    "svc": "service",
    "grp": "group",
    "dept": "department",
    "&": "and",
}

# Address abbreviation expansions (street types, directions, unit types).
ADDRESS_ABBREVIATIONS = {
    "st": "street", "str": "street",
    "rd": "road",
    "ave": "avenue", "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "sq": "square",
    "ct": "court",
    "pl": "place",
    "ter": "terrace",
    "pkwy": "parkway",
    "cir": "circle",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "blk": "block",
    "bldg": "building",
    "no": "number",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "pkg": "parking",
    "opp": "opposite",
    "nr": "near",
}

# NOTE (fix): "near", "opposite", "behind", "beside", "next", "to" were
# previously stripped as stopwords, which silently destroyed landmark
# references ("Near SBI ATM" -> "SBI ATM"). That contradicts this module's
# own design principle of preserving landmark-based address text. Only
# genuine filler words are stripped now.
STOPWORDS_ADDRESS = {"the", "of", "at"}

PUNCT_RE = re.compile(r"[^\w\s&]", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
POSTAL_CODE_RE = re.compile(r"\b(\d{4,10}(?:-\d{2,4})?)\b")
NUMERIC_TOKEN_RE = re.compile(r"\d+")


def _unicode_normalize(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def _basic_clean(text: str) -> str:
    """Unicode-normalize, casefold, standardize '&'/'and', strip punctuation, collapse whitespace."""
    text = _unicode_normalize(text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = PUNCT_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def _expand_tokens(tokens: List[str], mapping: Dict[str, str]) -> List[str]:
    return [mapping.get(tok, tok) for tok in tokens]


def _strip_legal_suffix(name_clean: str) -> str:
    """Remove a trailing legal-entity suffix phrase, longest match first."""
    for suffix in LEGAL_SUFFIXES:
        pattern = r"\b" + re.escape(suffix) + r"\b\.?\s*$"
        new_name = re.sub(pattern, "", name_clean).strip()
        if new_name != name_clean and new_name:
            return new_name
    return name_clean


@dataclass
class NormalizedName:
    raw: str
    normalized: str          # cleaned, abbreviation-expanded, suffix-stripped
    tokens: List[str] = field(default_factory=list)
    token_sorted: str = ""   # tokens sorted -> order-invariant key
    dedup_tokens: List[str] = field(default_factory=list)  # repeated tokens removed
    alnum: str = ""          # normalized with all non-alphanumerics removed
    is_missing: bool = False


@dataclass
class NormalizedAddress:
    raw: str
    normalized: str
    tokens: List[str] = field(default_factory=list)
    numeric_tokens: List[str] = field(default_factory=list)
    postal_code: Optional[str] = None
    alnum: str = ""
    is_missing: bool = False


def normalize_business_name(raw_name: Optional[str], *, strip_suffix: bool = True,
                             expand_abbrev: bool = True) -> NormalizedName:
    if raw_name is None or (isinstance(raw_name, float)) or str(raw_name).strip() == "" or str(raw_name).lower() == "nan":
        return NormalizedName(raw="", normalized="", tokens=[], token_sorted="", dedup_tokens=[], alnum="", is_missing=True)

    raw = str(raw_name)
    cleaned = _basic_clean(raw)
    tokens = cleaned.split()
    if expand_abbrev:
        tokens = _expand_tokens(tokens, ABBREVIATION_MAP)
    normalized = " ".join(tokens)
    if strip_suffix:
        normalized = _strip_legal_suffix(normalized)
    tokens = normalized.split()

    seen: Set[str] = set()
    dedup: List[str] = []
    for t in tokens:
        if t not in seen:
            dedup.append(t)
            seen.add(t)

    token_sorted = " ".join(sorted(tokens))
    alnum = re.sub(r"[^a-z0-9]", "", normalized)

    return NormalizedName(
        raw=raw, normalized=normalized, tokens=tokens, token_sorted=token_sorted,
        dedup_tokens=dedup, alnum=alnum, is_missing=(normalized == ""),
    )


def normalize_address(raw_address: Optional[str], *, expand_abbrev: bool = True) -> NormalizedAddress:
    if raw_address is None or (isinstance(raw_address, float)) or str(raw_address).strip() == "" or str(raw_address).lower() == "nan":
        return NormalizedAddress(raw="", normalized="", tokens=[], numeric_tokens=[], postal_code=None, alnum="", is_missing=True)

    raw = str(raw_address)
    postal_match = POSTAL_CODE_RE.search(raw)
    postal_code = postal_match.group(1) if postal_match else None

    cleaned = _basic_clean(raw)
    tokens = cleaned.split()
    if expand_abbrev:
        tokens = _expand_tokens(tokens, ADDRESS_ABBREVIATIONS)
    tokens = [t for t in tokens if t not in STOPWORDS_ADDRESS]
    normalized = " ".join(tokens)
    numeric_tokens = [t for t in tokens if NUMERIC_TOKEN_RE.fullmatch(t)]
    alnum = re.sub(r"[^a-z0-9]", "", normalized)

    return NormalizedAddress(
        raw=raw, normalized=normalized, tokens=tokens, numeric_tokens=numeric_tokens,
        postal_code=postal_code, alnum=alnum, is_missing=(normalized == ""),
    )


def char_ngrams(text: str, n: int = 3) -> List[str]:
    """Character n-grams over a normalized string, padded so short strings still yield a gram."""
    if not text:
        return []
    padded = f" {text} "
    if len(padded) < n:
        return [padded]
    return [padded[i:i + n] for i in range(len(padded) - n + 1)]


def normalize_country(raw_country: Optional[str]) -> str:
    """Open-set country normalization: casefold + whitespace only. Never mapped to a fixed enum."""
    if raw_country is None or (isinstance(raw_country, float)) or str(raw_country).strip() == "":
        return ""
    return WHITESPACE_RE.sub(" ", _unicode_normalize(str(raw_country)).strip().lower())