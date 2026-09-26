"""Optional script-neutral alternate representation, preserving raw records.

ICU rules transliterate characters; no business lookup or translation API is
used. PyICU and ICU versions are recorded with fitted retrieval artifacts.
"""
from functools import lru_cache
from .text import normalize_text


@lru_cache(maxsize=1)
def _transform():
    import icu
    return icu.Transliterator.createInstance('Any-Latin; Latin-ASCII')


def romanize(value):
    text=normalize_text(value)
    if text.isascii():return text
    return normalize_text(_transform().transliterate(text))


def representation_metadata():
    import icu
    from pathlib import Path
    from .retrieval import _digest
    return {'representation':'ICU Any-Latin; Latin-ASCII', 'PyICU':icu.VERSION,
            'ICU':icu.ICU_VERSION,'implementation_sha256':_digest(Path(__file__))}
