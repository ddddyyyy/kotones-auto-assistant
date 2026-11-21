"""Shared cached digit classifier for current screenshot readers."""
from functools import lru_cache


@lru_cache(maxsize=1)
def digit_classifier():
    import ddddocr
    return ddddocr.DdddOcr(show_ad=False)
