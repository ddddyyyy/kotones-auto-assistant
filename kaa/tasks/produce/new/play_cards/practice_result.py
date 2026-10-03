"""Read a lesson's transient result banner without guessing from scene exit."""

import re
import unicodedata

from cv2.typing import MatLike
from kotonebot import logging
from kotonebot.backend.ocr import en
from kotonebot.primitives import Rect


RESULT_BANNER = Rect(0, 500, 720, 400)
logger = logging.getLogger(__name__)


def classify_practice_result(text: str) -> str | None:
    compact = re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).upper()
    if 'PERFECT' in compact:
        return 'PERFECT'
    if 'NOTCLEAR' in compact or 'FAILED' in compact or 'FAILURE' in compact:
        return 'FAILED'
    if 'CLEAR' in compact:
        return 'CLEAR'
    return None


def read_practice_result(screen: MatLike) -> str | None:
    text = en().ocr(screen, rect=RESULT_BANNER).squash().text
    result = classify_practice_result(text)
    logger.debug('Practice result OCR: %r -> %s', text, result)
    return result
