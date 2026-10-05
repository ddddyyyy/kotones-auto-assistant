"""Same-frame card preview values; never post-play outcome evidence."""
from typing import NamedTuple

import cv2
from cv2.typing import MatLike
from kotonebot.backend.ocr import en
from kotonebot.primitives import Rect


class CardOutputPreview(NamedTuple):
    per_hit: int
    count: int | None


def _integer_read(image: MatLike) -> int | None:
    results = en().ocr(image)
    if (len(results) == 1 and results[0].confidence >= 0.98
            and results[0].text.isascii() and results[0].text.isdecimal()):
        return int(results[0].text)
    return None


def read_card_output_preview(screen: MatLike, letter: Rect) -> CardOutputPreview | None:
    """Read the header anchored to its own letter; no DB count fallback."""
    x, y = int(letter.center.x), letter.y1
    height, width = screen.shape[:2]
    x1, x2, y1, y2 = x - 58, x + 75, y - 200, y - 153
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return None
    crop = screen[y1:y2, x1:x2]
    try:
        amount = _integer_read(crop)
        white = (crop.min(axis=2) > 230).astype('uint8') * 255
        if amount is None or _integer_read(white) != amount:
            return None
        # The hit-count bubble is neutral gray; the output banner is colored.
        low, high = crop.min(axis=2), crop.max(axis=2)
        gray = ((high.astype('int16') - low < 50) & (high < 180) & (low > 25))
        _, _, stats, _ = cv2.connectedComponentsWithStats(gray.astype('uint8'))
        bubbles = [(x, y, w, h) for x, y, w, h, area in stats[1:]
                   if x >= 40 and 12 <= w <= 45 and 12 <= h <= 35 and area >= 100]
        if len(bubbles) != 1:
            return CardOutputPreview(amount, None)
        bx, by, bw, bh = bubbles[0]
        badge = crop[by:by+bh, bx:bx+bw]
        ink = (badge.min(axis=2) > 210).astype('uint8') * 255
        _, _, digits, _ = cv2.connectedComponentsWithStats(ink)
        glyphs = [(x, y, w, h) for x, y, w, h, area in digits[1:]
                  if 0 < x and x+w < bw and 1 < y and y+h < bh
                  and 4 <= w <= 14 and 10 <= h <= 18 and area >= 20]
        # Multi-digit/merged/occluded counts are outside this reader's scope.
        if len(glyphs) != 1:
            return CardOutputPreview(amount, None)
        gx, gy, gw, gh = glyphs[0]
        digit = ink[gy:gy+gh, gx:gx+gw]
        digit = cv2.copyMakeBorder(digit, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0,))
        counts = [_integer_read(cv2.resize(digit, None, fx=k, fy=k,
                   interpolation=cv2.INTER_NEAREST)) for k in (2, 3)]
        if counts[0] is None or counts[0] != counts[1] or not 1 <= counts[0] <= 9:
            return CardOutputPreview(amount, None)
        raw = cv2.copyMakeBorder(badge[gy:gy+gh, gx:gx+gw], 4, 4, 4, 4,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
        raw_count = _integer_read(cv2.resize(raw, None, fx=3, fy=3,
                                           interpolation=cv2.INTER_NEAREST))
        if raw_count is not None and raw_count != counts[0]:
            return CardOutputPreview(amount, None)
        return CardOutputPreview(amount, counts[0])
    except Exception:
        return None
