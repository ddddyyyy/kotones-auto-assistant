"""Recognize a lesson's parameter result without inferring its gain values."""

import re
import unicodedata
from cv2.typing import MatLike
from kotonebot.backend.ocr import jp


def is_practice_result_screen(screen: MatLike) -> bool:
    results = jp().ocr(screen)
    limits = [r for r in results if r.confidence >= 0.98
              and re.fullmatch(r'/[1-9]\d{1,4}', r.text.strip())]
    if len(limits) not in (2, 3):
        return False
    # The result gauges share a limit and baseline. Require separate columns;
    # unrelated numbers and duplicated OCR boxes must not establish this page.
    limits.sort(key=lambda r: r.rect.center.x)
    if len({r.text for r in limits}) != 1:
        return False
    if max(r.rect.center.y for r in limits) - min(r.rect.center.y for r in limits) > min(r.rect.h for r in limits) / 2:
        return False
    if any(b.rect.x1 - a.rect.x2 < min(a.rect.w, b.rect.w)
           for a, b in zip(limits, limits[1:])):
        return False
    summaries = []
    for result in results:
        match = re.search(r'(Vo|Da|Vi)\.が\d+上昇した[!！]?',
                          unicodedata.normalize('NFKC', result.text))
        if (result.confidence >= 0.8 and match is not None
                and result.rect.y1 > max(limit.rect.y2 for limit in limits)):
            summaries.append((result, match.group(1)))
    if len(limits) == 2:
        if any(r.confidence >= 0.98 and r.text.strip().upper() == 'UP!'
                   and r.rect.y2 < min(limit.rect.y1 for limit in limits)
                   for r in results):
            return True
        # A single denominator can be misread while the dedicated summary
        # remains visible. Require different parameter rows, not a guessed
        # third limit or duplicate OCR boxes.
        summaries.sort(key=lambda item: item[0].rect.y1)
        return (
            len({label for _, label in summaries}) >= 2
            and max(r.rect.x1 for r, _ in summaries) - min(r.rect.x1 for r, _ in summaries)
            <= min(r.rect.w for r, _ in summaries) / 2
            and all(a.rect.y2 <= b.rect.y1
                    for (a, _), (b, _) in zip(summaries, summaries[1:]))
        )
    return any(r.confidence >= 0.8
               and re.search(r'(?:Vo|Da|Vi)\.が\d+上昇した[!！]?',
                             unicodedata.normalize('NFKC', r.text))
               and r.rect.y1 > max(limit.rect.y2 for limit in limits)
               for r in results)
