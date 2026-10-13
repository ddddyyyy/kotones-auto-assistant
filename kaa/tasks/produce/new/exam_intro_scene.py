"""Recognize the judging intro without interpreting parameter multipliers."""

import re
from cv2.typing import MatLike
from kotonebot.backend.ocr import jp


def is_exam_intro_screen(screen: MatLike) -> bool:
    results = jp().ocr(screen)
    titles = {r.text.strip(): r for r in results if r.confidence >= 0.98
              and r.text.strip() in {'審査基準', '合格条件'}}
    if len(titles) != 2:
        return False
    judging, passing = titles['審査基準'], titles['合格条件']
    if (judging.rect.x2 >= passing.rect.x1
            or abs(judging.rect.center.y - passing.rect.center.y)
            > min(judging.rect.h, passing.rect.h) / 2):
        return False
    conditions = [r for r in results if r.confidence >= 0.98
                  and re.fullmatch(r'[1-9]位以上', r.text.strip())
                  and r.rect.y1 > passing.rect.y2
                  and r.rect.y1 - passing.rect.y2 < passing.rect.h * 3
                  and abs(r.rect.x1 - passing.rect.x1) < passing.rect.h]
    if len(conditions) != 1:
        return False
    # Three separate left-hand parameter rows sit below the pass condition.
    # Percent signs and footer hints are unreliable OCR anchors on this page.
    values = [r for r in results if r.confidence >= 0.98
              and re.fullmatch(r'\d{1,4}', r.text.strip())
              and r.rect.y1 > conditions[0].rect.y2
              and r.rect.x2 < passing.rect.x1]
    if len(values) != 3:
        return False
    values.sort(key=lambda r: r.rect.y1)
    return (
        max(r.rect.x1 for r in values) - min(r.rect.x1 for r in values)
        <= min(r.rect.w for r in values) / 2
        and all(a.rect.y2 < b.rect.y1 for a, b in zip(values, values[1:]))
    )
