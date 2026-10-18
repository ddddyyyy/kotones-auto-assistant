"""Recognize the exam ranking page without attributing single-card output."""

import re
from cv2.typing import MatLike
from kotonebot.backend.ocr import jp


def is_exam_result_screen(screen: MatLike) -> bool:
    results = jp().ocr(screen)
    scores = [r for r in results if r.confidence >= 0.9
              and re.fullmatch(r'\d{1,3}(?:,\d{3})*(?:Pt|pt)', r.text.strip())]
    if len(scores) < 3:
        return False
    # The ranking score rows form a column; isolated battle score badges do not.
    if max(r.rect.x1 for r in scores) - min(r.rect.x1 for r in scores) > min(r.rect.w for r in scores) / 2:
        return False
    ranks = [r for r in results if r.confidence >= 0.9
             and re.fullmatch(r'(?:1st|2nd|3rd|[4-9]th)', r.text.strip())]
    if len({r.text for r in ranks}) < 2:
        return False
    if sum(any(rank.rect.x2 < score.rect.x1
               and abs(rank.rect.center.y - score.rect.center.y) < rank.rect.h
               for rank in ranks) for score in scores) < 2:
        return False
    controls = [r for r in results if r.confidence >= 0.98
                and r.rect.y1 > max(score.rect.y2 for score in scores)]
    return {'再挑戦', '次へ'} <= {r.text.strip() for r in controls}
