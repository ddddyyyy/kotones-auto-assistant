"""匹配当前正文内的两个已知图标，遮罩后保留独立 OCR 证据。"""

from typing import Literal
import cv2
from cv2.typing import MatLike
from kotonebot.backend.ocr import jp

from kaa.tasks import R
from .rules import RuleTextLine, RuleTextRead
from .conditional_rule import (ConditionalPlayAnalysis, ConditionalPlayEvidence, RuleIconEvidence,
                               interpret_conditional_play)
from .rule_symbols_page import find_rule_symbols


def _icons(body: MatLike, frame_id: str) -> tuple[RuleIconEvidence, ...]:
    road = R.IdolRoad
    kinds: tuple[tuple[Literal['concentration', 'extra_play'], tuple], ...] = (
        ('concentration', (road.RuleConcentrationIcon, road.RuleConcentrationIconScrolled)),
        ('extra_play', (road.RuleExtraPlayIcon, road.RuleExtraPlayIconScrolled, road.RuleExtraPlayIconScrolledWide)),
    )
    return find_rule_symbols(body, frame_id, kinds)


def read_conditional_play(body: MatLike, frame_id: str, turn: int | None,
                           original_reads: tuple[RuleTextRead, ...]) -> ConditionalPlayAnalysis:
    icons = _icons(body, frame_id)
    reads = []
    if len(icons) == 2 and {icon.kind for icon in icons} == {'concentration', 'extra_play'}:
        masked = body.copy()
        for icon in icons:
            x, y, w, h = icon.rect
            masked[y:y+h, x:x+w] = 255
        variants: tuple[tuple[Literal['original', 'grayscale'], MatLike], ...] = (
            ('original', masked), ('grayscale', cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)),
        )
        for kind, image in variants:
            # OCR 返回顺序不作拼接依据；明确按本帧行位置和横向位置排序。
            rows = sorted(jp().ocr(image), key=lambda row: (round(row.original_rect.center.y/10), row.original_rect.x1))
            reads.append(RuleTextRead(frame_id, kind, tuple(RuleTextLine(row.text, row.confidence) for row in rows)))
    evidence = ConditionalPlayEvidence(frame_id, icons, tuple(reads))
    return interpret_conditional_play(turn, evidence, original_reads)
