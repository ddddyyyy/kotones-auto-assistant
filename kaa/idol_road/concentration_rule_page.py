"""已知集中图标遮罩后的同帧文字核验，其他图标／规则不简化。"""
from typing import Literal
import cv2
from cv2.typing import MatLike
from kotonebot.backend.ocr import jp
from kaa.tasks import R
from .rules import RuleTextRead, RuleTextLine
from .rule_symbols_page import find_rule_symbols
from .concentration_rule import (ConcentrationRuleEvidence, ConcentrationRuleAnalysis,
                                 interpret_concentration_rule)


def read_concentration_rule(body: MatLike, frame_id: str, turn: int | None,
                             original: tuple[RuleTextRead, ...]) -> ConcentrationRuleAnalysis | None:
    # Don't add OCR work to unrelated cards. This is a proposal, not proof.
    if not any('集中2倍' in line.text for read in original for line in read.lines):
        return None
    road = R.IdolRoad
    icons = find_rule_symbols(body, frame_id, (
        ('concentration', (road.RuleConcentrationIcon, road.RuleConcentrationIconScrolled)),))
    masked = body.copy()
    for icon in icons:
        x,y,w,h = icon.rect
        masked[y:y+h,x:x+w] = 255
    reads = []
    variants: tuple[tuple[Literal['original', 'grayscale'], MatLike], ...] = (
        ('original', masked), ('grayscale', cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)))
    for kind, image in variants:
        rows = sorted(jp().ocr(image), key=lambda row: (round(row.original_rect.center.y/10), row.original_rect.x1))
        reads.append(RuleTextRead(frame_id, kind, tuple(RuleTextLine(row.text,row.confidence) for row in rows)))
    return interpret_concentration_rule(turn, ConcentrationRuleEvidence(frame_id,icons,tuple(reads)),original)
