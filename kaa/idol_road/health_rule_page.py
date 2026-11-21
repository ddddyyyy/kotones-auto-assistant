"""同帧核验 mental 图标、完整禁用短语和两回合标记，再读取体力条件。"""
from typing import Literal
import cv2
from cv2.typing import MatLike
from kotonebot.backend.ocr import jp, en
from kaa.tasks import R
from .rule_symbols_page import find_rule_symbols
from .rules import RuleTextRead, RuleTextLine
from .health_rule import HealthBlockAnalysis, HealthBlockEvidence, interpret_health_block


def read_health_block(body: MatLike, frame_id: str, turn: int | None,
                      original_reads: tuple[RuleTextRead, ...]) -> HealthBlockAnalysis:
    road = R.IdolRoad
    symbols = find_rule_symbols(body, frame_id, (
        ('mental', (road.RuleMentalIcon, road.RuleMentalIconScrolled)),
        ('mental_skill_label', (road.RuleMentalSkillLabel, road.RuleMentalSkillLabelScrolled)),
        ('mental_block_two_turns', (road.RuleMentalBlockTwoTurns, road.RuleMentalBlockTwoTurnsScrolled)),
    ))
    reads = []
    duration_reads = []
    blocks = [symbol for symbol in symbols if symbol.kind == 'mental_block_two_turns']
    if len(blocks) == 1:
        reference, number = road.BoxMentalBlockReference, road.BoxMentalDurationDigits
        x, y, _, _ = blocks[0].rect
        x, y = x+number.x1-reference.x1, y+number.y1-reference.y1
        if 0 <= x and 0 <= y and x+number.w <= body.shape[1] and y+number.h <= body.shape[0]:
            digits = body[y:y+number.h, x:x+number.w]
            number_variants: tuple[tuple[Literal['original', 'grayscale'], MatLike], ...] = (
                ('original', digits), ('grayscale', cv2.cvtColor(digits, cv2.COLOR_BGR2GRAY)),
            )
            for kind, image in number_variants:
                duration_reads.append(RuleTextRead(frame_id, kind, tuple(RuleTextLine(row.text, row.confidence) for row in en().ocr(image))))
    icons = [symbol for symbol in symbols if symbol.kind == 'mental']
    if len(icons) == 1:
        x, y, _, height = icons[0].rect
        condition = body[y:y+height, :x]
        if condition.size:
            variants: tuple[tuple[Literal['original', 'grayscale'], MatLike], ...] = (
                ('original', condition), ('grayscale', cv2.cvtColor(condition, cv2.COLOR_BGR2GRAY)),
            )
            for kind, image in variants:
                reads.append(RuleTextRead(frame_id, kind, tuple(RuleTextLine(row.text, row.confidence) for row in jp().ocr(image))))
    return interpret_health_block(turn, HealthBlockEvidence(frame_id, symbols, tuple(reads), tuple(duration_reads)), original_reads)
