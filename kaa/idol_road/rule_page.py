"""当前完整可见应援／妨碍卡片，不把带图标的 OCR 当作完整效果。"""

import hashlib
import re
from typing import Literal
from uuid import uuid4

import cv2
from cv2.typing import MatLike
import numpy as np
from kotonebot.backend.ocr import en, jp
from kotonebot.primitives import Rect

from kaa.tasks import R
from kaa.db.tower_rules import gimmick_descriptions_at_turn
from .rule_references import RuleDescriptionCandidate, match_rule_description
from .rule_program_loader import analyze_rule_programs
from .conditional_rule_page import read_conditional_play
from .good_condition_rule import read_good_condition_requirement
from .good_condition_clause_page import read_good_condition_lexical
from .good_condition_support_page import read_good_condition_support
from .category_rule_page import read_category_condition_evidence
from .turn_budget_page import read_turn_addition
from .recurring_hp_cost_page import read_recurring_hp_timing
from .health_rule_page import read_health_block
from .concentration_rule_page import read_concentration_rule
from .threshold_turn_end_page import read_threshold_turn_end
from .rules import (RuleTextLine, RuleTextRead, RuleTurnNumberEvidence, RuleViewport,
                    VisibleStageRule, confirmed_rule_turn, confirmed_rule_turn_with_number)


def read_support_stage(screen: MatLike) -> int | None:
    box = R.IdolRoad.BoxSupportStage
    crop = screen[box.y1:box.y2, box.x1:box.x2]
    variants = (crop, cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    original = [en().ocr(variant) for variant in variants]
    # Only recheck a complete, agreeing numeric proposal from this same crop.
    if any(len(rows) != 1 or re.fullmatch(r'[1-9][0-9]*', rows[0].text) is None for rows in original):
        return None
    values = [int(rows[0].text) for rows in original]
    if values[0] != values[1]:
        return None
    if all(.98 <= rows[0].confidence <= 1 for rows in original):
        return values[0]
    for variant in variants:
        rows = en().ocr(cv2.resize(variant, None, fx=2, fy=2))
        if (len(rows) != 1 or not .98 <= rows[0].confidence <= 1
            or re.fullmatch(r'[1-9][0-9]*', rows[0].text) is None or int(rows[0].text) != values[0]):
            return None
    return values[0]


def _read(crop: MatLike, frame_id: str) -> tuple[RuleTextRead, ...]:
    variants: tuple[tuple[Literal['original', 'grayscale'], MatLike], ...] = (
        ('original', crop), ('grayscale', cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)),
    )
    return tuple(RuleTextRead(frame_id, representation,
                             tuple(RuleTextLine(row.text, row.confidence) for row in jp().ocr(variant)))
                 for representation, variant in variants)


def _turn_number(title: MatLike, frame_id: str) -> RuleTurnNumberEvidence | None:
    suffix = R.IdolRoad.SupportTurnSuffix.template.pixels
    if title.shape[0] < suffix.shape[0] or title.shape[1] < suffix.shape[1]:
        return None
    scores = cv2.matchTemplate(title, suffix, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(scores)
    # 较低匹配只定位复核区域；不能确认后缀，仍须同帧双表示 OCR >= .98。
    if score < .8 or x <= 0:
        return None
    digits = title[:, :x]
    variants: tuple[tuple[Literal['original', 'grayscale'], MatLike], ...] = (
        ('original', digits), ('grayscale', cv2.cvtColor(digits, cv2.COLOR_BGR2GRAY)),
    )
    reads = tuple(RuleTextRead(frame_id, kind,
                             tuple(RuleTextLine(row.text, row.confidence) for row in en().ocr(image)))
                  for kind, image in variants)
    suffix_reads = _read(title[y:y+suffix.shape[0], x:x+suffix.shape[1]], frame_id)
    return RuleTurnNumberEvidence(frame_id, reads, score, (x, y, suffix.shape[1], suffix.shape[0]), suffix_reads)


def read_rule_viewport(screen: MatLike) -> RuleViewport:
    road = R.IdolRoad
    box = road.BoxSupportCards
    crop = screen[box.y1:box.y2, box.x1:box.x2]
    mask = cv2.inRange(crop, np.array([210]*3), np.array([225]*3))
    reference = road.BoxSupportCardReference
    # 已标注卡片右边界外是滚动条；先隔离它，避免把相邻卡片连成一框。
    mask[:, reference.x2-box.x1:] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    title_box, body_box = road.BoxSupportTurn, road.BoxSupportDescription
    cards = sorted((cv2.boundingRect(contour) for contour in contours), key=lambda rect: rect[1])
    frame_id = uuid4().hex
    rules = []
    fingerprint = hashlib.sha256()
    for x, y, width, height in cards:
        if not reference.w * .95 <= width <= reference.w * 1.05 or not reference.h * .9 <= height <= reference.h * 1.1:
            continue
        if y <= 0 or y + height >= crop.shape[0] or x <= 0 or x + width >= crop.shape[1]:
            continue
        x, y = x + box.x1, y + box.y1

        def region(rect: Rect) -> MatLike:
            dx, dy = rect.x1 - reference.x1, rect.y1 - reference.y1
            return screen[y+dy:y+dy+rect.h, x+dx:x+dx+rect.w]

        title = region(title_box)
        turn_reads = _read(title, frame_id)
        number = _turn_number(title, frame_id)
        turn = confirmed_rule_turn_with_number(turn_reads, number)
        turn_source = ('full_title' if confirmed_rule_turn(turn_reads) == turn else 'number_and_suffix') if turn is not None else None
        body = region(body_box)
        body_reads = _read(body, frame_id)
        category_box=road.BoxSupportCategoryDescription
        category_evidence=read_category_condition_evidence(region(category_box),frame_id,body_reads,
            (x+category_box.x1-reference.x1,y+category_box.y1-reference.y1))
        lexical=read_good_condition_lexical(body,frame_id,
            (x+body_box.x1-reference.x1,y+body_box.y1-reference.y1),turn,body_reads)
        conditional_play = read_conditional_play(body, frame_id, turn, body_reads)
        health_block = read_health_block(body, frame_id, turn, body_reads)
        concentration_rule = read_concentration_rule(body, frame_id, turn, body_reads)
        threshold_turn_end = read_threshold_turn_end(body, frame_id, turn, body_reads)
        facts = gimmick_descriptions_at_turn(turn) if turn is not None else None
        candidates = tuple(RuleDescriptionCandidate(row.group_id, row.priority, row.start_turn,
                                                     row.exam_effect_id, row.description_json)
                           for row in facts) if facts is not None else None
        description_match = match_rule_description(turn, body_reads, candidates)
        program_analysis = analyze_rule_programs(description_match) if description_match.candidates else None
        rules.append(VisibleStageRule((x, y, width, height), turn, turn_reads, body_reads,
                                      turn_number_evidence=number, turn_source=turn_source,
                                      description_match=description_match, program_analysis=program_analysis,
                                      conditional_play=conditional_play, health_block=health_block,
                                      concentration_rule=concentration_rule, threshold_turn_end=threshold_turn_end,
                                      good_condition=read_good_condition_requirement(turn,body_reads,lexical),
                                      good_condition_lexical=lexical,
                                      category_condition=category_evidence,
                                      good_condition_support=read_good_condition_support(
                                          screen[y:y+height,x:x+width],frame_id,(x,y)),
                                      recurring_hp_timing=read_recurring_hp_timing(body,frame_id,
                                          (x+body_box.x1-reference.x1,y+body_box.y1-reference.y1)),
                                      turn_addition=read_turn_addition(body,frame_id,
                                          (x+body_box.x1-reference.x1,y+body_box.y1-reference.y1))))
        fingerprint.update(str((x, y, width, height)).encode('ascii'))
        fingerprint.update(screen[y:y+height, x:x+width].tobytes())
    return RuleViewport(fingerprint.hexdigest() if rules else None, tuple(rules) if rules else None)
