"""应援／妨碍的可见文字观察；图标、效果语义与完整性尚未确认。"""

from dataclasses import dataclass
import re
from typing import Literal, TYPE_CHECKING
import unicodedata

if TYPE_CHECKING:
    from .category_rule import CategoryConditionEvidence
    from .rule_references import RuleDescriptionMatch
    from .rule_program import RuleProgramAnalysis
    from .conditional_rule import ConditionalPlayAnalysis
    from .health_rule import HealthBlockAnalysis
    from .concentration_rule import ConcentrationRuleAnalysis
    from .threshold_turn_end import ThresholdTurnEndAnalysis
    from .good_condition_rule import GoodConditionAnalysis,GoodConditionLexicalEvidence
    from .rule_symbols import RuleSymbolEvidence
    from .recurring_hp_cost import RecurringHpTiming


@dataclass(frozen=True)
class RuleTextLine:
    text: str
    confidence: float


@dataclass(frozen=True)
class RuleTextRead:
    frame_id: str
    representation: Literal['original', 'grayscale']
    lines: tuple[RuleTextLine, ...]


@dataclass(frozen=True)
class RuleTurnNumberEvidence:
    frame_id: str
    reads: tuple[RuleTextRead, ...]
    suffix_score: float
    suffix_rect: tuple[int, int, int, int]
    suffix_reads: tuple[RuleTextRead, ...] = ()


def _turn_text(text: str) -> int | None:
    normalized = re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))
    match = re.fullmatch(r'([1-9][0-9]*)ターン目', normalized)
    return int(match[1]) if match is not None else None


def confirmed_rule_turn(reads: tuple[RuleTextRead, ...]) -> int | None:
    if (len(reads) != 2 or not reads[0].frame_id or reads[0].frame_id != reads[1].frame_id
        or {row.representation for row in reads} != {'original', 'grayscale'}):
        return None
    turns = []
    for read in reads:
        if len(read.lines) != 1 or not .98 <= read.lines[0].confidence <= 1:
            return None
        turn = _turn_text(read.lines[0].text)
        if turn is None:
            return None
        turns.append(turn)
    return turns[0] if turns[0] == turns[1] else None


def confirmed_rule_turn_with_number(
    reads: tuple[RuleTextRead, ...], number: RuleTurnNumberEvidence | None,
) -> int | None:
    """同帧固定后缀＋完整数字可补低置信混排标题；高置信冲突仍未知。"""
    primary = confirmed_rule_turn(reads)
    if number is None:
        return primary
    suffix_confirmed = .97 <= number.suffix_score <= 1
    if number.suffix_reads:
        suffix_reads = number.suffix_reads
        if (len(suffix_reads) != 2 or any(row.frame_id != number.frame_id for row in suffix_reads)
            or {row.representation for row in suffix_reads} != {'original', 'grayscale'}):
            return None
        for row in suffix_reads:
            if any(.98 <= line.confidence <= 1 for line in row.lines):
                if len(row.lines) != 1 or re.sub(r'\s+', '', row.lines[0].text) != 'ターン目':
                    return None
        suffix_confirmed = suffix_confirmed or all(
            len(row.lines) == 1 and .98 <= row.lines[0].confidence <= 1
            and re.sub(r'\s+', '', row.lines[0].text) == 'ターン目' for row in suffix_reads)
    if not suffix_confirmed:
        return primary
    if (len(reads) != 2 or not number.frame_id or any(row.frame_id != number.frame_id for row in reads)
        or {row.representation for row in reads} != {'original', 'grayscale'}
        or len(number.reads) != 2 or any(row.frame_id != number.frame_id for row in number.reads)
        or {row.representation for row in number.reads} != {'original', 'grayscale'}):
        return None
    values = []
    for row in number.reads:
        if len(row.lines) != 1 or not .98 <= row.lines[0].confidence <= 1:
            return primary
        text = unicodedata.normalize('NFKC', row.lines[0].text).strip()
        if re.fullmatch(r'[1-9][0-9]*', text) is None:
            return primary
        values.append(int(text))
    if values[0] != values[1]:
        return None
    for row in reads:
        if any(.98 <= line.confidence <= 1 for line in row.lines):
            if len(row.lines) != 1:
                return None
            # 原标题任一高置信表示都必须与补读数字相符，不能挑有利的版本。
            if _turn_text(row.lines[0].text) != values[0]:
                return None
    return values[0]


@dataclass(frozen=True)
class VisibleStageRule:
    card_rect: tuple[int, int, int, int]
    turn: int | None
    turn_reads: tuple[RuleTextRead, ...]
    description_reads: tuple[RuleTextRead, ...]
    turn_number_evidence: RuleTurnNumberEvidence | None = None
    turn_source: Literal['full_title', 'number_and_suffix'] | None = None
    description_match: 'RuleDescriptionMatch | None' = None
    program_analysis: 'RuleProgramAnalysis | None' = None
    conditional_play: 'ConditionalPlayAnalysis | None' = None
    health_block: 'HealthBlockAnalysis | None' = None
    effect_id: None = None
    """OCR 文字包含图标缺口，不据此推导效果 ID 或执行事件。"""
    concentration_rule: 'ConcentrationRuleAnalysis | None' = None
    threshold_turn_end: 'ThresholdTurnEndAnalysis | None' = None
    good_condition: 'GoodConditionAnalysis | None' = None
    good_condition_support: 'RuleSymbolEvidence | None' = None
    turn_addition: 'RuleSymbolEvidence | None' = None
    recurring_hp_timing: 'RecurringHpTiming | None' = None
    good_condition_lexical: 'GoodConditionLexicalEvidence | None' = None
    category_condition: 'CategoryConditionEvidence | None' = None


@dataclass(frozen=True)
class RuleViewport:
    key: str | None
    visible_rules: tuple[VisibleStageRule, ...] | None


@dataclass(frozen=True)
class StageRules:
    character_id: str
    stage: int
    viewports: tuple[RuleViewport, ...]
    complete: Literal[False] = False
    semantics_confirmed: Literal[False] = False
    source: str = 'visible_support_details'
    identity_source: str = 'opened_from_confirmed_formation'
