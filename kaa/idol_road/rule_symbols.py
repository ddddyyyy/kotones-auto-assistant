"""当前规则正文中的可核验符号；矩形相对于当前正文裁剪。"""
from dataclasses import dataclass
from typing import Literal

RuleSymbolKind = Literal['concentration', 'extra_play', 'mental', 'active', 'mental_skill_label', 'mental_block_two_turns', 'positive_turn_end_support', 'positive_debuff_recovery', 'turn_addition']


@dataclass(frozen=True)
class RuleSymbolEvidence:
    frame_id: str
    kind: RuleSymbolKind
    rect: tuple[int, int, int, int]
    score: float
