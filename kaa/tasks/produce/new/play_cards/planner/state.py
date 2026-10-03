"""规划器使用的纯战斗状态模型。

状态模型刻意不依赖截图或设备 API，后续可以直接用于模拟、回放和单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from kaa.db.constants import ProduceExamEffectType, ProducePlanType


class AnomalyStance(str, Enum):
    """非凡体系的姿态状态。"""

    NORMAL = 'normal'
    PRESERVATION = 'preservation'
    PRESERVATION_2 = 'preservation_2'
    AGGRESSIVE = 'aggressive'
    AGGRESSIVE_2 = 'aggressive_2'


@dataclass(frozen=True)
class AnomalyState:
    """温存、全力、热情采用独立维度，避免错误地视为互斥流派。"""

    stance: AnomalyStance = AnomalyStance.NORMAL
    stance_locked: bool = False
    over_preservation: bool = False
    full_power_points: int = 0
    cumulative_full_power_points: int = 0
    full_power_pending: bool = False
    full_power_active: bool = False
    full_power_turns: int = 0
    stance_change_count: int = 0
    preservation_change_count: int = 0
    concentration_change_count: int = 0
    full_power_change_count: int = 0
    enthusiasm: int = 0
    extra_play_count: int = 0


@dataclass(frozen=True)
class ExamBuffState:
    """Sense/logic values that affect later cards and end-of-turn scoring."""

    lesson_buff: int = 0
    parameter_buff_turns: int = 0
    parameter_buff_multiple_turns: int = 0
    review: int = 0
    aggressive: int = 0


@dataclass(frozen=True)
class ProduceItemState:
    """局内 P 道具及其触发进度。"""

    item_id: str
    fired_count: int = 0
    fire_limit: int | None = None
    cooldown_turns: int = 0
    enabled: bool = True

    @property
    def can_fire(self) -> bool:
        if not self.enabled or self.cooldown_turns > 0:
            return False
        return self.fire_limit is None or self.fired_count < self.fire_limit


@dataclass(frozen=True)
class BattleState:
    """一次决策所需的完整、可扩展战斗快照。"""

    remaining_turns: int
    hp: int
    genki: int
    current_score: int | None = None
    """Player's observed ranked-exam score; not used as a lesson score gap."""
    target_score: int | None = None
    """Current first-place score + 1 when the player is behind; not a final target."""
    score_gap: int | None = None
    """距离当前课程 CLEAR/PERFECT 目标还差的参数值。考试中通常未知。"""
    score_target_is_final: bool | None = None
    """True when the visible lesson target is PERFECT, False for CLEAR."""
    is_exam: bool = False
    playable_count: int = 1
    archetype: ProduceExamEffectType | None = None
    plan_type: ProducePlanType | None = None
    hand_card_ids: tuple[str, ...] = ()
    hand_size: int | None = None
    """Visible hand count, including cards whose identity could not be read."""
    draw_pile_card_ids: tuple[str, ...] = ()
    draw_pile_empty: bool | None = None
    """None means the unseen draw pile has not been checked, not that it is empty."""
    discard_card_ids: tuple[str, ...] = ()
    removed_card_ids: tuple[str, ...] = ()
    observed_effects: frozenset[str] = frozenset()
    """HUD-confirmed effects; used when their exact numeric value is unreadable."""
    confirmed_absent_effects: frozenset[str] = frozenset()
    """Effects whose calibrated HUD icons were absent on repeated observations."""
    items: tuple[ProduceItemState, ...] = ()
    anomaly: AnomalyState = field(default_factory=AnomalyState)
    buffs: ExamBuffState = field(default_factory=ExamBuffState)
    observation_confidence: float = 1.0
    hp_known: bool = True
    genki_known: bool = True
    observed_remaining_turns: int | None = None
    observed_hp: int | None = None
    observed_genki: int | None = None
    """Raw numeric OCR before reconciliation; None means no reading."""
    remaining_turns_known: bool = True
    """False means remaining_turns is a conservative planning horizon, not a HUD fact."""
    observed_effect_values: tuple[tuple[str, int | None], ...] = ()
    """Raw HUD values; inferred buffs must not confirm their own speculative play."""

    def __post_init__(self) -> None:
        if self.remaining_turns < 0:
            raise ValueError('remaining_turns must be non-negative')
        if not 0.0 <= self.observation_confidence <= 1.0:
            raise ValueError('observation_confidence must be between 0 and 1')
