"""规划器使用的纯战斗状态模型。

状态模型刻意不依赖截图或设备 API，后续可以直接用于模拟、回放和单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

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
    """决策快照，含观察、推断和预留字段；并非完整游戏真值。"""

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
    exam_score_percent: int | None = None
    """Current exam attribute percentage from the same HUD, never a future-turn value."""
    playable_count: int = 1
    archetype: ProduceExamEffectType | None = None
    plan_type: ProducePlanType | None = None
    hand_card_ids: tuple[str, ...] = ()
    hand_size: int | None = None
    """Visible hand count, including cards whose identity could not be read."""
    draw_pile_card_ids: tuple[str, ...] = ()
    draw_pile_empty: bool | None = None
    """Reserved input: production never observes emptiness; None stays unknown."""
    discard_card_ids: tuple[str, ...] = ()
    removed_card_ids: tuple[str, ...] = ()
    observed_effects: frozenset[str] = frozenset()
    """HUD-confirmed effects; used when their exact numeric value is unreadable."""
    confirmed_absent_effects: frozenset[str] = frozenset()
    """Effects whose calibrated HUD icons were absent on repeated observations."""
    items: tuple[ProduceItemState, ...] = ()
    """Reserved only: production does not populate or simulate P-item triggers."""
    anomaly: AnomalyState = field(default_factory=AnomalyState)
    buffs: ExamBuffState = field(default_factory=ExamBuffState)
    observation_confidence: float = 1.0
    """Legacy reserved scalar, not measured or used by production decisions.

    Its default must not be interpreted as 100% confidence; use field-level
    known flags, raw observations and CardEvaluation confidence instead.
    """
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

    projected_effect_values: tuple[tuple[str, int | None], ...] | None = None
    """Trusted numeric balances propagated by lookahead, never HUD evidence.

    None denotes an observation state; an empty tuple denotes a simulation
    with no trusted numeric balances. Raw observed_effect_values stay intact.
    """

    projected_full_power_uncertain: bool = False
    """A possible same-turn full-power entry, never an observed active state.

    Confirmed entry resolves current activity, but does not certify a new
    entry reward when the prior search state may already have been active.
    """

    projected_stance_lock_uncertain: bool = False
    """A possible search-only lock; the current stance remains known until a switch.

    A later confirmed lock resolves this marker, without resolving earlier
    uncertainty about stance changes. Fresh HUD states do not inherit it.
    """

    projected_stance_uncertain: bool = False
    """A possible conditional stance change in search, never a HUD reading.

    Keep the modeled stance for ranking; it cannot prove later costs or output
    multipliers. Further grants do not resolve its level or switching history.
    """

    projected_stamina_consumption: Literal['half', 'unknown'] | None = None
    """Same-turn search modifier, never a HUD observation.

    None retains the existing stance-only estimate; it does not prove the
    absence of current modifiers. Unknown forbids proving a future payment.
    """

    projected_resource_restrictions: frozenset[str] = frozenset()
    """Possible same-turn gain restrictions, used only by search.

    A known or conditionally unknown restriction prevents certifying later
    grants. This does not claim the restriction was observed on the HUD.
    """

    projected_resource_gain_modifiers: frozenset[str] = frozenset()
    """Possible unmodeled gain modifiers, carried only by search states.

    They leave the current balance intact but prevent certifying a later
    matching grant. Expiry and amounts are unverified; fresh HUD snapshots
    rebuild state rather than treating this marker as an observation.
    """

    observed_card_output: tuple[str, int, int] | None = None
    """Candidate-local current header (card kind, per-hit, count), ranking only.

    Set only for the individual evaluation; never shared with other slots,
    lookahead, memory, or a later observation.
    """

    unresolved_growth_card_ids: frozenset[str] = frozenset()
    """Card kinds with a visible, unparsed growth marker in this snapshot.

    Copy identity is unavailable: matching copies all remain uncertain, never
    share an inferred growth amount. Refreshed observations replace this set.
    """

    projected_card_growth: tuple[tuple[str, bool], ...] = ()
    """Growth effect IDs and confirmed-trigger flags, only for same-turn search.

    Rules match each candidate's metadata rather than changing DB cards or
    treating every copy of a card ID as one observed instance.
    """

    @property
    def effect_values_for_planning(self) -> tuple[tuple[str, int | None], ...]:
        return (
            self.observed_effect_values
            if self.projected_effect_values is None
            else self.projected_effect_values
        )

    def __post_init__(self) -> None:
        if self.remaining_turns < 0:
            raise ValueError('remaining_turns must be non-negative')
        if not 0.0 <= self.observation_confidence <= 1.0:
            raise ValueError('observation_confidence must be between 0 and 1')
