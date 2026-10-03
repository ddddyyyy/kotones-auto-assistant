"""Shared payment rules for evaluation and shallow search.

Integer rounding preserves the existing simulator convention; game rounding
and persistent stamina modifiers still need calibration.
"""
from __future__ import annotations

from kaa.db.skill_card import SkillCard
from .state import BattleState

SPECIAL_COST_EFFECTS = {
    'ExamCostType_ExamFullPowerPoint': 'full_power_point',
    'ExamCostType_ExamLessonBuff': 'lesson_buff',
    'ExamCostType_ExamReview': 'review',
    'ExamCostType_ExamCardPlayAggressive': 'aggressive',
    'ExamCostType_ExamParameterBuff': 'parameter_buff',
    'ExamCostType_ExamParameterBuffMultiplePerTurn': 'parameter_buff_multiple',
}


def play_cost(state: BattleState, card: SkillCard) -> tuple[int, int]:
    """Green (genki first) and red (HP only) costs before card effects."""
    multiplier = 1.0
    if not state.anomaly.full_power_active:
        stance = state.anomaly.stance.value
        if stance in {'aggressive', 'aggressive_2'}:
            multiplier = 2.0
        elif stance == 'preservation':
            multiplier = 0.5
        elif stance == 'preservation_2':
            multiplier = 0.25
    return (
        round(max(0, getattr(card, 'stamina', 0) or 0) * multiplier),
        round(max(0, getattr(card, 'force_stamina', 0) or 0) * multiplier),
    )


def special_cost_affordability(state: BattleState, card: SkillCard) -> bool | None:
    """Require a numeric observation to prove a special-resource payment.

    A missing observation must not turn the inferred default zero into a
    rejection. Unknown costs remain unknown rather than free in lookahead.
    """
    value = max(0, getattr(card, 'cost_value', 0) or 0)
    kind = getattr(card, 'cost_type', None)
    if value == 0:
        return True
    effect = SPECIAL_COST_EFFECTS.get(kind or '')
    if effect is None:
        return None
    observed = dict(state.observed_effect_values).get(effect)
    balances = special_resource_balances(state)
    # A raw OCR value still awaiting reconciliation is not a trusted balance.
    if observed is None or observed != balances[effect]:
        return None
    return observed >= value


def special_resource_balances(state: BattleState) -> dict[str, int]:
    """Modeled balances, which can be inferred or hypothetical."""
    return {
        'full_power_point': state.anomaly.full_power_points,
        'lesson_buff': state.buffs.lesson_buff,
        'review': state.buffs.review,
        'aggressive': state.buffs.aggressive,
        'parameter_buff': state.buffs.parameter_buff_turns,
        'parameter_buff_multiple': state.buffs.parameter_buff_multiple_turns,
    }
