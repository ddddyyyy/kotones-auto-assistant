"""Shared payment rules for evaluation and shallow search.

Integer rounding preserves the existing simulator convention; game rounding
and persistent stamina modifiers still need calibration.
"""
from __future__ import annotations

from kaa.db.constants import ProduceExamEffectType
from kaa.db.skill_card import SkillCard
from kaa.db.card_growth import get_card_growth_rule
from kaa.db.card_status_enchant import card_enchant_uncertainty
from .state import AnomalyStance, BattleState

SPECIAL_COST_EFFECTS = {
    'ExamCostType_ExamFullPowerPoint': 'full_power_point',
    'ExamCostType_ExamLessonBuff': 'lesson_buff',
    'ExamCostType_ExamReview': 'review',
    'ExamCostType_ExamCardPlayAggressive': 'aggressive',
    'ExamCostType_ExamParameterBuff': 'parameter_buff',
    'ExamCostType_ExamParameterBuffMultiplePerTurn': 'parameter_buff_multiple',
}


def projected_growth_payment(state: BattleState, card: SkillCard) -> tuple[int, int, bool]:
    green = max(0, getattr(card, 'stamina', 0) or 0)
    red = max(0, getattr(card, 'force_stamina', 0) or 0)
    known = not card_enchant_uncertainty(card)[0]
    for effect_id, active in state.projected_card_growth:
        rule = get_card_growth_rule(effect_id)
        match = rule.matches(card)
        if match is False:
            continue
        if match is None or not active or not rule.costs_known:
            known = False
            continue
        for change in rule.green_changes:
            green = max(0, green + change)
        for change in rule.red_changes:
            red = max(0, red + change)
    return green, red, known


def projected_growth_output(
    state: BattleState, card: SkillCard, kind: ProduceExamEffectType,
    value: float, count: int, *, applied_count: int,
) -> tuple[float, int, bool]:
    """Apply supported pre-card growth; in-card timing stays unknown."""
    known = not card_enchant_uncertainty(card)[1]
    for index, (effect_id, active) in enumerate(state.projected_card_growth):
        rule = get_card_growth_rule(effect_id)
        match = rule.matches(card)
        if match is False or not rule.output_changed:
            continue
        if (
            match is None or not active or index >= applied_count
            or not rule.output_known
        ):
            known = False
            continue
        if kind == ProduceExamEffectType.ExamLesson:
            for change in rule.lesson_changes:
                value = max(0.0, value + change)
            count += sum(rule.lesson_count_changes)
        elif kind == ProduceExamEffectType.ExamBlock:
            value += sum(rule.block_changes)
        elif kind == ProduceExamEffectType.ExamReview:
            value += sum(rule.review_changes)
        elif kind == ProduceExamEffectType.ExamFullPowerPoint:
            value += sum(rule.full_power_changes)
        elif kind in {
            ProduceExamEffectType.ExamLessonDependBlock,
            ProduceExamEffectType.ExamLessonDependExamCardPlayAggressive,
            ProduceExamEffectType.ExamLessonDependExamReview,
        } and not (rule.lesson_changes or rule.lesson_count_changes):
            # Resource growth changes the simulated balance, not this coefficient.
            pass
        else:
            known = False
    return value, count, known


def projected_cost_supported(state: BattleState, card: SkillCard) -> bool:
    """Keep uncalibrated modifier combinations out of payment proofs."""
    green, red, growth_known = projected_growth_payment(state, card)
    if not growth_known:
        return False
    if state.projected_effect_values is None or state.projected_stamina_consumption is None:
        return True
    if any(
        rule.matches(card) is not False and (rule.green_changes or rule.red_changes)
        for effect_id, _active in state.projected_card_growth
        for rule in (get_card_growth_rule(effect_id),)
    ):
        # Growth plus temporary consumption modifiers has no payment evidence.
        return False
    if not (green or red):
        return True
    return (
        state.projected_stamina_consumption == 'half'
        and state.anomaly.stance == AnomalyStance.NORMAL
        and not state.anomaly.full_power_active
        and not red
    )


def play_cost(state: BattleState, card: SkillCard) -> tuple[int, int]:
    """Green (genki first) and red (HP only) costs before card effects."""
    green, red, _growth_known = projected_growth_payment(state, card)
    if (
        state.projected_effect_values is not None
        and state.projected_stamina_consumption == 'half'
        and projected_cost_supported(state, card)
    ):
        # Calibrated green costs 4->2, 3->2 and 2->1. The modifier starts
        # after the granting card pays; repeated grants extend duration,
        # rather than halving an already reduced cost again.
        return ((green + 1) // 2, 0)
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
        round(green * multiplier),
        round(red * multiplier),
    )


def special_cost_affordability(state: BattleState, card: SkillCard) -> bool | None:
    """Require a numeric observation to prove a special-resource payment.

    A missing observation must not turn the inferred default zero into a
    rejection. Unknown costs remain unknown rather than free in lookahead.
    """
    if not projected_growth_payment(state, card)[2]:
        return None
    value = max(0, getattr(card, 'cost_value', 0) or 0)
    kind = getattr(card, 'cost_type', None)
    if value == 0:
        return True
    effect = SPECIAL_COST_EFFECTS.get(kind or '')
    if effect is None:
        return None
    observed = dict(state.effect_values_for_planning).get(effect)
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
