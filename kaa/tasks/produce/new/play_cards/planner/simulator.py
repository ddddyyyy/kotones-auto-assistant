"""Small deterministic state transitions used by shallow card lookahead."""

from __future__ import annotations

from dataclasses import replace

from kaa.db.constants import ProduceExamEffectType
from kaa.db.skill_card import SkillCard

from .costs import (
    SPECIAL_COST_EFFECTS, play_cost, projected_cost_supported, special_cost_affordability,
    special_resource_balances,
)
from .memory import BattleMemory
from .state import BattleState
from .trigger import TriggerVerdict, evaluate_trigger_id


def can_afford_play(state: BattleState, card: SkillCard) -> bool | None:
    """Prove affordability from the available HUD values, or return unknown."""
    special = special_cost_affordability(state, card)
    if special is False:
        return False
    if not projected_cost_supported(state, card):
        return None
    green_cost, red_cost = play_cost(state, card)
    if green_cost == 0 and red_cost == 0:
        return special
    if state.hp_known and state.hp < red_cost:
        return False
    if red_cost == 0 and state.genki_known and state.genki >= green_cost:
        return special
    if state.hp_known and state.hp >= green_cost + red_cost:
        # Even zero genki would pay the cost, so its OCR value is unnecessary.
        return special
    if state.hp_known and state.genki_known:
        if state.hp < max(0, green_cost - state.genki) + red_cost:
            return False
        return special
    return None


def simulate_play(state: BattleState, card: SkillCard) -> BattleState:
    """Return the state after playing one card, for ordering comparisons.

    This deliberately models only information already visible to the planner.
    Random draws and item triggers remain outside the shallow search.
    """
    memory = BattleMemory(anomaly=state.anomaly, buffs=state.buffs)
    memory.record_play(card, state)
    added_plays = (
        memory.anomaly.extra_play_count - state.anomaly.extra_play_count
    )

    hp = state.hp
    genki = state.genki
    green_cost, red_cost = play_cost(state, card)
    genki_spent = min(genki, green_cost)
    genki -= genki_spent
    hp = max(0, hp - (green_cost - genki_spent) - red_cost)

    modifier = state.projected_stamina_consumption if state.projected_effect_values is not None else None
    modifier_effects = {
        ProduceExamEffectType.ExamStaminaConsumptionDown,
        ProduceExamEffectType.ExamStaminaConsumptionAdd,
        ProduceExamEffectType.ExamStaminaConsumptionDownFix,
        ProduceExamEffectType.ExamStaminaConsumptionAddFix,
        ProduceExamEffectType.ExamSearchPlayCardStaminaConsumptionChange,
    }
    for play_effect in getattr(card, 'play_effects', ()):
        effect = play_effect.produce_exam_effect
        if effect is None or effect.effect_type is None:
            continue
        verdict = evaluate_trigger_id(
            getattr(play_effect, '_produce_exam_trigger_id', ''), state,
        )
        if effect.effect_type in modifier_effects and verdict == TriggerVerdict.UNKNOWN:
            modifier = 'unknown'
        if verdict != TriggerVerdict.ACTIVE:
            continue
        if effect.effect_type in modifier_effects:
            if (
                effect.effect_type == ProduceExamEffectType.ExamStaminaConsumptionDown
                and (effect.effect_turn == -1 or (effect.effect_turn or 0) > 0)
                and modifier != 'unknown'
            ):
                modifier = 'half'
            else:
                modifier = 'unknown'
        value = max(0, effect.effect_value1 or 0)
        if effect.effect_type in {
            ProduceExamEffectType.ExamBlock,
            ProduceExamEffectType.ExamBlockFix,
        }:
            genki += value
        elif effect.effect_type == ProduceExamEffectType.ExamStaminaRecoverFix:
            hp += value

    projected_values = dict(state.effect_values_for_planning)
    balances = {
        'full_power_point': memory.anomaly.full_power_points,
        'lesson_buff': memory.buffs.lesson_buff,
        'review': memory.buffs.review,
        'aggressive': memory.buffs.aggressive,
        'parameter_buff': memory.buffs.parameter_buff_turns,
        'parameter_buff_multiple': memory.buffs.parameter_buff_multiple_turns,
    }
    before_balances = special_resource_balances(state)
    for effect in SPECIAL_COST_EFFECTS.values():
        if projected_values.get(effect) is not None:
            projected_values[effect] = (
                balances[effect]
                if projected_values[effect] == before_balances[effect]
                else None
            )

    hand = list(state.hand_card_ids)
    try:
        hand.remove(card._id)
    except ValueError:
        pass
    return replace(
        state,
        hp=hp,
        genki=genki,
        playable_count=max(0, state.playable_count - 1 + added_plays),
        hand_card_ids=tuple(hand),
        hand_size=(
            max(0, state.hand_size - 1)
            if state.hand_size is not None
            else None
        ),
        anomaly=memory.anomaly,
        buffs=memory.buffs,
        projected_effect_values=tuple(projected_values.items()),
        projected_stamina_consumption=modifier,
        hp_known=state.hp_known and projected_cost_supported(state, card),
        genki_known=state.genki_known and projected_cost_supported(state, card),
    )
