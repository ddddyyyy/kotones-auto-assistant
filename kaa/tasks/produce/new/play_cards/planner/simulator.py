"""Small deterministic state transitions used by shallow card lookahead."""

from __future__ import annotations

from copy import copy
from dataclasses import replace

from kaa.db.constants import ProduceExamEffectType
from kaa.db.skill_card import SkillCard

from .costs import (
    SPECIAL_COST_EFFECTS, play_cost, projected_cost_supported, projected_growth_output,
    projected_growth_payment, special_cost_affordability,
    special_resource_balances,
)
from .memory import BattleMemory
from .state import BattleState
from .trigger import TriggerVerdict, evaluate_trigger_id


_RESOURCE_EFFECTS = {
    ProduceExamEffectType.ExamLessonBuff: 'lesson_buff',
    ProduceExamEffectType.ExamLessonBuffReduce: 'lesson_buff',
    ProduceExamEffectType.ExamReview: 'review',
    ProduceExamEffectType.ExamReviewReduce: 'review',
    ProduceExamEffectType.ExamCardPlayAggressive: 'aggressive',
    ProduceExamEffectType.ExamAggressiveReduce: 'aggressive',
    ProduceExamEffectType.ExamParameterBuff: 'parameter_buff',
    ProduceExamEffectType.ExamParameterBuffMultiplePerTurn: 'parameter_buff_multiple',
    ProduceExamEffectType.ExamFullPowerPoint: 'full_power_point',
    ProduceExamEffectType.ExamFullPowerPointReduce: 'full_power_point',
}


_RESOURCE_GAIN_EFFECTS = {
    ProduceExamEffectType.ExamLessonBuff,
    ProduceExamEffectType.ExamReview,
    ProduceExamEffectType.ExamCardPlayAggressive,
    ProduceExamEffectType.ExamParameterBuff,
    ProduceExamEffectType.ExamParameterBuffMultiplePerTurn,
    ProduceExamEffectType.ExamFullPowerPoint,
}
_RESOURCE_GAIN_MODIFIERS = {
    ProduceExamEffectType.ExamAggressiveAdditive: 'aggressive',
    ProduceExamEffectType.ExamAggressiveAdditiveFix: 'aggressive',
    ProduceExamEffectType.ExamLessonBuffAdditive: 'lesson_buff',
    ProduceExamEffectType.ExamLessonBuffAdditiveFix: 'lesson_buff',
    ProduceExamEffectType.ExamParameterBuffAdditive: 'parameter_buff',
    ProduceExamEffectType.ExamReviewAdditive: 'review',
    ProduceExamEffectType.ExamFullPowerPointAdditive: 'full_power_point',
}


_UNMODELED_SPECIAL_RESOURCE_EFFECTS = {
    ProduceExamEffectType.ExamAggressiveValueMultiple: 'aggressive',
    ProduceExamEffectType.ExamLessonBuffDependParameterBuff: 'lesson_buff',
    ProduceExamEffectType.ExamLessonBuffPerSearchCount: 'lesson_buff',
    ProduceExamEffectType.ExamParameterBuffDependLessonBuff: 'parameter_buff',
    ProduceExamEffectType.ExamParameterBuffReduce: 'parameter_buff',
    ProduceExamEffectType.ExamParameterBuffMultiplePerTurnReduce: 'parameter_buff_multiple',
    ProduceExamEffectType.ExamReviewDependExamBlock: 'review',
    ProduceExamEffectType.ExamReviewDependExamCardPlayAggressive: 'review',
    ProduceExamEffectType.ExamReviewPerSearchCount: 'review',
    ProduceExamEffectType.ExamReviewValueMultiple: 'review',
}


_UNMODELED_GENKI_EFFECTS = {
    ProduceExamEffectType.ExamBlockAddMultipleAggressive,
    ProduceExamEffectType.ExamBlockPerUseCardCount,
    ProduceExamEffectType.ExamBlockDependExamReview,
    ProduceExamEffectType.ExamBlockDependBlockConsumptionSum,
    ProduceExamEffectType.ExamBlockDown,
}
_UNMODELED_HP_EFFECTS = {
    ProduceExamEffectType.ExamStaminaRecoverMultiple,
    ProduceExamEffectType.ExamStaminaReduce,
    ProduceExamEffectType.ExamStaminaReduceFix,
    ProduceExamEffectType.ExamStaminaDamage,
}


def estimate_genki_gain(
    kind: ProduceExamEffectType, base: float, aggressive: int, bonus: float = 0.0,
) -> float:
    """Ranking amount; callers separately preserve uncertain resource budgets."""
    if kind == ProduceExamEffectType.ExamBlock:
        return base + max(0, aggressive)
    if kind == ProduceExamEffectType.ExamBlockAddMultipleAggressive:
        return base + max(0, aggressive) * (1 + bonus / 1000.0)
    return base


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


def _growth_adjusted_resource_card(state: BattleState, card: SkillCard) -> tuple[SkillCard, set[str]]:
    """Copy search-only resource segments before memory applies ordered effects."""
    adjusted = copy(card)
    adjusted.play_effects = []
    growth = list(state.projected_card_growth)
    uncertain: set[str] = set()
    for play_effect in getattr(card, 'play_effects', ()):
        effect = play_effect.produce_exam_effect
        if (
            effect is not None and effect.effect_type in _RESOURCE_EFFECTS
            and evaluate_trigger_id(getattr(play_effect, '_produce_exam_trigger_id', ''), state)
            != TriggerVerdict.INACTIVE
        ):
            value, _count, known = projected_growth_output(
                replace(state, projected_card_growth=tuple(growth)), card,
                effect.effect_type, effect.effect_value1 or 0, 1,
                applied_count=len(state.projected_card_growth),
            )
            if not known:
                uncertain.add(_RESOURCE_EFFECTS[effect.effect_type])
            segment = copy(play_effect)
            segment.produce_exam_effect = copy(effect)
            segment.produce_exam_effect.effect_value1 = int(value)
            adjusted.play_effects.append(segment)
        else:
            adjusted.play_effects.append(play_effect)
        if effect is not None and effect.effect_type == ProduceExamEffectType.ExamAddGrowEffect:
            verdict = evaluate_trigger_id(getattr(play_effect, '_produce_exam_trigger_id', ''), state)
            if verdict != TriggerVerdict.INACTIVE:
                growth.append((getattr(effect, '_id', ''), verdict == TriggerVerdict.ACTIVE))
    return adjusted, uncertain


def simulate_play(state: BattleState, card: SkillCard) -> BattleState:
    """Return the state after playing one card, for ordering comparisons.

    This deliberately models only information already visible to the planner.
    Random draws and item triggers remain outside the shallow search.
    """
    memory = BattleMemory(anomaly=state.anomaly, buffs=state.buffs)
    resource_card, growth_uncertain_balances = _growth_adjusted_resource_card(state, card)
    memory.record_play(resource_card, state)
    added_plays = (
        memory.anomaly.extra_play_count - state.anomaly.extra_play_count
    )

    extra_turns = 0
    turns_known = state.remaining_turns_known
    stance_uncertain = state.projected_stance_uncertain
    stance_locked = state.anomaly.stance_locked
    lock_uncertain = state.projected_stance_lock_uncertain
    full_power_active = state.anomaly.full_power_active
    full_power_uncertain = state.projected_full_power_uncertain
    hp = state.hp
    genki = state.genki
    aggressive = state.buffs.aggressive
    aggressive_values = dict(state.effect_values_for_planning)
    aggressive_known = aggressive_values.get('aggressive') == aggressive or (
        aggressive == 0
        and state.projected_effect_values is None
        and 'aggressive' not in aggressive_values
        and 'aggressive' not in state.observed_effects
    )
    if getattr(card, 'cost_type', None) == 'ExamCostType_ExamCardPlayAggressive':
        aggressive = max(0, aggressive - max(0, getattr(card, 'cost_value', 0) or 0))
    green_cost, red_cost = play_cost(state, card)
    genki_spent = min(genki, green_cost)
    genki -= genki_spent
    hp = max(0, hp - (green_cost - genki_spent) - red_cost)

    pride_value = dict(state.effect_values_for_planning).get('pride', 0)
    pride = (
        'pride' in state.observed_effects if state.projected_effect_values is None
        else None if pride_value is None else pride_value == 1
    )
    modifier = (
        state.projected_stamina_consumption
        if state.projected_effect_values is not None
        else 'unknown' if 'stamina_consumption_modifier' in state.observed_effects else None
    )
    modifier_effects = {
        ProduceExamEffectType.ExamStaminaConsumptionDown,
        ProduceExamEffectType.ExamStaminaConsumptionAdd,
        ProduceExamEffectType.ExamStaminaConsumptionDownFix,
        ProduceExamEffectType.ExamStaminaConsumptionAddFix,
        ProduceExamEffectType.ExamSearchPlayCardStaminaConsumptionChange,
    }
    restrictions = set(
        state.projected_resource_restrictions
        if state.projected_effect_values is not None else ()
    )
    hp_known = state.hp_known and projected_cost_supported(state, card) and (
        green_cost == 0 or state.genki_known
    )
    genki_known = state.genki_known and projected_cost_supported(state, card)
    restriction_effects = {
        ProduceExamEffectType.ExamBlockRestriction: 'genki_gain',
        ProduceExamEffectType.ExamStaminaRecoverRestriction: 'hp_recovery',
    }
    gain_modifiers = set(
        state.projected_resource_gain_modifiers
        if state.projected_effect_values is not None else ()
    )
    growth = list(state.projected_card_growth)
    uncertain_balances = growth_uncertain_balances
    unmodeled_zero_balances: set[str] = set()
    resource_prefixes = {
        'lesson_buff': 'ExamLessonBuff',
        'review': 'ExamReview',
        'aggressive': 'ExamAggressive',
        'parameter_buff_multiple': 'ExamParameterBuffMultiplePerTurn',
        'parameter_buff': 'ExamParameterBuff',
        'full_power_point': 'ExamFullPowerPoint',
    }
    if not projected_growth_payment(state, card)[2]:
        # Unknown growth may change a special fee or replace its type. Paying
        # the DB base amount does not establish the next card's resource budget.
        uncertain_balances.update(SPECIAL_COST_EFFECTS.values())
    for play_effect in getattr(card, 'play_effects', ()):
        effect = play_effect.produce_exam_effect
        if effect is None or effect.effect_type is None:
            continue
        verdict = evaluate_trigger_id(
            getattr(play_effect, '_produce_exam_trigger_id', ''), state,
        )
        if effect.effect_type == ProduceExamEffectType.ExamExtraTurn:
            if verdict != TriggerVerdict.INACTIVE:
                # The current DB primitive grants one turn with zero numeric
                # fields. Restrict this forecast to the verified plain form;
                # growth/chains/unknown conditions cannot invent opportunities.
                plain = (
                    not any(getattr(effect, name, 0) for name in (
                        'effect_value1', 'effect_value2', 'effect_count', 'effect_turn',
                    ))
                    and getattr(effect, '_produce_card_grow_effect_ids', None) == '[]'
                    and not getattr(effect, '_chain_produce_exam_effect_id', '')
                    and not getattr(effect, 'chain_effect_ids', ())
                    and not getattr(effect, '_produce_card_search_id', '')
                    and card._id not in state.unresolved_growth_card_ids
                    and not state.projected_card_growth
                )
                if verdict == TriggerVerdict.ACTIVE and plain:
                    extra_turns += 1
                # Even a deterministic extension is a search horizon, never
                # new HUD evidence or proof of the next turn's attribute/draw.
                turns_known = False
        if (
            (verdict == TriggerVerdict.UNKNOWN
             or (verdict == TriggerVerdict.ACTIVE and (lock_uncertain or full_power_uncertain)))
            and effect.effect_type in {
                ProduceExamEffectType.ExamConcentration,
                ProduceExamEffectType.ExamPreservation,
                ProduceExamEffectType.ExamStanceReset,
            }
            and not full_power_active
            and not stance_locked
        ):
            # Skipping a conditional switch leaves only the old estimate.
            # Its possible level and switching history are not known.
            stance_uncertain = True
        if (
            verdict == TriggerVerdict.UNKNOWN
            and effect.effect_type == ProduceExamEffectType.StanceLock
            and not stance_locked
        ):
            lock_uncertain = True
        if (
            verdict == TriggerVerdict.UNKNOWN
            and effect.effect_type == ProduceExamEffectType.ExamFullPower
            and not full_power_active
        ):
            full_power_uncertain = True
            stance_uncertain = True
        if verdict == TriggerVerdict.ACTIVE:
            if effect.effect_type == ProduceExamEffectType.StanceLock:
                stance_locked = True
                lock_uncertain = False
            elif effect.effect_type == ProduceExamEffectType.ExamFullPower:
                full_power_active = True
                full_power_uncertain = False
        if (
            verdict != TriggerVerdict.INACTIVE
            and effect.effect_type == ProduceExamEffectType.ExamLessonValueMultipleDependReviewOrAggressive
        ):
            if verdict == TriggerVerdict.ACTIVE and (
                effect.effect_turn == -1 or (effect.effect_turn or 0) > 0
            ):
                pride = True
            elif pride is not True:
                pride = None
        if verdict != TriggerVerdict.INACTIVE:
            if effect.effect_type in _RESOURCE_GAIN_MODIFIERS:
                gain_modifiers.add(_RESOURCE_GAIN_MODIFIERS[effect.effect_type])
            if (
                effect.effect_type in _RESOURCE_GAIN_EFFECTS
                and _RESOURCE_EFFECTS[effect.effect_type] in gain_modifiers
            ):
                # The prior balance remains valid until a matching grant uses
                # this uncalibrated modifier; reductions are unaffected.
                uncertain_balances.add(_RESOURCE_EFFECTS[effect.effect_type])
        if (
            verdict != TriggerVerdict.INACTIVE
            and effect.effect_type in _UNMODELED_SPECIAL_RESOURCE_EFFECTS
        ):
            # An unmodeled direct transfer/scaling invalidates an existing
            # numeric budget too, not just a newly seeded absent-resource zero.
            uncertain_balances.add(_UNMODELED_SPECIAL_RESOURCE_EFFECTS[effect.effect_type])
        if verdict != TriggerVerdict.INACTIVE and effect.effect_type not in _RESOURCE_EFFECTS:
            # A zero observed before an unmodeled related modifier/transfer
            # cannot establish this resource's balance after the card.
            kind = effect.effect_type.value.removeprefix('ProduceExamEffectType_')
            for resource, prefix in resource_prefixes.items():
                if kind.startswith(prefix):
                    unmodeled_zero_balances.add(resource)
                    break
        if verdict != TriggerVerdict.INACTIVE and effect.effect_type == ProduceExamEffectType.ExamAddGrowEffect:
            growth.append((getattr(effect, '_id', ''), verdict == TriggerVerdict.ACTIVE))
        if verdict == TriggerVerdict.UNKNOWN and effect.effect_type in _RESOURCE_EFFECTS:
            # Skipping an uncertain effect does not prove that the old balance
            # survives. It may change before the second card pays.
            uncertain_balances.add(_RESOURCE_EFFECTS[effect.effect_type])
        if effect.effect_type in modifier_effects and verdict == TriggerVerdict.UNKNOWN:
            modifier = 'unknown'
        if verdict != TriggerVerdict.INACTIVE and effect.effect_type in restriction_effects:
            restrictions.add(restriction_effects[effect.effect_type])
        if verdict == TriggerVerdict.UNKNOWN and effect.effect_type in {
            ProduceExamEffectType.ExamBlock, ProduceExamEffectType.ExamBlockFix,
        }:
            genki_known = False
        if verdict == TriggerVerdict.UNKNOWN and effect.effect_type == ProduceExamEffectType.ExamStaminaRecoverFix:
            hp_known = False
        if (
            verdict != TriggerVerdict.INACTIVE
            and effect.effect_type == ProduceExamEffectType.ExamLessonDependBlock
            and (effect.effect_value2 or 0) != 0
        ):
            # Consumption/rounding is not a trusted future payment balance.
            # A definitely active full conversion has no genki left even as
            # a ranking estimate; retain UNKNOWN for its resource budget.
            genki_known = False
            if verdict == TriggerVerdict.ACTIVE and effect.effect_value2 == 1000:
                genki = 0
        if verdict != TriggerVerdict.INACTIVE:
            # These direct resource changes need unobserved maxima, counters
            # or uncalibrated rules. Skipping their amount is only an estimate;
            # it cannot certify the old balance for payment or conversion.
            if effect.effect_type in _UNMODELED_GENKI_EFFECTS:
                genki_known = False
            if effect.effect_type in _UNMODELED_HP_EFFECTS:
                hp_known = False
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
            ProduceExamEffectType.ExamCardPlayAggressive,
            ProduceExamEffectType.ExamAggressiveReduce,
        }:
            aggressive_value, _, _ = projected_growth_output(
                replace(state, projected_card_growth=tuple(growth)), card,
                effect.effect_type, value, 1,
                applied_count=len(state.projected_card_growth),
            )
            aggressive = max(0, aggressive + (
                aggressive_value if effect.effect_type == ProduceExamEffectType.ExamCardPlayAggressive
                else -aggressive_value
            ))
        if effect.effect_type in {
            ProduceExamEffectType.ExamBlock,
            ProduceExamEffectType.ExamBlockFix,
        }:
            block_value, _count, growth_known = projected_growth_output(
                replace(state, projected_card_growth=tuple(growth)), card,
                effect.effect_type, value, 1, applied_count=len(state.projected_card_growth),
            )
            if effect.effect_type == ProduceExamEffectType.ExamBlock:
                block_value = estimate_genki_gain(effect.effect_type, block_value, int(aggressive))
                genki_known &= aggressive_known and 'aggressive' not in uncertain_balances
            genki += int(block_value)
            genki_known &= growth_known
            if block_value and 'genki_gain' in restrictions:
                genki_known = False
        elif effect.effect_type == ProduceExamEffectType.ExamStaminaRecoverFix:
            recovery, _count, growth_known = projected_growth_output(
                replace(state, projected_card_growth=tuple(growth)), card,
                effect.effect_type, value, 1, applied_count=len(state.projected_card_growth),
            )
            hp += int(recovery)
            hp_known &= growth_known
            if recovery and 'hp_recovery' in restrictions:
                hp_known = False

    if stance_uncertain or state.projected_full_power_uncertain:
        # Leaving modeled preservation may grant a play that the unknown
        # actual stance would not grant. Count only independent additions;
        # retain the original anomaly as an estimate for ranking/history.
        opportunity_memory = BattleMemory(
            anomaly=replace(
                state.anomaly, stance_locked=True,
                full_power_active=(state.anomaly.full_power_active
                                   or state.projected_full_power_uncertain),
            ), buffs=state.buffs,
        )
        opportunity_memory.record_play(
            resource_card, replace(state, projected_stance_uncertain=True),
        )
        added_plays = (
            opportunity_memory.anomaly.extra_play_count
            - state.anomaly.extra_play_count
        )

    projected_values = dict(state.effect_values_for_planning)
    if pride is not False or 'pride' in projected_values:
        projected_values['pride'] = None if pride is None else int(pride)
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
        if (
            state.projected_effect_values is None
            and effect not in projected_values
            and effect in state.confirmed_absent_effects
            and effect not in state.observed_effects
            and before_balances[effect] == 0
            and effect not in unmodeled_zero_balances
        ):
            # Current reconciled absence establishes a zero starting balance.
            # It is a search input, never a fabricated numeric HUD reading.
            # Do not reuse old absence after projection, or override conflicts.
            projected_values[effect] = 0
        if projected_values.get(effect) is not None:
            projected_values[effect] = (
                balances[effect]
                if projected_values[effect] == before_balances[effect]
                and effect not in uncertain_balances
                else None
            )

    hand = list(state.hand_card_ids)
    try:
        hand.remove(card._id)
    except ValueError:
        pass
    return replace(
        state,
        observed_card_output=None,
        remaining_turns=state.remaining_turns + extra_turns,
        remaining_turns_known=turns_known,
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
        projected_stance_uncertain=stance_uncertain,
        projected_stance_lock_uncertain=lock_uncertain,
        projected_full_power_uncertain=full_power_uncertain,
        # Green payment spills into HP when genki is insufficient. An unknown
        # genki balance cannot establish exact remaining HP after that payment.
        hp_known=hp_known,
        genki_known=genki_known,
        projected_resource_restrictions=frozenset(restrictions),
        projected_resource_gain_modifiers=frozenset(gain_modifiers),
        projected_card_growth=tuple(growth),
    )


def simulate_play_prefix(state: BattleState, card: SkillCard, effect_count: int) -> BattleState:
    """Pay once, then apply only effects preceding an in-card conversion.

    Work on a shallow card copy; the DB-owned effect list stays unchanged.
    This reuses search's supported transitions and uncertainty propagation.
    """
    prefix = copy(card)
    prefix.play_effects = card.play_effects[:effect_count]
    return simulate_play(state, prefix)
