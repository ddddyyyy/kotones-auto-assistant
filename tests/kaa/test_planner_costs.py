from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from kaa.tasks.produce.new.play_cards.planner.costs import play_cost
from kaa.tasks.produce.new.play_cards.planner.evaluator import EffectHeuristicEvaluator
from kaa.tasks.produce.new.play_cards.planner.simulator import can_afford_play, simulate_play
from kaa.tasks.produce.new.play_cards.planner.state import (
    AnomalyState, AnomalyStance, BattleState, ExamBuffState,
)


def card(cost_type=None, cost_value=0, stamina=0, force=0):
    return cast(Any, SimpleNamespace(
        _id='cost-card', cost_type=cost_type, cost_value=cost_value,
        stamina=stamina, force_stamina=force, play_effects=[],
        plan_type=None, evaluation=0, play_move_position_type=None,
    ))


@pytest.mark.parametrize('cost_type, effect, buffs', [
    ('ExamLessonBuff', 'lesson_buff', ExamBuffState(lesson_buff=4)),
    ('ExamReview', 'review', ExamBuffState(review=4)),
    ('ExamCardPlayAggressive', 'aggressive', ExamBuffState(aggressive=4)),
    ('ExamParameterBuff', 'parameter_buff', ExamBuffState(parameter_buff_turns=4)),
    ('ExamParameterBuffMultiplePerTurn', 'parameter_buff_multiple', ExamBuffState(parameter_buff_multiple_turns=4)),
])
def test_special_payment_is_consumed_before_a_second_card(cost_type, effect, buffs):
    first = card('ExamCostType_' + cost_type, 3)
    state = BattleState(remaining_turns=3, hp=20, genki=20, buffs=buffs,
                        observed_effect_values=((effect, 4),))
    assert can_afford_play(state, first) is True
    after = simulate_play(state, first)
    assert after.observed_effect_values == state.observed_effect_values
    assert after.projected_effect_values is not None
    assert dict(after.projected_effect_values)[effect] == 1
    assert can_afford_play(after, first) is False


def test_full_power_payment_clears_pending_and_does_not_reuse_points():
    payment = card('ExamCostType_ExamFullPowerPoint', 6)
    state = BattleState(remaining_turns=3, hp=20, genki=20,
                        anomaly=AnomalyState(full_power_points=10, full_power_pending=True,
                                             cumulative_full_power_points=18),
                        observed_effect_values=(('full_power_point', 10),))
    assert can_afford_play(state, payment) is True
    after = simulate_play(state, payment)
    assert after.anomaly.full_power_points == 4
    assert after.anomaly.cumulative_full_power_points == 18
    assert not after.anomaly.full_power_pending
    assert can_afford_play(after, payment) is False


@pytest.mark.parametrize('raw', [None, 0, 8])
def test_missing_or_unreconciled_special_balance_remains_unknown(raw):
    state = BattleState(remaining_turns=3, hp=20, genki=20,
                        anomaly=AnomalyState(full_power_points=5),
                        observed_effect_values=(('full_power_point', raw),))
    payment = card('ExamCostType_ExamFullPowerPoint', 3)
    assert can_afford_play(state, payment) is None
    assert can_afford_play(simulate_play(state, payment), payment) is None
    assert can_afford_play(replace(state, hp=0, genki=0),
                           card('ExamCostType_ExamFullPowerPoint', 3, stamina=1)) is False


def test_zero_and_unsupported_special_costs():
    state = BattleState(remaining_turns=3, hp=20, genki=20)
    assert can_afford_play(state, card('future-cost', 1)) is None
    assert can_afford_play(state, card('future-cost', 0)) is True


@pytest.mark.parametrize('stance', list(AnomalyStance))
@pytest.mark.parametrize('full_power', [False, True])
def test_evaluation_penalty_and_simulation_use_the_same_integer_cost(stance, full_power):
    state = BattleState(remaining_turns=3, hp=30, genki=20,
                        anomaly=AnomalyState(stance=stance, full_power_active=full_power))
    paid = card(stamina=3, force=1)
    green, red = play_cost(state, paid)
    evaluator = EffectHeuristicEvaluator()
    assert evaluator.evaluate(paid, state).score - evaluator.evaluate(card(), state).score == -(green + red)
    after = simulate_play(state, paid)
    assert after.genki == 20 - green
    assert after.hp == 30 - red


def test_repeated_lookahead_keeps_raw_hud_and_uses_current_projected_balance():
    from kaa.tasks.produce.new.play_cards.planner.trigger import (
        ExamTrigger, TriggerVerdict, evaluate_exam_trigger,
    )
    payment = card('ExamCostType_ExamFullPowerPoint', 3)
    state = BattleState(remaining_turns=3, hp=20, genki=20,
                        anomaly=AnomalyState(full_power_points=10),
                        observed_effect_values=(('full_power_point', 10),))
    once = simulate_play(state, payment)
    twice = simulate_play(once, payment)
    assert state.projected_effect_values is None
    assert once.observed_effect_values == twice.observed_effect_values == state.observed_effect_values
    assert once.projected_effect_values is not None
    assert dict(once.projected_effect_values)['full_power_point'] == 7
    assert twice.projected_effect_values is not None
    assert dict(twice.projected_effect_values)['full_power_point'] == 4
    assert can_afford_play(twice, card('ExamCostType_ExamFullPowerPoint', 5)) is False
    condition = ExamTrigger(field_status_types=('ProduceExamFieldStatusType_FullPowerPointUp',),
                            field_status_values=(5,))
    assert evaluate_exam_trigger(condition, once) == TriggerVerdict.ACTIVE
    assert evaluate_exam_trigger(condition, twice) == TriggerVerdict.INACTIVE


@pytest.mark.parametrize('raw', [None, 0, 8])
def test_projection_does_not_promote_missing_or_conflicting_hud(raw):
    state = BattleState(remaining_turns=3, hp=20, genki=20,
                        anomaly=AnomalyState(full_power_points=5),
                        observed_effect_values=(('full_power_point', raw),))
    after = simulate_play(state, card('ExamCostType_ExamFullPowerPoint', 3))
    assert after.observed_effect_values == state.observed_effect_values
    assert after.projected_effect_values is not None
    assert dict(after.projected_effect_values)['full_power_point'] is None
    assert can_afford_play(after, card('ExamCostType_ExamFullPowerPoint', 1)) is None


def test_empty_projection_is_not_replaced_with_old_raw_values():
    state = BattleState(remaining_turns=3, hp=20, genki=20,
                        anomaly=AnomalyState(full_power_points=5),
                        observed_effect_values=(('full_power_point', 5),),
                        projected_effect_values=())
    assert state.effect_values_for_planning == ()
    assert can_afford_play(state, card('ExamCostType_ExamFullPowerPoint', 1)) is None


@pytest.mark.parametrize('raw', [None, 0, 8])
@pytest.mark.parametrize('negated', [False, True])
def test_untrusted_projected_balance_cannot_enable_or_disable_a_condition(raw, negated):
    from kaa.tasks.produce.new.play_cards.planner.trigger import (
        ExamTrigger, TriggerVerdict, evaluate_exam_trigger,
    )
    state = BattleState(remaining_turns=3, hp=20, genki=20,
                        anomaly=AnomalyState(full_power_points=5),
                        observed_effect_values=(('full_power_point', raw),))
    after = simulate_play(state, card('ExamCostType_ExamFullPowerPoint', 3))
    assert after.anomaly.full_power_points == 2
    assert after.projected_effect_values is not None
    assert dict(after.projected_effect_values)['full_power_point'] is None
    trigger = ExamTrigger(
        field_status_types=('ProduceExamFieldStatusType_FullPowerPointUp',),
        field_status_values=(1,),
        field_status_check_types=('ProduceExamTriggerCheckType_Not',) if negated else (),
    )
    assert evaluate_exam_trigger(trigger, after) == TriggerVerdict.UNKNOWN


@pytest.mark.parametrize('effect, status, buffs', [
    ('lesson_buff', 'LessonBuffUp', ExamBuffState(lesson_buff=4)),
    ('review', 'ReviewUp', ExamBuffState(review=4)),
    ('aggressive', 'CardPlayAggressiveUp', ExamBuffState(aggressive=4)),
    ('parameter_buff', 'ParameterBuffUp', ExamBuffState(parameter_buff_turns=4)),
    ('parameter_buff_multiple', 'ParameterBuffMultiplePerTurnUp', ExamBuffState(parameter_buff_multiple_turns=4)),
])
def test_empty_projection_does_not_promote_modeled_buffs_to_numeric_evidence(effect, status, buffs):
    from kaa.tasks.produce.new.play_cards.planner.trigger import (
        ExamTrigger, TriggerVerdict, evaluate_exam_trigger,
    )
    state = BattleState(remaining_turns=3, hp=20, genki=20, buffs=buffs,
                        observed_effect_values=((effect, 4),), projected_effect_values=())
    trigger = ExamTrigger(field_status_types=('ProduceExamFieldStatusType_' + status,),
                          field_status_values=(1,))
    assert evaluate_exam_trigger(trigger, state) == TriggerVerdict.UNKNOWN


def reduction_card(kind='ExamStaminaConsumptionDown', turn=4, stamina=2):
    from kaa.db.constants import ProduceExamEffectType
    result = card(stamina=stamina)
    result.play_effects = [SimpleNamespace(
        _produce_exam_trigger_id='',
        produce_exam_effect=SimpleNamespace(
            effect_type=ProduceExamEffectType('ProduceExamEffectType_' + kind),
            effect_value1=0, effect_value2=0, effect_turn=turn,
        ),
    )]
    return result


def test_reduction_starts_after_payment_and_can_enable_second_card():
    initial = BattleState(remaining_turns=3, hp=0, genki=4,
                          observed_effect_values=(('review', 7),))
    grant = reduction_card()
    after = simulate_play(initial, grant)
    assert after.genki == 2  # Grant itself pays its full two points.
    assert after.projected_stamina_consumption == 'half'
    assert after.observed_effect_values == initial.observed_effect_values
    assert initial.projected_stamina_consumption is None
    assert can_afford_play(replace(initial, genki=2), card(stamina=3)) is False
    assert can_afford_play(after, card(stamina=3)) is True
    assert simulate_play(after, card(stamina=3)).genki == 0
    twice = simulate_play(after, grant)
    assert twice.genki == 1
    assert play_cost(twice, card(stamina=3)) == (2, 0)


@pytest.mark.parametrize('cost, expected', [(0, 0), (1, 1), (2, 1), (3, 2), (4, 2), (5, 3)])
def test_projected_half_cost_rounding_and_evaluation_agree(cost, expected):
    state = simulate_play(BattleState(remaining_turns=3, hp=20, genki=20), reduction_card())
    paid = card(stamina=cost)
    assert play_cost(state, paid) == (expected, 0)
    evaluator = EffectHeuristicEvaluator()
    assert evaluator.evaluate(paid, state).score - evaluator.evaluate(card(), state).score == -expected
    assert simulate_play(state, paid).genki == state.genki - expected


@pytest.mark.parametrize('verdict', ['inactive', 'unknown'])
def test_conditional_reduction_does_not_become_confirmed(monkeypatch, verdict):
    from kaa.tasks.produce.new.play_cards.planner import simulator
    from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *args: TriggerVerdict(verdict))
    state = BattleState(remaining_turns=3, hp=0, genki=4)
    after = simulate_play(state, reduction_card())
    assert after.projected_stamina_consumption == (None if verdict == 'inactive' else 'unknown')
    assert can_afford_play(after, card(stamina=3)) is (False if verdict == 'inactive' else None)


@pytest.mark.parametrize('kind', ['ExamStaminaConsumptionAdd', 'ExamStaminaConsumptionDownFix', 'ExamStaminaConsumptionAddFix', 'ExamSearchPlayCardStaminaConsumptionChange'])
def test_uncalibrated_modifier_invalidates_an_existing_half_projection(kind):
    initial = BattleState(remaining_turns=3, hp=20, genki=20)
    half = simulate_play(initial, reduction_card())
    after = simulate_play(half, reduction_card(kind))
    assert after.projected_stamina_consumption == 'unknown'
    assert can_afford_play(after, card(stamina=1)) is None
    next_state = simulate_play(after, card(stamina=1))
    assert not next_state.hp_known and not next_state.genki_known
    assert can_afford_play(after, card()) is True


@pytest.mark.parametrize('stance, full_power, force', [
    (AnomalyStance.AGGRESSIVE, False, 0),
    (AnomalyStance.PRESERVATION, False, 0),
    (AnomalyStance.NORMAL, True, 0),
    (AnomalyStance.NORMAL, False, 1),
])
def test_uncalibrated_half_combinations_cannot_prove_payment(stance, full_power, force):
    after = simulate_play(BattleState(remaining_turns=3, hp=20, genki=20), reduction_card())
    after = replace(after, anomaly=AnomalyState(stance=stance, full_power_active=full_power))
    assert can_afford_play(after, card(stamina=3, force=force)) is None


def test_raw_state_cannot_use_a_projection_as_a_hud_discount():
    raw = BattleState(remaining_turns=3, hp=0, genki=2,
                      projected_stamina_consumption='half')
    assert play_cost(raw, card(stamina=3)) == (3, 0)
    assert can_afford_play(raw, card(stamina=3)) is False


def resource_effect_card(kind, value=2):
    result = reduction_card(kind, turn=0, stamina=0)
    result.play_effects[0].produce_exam_effect.effect_value1 = value
    result.play_effects[0].produce_exam_effect.effect_count = 1
    return result


def test_unknown_genki_payment_does_not_prove_remaining_hp_budget():
    initial = BattleState(remaining_turns=3, hp=6, genki=20, genki_known=False)
    assert can_afford_play(initial, card(stamina=6)) is True
    after = simulate_play(initial, card(stamina=6))
    assert not after.hp_known and not after.genki_known
    assert can_afford_play(after, card(stamina=6)) is None
    assert initial.hp_known and initial.genki == 20


def test_unknown_genki_does_not_invalidate_hp_after_red_only_payment():
    initial = BattleState(remaining_turns=3, hp=6, genki=20, genki_known=False)
    after = simulate_play(initial, card(force=2))
    assert after.hp_known and after.hp == 4
    assert can_afford_play(after, card(force=5)) is False
    assert can_afford_play(after, card(force=4)) is True


@pytest.mark.parametrize('kind, name, buffs, cost', [
    ('ExamLessonBuffReduce', 'lesson_buff', ExamBuffState(lesson_buff=5), 'ExamLessonBuff'),
    ('ExamReviewReduce', 'review', ExamBuffState(review=5), 'ExamReview'),
    ('ExamAggressiveReduce', 'aggressive', ExamBuffState(aggressive=5), 'ExamCardPlayAggressive'),
])
def test_resource_reduction_precedes_next_special_payment(kind, name, buffs, cost):
    initial = BattleState(remaining_turns=3, hp=20, genki=20, buffs=buffs,
                          observed_effect_values=((name, 5),))
    after = simulate_play(initial, resource_effect_card(kind))
    assert after.projected_effect_values is not None
    assert dict(after.projected_effect_values)[name] == 3
    assert can_afford_play(after, card('ExamCostType_' + cost, 4)) is False
    assert can_afford_play(after, card('ExamCostType_' + cost, 3)) is True
    assert after.observed_effect_values == initial.observed_effect_values


@pytest.mark.parametrize('kind', ['ExamReview', 'ExamReviewReduce'])
@pytest.mark.parametrize('verdict', ['inactive', 'unknown'])
def test_uncertain_resource_changes_do_not_reuse_old_budget(monkeypatch, kind, verdict):
    from kaa.tasks.produce.new.play_cards.planner import simulator, memory
    from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *args: TriggerVerdict(verdict))
    monkeypatch.setattr(memory, 'evaluate_trigger_id', lambda *args: TriggerVerdict(verdict))
    initial = BattleState(remaining_turns=3, hp=20, genki=20,
                          buffs=ExamBuffState(review=5), observed_effect_values=(('review', 5),))
    after = simulate_play(initial, resource_effect_card(kind))
    expected = True if verdict == 'inactive' else None
    assert can_afford_play(after, card('ExamCostType_ExamReview', 4)) is expected
    assert after.observed_effect_values == initial.observed_effect_values


def test_restriction_starts_after_earlier_gain_and_invalidates_later_budget():
    gain = resource_effect_card('ExamBlock', 2)
    restriction = resource_effect_card('ExamBlockRestriction', 0)
    initial = BattleState(remaining_turns=3, hp=0, genki=0)
    gain.play_effects += restriction.play_effects
    after = simulate_play(initial, gain)
    assert after.genki_known and after.genki == 2
    assert after.projected_resource_restrictions == frozenset({'genki_gain'})
    next_state = simulate_play(after, resource_effect_card('ExamBlock', 4))
    assert not next_state.genki_known
    assert can_afford_play(next_state, card(stamina=5)) is None
    assert initial.projected_resource_restrictions == frozenset()


@pytest.mark.parametrize('kind,gain,known', [
    ('ExamBlockRestriction', 'ExamBlock', 'genki_known'),
    ('ExamStaminaRecoverRestriction', 'ExamStaminaRecoverFix', 'hp_known'),
])
def test_unknown_restriction_cannot_certify_later_recovery(monkeypatch, kind, gain, known):
    from kaa.tasks.produce.new.play_cards.planner import simulator
    from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda trigger, state:
                        TriggerVerdict.UNKNOWN if trigger == 'uncertain' else TriggerVerdict.ACTIVE)
    restriction = resource_effect_card(kind, 0)
    restriction.play_effects[0]._produce_exam_trigger_id = 'uncertain'
    initial = BattleState(remaining_turns=3, hp=0, genki=0)
    after = simulate_play(initial, restriction)
    next_state = simulate_play(after, resource_effect_card(gain, 4))
    assert not getattr(next_state, known)
    assert can_afford_play(next_state, card(force=3) if known == 'hp_known' else card(stamina=3)) is None


@pytest.mark.parametrize('green, red', [(4, 0), (0, 2), (4, 2)])
@pytest.mark.parametrize('balance', [0, 30])
def test_ambiguous_current_modifier_does_not_prove_payment_or_future_balance(green, red, balance):
    state = BattleState(remaining_turns=3, hp=balance, genki=balance,
                        observed_effects=frozenset({'stamina_consumption_modifier'}))
    paid = card(stamina=green, force=red)
    assert can_afford_play(state, paid) is None
    after = simulate_play(state, paid)
    assert not after.hp_known and not after.genki_known
    assert after.projected_stamina_consumption == 'unknown'
    assert after.observed_effect_values == state.observed_effect_values
    assert can_afford_play(after, paid) is None


def test_free_grant_does_not_resolve_ambiguous_current_modifier():
    state = BattleState(remaining_turns=3, hp=20, genki=20,
                        observed_effects=frozenset({'stamina_consumption_modifier'}))
    grant = reduction_card(stamina=0)
    assert can_afford_play(state, grant) is True
    after = simulate_play(state, grant)
    assert after.hp_known and after.genki_known
    assert after.projected_stamina_consumption == 'unknown'
    assert can_afford_play(after, card(stamina=4)) is None
    # A fresh observation without the glyph must not inherit old uncertainty.
    assert can_afford_play(BattleState(remaining_turns=2, hp=20, genki=20), card(stamina=4)) is True


def conversion_card(ratio=1000, trigger=''):
    from kaa.db.constants import ProduceExamEffectType
    result = card()
    result.play_effects = [SimpleNamespace(
        _produce_exam_trigger_id=trigger,
        produce_exam_effect=SimpleNamespace(
            effect_type=ProduceExamEffectType.ExamLessonDependBlock,
            effect_value1=3200, effect_value2=ratio, effect_count=1,
            effect_turn=0,
        ),
    )]
    return result


@pytest.mark.parametrize('ratio', [500, 1000])
def test_conversion_cannot_reuse_spent_genki_for_followup_payment_or_output(ratio):
    before = BattleState(remaining_turns=4, hp=0, genki=45, playable_count=2)
    after = simulate_play(before, conversion_card(ratio))
    assert not after.genki_known
    assert can_afford_play(after, card(stamina=30)) is None
    result = EffectHeuristicEvaluator().evaluate(conversion_card(0), after)
    assert result.confirmed_output == 0
    if ratio == 1000:
        assert after.genki == 0
        assert result.projected_output == 0
    else:
        # Half-consumption rounding is uncalibrated: no invented exact value.
        assert after.genki == before.genki
    assert before.genki_known and before.genki == 45


def test_nonconsuming_conversion_preserves_known_genki_budget():
    before = BattleState(remaining_turns=4, hp=0, genki=45, playable_count=2)
    after = simulate_play(before, conversion_card(0))
    assert after.genki == 45 and after.genki_known
    assert can_afford_play(after, card(stamina=30)) is True


@pytest.mark.parametrize('verdict_name', ['INACTIVE', 'UNKNOWN', 'ACTIVE'])
def test_conversion_only_clears_estimate_for_definitely_active_full_consumption(monkeypatch, verdict_name):
    from kaa.tasks.produce.new.play_cards.planner import simulator
    from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
    verdict = TriggerVerdict[verdict_name]
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *_: verdict)
    before = BattleState(remaining_turns=4, hp=20, genki=45)
    after = simulate_play(before, conversion_card(trigger='condition'))
    assert after.genki == (0 if verdict_name == 'ACTIVE' else 45)
    assert after.genki_known == (verdict_name == 'INACTIVE')


def test_later_genki_grant_does_not_restore_consumed_unknown_budget():
    from kaa.db.constants import ProduceExamEffectType
    full = conversion_card()
    full.play_effects.append(SimpleNamespace(
        _produce_exam_trigger_id='',
        produce_exam_effect=SimpleNamespace(
            effect_type=ProduceExamEffectType.ExamBlockFix,
            effect_value1=10, effect_value2=0, effect_turn=0,
        ),
    ))
    after = simulate_play(BattleState(remaining_turns=4, hp=20, genki=45), full)
    assert after.genki == 10 and not after.genki_known


def test_consumption_prevents_same_card_second_conversion_from_claiming_output():
    full = conversion_card()
    full.play_effects.extend(conversion_card(0).play_effects)
    state = BattleState(remaining_turns=4, hp=20, genki=45)
    result = EffectHeuristicEvaluator().evaluate(full, state)
    assert result.projected_output == 144
    assert result.confirmed_output == 144


@pytest.mark.parametrize('stance,amount', [
    (AnomalyStance.PRESERVATION, 1),
    (AnomalyStance.PRESERVATION, 3),
    (AnomalyStance.PRESERVATION_2, 1),
    (AnomalyStance.PRESERVATION_2, 2),
    (AnomalyStance.PRESERVATION_2, 3),
    (AnomalyStance.PRESERVATION_2, 5),
])
@pytest.mark.parametrize('red', [False, True])
def test_fractional_stance_cost_does_not_certify_payment_or_later_balance(stance, amount, red):
    from kaa.tasks.produce.new.play_cards.planner.costs import projected_cost_supported
    state = BattleState(remaining_turns=4, hp=20, genki=20,
                        anomaly=AnomalyState(stance=stance))
    paid = card(force=amount) if red else card(stamina=amount)
    assert not projected_cost_supported(state, paid)
    assert can_afford_play(state, paid) is None
    after = simulate_play(state, paid)
    assert not after.hp_known and not after.genki_known
    assert can_afford_play(after, card(stamina=1, force=1)) is None
    assert state.hp == state.genki == 20


@pytest.mark.parametrize('stance,amount', [
    (AnomalyStance.NORMAL, 1),
    (AnomalyStance.AGGRESSIVE, 1),
    (AnomalyStance.PRESERVATION, 2),
    (AnomalyStance.PRESERVATION_2, 4),
    (AnomalyStance.PRESERVATION_2, 0),
])
def test_integral_stance_cost_keeps_existing_payment_proof(stance, amount):
    state = BattleState(remaining_turns=4, hp=20, genki=20,
                        anomaly=AnomalyState(stance=stance))
    paid = card(stamina=amount, force=amount)
    assert can_afford_play(state, paid) is True
    after = simulate_play(state, paid)
    assert after.hp_known and after.genki_known


@pytest.mark.parametrize('stance', [AnomalyStance.PRESERVATION, AnomalyStance.PRESERVATION_2])
def test_full_power_bypasses_fractional_stance_cost(stance):
    state = BattleState(remaining_turns=4, hp=20, genki=20,
                        anomaly=AnomalyState(stance=stance, full_power_active=True))
    paid = card(stamina=1, force=1)
    assert play_cost(state, paid) == (1, 1)
    assert can_afford_play(state, paid) is True
    after = simulate_play(state, paid)
    assert after.hp_known and after.genki_known


@pytest.mark.parametrize('stance', [AnomalyStance.PRESERVATION, AnomalyStance.PRESERVATION_2])
def test_unfunded_fractional_stance_cost_is_not_certified_free(stance):
    state = BattleState(remaining_turns=4, hp=0, genki=0,
                        anomaly=AnomalyState(stance=stance))
    assert play_cost(state, card(stamina=1)) == (0, 0)
    assert can_afford_play(state, card(stamina=1)) is None
    assert can_afford_play(state, card()) is True


@pytest.mark.parametrize('kind', ['ExamConcentration', 'ExamPreservation', 'ExamStanceReset'])
def test_unknown_stance_switch_cannot_prove_followup_cost_or_output(monkeypatch, kind):
    from kaa.tasks.produce.new.play_cards.planner import simulator, memory
    from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *_: TriggerVerdict.UNKNOWN)
    monkeypatch.setattr(memory, 'evaluate_trigger_id', lambda *_: TriggerVerdict.UNKNOWN)
    state = BattleState(remaining_turns=4, hp=20, genki=20)
    future = simulate_play(state, resource_effect_card(kind, 1))
    assert future.projected_stance_uncertain
    assert future.hp_known and future.genki_known
    assert can_afford_play(future, card(stamina=2)) is None
    assert can_afford_play(future, card()) is True
    output = resource_effect_card('ExamLesson', 10)
    result = EffectHeuristicEvaluator().evaluate(output, future)
    assert result.projected_output == 10 and result.confirmed_output == 0
    assert not state.projected_stance_uncertain


@pytest.mark.parametrize('kind', ['ExamConcentration', 'ExamPreservation', 'ExamStanceReset'])
@pytest.mark.parametrize('verdict,locked,full', [
    ('inactive', False, False), ('unknown', True, False), ('unknown', False, True),
])
def test_ineffective_unknown_stance_switch_keeps_payment_proof(monkeypatch, kind, verdict, locked, full):
    from kaa.tasks.produce.new.play_cards.planner import simulator
    from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *_: TriggerVerdict(verdict))
    state = BattleState(remaining_turns=4, hp=20, genki=20,
                        anomaly=AnomalyState(stance_locked=locked, full_power_active=full))
    future = simulate_play(state, resource_effect_card(kind, 1))
    assert not future.projected_stance_uncertain
    assert can_afford_play(future, card(stamina=2)) is True


@pytest.mark.parametrize('kind', ['ExamConcentration', 'ExamPreservation', 'ExamStanceReset'])
def test_known_later_switch_does_not_resolve_unknown_stance_history(kind):
    state = BattleState(remaining_turns=4, hp=20, genki=20, projected_stance_uncertain=True)
    future = simulate_play(state, resource_effect_card(kind, 1))
    assert future.projected_stance_uncertain
    assert can_afford_play(future, card(stamina=2)) is None
    assert EffectHeuristicEvaluator().evaluate(resource_effect_card('ExamLesson', 10), future).confirmed_output == 0


def test_full_power_output_and_cost_bypass_uncertain_stance_estimate():
    state = BattleState(remaining_turns=4, hp=20, genki=20, projected_stance_uncertain=True,
                        anomaly=AnomalyState(full_power_active=True))
    assert can_afford_play(state, card(stamina=2)) is True
    result = EffectHeuristicEvaluator().evaluate(resource_effect_card('ExamLesson', 10), state)
    assert result.projected_output == 30 and result.confirmed_output == 0
    conversion = resource_effect_card('ExamLessonDependBlock', 1000)
    independent = EffectHeuristicEvaluator().evaluate(conversion, state)
    assert independent.projected_output == independent.confirmed_output == 60


@pytest.mark.parametrize('kind,value,expected', [
    ('ExamLesson', 10, 45),
    ('ExamLessonFix', 10, 45),
    ('ExamLessonFullPowerPoint', 10, 45),
    ('ExamMultipleLessonBuffLesson', 10, 45),
    ('ExamMultipleEnthusiasticLesson', 1000, 15),
])
def test_full_power_does_not_certify_enthusiasm_after_unknown_stance_history(kind, value, expected):
    state = BattleState(remaining_turns=4, hp=20, genki=20, projected_stance_uncertain=True,
                        anomaly=AnomalyState(full_power_active=True, enthusiasm=5))
    result = EffectHeuristicEvaluator().evaluate(resource_effect_card(kind, value), state)
    assert result.projected_output == expected
    assert result.confirmed_output == 0


@pytest.mark.parametrize('kind', ['ExamConcentration', 'ExamStanceReset'])
@pytest.mark.parametrize('stance', [AnomalyStance.PRESERVATION, AnomalyStance.PRESERVATION_2])
def test_unknown_preservation_history_does_not_grant_a_search_play(kind, stance):
    state = BattleState(remaining_turns=4, hp=20, genki=20, playable_count=1,
                        projected_stance_uncertain=True, anomaly=AnomalyState(stance=stance))
    future = simulate_play(state, resource_effect_card(kind, 1))
    assert future.anomaly.extra_play_count == state.anomaly.extra_play_count + 1
    assert future.playable_count == 0
    assert state.playable_count == 1


@pytest.mark.parametrize('kind,expected', [('ExamPlayableValueAdd', 2), ('ExamFullPower', 1)])
def test_unknown_stance_preserves_independent_added_search_plays(kind, expected):
    state = BattleState(remaining_turns=4, hp=20, genki=20, playable_count=1,
                        projected_stance_uncertain=True, anomaly=AnomalyState(stance=AnomalyStance.PRESERVATION))
    grant = resource_effect_card(kind, 2)
    switch = resource_effect_card('ExamConcentration', 1)
    grant.play_effects = switch.play_effects + grant.play_effects
    future = simulate_play(state, grant)
    assert future.playable_count == expected


def test_known_preservation_exit_keeps_search_play():
    state = BattleState(remaining_turns=4, hp=20, genki=20, playable_count=1,
                        anomaly=AnomalyState(stance=AnomalyStance.PRESERVATION))
    assert simulate_play(state, resource_effect_card('ExamConcentration', 1)).playable_count == 1


def test_unknown_stance_does_not_create_a_second_full_power_entry_play():
    state = BattleState(remaining_turns=4, hp=20, genki=20, playable_count=1,
                        projected_stance_uncertain=True, anomaly=AnomalyState(full_power_active=True))
    assert simulate_play(state, resource_effect_card('ExamFullPower', 1)).playable_count == 0


def test_unknown_switch_before_known_exit_in_same_card_does_not_grant_play(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner import simulator, memory
    from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
    def verdict(trigger_id, _state):
        return TriggerVerdict.UNKNOWN if trigger_id == 'maybe' else TriggerVerdict.ACTIVE
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', verdict)
    monkeypatch.setattr(memory, 'evaluate_trigger_id', verdict)
    first = resource_effect_card('ExamPreservation', 1)
    first.play_effects[0]._produce_exam_trigger_id = 'maybe'
    first.play_effects += resource_effect_card('ExamConcentration', 1).play_effects
    state = BattleState(remaining_turns=4, hp=20, genki=20, playable_count=1,
                        anomaly=AnomalyState(stance=AnomalyStance.PRESERVATION))
    future = simulate_play(state, first)
    assert future.projected_stance_uncertain
    assert future.playable_count == 0


@pytest.mark.parametrize('switch', ['ExamConcentration', 'ExamPreservation', 'ExamStanceReset'])
@pytest.mark.parametrize('blocker', ['StanceLock', 'ExamFullPower'])
@pytest.mark.parametrize('block_first', [False, True])
def test_known_stance_blocker_only_blocks_later_unknown_switch(monkeypatch, switch, blocker, block_first):
    from kaa.tasks.produce.new.play_cards.planner import simulator, memory
    from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
    def verdict(trigger_id, _state):
        return TriggerVerdict.UNKNOWN if trigger_id == 'maybe' else TriggerVerdict.ACTIVE
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', verdict)
    monkeypatch.setattr(memory, 'evaluate_trigger_id', verdict)
    switch_card = resource_effect_card(switch, 1)
    switch_card.play_effects[0]._produce_exam_trigger_id = 'maybe'
    block = resource_effect_card(blocker, 1)
    block.play_effects = (block.play_effects + switch_card.play_effects if block_first
                         else switch_card.play_effects + block.play_effects)
    state = BattleState(remaining_turns=4, hp=20, genki=20)
    future = simulate_play(state, block)
    assert future.projected_stance_uncertain is not block_first
    if blocker == 'StanceLock':
        assert future.anomaly.stance_locked
    else:
        assert future.anomaly.full_power_active


@pytest.mark.parametrize('blocker', ['StanceLock', 'ExamFullPower'])
@pytest.mark.parametrize('verdict_name', ['inactive', 'unknown'])
def test_unconfirmed_blocker_does_not_hide_later_unknown_switch(monkeypatch, blocker, verdict_name):
    from kaa.tasks.produce.new.play_cards.planner import simulator, memory
    from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
    def verdict(trigger_id, _state):
        return TriggerVerdict(verdict_name) if trigger_id == 'block' else TriggerVerdict.UNKNOWN
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', verdict)
    monkeypatch.setattr(memory, 'evaluate_trigger_id', verdict)
    block = resource_effect_card(blocker, 1)
    block.play_effects[0]._produce_exam_trigger_id = 'block'
    block.play_effects += resource_effect_card('ExamConcentration', 1).play_effects
    future = simulate_play(BattleState(remaining_turns=4, hp=20, genki=20), block)
    assert future.projected_stance_uncertain


@pytest.mark.parametrize('blocker', ['StanceLock', 'ExamFullPower'])
def test_known_blocker_does_not_restore_prior_unknown_stance_history(blocker):
    state = BattleState(remaining_turns=4, hp=20, genki=20, projected_stance_uncertain=True)
    future = simulate_play(state, resource_effect_card(blocker, 1))
    assert future.projected_stance_uncertain


@pytest.mark.parametrize('switch', ['ExamConcentration', 'ExamStanceReset'])
@pytest.mark.parametrize('same_card', [False, True])
def test_possible_stance_lock_invalidates_a_later_confirmed_switch(monkeypatch, switch, same_card):
    from kaa.tasks.produce.new.play_cards.planner import simulator, memory
    from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
    def verdict(trigger_id, _state):
        return TriggerVerdict.UNKNOWN if trigger_id == 'maybe-lock' else TriggerVerdict.ACTIVE
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', verdict)
    monkeypatch.setattr(memory, 'evaluate_trigger_id', verdict)
    lock = resource_effect_card('StanceLock', 1)
    lock.play_effects[0]._produce_exam_trigger_id = 'maybe-lock'
    switch_card = resource_effect_card(switch, 1)
    state = BattleState(remaining_turns=4, hp=20, genki=20, playable_count=2,
                        anomaly=AnomalyState(stance=AnomalyStance.PRESERVATION))
    if same_card:
        lock.play_effects += switch_card.play_effects
        future = simulate_play(state, lock)
        assert future.playable_count == 1
    else:
        after_lock = simulate_play(state, lock)
        assert after_lock.projected_stance_lock_uncertain
        assert not after_lock.projected_stance_uncertain
        assert can_afford_play(after_lock, card(stamina=2)) is True
        future = simulate_play(after_lock, switch_card)
        assert future.playable_count == 0
    assert future.projected_stance_uncertain
    assert future.projected_stance_lock_uncertain
    assert can_afford_play(future, card(stamina=2)) is None


def test_confirmed_lock_resolves_possible_lock_without_changing_stance_certainty():
    state = BattleState(remaining_turns=4, hp=20, genki=20, projected_stance_lock_uncertain=True)
    future = simulate_play(state, resource_effect_card('StanceLock', 1))
    assert future.anomaly.stance_locked
    assert not future.projected_stance_lock_uncertain
    assert not future.projected_stance_uncertain
    after_switch = simulate_play(future, resource_effect_card('ExamPreservation', 1))
    assert not after_switch.projected_stance_uncertain
    assert can_afford_play(after_switch, card(stamina=1)) is True


def test_existing_confirmed_lock_ignores_another_unknown_lock(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner import simulator
    from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *_: TriggerVerdict.UNKNOWN)
    state = BattleState(remaining_turns=4, hp=20, genki=20, anomaly=AnomalyState(stance_locked=True))
    future = simulate_play(state, resource_effect_card('StanceLock', 1))
    assert not future.projected_stance_lock_uncertain
    assert not future.projected_stance_uncertain


@pytest.mark.parametrize('stance', [AnomalyStance.NORMAL, AnomalyStance.PRESERVATION, AnomalyStance.AGGRESSIVE])
def test_unknown_full_power_entry_invalidates_followup_proofs(monkeypatch, stance):
    from kaa.tasks.produce.new.play_cards.planner import simulator, memory
    from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *_: TriggerVerdict.UNKNOWN)
    monkeypatch.setattr(memory, 'evaluate_trigger_id', lambda *_: TriggerVerdict.UNKNOWN)
    state = BattleState(remaining_turns=4, hp=20, genki=20, playable_count=2,
                        anomaly=AnomalyState(stance=stance))
    future = simulate_play(state, resource_effect_card('ExamFullPower', 1))
    assert future.projected_full_power_uncertain and future.projected_stance_uncertain
    assert not future.anomaly.full_power_active and future.playable_count == 1
    assert can_afford_play(future, card(stamina=2)) is None
    assert can_afford_play(future, card()) is True
    assert EffectHeuristicEvaluator().evaluate(resource_effect_card('ExamLessonDependBlock', 1000), future).confirmed_output == 0


def test_known_entry_resolves_possible_full_power_without_regranting_entry_play():
    state = BattleState(remaining_turns=4, hp=20, genki=20, playable_count=1,
                        projected_full_power_uncertain=True, projected_stance_uncertain=True)
    future = simulate_play(state, resource_effect_card('ExamFullPower', 1))
    assert future.anomaly.full_power_active and not future.projected_full_power_uncertain
    assert future.projected_stance_uncertain and future.playable_count == 0
    assert can_afford_play(future, card(stamina=2)) is True
    assert EffectHeuristicEvaluator().evaluate(resource_effect_card('ExamLessonDependBlock', 1000), future).confirmed_output == 60


def test_direct_added_plays_survive_possible_prior_full_power_entry():
    state = BattleState(remaining_turns=4, hp=20, genki=20, playable_count=1,
                        projected_full_power_uncertain=True, projected_stance_uncertain=True)
    paid = resource_effect_card('ExamFullPower', 1)
    paid.play_effects += resource_effect_card('ExamPlayableValueAdd', 2).play_effects
    assert simulate_play(state, paid).playable_count == 2


def test_same_card_unknown_then_known_full_power_keeps_single_entry_play(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner import simulator, memory
    from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
    def verdict(trigger_id, _state):
        return TriggerVerdict.UNKNOWN if trigger_id == 'maybe' else TriggerVerdict.ACTIVE
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', verdict)
    monkeypatch.setattr(memory, 'evaluate_trigger_id', verdict)
    paid = resource_effect_card('ExamFullPower', 1)
    paid.play_effects[0]._produce_exam_trigger_id = 'maybe'
    paid.play_effects += resource_effect_card('ExamFullPower', 1).play_effects
    future = simulate_play(BattleState(remaining_turns=4, hp=20, genki=20, playable_count=1), paid)
    assert future.anomaly.full_power_active and not future.projected_full_power_uncertain
    assert future.playable_count == 1


def test_unknown_reentry_cannot_invalidate_existing_confirmed_full_power(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner import simulator
    from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *_: TriggerVerdict.UNKNOWN)
    state = BattleState(remaining_turns=4, hp=20, genki=20, anomaly=AnomalyState(full_power_active=True))
    future = simulate_play(state, resource_effect_card('ExamFullPower', 1))
    assert not future.projected_full_power_uncertain and not future.projected_stance_uncertain
    assert can_afford_play(future, card(stamina=1)) is True
