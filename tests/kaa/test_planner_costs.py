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
