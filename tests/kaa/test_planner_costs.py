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
    assert dict(after.observed_effect_values)[effect] == 1
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
