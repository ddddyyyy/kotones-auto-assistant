from types import SimpleNamespace
from typing import Any, cast
from dataclasses import replace

import pytest

from kaa.db.card_growth import CardGrowthRule
from kaa.db.constants import ProduceExamEffectType
from kaa.tasks.produce.new.play_cards.planner import costs, simulator
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from kaa.tasks.produce.new.play_cards.planner.simulator import can_afford_play, simulate_play
from kaa.tasks.produce.new.play_cards.planner.evaluator import EffectHeuristicEvaluator
from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict


def _card(category='ProduceCardCategory_ActiveSkill', green=4, red=0, effects=()):
    return cast(Any, SimpleNamespace(_id='same-id', category=category, stamina=green,
        force_stamina=red, cost_type=None, cost_value=0, play_effects=list(effects),
        plan_type=None, evaluation=0))


def _state(hp=4, growth=(('growth', True),)):
    return BattleState(remaining_turns=3, hp=hp, genki=0, playable_count=2,
                       projected_effect_values=(), projected_card_growth=growth)


def _rule(**kwargs):
    return CardGrowthRule(categories=('ProduceCardCategory_ActiveSkill',),
                          target_known=True, costs_known=True, **kwargs)


def test_growth_increase_prevents_original_cost_affordability_proof(monkeypatch):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(green_changes=(1,)))
    card = _card()
    state = _state()
    assert can_afford_play(replace(state, projected_card_growth=()), card) is True
    assert costs.play_cost(state, card) == (5, 0)
    assert can_afford_play(state, card) is False
    after = simulate_play(_state(hp=10), card)
    assert after.hp == 5 and after.hp_known
    assert card.stamina == 4 and state.hp == 4


def test_growth_does_not_affect_other_categories_or_card_id_copies(monkeypatch):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(green_changes=(1,)))
    active, mental = _card(), _card('ProduceCardCategory_MentalSkill')
    assert active._id == mental._id
    assert costs.play_cost(_state(), active) == (5, 0)
    assert costs.play_cost(_state(), mental) == (4, 0)
    assert can_afford_play(_state(), mental) is True


@pytest.mark.parametrize('rule,active', [
    (CardGrowthRule(), True),
    (CardGrowthRule(target_known=False, costs_known=True, green_changes=(1,)), True),
    (CardGrowthRule(target_known=True, costs_known=False), True),
    (_rule(green_changes=(1,)), False),
])
def test_unknown_target_trigger_or_cost_never_proves_payment(monkeypatch, rule, active):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: rule)
    state = _state(hp=100, growth=(('growth', active),))
    assert can_afford_play(state, _card(green=0)) is None
    after = simulate_play(state, _card())
    assert not after.hp_known and not after.genki_known


def test_sequential_growth_reductions_clamp_before_later_increase(monkeypatch):
    rules = {'reduce': _rule(green_changes=(-10,)), 'increase': _rule(green_changes=(3,), red_changes=(2,))}
    monkeypatch.setattr(costs, 'get_card_growth_rule', rules.__getitem__)
    state = _state(hp=10, growth=(('reduce', True), ('increase', True)))
    assert costs.play_cost(state, _card()) == (3, 2)
    assert simulate_play(state, _card()).hp == 5


def test_growth_plus_temporary_half_consumption_remains_unproven(monkeypatch):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(green_changes=(1,)))
    state = replace(_state(hp=100), projected_stamina_consumption='half')
    assert can_afford_play(state, _card()) is None
    assert not costs.projected_cost_supported(state, _card())


def test_growth_output_not_yet_modeled_is_not_confirmed(monkeypatch):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(green_changes=(1,), output_changed=True))
    effect = SimpleNamespace(produce_exam_effect=SimpleNamespace(
        effect_type=ProduceExamEffectType.ExamLesson, effect_value1=10,
        effect_value2=0, effect_count=1, effect_turn=0))
    card = _card(effects=(effect,))
    result = EffectHeuristicEvaluator().evaluate(card, _state(hp=100))
    assert result.projected_output == 10 and result.confirmed_output == 0


@pytest.mark.parametrize('verdict,expected', [
    (TriggerVerdict.ACTIVE, (('grant', True),)),
    (TriggerVerdict.UNKNOWN, (('grant', False),)),
    (TriggerVerdict.INACTIVE, ()),
])
def test_growth_is_projected_only_after_the_granting_card_pays(monkeypatch, verdict, expected):
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *_: verdict)
    grant = SimpleNamespace(produce_exam_effect=SimpleNamespace(_id='grant',
        effect_type=ProduceExamEffectType.ExamAddGrowEffect, effect_value1=0))
    before = _state(hp=10, growth=())
    card = _card(green=4, effects=(grant,))
    after = simulate_play(before, card)
    assert after.hp == 6 and after.projected_card_growth == expected
    assert not before.projected_card_growth



def test_two_step_search_does_not_reward_followup_made_unaffordable_by_growth(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner.strategy import PlannerStrategy
    from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluation, EvaluationConfidence
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(green_changes=(1,)))
    grant = SimpleNamespace(produce_exam_effect=SimpleNamespace(_id='growth',
        effect_type=ProduceExamEffectType.ExamAddGrowEffect, effect_value1=0))
    first = _card('ProduceCardCategory_MentalSkill', green=0, effects=(grant,))
    first._id = 'first'
    second = _card()
    second._id = 'second'
    def evaluation(card, state):
        return CardEvaluation(card._id, 10 if card._id == 'first' else 100,
                              EvaluationConfidence.LOW, 'rule test')
    planner = PlannerStrategy(cast(Any, SimpleNamespace(evaluate=evaluation)))
    first_hand, second_hand = SimpleNamespace(card=first), SimpleNamespace(card=second)
    state = _state(growth=())
    ranked = planner._add_shallow_lookahead(cast(Any, [
        (first_hand, evaluation(first, state)), (second_hand, evaluation(second, state)),
    ]), state)
    assert ranked[0][1].score == 10
    assert '两步预估后续' not in ranked[0][1].reason
