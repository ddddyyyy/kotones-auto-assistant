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



def _output(kind=ProduceExamEffectType.ExamLesson, value=10, count=1):
    return SimpleNamespace(produce_exam_effect=SimpleNamespace(
        effect_type=kind, effect_value1=value, effect_value2=0,
        effect_count=count, effect_turn=0))


@pytest.mark.parametrize('changes,count,expected', [
    ((5,), 1, 15), ((5,), 2, 30), ((-15,), 1, 0), ((-20, 3), 1, 3),
])
def test_pre_card_lesson_growth_changes_each_hit_without_mutating_db_card(monkeypatch, changes, count, expected):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(
        lesson_changes=changes, output_changed=True, output_known=True))
    effect = _output(count=count)
    card = _card(green=0, effects=(effect,))
    result = EffectHeuristicEvaluator().evaluate(card, _state(hp=100))
    assert result.projected_output == result.confirmed_output == expected
    assert effect.produce_exam_effect.effect_value1 == 10
    assert card.play_effects == [effect]


def test_lesson_growth_is_applied_before_enthusiasm_and_each_hit_multiplier(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner.state import AnomalyState, AnomalyStance, ExamBuffState
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(
        lesson_changes=(5,), output_known=True))
    state = replace(_state(hp=100), anomaly=AnomalyState(stance=AnomalyStance.AGGRESSIVE, enthusiasm=2),
                    buffs=ExamBuffState(parameter_buff_turns=2))
    result = EffectHeuristicEvaluator().evaluate(_card(green=0, effects=(_output(count=2),)), state)
    assert result.projected_output == result.confirmed_output == 102


@pytest.mark.parametrize('rule,active', [
    (_rule(lesson_changes=(5,), output_known=True), False),
    (CardGrowthRule(lesson_changes=(5,), output_known=True), True),
    (_rule(lesson_changes=(5,), output_known=False), True),
])
def test_unknown_growth_does_not_add_or_confirm_lesson_points(monkeypatch, rule, active):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: rule)
    result = EffectHeuristicEvaluator().evaluate(_card(green=0, effects=(_output(),)),
        _state(hp=100, growth=(('growth', active),)))
    assert result.projected_output == 10 and result.confirmed_output == 0


def test_other_category_does_not_receive_growth_or_lose_confirmed_output(monkeypatch):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(
        lesson_changes=(5,), output_known=True))
    result = EffectHeuristicEvaluator().evaluate(
        _card('ProduceCardCategory_MentalSkill', green=0, effects=(_output(),)), _state(hp=100))
    assert result.projected_output == result.confirmed_output == 10


def test_lesson_growth_is_not_assumed_to_change_fixed_lesson(monkeypatch):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(
        lesson_changes=(5,), output_known=True))
    result = EffectHeuristicEvaluator().evaluate(
        _card(green=0, effects=(_output(ProduceExamEffectType.ExamLessonFix),)), _state(hp=100))
    assert result.projected_output == 10 and result.confirmed_output == 0


def test_growth_granted_inside_current_card_keeps_timing_unknown(monkeypatch):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(
        lesson_changes=(5,), output_known=True))
    grant = SimpleNamespace(produce_exam_effect=SimpleNamespace(_id='growth',
        effect_type=ProduceExamEffectType.ExamAddGrowEffect, effect_value1=0,
        effect_value2=0, effect_count=1, effect_turn=0))
    result = EffectHeuristicEvaluator().evaluate(_card(green=0, effects=(_output(), grant, _output())),
        _state(hp=100, growth=()))
    assert result.projected_output == 20 and result.confirmed_output == 10


def test_growth_gain_and_extra_fee_are_evaluated_together(monkeypatch):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(
        green_changes=(1,), lesson_changes=(5,), output_known=True))
    card = _card(green=4, effects=(_output(),))
    state = _state(hp=100)
    result = EffectHeuristicEvaluator().evaluate(card, state)
    assert costs.play_cost(state, card) == (5, 0)
    assert result.projected_output == result.confirmed_output == 15
    assert simulate_play(state, card).hp == 95



@pytest.mark.parametrize('groups,fee,output,confirmed,affordable', [
    ('["strong"]', 11, 47, 47, False),
    ('["other"]', 4, 10, 10, True),
    (None, 4, 10, 0, None),
])
def test_effect_group_filter_controls_growth_cost_and_output_together(
        monkeypatch, groups, fee, output, confirmed, affordable):
    rule = _rule(green_changes=(7,), lesson_changes=(37,), output_known=True,
                 effect_groups=('strong',))
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: rule)
    card = _card(green=4, effects=(_output(),))
    card._effect_group_ids = groups
    state = _state(hp=4)
    result = EffectHeuristicEvaluator().evaluate(card, state)
    assert costs.play_cost(state, card) == (fee, 0)
    assert result.projected_output == output and result.confirmed_output == confirmed
    assert can_afford_play(state, card) is affordable
    assert card.stamina == 4 and card.play_effects[0].produce_exam_effect.effect_value1 == 10



@pytest.mark.parametrize('base_count,additions,expected', [(1, (1,), 20), (2, (1,), 30), (1, (1, 2), 40)])
def test_lesson_count_growth_adds_hits_to_existing_card_count(monkeypatch, base_count, additions, expected):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(
        lesson_count_changes=additions, output_known=True))
    effect = _output(count=base_count)
    result = EffectHeuristicEvaluator().evaluate(_card(green=0, effects=(effect,)), _state(hp=100))
    assert result.projected_output == result.confirmed_output == expected
    assert effect.produce_exam_effect.effect_count == base_count


def test_lesson_count_growth_repeats_enthusiasm_and_multiplier_for_added_hits(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner.state import AnomalyState, AnomalyStance
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(
        lesson_changes=(5,), lesson_count_changes=(1,), output_known=True))
    state = replace(_state(hp=100), anomaly=AnomalyState(enthusiasm=2, stance=AnomalyStance.AGGRESSIVE))
    result = EffectHeuristicEvaluator().evaluate(_card(green=0, effects=(_output(),)), state)
    assert result.projected_output == result.confirmed_output == 68


@pytest.mark.parametrize('candidate,expected', [('same-id', 20), ('different-id', 10)])
def test_named_growth_and_cost_apply_only_to_matching_card_type(monkeypatch, candidate, expected):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(
        card_ids=('same-id',), green_changes=(3,), lesson_count_changes=(1,), output_known=True))
    card = _card(green=4, effects=(_output(),))
    card._id = candidate
    result = EffectHeuristicEvaluator().evaluate(card, _state(hp=100))
    assert result.projected_output == result.confirmed_output == expected
    assert costs.play_cost(_state(hp=100), card) == ((7 if candidate == 'same-id' else 4), 0)



def test_block_growth_feeds_later_payment_without_changing_observation_or_card(monkeypatch):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(
        block_changes=(5,), output_known=True))
    state = replace(_state(hp=1), observed_genki=0)
    card = _card(green=0, effects=(_output(ProduceExamEffectType.ExamBlock, 3),))
    after = simulate_play(state, card)
    assert after.genki == 8 and after.genki_known and after.observed_genki == 0
    assert can_afford_play(after, _card(green=9)) is True
    assert can_afford_play(replace(after, genki=3), _card(green=9)) is False
    assert state.genki == 0 and card.play_effects[0].produce_exam_effect.effect_value1 == 3


def test_block_growth_changes_survival_score_and_same_card_conversion(monkeypatch):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(
        block_changes=(5,), output_known=True))
    card = _card(green=0, effects=(
        _output(ProduceExamEffectType.ExamBlock, 3),
        _output(ProduceExamEffectType.ExamLessonDependBlock, 1000),
    ))
    state = _state(hp=100)
    grown = EffectHeuristicEvaluator().evaluate(card, state)
    base = EffectHeuristicEvaluator().evaluate(card, replace(state, projected_card_growth=()))
    assert grown.projected_output == grown.confirmed_output == 8
    assert base.projected_output == base.confirmed_output == 3
    assert grown.score > base.score


@pytest.mark.parametrize('active,restriction,known', [
    (True, False, True), (False, False, False), (True, True, False),
])
def test_uncertain_or_restricted_block_growth_cannot_prove_followup_payment(
    monkeypatch, active, restriction, known,
):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(
        block_changes=(5,), output_known=True))
    state = replace(_state(hp=1, growth=(('growth', active),)),
        projected_resource_restrictions=frozenset({'genki_gain'}) if restriction else frozenset())
    after = simulate_play(state, _card(green=0, effects=(_output(ProduceExamEffectType.ExamBlock, 3),)))
    assert after.genki_known is known
    assert can_afford_play(after, _card(green=9)) is (True if known else None)


def test_in_card_block_growth_timing_and_fixed_block_remain_unknown(monkeypatch):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(
        block_changes=(5,), output_known=True))
    grant = _output(ProduceExamEffectType.ExamAddGrowEffect, 0)
    grant.produce_exam_effect._id = 'growth'
    after = simulate_play(_state(hp=1, growth=()), _card(green=0, effects=(
        grant, _output(ProduceExamEffectType.ExamBlock, 3))))
    assert after.genki == 3 and not after.genki_known
    fixed = simulate_play(_state(hp=1), _card(green=0, effects=(
        _output(ProduceExamEffectType.ExamBlockFix, 3),)))
    assert fixed.genki == 3 and not fixed.genki_known


def test_unknown_block_trigger_invalidates_old_balance(monkeypatch):
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *_: TriggerVerdict.UNKNOWN)
    after = simulate_play(replace(_state(hp=1, growth=()), genki=3), _card(green=0, effects=(
        _output(ProduceExamEffectType.ExamBlock, 3),)))
    assert after.genki == 3 and not after.genki_known


@pytest.mark.parametrize('kind,field,name', [
    (ProduceExamEffectType.ExamReview, 'review_changes', 'review'),
    (ProduceExamEffectType.ExamFullPowerPoint, 'full_power_changes', 'full_power_point'),
])
def test_resource_growth_updates_search_balance_without_modifying_card_or_hud(monkeypatch, kind, field, name):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(**{field: (2,)}, output_known=True))
    state = replace(_state(hp=100), projected_effect_values=((name, 0),), observed_effect_values=((name, 0),))
    effect = _output(kind, 3)
    after = simulate_play(state, _card(green=0, effects=(effect,)))
    assert dict(after.effect_values_for_planning)[name] == 5
    assert dict(after.observed_effect_values)[name] == 0
    assert effect.produce_exam_effect.effect_value1 == 3
    if name == 'full_power_point':
        assert after.anomaly.full_power_points == after.anomaly.cumulative_full_power_points == 5
    else:
        assert after.buffs.review == 5


def test_review_growth_is_applied_before_later_reduction_and_cost_is_paid_first(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(review_changes=(2,), output_known=True))
    state = replace(_state(hp=100), buffs=ExamBuffState(review=2), projected_effect_values=(('review', 2),))
    card = _card(green=0, effects=(
        _output(ProduceExamEffectType.ExamReview, 3),
        _output(ProduceExamEffectType.ExamReviewReduce, 6),
    ))
    card.cost_type = 'ExamCostType_ExamReview'
    card.cost_value = 2
    after = simulate_play(state, card)
    assert after.buffs.review == 0 and dict(after.effect_values_for_planning)['review'] == 0


@pytest.mark.parametrize('active', [True, False])
def test_review_growth_known_and_unknown_propagate_to_same_card_conversion(monkeypatch, active):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(review_changes=(2,), output_known=True))
    state = replace(_state(hp=100, growth=(('growth', active),)), projected_effect_values=(('review', 0),))
    card = _card(green=0, effects=(
        _output(ProduceExamEffectType.ExamReview, 3),
        _output(ProduceExamEffectType.ExamLessonDependExamReview, 1000),
    ))
    result = EffectHeuristicEvaluator().evaluate(card, state)
    assert result.projected_output == (5 if active else 3)
    assert result.confirmed_output == (5 if active else 0)
    after = simulate_play(state, card)
    assert dict(after.effect_values_for_planning)['review'] == (5 if active else None)


def test_in_card_review_growth_timing_remains_unknown(monkeypatch):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: _rule(review_changes=(2,), output_known=True))
    grant = _output(ProduceExamEffectType.ExamAddGrowEffect, 0)
    grant.produce_exam_effect._id = 'growth'
    state = replace(_state(hp=100, growth=()), projected_effect_values=(('review', 0),))
    after = simulate_play(state, _card(green=0, effects=(grant, _output(ProduceExamEffectType.ExamReview, 3))))
    assert after.buffs.review == 3 and dict(after.effect_values_for_planning)['review'] is None


def test_inactive_resource_effect_does_not_invalidate_known_balance(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner import memory
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *_: TriggerVerdict.INACTIVE)
    monkeypatch.setattr(memory, 'evaluate_trigger_id', lambda *_: TriggerVerdict.INACTIVE)
    state = replace(_state(hp=100), projected_effect_values=(('review', 0),))
    after = simulate_play(state, _card(green=0, effects=(_output(ProduceExamEffectType.ExamReview, 3),)))
    assert dict(after.effect_values_for_planning)['review'] == 0


@pytest.mark.parametrize('cost_changed,output_changed', [(True, True), (False, True), (True, False)])
def test_intrinsic_enchant_does_not_prove_base_payment_or_output(monkeypatch, cost_changed, output_changed):
    from kaa.db import card_status_enchant as db
    monkeypatch.setattr(db, 'get_card_status_enchant', lambda key: SimpleNamespace(
        cost_changed=cost_changed, output_changed=output_changed))
    card = _card(green=4, effects=(_output(),))
    card._produce_card_status_enchant_id = 'enchant'
    state = _state(hp=100, growth=())
    assert can_afford_play(state, card) is (None if cost_changed else True)
    result = EffectHeuristicEvaluator().evaluate(card, state)
    assert result.projected_output == 10
    assert result.confirmed_output == (0 if output_changed else 10)
    assert simulate_play(state, card).hp_known is (not cost_changed)


def test_intrinsic_enchant_diagnostic_does_not_invent_remaining_count(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner import decision_trace
    from kaa.db.card_status_enchant import CardStatusEnchant
    monkeypatch.setattr(decision_trace, 'get_card_status_enchant', lambda key: CardStatusEnchant(
        key, 'trigger', 2, ('growth',), ('ProduceCardGrowEffectType_LessonAdd',), True, False, True))
    card = _card()
    card._produce_card_status_enchant_id = 'enchant'
    metadata = decision_trace._card_enchant_metadata(card)
    assert metadata is not None
    assert metadata['definition']['trigger_count'] == 2
    assert metadata['remaining_trigger_count'] is None
