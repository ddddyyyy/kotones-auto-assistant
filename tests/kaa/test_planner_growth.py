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
                    buffs=ExamBuffState(parameter_buff_turns=2),
                    projected_effect_values=(('parameter_buff', 2),))
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
    state = replace(_state(hp=1), observed_genki=0, projected_effect_values=(('aggressive', 0),))
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
    state = replace(_state(hp=100), projected_effect_values=(('aggressive', 0),))
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
        projected_effect_values=(('aggressive', 0),),
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
    state = replace(_state(hp=100, growth=()), projected_effect_values=(('review', 0),))
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


@pytest.mark.parametrize('resource,cost_type,field', [
    ('review', 'ExamCostType_ExamReview', 'review'),
    ('lesson_buff', 'ExamCostType_ExamLessonBuff', 'lesson_buff'),
    ('aggressive', 'ExamCostType_ExamCardPlayAggressive', 'aggressive'),
    ('parameter_buff', 'ExamCostType_ExamParameterBuff', 'parameter_buff_turns'),
    ('parameter_buff_multiple', 'ExamCostType_ExamParameterBuffMultiplePerTurn', 'parameter_buff_multiple_turns'),
    ('full_power_point', 'ExamCostType_ExamFullPowerPoint', None),
])
def test_unknown_intrinsic_fee_cannot_certify_followup_special_budget(monkeypatch, resource, cost_type, field):
    from kaa.db import card_status_enchant as db
    from kaa.tasks.produce.new.play_cards.planner.state import AnomalyState, ExamBuffState
    monkeypatch.setattr(db, 'get_card_status_enchant', lambda key: SimpleNamespace(
        cost_changed=True, output_changed=False))
    state = replace(_state(hp=100, growth=()),
        anomaly=AnomalyState(full_power_points=5) if field is None else AnomalyState(),
        buffs=ExamBuffState(**{field: 5}) if field else ExamBuffState(),
        observed_effect_values=((resource, 5),), projected_effect_values=((resource, 5),))
    first = _card(green=0)
    first._produce_card_status_enchant_id = 'enchant'
    first.cost_type, first.cost_value = cost_type, 2
    after = simulate_play(state, first)
    assert costs.special_resource_balances(after)[resource] == 3
    assert dict(after.effect_values_for_planning)[resource] is None
    assert dict(after.observed_effect_values)[resource] == 5
    follow = _card(green=0)
    follow.cost_type, follow.cost_value = cost_type, 3
    assert can_afford_play(after, follow) is None
    known = _card(green=0)
    known.cost_type, known.cost_value = cost_type, 2
    known_after = simulate_play(state, known)
    assert dict(known_after.effect_values_for_planning)[resource] == 3
    assert can_afford_play(known_after, follow) is True


def test_unknown_growth_fee_cannot_certify_original_zero_special_cost(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: CardGrowthRule(target_known=True, costs_known=False))
    state = replace(_state(hp=100), buffs=ExamBuffState(review=5), projected_effect_values=(('review', 5),))
    after = simulate_play(state, _card(green=0))
    assert after.buffs.review == 5 and dict(after.effect_values_for_planning)['review'] is None


@pytest.mark.parametrize('green,red', [(4, 0), (0, 2), (0, 0)])
def test_visible_unparsed_growth_cannot_prove_base_payment(green, red):
    card = _card(green=green, red=red)
    state = replace(_state(hp=100, growth=()),
                    unresolved_growth_card_ids=frozenset({card._id}))
    assert can_afford_play(state, card) is None
    assert costs.play_cost(state, card) == (green, red)
    after = simulate_play(state, card)
    assert not after.hp_known and not after.genki_known
    assert after.unresolved_growth_card_ids == state.unresolved_growth_card_ids


def test_visible_unparsed_growth_keeps_output_estimate_but_not_confirmation():
    effect = SimpleNamespace(produce_exam_effect=SimpleNamespace(
        effect_type=ProduceExamEffectType.ExamLesson, effect_value1=6,
        effect_value2=0, effect_count=2, effect_turn=0))
    card = _card(green=0, effects=(effect,))
    state = replace(_state(growth=()), unresolved_growth_card_ids=frozenset({card._id}))
    evaluation = EffectHeuristicEvaluator().evaluate(card, state)
    assert evaluation.projected_output == 12
    assert evaluation.confirmed_output == 0


def test_uncertain_copy_identity_does_not_share_growth_amounts():
    first, second = _card(), _card()
    other = _card(); other._id = 'different-id'
    state = replace(_state(hp=100, growth=()), unresolved_growth_card_ids=frozenset({first._id}))
    assert first is not second
    assert can_afford_play(state, first) is None
    assert can_afford_play(state, second) is None
    assert can_afford_play(state, other) is True
    assert costs.play_cost(state, first) == costs.play_cost(state, second) == (4, 0)


def test_refreshed_snapshot_does_not_retain_old_growth_marker():
    state = replace(_state(hp=100, growth=()), unresolved_growth_card_ids=frozenset({'same-id'}))
    assert can_afford_play(state, _card()) is None
    refreshed = replace(state, unresolved_growth_card_ids=frozenset())
    assert can_afford_play(refreshed, _card()) is True


_RESOURCE_GROWTH_CASES = [
    ('ExamLessonBuff', 'lesson_buff', 'ExamCostType_ExamLessonBuff'),
    ('ExamLessonBuffReduce', 'lesson_buff', 'ExamCostType_ExamLessonBuff'),
    ('ExamReview', 'review', 'ExamCostType_ExamReview'),
    ('ExamReviewReduce', 'review', 'ExamCostType_ExamReview'),
    ('ExamCardPlayAggressive', 'aggressive', 'ExamCostType_ExamCardPlayAggressive'),
    ('ExamAggressiveReduce', 'aggressive', 'ExamCostType_ExamCardPlayAggressive'),
    ('ExamParameterBuff', 'parameter_buff', 'ExamCostType_ExamParameterBuff'),
    ('ExamParameterBuffMultiplePerTurn', 'parameter_buff_multiple', 'ExamCostType_ExamParameterBuffMultiplePerTurn'),
    ('ExamFullPowerPoint', 'full_power_point', 'ExamCostType_ExamFullPowerPoint'),
    ('ExamFullPowerPointReduce', 'full_power_point', 'ExamCostType_ExamFullPowerPoint'),
]


@pytest.mark.parametrize('kind,resource,cost_type', _RESOURCE_GROWTH_CASES)
@pytest.mark.parametrize('source', ['intrinsic', 'projected'])
def test_uncertain_output_growth_cannot_prove_special_balance(monkeypatch, kind, resource, cost_type, source):
    from kaa.db import card_status_enchant as db
    from kaa.tasks.produce.new.play_cards.planner.state import AnomalyState, ExamBuffState
    initial = replace(_state(hp=100, growth=()), anomaly=AnomalyState(full_power_points=5),
        buffs=ExamBuffState(lesson_buff=5, review=5, aggressive=5,
            parameter_buff_turns=5, parameter_buff_multiple_turns=5),
        projected_effect_values=((resource, 5),), observed_effect_values=((resource, 5),))
    growing = _card(green=0, effects=(_output(getattr(ProduceExamEffectType, kind), 2),))
    if source == 'intrinsic':
        growing._produce_card_status_enchant_id = 'unobserved'
        monkeypatch.setattr(db, 'get_card_status_enchant', lambda _: SimpleNamespace(
            cost_changed=False, output_changed=True))
    else:
        monkeypatch.setattr(costs, 'get_card_growth_rule', lambda _: _rule(
            output_changed=True, output_known=False))
        initial = replace(initial, projected_card_growth=(('growth', True),))
    after = simulate_play(initial, growing)
    follow = _card(green=0); follow.cost_type = cost_type; follow.cost_value = 1
    assert dict(after.effect_values_for_planning)[resource] is None
    assert can_afford_play(after, follow) is None
    assert after.observed_effect_values == initial.observed_effect_values
    assert growing.play_effects[0].produce_exam_effect.effect_value1 == 2


@pytest.mark.parametrize('uncertain', [False, True])
def test_hp_recovery_growth_preserves_estimate_but_only_known_value_proves_payment(monkeypatch, uncertain):
    from kaa.db import card_status_enchant as db
    monkeypatch.setattr(db, 'get_card_status_enchant', lambda _: SimpleNamespace(
        cost_changed=False, output_changed=uncertain))
    recovery = _card(green=0, effects=(_output(ProduceExamEffectType.ExamStaminaRecoverFix, 5),))
    recovery._produce_card_status_enchant_id = 'enchant'
    initial = _state(hp=1, growth=())
    after = simulate_play(initial, recovery)
    assert after.hp == 6 and after.hp_known is (not uncertain)
    assert after.genki_known
    assert can_afford_play(after, _card(green=0, red=4)) is (None if uncertain else True)
    assert initial.hp == 1 and initial.hp_known


@pytest.mark.parametrize('verdict', [TriggerVerdict.UNKNOWN, TriggerVerdict.INACTIVE])
def test_unconfirmed_recovery_trigger_does_not_prove_unchanged_hp(monkeypatch, verdict):
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *_: verdict)
    recovery = _card(green=0, effects=(_output(ProduceExamEffectType.ExamStaminaRecoverFix, 5),))
    after = simulate_play(_state(hp=0, growth=()), recovery)
    assert after.hp == 0
    assert after.hp_known is (verdict == TriggerVerdict.INACTIVE)
    assert can_afford_play(after, _card(green=0, red=4)) is (False if verdict == TriggerVerdict.INACTIVE else None)


def test_cost_only_growth_preserves_hp_recovery_and_special_balance(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda _: _rule(output_changed=False))
    initial = replace(_state(hp=1), buffs=ExamBuffState(lesson_buff=5),
        projected_effect_values=(('lesson_buff', 5),))
    recovering = _card(green=0, effects=(
        _output(ProduceExamEffectType.ExamLessonBuff, 2),
        _output(ProduceExamEffectType.ExamStaminaRecoverFix, 5)))
    after = simulate_play(initial, recovering)
    assert after.hp == 6 and after.hp_known
    assert dict(after.effect_values_for_planning)['lesson_buff'] == 7


@pytest.mark.parametrize('kind,resource,cost_type', [
    row for row in _RESOURCE_GROWTH_CASES if row[0] not in {'ExamReview', 'ExamFullPowerPoint'}
])
def test_decoded_review_growth_does_not_change_other_resource_segments(monkeypatch, kind, resource, cost_type):
    from kaa.tasks.produce.new.play_cards.planner.state import AnomalyState, ExamBuffState
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda _: _rule(
        output_changed=True, output_known=True, review_changes=(3,)))
    initial = replace(_state(hp=100), anomaly=AnomalyState(full_power_points=5),
        buffs=ExamBuffState(lesson_buff=5, review=5, aggressive=5,
            parameter_buff_turns=5, parameter_buff_multiple_turns=5),
        projected_effect_values=((resource, 5),))
    after = simulate_play(initial, _card(green=0, effects=(
        _output(getattr(ProduceExamEffectType, kind), 2),)))
    assert dict(after.effect_values_for_planning)[resource] is not None
    follow = _card(green=0); follow.cost_type = cost_type; follow.cost_value = 1
    assert can_afford_play(after, follow) is True


def test_decoded_resource_growth_does_not_change_fixed_hp_recovery(monkeypatch):
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda _: _rule(
        output_changed=True, output_known=True, review_changes=(3,)))
    after = simulate_play(_state(hp=1), _card(green=0, effects=(
        _output(ProduceExamEffectType.ExamStaminaRecoverFix, 5),)))
    assert after.hp == 6 and after.hp_known


@pytest.mark.parametrize('uncertain', [False, True])
def test_next_card_does_not_confirm_output_from_unobserved_growth_balance(monkeypatch, uncertain):
    from kaa.db import card_status_enchant as db
    from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState
    monkeypatch.setattr(db, 'get_card_status_enchant', lambda _: SimpleNamespace(
        cost_changed=False, output_changed=uncertain))
    setup = _card(green=0, effects=(_output(ProduceExamEffectType.ExamLessonBuff, 2),))
    setup._produce_card_status_enchant_id = 'enchant'
    initial = replace(_state(hp=100, growth=()), buffs=ExamBuffState(),
        projected_effect_values=(('lesson_buff', 0),))
    after = simulate_play(initial, setup)
    segment = _output(ProduceExamEffectType.ExamMultipleLessonBuffLesson, 10)
    segment.produce_exam_effect.effect_value2 = 1000
    follow = _card(green=0, effects=(segment,))
    evaluation = EffectHeuristicEvaluator().evaluate(follow, after)
    assert evaluation.projected_output == 12
    assert evaluation.confirmed_output == (0 if uncertain else 12)


_UNMODELED_BALANCE_EFFECTS = [
    ('ExamBlockAddMultipleAggressive', 'genki'),
    ('ExamBlockPerUseCardCount', 'genki'),
    ('ExamBlockDependExamReview', 'genki'),
    ('ExamBlockDependBlockConsumptionSum', 'genki'),
    ('ExamBlockDown', 'genki'),
    ('ExamStaminaRecoverMultiple', 'hp'),
    ('ExamStaminaReduce', 'hp'),
    ('ExamStaminaReduceFix', 'hp'),
    ('ExamStaminaDamage', 'hp'),
]


@pytest.mark.parametrize('kind,resource', _UNMODELED_BALANCE_EFFECTS)
@pytest.mark.parametrize('verdict', list(TriggerVerdict))
def test_unmodeled_resource_effect_never_certifies_old_budget(monkeypatch, kind, resource, verdict):
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *_: verdict)
    initial = replace(_state(hp=20, growth=()), genki=20)
    after = simulate_play(initial, _card(green=0, effects=(
        _output(getattr(ProduceExamEffectType, kind), 5),)))
    assert after.hp == 20 and after.genki == 20
    assert getattr(after, resource + '_known') == (verdict == TriggerVerdict.INACTIVE)
    other = 'hp' if resource == 'genki' else 'genki'
    assert getattr(after, other + '_known')
    follow = _card(green=41) if resource == 'genki' else _card(green=0, red=1)
    assert can_afford_play(after, follow) == (
        (False if resource == 'genki' else True)
        if verdict == TriggerVerdict.INACTIVE else None
    )
    assert initial.hp_known and initial.genki_known


@pytest.mark.parametrize('kind,resource,gain', [
    ('ExamStaminaReduce', 'hp', 'ExamStaminaRecoverFix'),
    ('ExamBlockDown', 'genki', 'ExamBlockFix'),
])
def test_known_gain_after_unmodeled_balance_change_does_not_restore_certainty(kind, resource, gain):
    after = simulate_play(replace(_state(hp=20, growth=()), genki=20),
        _card(green=0, effects=(
            _output(getattr(ProduceExamEffectType, kind), 5),
            _output(getattr(ProduceExamEffectType, gain), 3))))
    assert getattr(after, resource) == 23
    assert not getattr(after, resource + '_known')


def test_unmodeled_genki_change_does_not_confirm_in_card_conversion():
    initial = replace(_state(hp=20, growth=()), genki=20)
    card = _card(green=0, effects=(
        _output(ProduceExamEffectType.ExamBlockAddMultipleAggressive, 5),
        _output(ProduceExamEffectType.ExamLessonDependBlock, 1000)))
    evaluation = EffectHeuristicEvaluator().evaluate(card, initial)
    assert evaluation.projected_output == 20
    assert evaluation.confirmed_output == 0


@pytest.mark.parametrize('resource,field,multiplier', [
    ('parameter_buff', 'parameter_buff_turns', 1.5),
    ('parameter_buff_multiple', 'parameter_buff_multiple_turns', 1.1),
])
@pytest.mark.parametrize('reading,expected_known', [
    ('missing', False), (None, False), (0, False), (3, False), (5, True),
])
def test_projected_output_multiplier_requires_trusted_balance(resource, field, multiplier, reading, expected_known):
    from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState
    values = () if reading == 'missing' else ((resource, reading),)
    state = replace(_state(hp=20, growth=()), buffs=ExamBuffState(**{field: 5}),
        projected_effect_values=values, observed_effect_values=((resource, 5),))
    evaluation = EffectHeuristicEvaluator().evaluate(_card(green=0, effects=(_output(value=10),)), state)
    assert evaluation.projected_output == pytest.approx(10 * multiplier)
    assert evaluation.confirmed_output == pytest.approx(10 * multiplier if expected_known else 0)
    assert state.observed_effect_values == ((resource, 5),)


@pytest.mark.parametrize('resource', ['parameter_buff', 'parameter_buff_multiple'])
@pytest.mark.parametrize('reading,expected', [(None, 0), (0, 10)])
def test_unknown_zero_multiplier_balance_is_not_confirmed_absence(resource, reading, expected):
    state = replace(_state(hp=20, growth=()), projected_effect_values=((resource, reading),))
    result = EffectHeuristicEvaluator().evaluate(_card(green=0, effects=(_output(),)), state)
    assert result.projected_output == 10
    assert result.confirmed_output == expected


@pytest.mark.parametrize('resource,field,kind,multiplier', [
    ('parameter_buff', 'parameter_buff_turns', 'ExamParameterBuff', 1.5),
    ('parameter_buff_multiple', 'parameter_buff_multiple_turns', 'ExamParameterBuffMultiplePerTurn', 1.1),
])
@pytest.mark.parametrize('unknown_growth', [False, True])
def test_unknown_setup_growth_does_not_certify_next_card_multiplier(
    monkeypatch, resource, field, kind, multiplier, unknown_growth,
):
    from kaa.db import card_status_enchant as db
    from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState
    monkeypatch.setattr(db, 'get_card_status_enchant', lambda _: SimpleNamespace(
        cost_changed=False, output_changed=unknown_growth))
    initial = replace(_state(hp=20, growth=()), buffs=ExamBuffState(**{field: 5}),
        projected_effect_values=((resource, 5),), observed_effect_values=((resource, 5),))
    setup = _card(green=0, effects=(_output(getattr(ProduceExamEffectType, kind), 2),))
    setup._produce_card_status_enchant_id = 'enchant'
    after = simulate_play(initial, setup)
    evaluation = EffectHeuristicEvaluator().evaluate(_card(green=0, effects=(_output(),)), after)
    assert evaluation.projected_output == pytest.approx(10 * multiplier)
    assert evaluation.confirmed_output == pytest.approx(0 if unknown_growth else 10 * multiplier)


@pytest.mark.parametrize('reading,bonus', [(None, False), (5, True)])
def test_unknown_multiplier_cannot_claim_certain_perfect_target(reading, bonus):
    from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState
    state = replace(_state(hp=20, growth=()), buffs=ExamBuffState(parameter_buff_turns=5),
        score_gap=10, score_target_is_final=True,
        projected_effect_values=(('parameter_buff', reading),))
    result = EffectHeuristicEvaluator().evaluate(_card(green=0, effects=(_output(),)), state)
    assert result.projected_output == 15
    assert ('足量达标' in result.reason) == bonus
    assert result.confirmed_output == (15 if bonus else 0)


_CONFIRMED_ZERO_RESOURCE_CASES = [
    ('lesson_buff', 'lesson_buff', 'ExamLessonBuff'),
    ('review', 'review', 'ExamReview'),
    ('aggressive', 'aggressive', 'ExamCardPlayAggressive'),
    ('parameter_buff', 'parameter_buff_turns', 'ExamParameterBuff'),
    ('parameter_buff_multiple', 'parameter_buff_multiple_turns', 'ExamParameterBuffMultiplePerTurn'),
    ('full_power_point', 'full_power_points', 'ExamFullPowerPoint'),
]


@pytest.mark.parametrize('resource,field,kind', _CONFIRMED_ZERO_RESOURCE_CASES)
@pytest.mark.parametrize('source,expected', [
    ('absence', 3), ('numeric', 3), ('unknown_reading', None),
    ('old_absence', None), ('positive_model', None), ('present_icon', None),
])
def test_current_confirmed_zero_can_seed_resource_projection_without_reviving_unknown(
    resource, field, kind, source, expected,
):
    from kaa.tasks.produce.new.play_cards.planner.state import AnomalyState, ExamBuffState
    modeled = 5 if source == 'positive_model' else 0
    initial = BattleState(remaining_turns=3, hp=20, genki=0, playable_count=2,
        buffs=ExamBuffState(**({field: modeled} if resource != 'full_power_point' else {})),
        anomaly=AnomalyState(full_power_points=modeled if resource == 'full_power_point' else 0),
        confirmed_absent_effects=frozenset({resource}),
        observed_effect_values=((resource, 0),) if source == 'numeric' else (
            ((resource, None),) if source == 'unknown_reading' else ()),
        projected_effect_values=() if source == 'old_absence' else None,
        observed_effects=frozenset({resource}) if source == 'present_icon' else frozenset())
    card = _card(green=0, effects=(_output(getattr(ProduceExamEffectType, kind), 3),))
    after = simulate_play(initial, card)
    assert dict(after.effect_values_for_planning).get(resource) == expected
    assert after.observed_effect_values == initial.observed_effect_values
    assert initial.projected_effect_values == (() if source == 'old_absence' else None)


@pytest.mark.parametrize('resource,field,kind', _CONFIRMED_ZERO_RESOURCE_CASES)
def test_unknown_growth_still_invalidates_current_confirmed_zero_after_grant(monkeypatch, resource, field, kind):
    from kaa.db import card_status_enchant as db
    monkeypatch.setattr(db, 'get_card_status_enchant', lambda _: SimpleNamespace(
        cost_changed=False, output_changed=True))
    initial = BattleState(remaining_turns=3, hp=20, genki=0,
        confirmed_absent_effects=frozenset({resource}))
    card = _card(green=0, effects=(_output(getattr(ProduceExamEffectType, kind), 3),))
    card._produce_card_status_enchant_id = 'enchant'
    after = simulate_play(initial, card)
    assert dict(after.effect_values_for_planning).get(resource) is None
    assert can_afford_play(after, _card(green=0)) is True


@pytest.mark.parametrize('resource,kind,multiplier', [
    ('parameter_buff', 'ExamParameterBuff', 1.5),
    ('parameter_buff_multiple', 'ExamParameterBuffMultiplePerTurn', 1.1),
])
def test_confirmed_absent_multiplier_grant_can_certify_followup_output(resource, kind, multiplier):
    initial = BattleState(remaining_turns=3, hp=20, genki=0,
        confirmed_absent_effects=frozenset({resource}))
    after = simulate_play(initial, _card(green=0, effects=(
        _output(getattr(ProduceExamEffectType, kind), 3),)))
    result = EffectHeuristicEvaluator().evaluate(_card(green=0, effects=(_output(),)), after)
    assert result.projected_output == pytest.approx(10 * multiplier)
    assert result.confirmed_output == pytest.approx(10 * multiplier)


@pytest.mark.parametrize('resource,kind', [
    ('lesson_buff', 'ExamLessonBuffAdditive'),
    ('review', 'ExamReviewPerSearchCount'),
    ('aggressive', 'ExamAggressiveAdditive'),
    ('parameter_buff', 'ExamParameterBuffReduce'),
    ('parameter_buff_multiple', 'ExamParameterBuffMultiplePerTurnReduce'),
    ('full_power_point', 'ExamFullPowerPointAdditive'),
])
def test_confirmed_zero_is_not_promoted_through_unmodeled_related_effect(resource, kind):
    initial = BattleState(remaining_turns=3, hp=20, genki=0,
        confirmed_absent_effects=frozenset({resource}))
    after = simulate_play(initial, _card(green=0, effects=(
        _output(getattr(ProduceExamEffectType, kind), 3),)))
    assert dict(after.effect_values_for_planning).get(resource) is None


@pytest.mark.parametrize('resource,kind', [
    ('parameter_buff', 'ExamParameterBuffMultiplePerTurnReduce'),
    ('parameter_buff_multiple', 'ExamParameterBuffReduce'),
])
def test_unmodeled_modifier_does_not_block_unrelated_confirmed_zero(resource, kind):
    initial = BattleState(remaining_turns=3, hp=20, genki=0,
        confirmed_absent_effects=frozenset({resource}))
    after = simulate_play(initial, _card(green=0, effects=(
        _output(getattr(ProduceExamEffectType, kind), 3),)))
    assert dict(after.effect_values_for_planning)[resource] == 0


_UNMODELED_SPECIAL_BALANCE_CASES = [
    ('ExamAggressiveValueMultiple', 'aggressive', 'ExamCostType_ExamCardPlayAggressive'),
    ('ExamLessonBuffDependParameterBuff', 'lesson_buff', 'ExamCostType_ExamLessonBuff'),
    ('ExamLessonBuffPerSearchCount', 'lesson_buff', 'ExamCostType_ExamLessonBuff'),
    ('ExamParameterBuffDependLessonBuff', 'parameter_buff', 'ExamCostType_ExamParameterBuff'),
    ('ExamParameterBuffReduce', 'parameter_buff', 'ExamCostType_ExamParameterBuff'),
    ('ExamParameterBuffMultiplePerTurnReduce', 'parameter_buff_multiple', 'ExamCostType_ExamParameterBuffMultiplePerTurn'),
    ('ExamReviewDependExamBlock', 'review', 'ExamCostType_ExamReview'),
    ('ExamReviewDependExamCardPlayAggressive', 'review', 'ExamCostType_ExamReview'),
    ('ExamReviewPerSearchCount', 'review', 'ExamCostType_ExamReview'),
    ('ExamReviewValueMultiple', 'review', 'ExamCostType_ExamReview'),
]


@pytest.mark.parametrize('kind,resource,cost_type', _UNMODELED_SPECIAL_BALANCE_CASES)
@pytest.mark.parametrize('verdict', list(TriggerVerdict))
def test_unmodeled_direct_special_change_invalidates_only_its_budget(monkeypatch, kind, resource, cost_type, verdict):
    from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *_: verdict)
    values = tuple((name, 5) for name in (
        'lesson_buff', 'review', 'aggressive', 'parameter_buff', 'parameter_buff_multiple'))
    initial = replace(_state(hp=20, growth=()),
        buffs=ExamBuffState(lesson_buff=5, review=5, aggressive=5,
            parameter_buff_turns=5, parameter_buff_multiple_turns=5),
        projected_effect_values=values, observed_effect_values=values)
    after = simulate_play(initial, _card(green=0, effects=(
        _output(getattr(ProduceExamEffectType, kind), 2),)))
    expected: dict[str, int | None] = dict(values)
    if verdict != TriggerVerdict.INACTIVE:
        expected[resource] = None
    assert dict(after.effect_values_for_planning) == expected
    follow = _card(green=0)
    follow.cost_type = cost_type
    follow.cost_value = 1
    assert can_afford_play(after, follow) is (True if verdict == TriggerVerdict.INACTIVE else None)
    assert after.observed_effect_values == values
    assert after.buffs == initial.buffs


@pytest.mark.parametrize('kind', ['ExamReviewPerSearchCount', 'ExamReviewValueMultiple'])
def test_unmodeled_review_change_does_not_confirm_same_card_conversion(kind):
    from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState
    initial = replace(_state(hp=20, growth=()), buffs=ExamBuffState(review=5),
        projected_effect_values=(('review', 5),))
    card = _card(green=0, effects=(
        _output(getattr(ProduceExamEffectType, kind), 2),
        _output(ProduceExamEffectType.ExamLessonDependExamReview, 1000)))
    result = EffectHeuristicEvaluator().evaluate(card, initial)
    assert result.projected_output == 5
    assert result.confirmed_output == 0


@pytest.mark.parametrize('kind,resource,gain', [
    ('ExamReviewValueMultiple', 'review', 'ExamReview'),
    ('ExamLessonBuffDependParameterBuff', 'lesson_buff', 'ExamLessonBuff'),
])
def test_fixed_gain_does_not_restore_balance_after_unmodeled_direct_change(kind, resource, gain):
    from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState
    initial = replace(_state(hp=20, growth=()), buffs=ExamBuffState(review=5, lesson_buff=5),
        projected_effect_values=(('review', 5), ('lesson_buff', 5)))
    after = simulate_play(initial, _card(green=0, effects=(
        _output(getattr(ProduceExamEffectType, kind), 2),
        _output(getattr(ProduceExamEffectType, gain), 3))))
    assert dict(after.effect_values_for_planning)[resource] is None
    assert getattr(after.buffs, resource) == 8


_GAIN_MODIFIER_CASES = [
    ('ExamAggressiveAdditive', 'aggressive', 'ExamCardPlayAggressive'),
    ('ExamAggressiveAdditiveFix', 'aggressive', 'ExamCardPlayAggressive'),
    ('ExamLessonBuffAdditive', 'lesson_buff', 'ExamLessonBuff'),
    ('ExamLessonBuffAdditiveFix', 'lesson_buff', 'ExamLessonBuff'),
    ('ExamParameterBuffAdditive', 'parameter_buff', 'ExamParameterBuff'),
    ('ExamReviewAdditive', 'review', 'ExamReview'),
    ('ExamFullPowerPointAdditive', 'full_power_point', 'ExamFullPowerPoint'),
]


def _all_known_resource_state():
    from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState, AnomalyState
    return replace(_state(hp=20, growth=()),
        buffs=ExamBuffState(lesson_buff=5, review=5, aggressive=5,
            parameter_buff_turns=5, parameter_buff_multiple_turns=5),
        anomaly=AnomalyState(full_power_points=5),
        projected_effect_values=tuple((name, 5) for name in costs.SPECIAL_COST_EFFECTS.values()))


@pytest.mark.parametrize('kind,resource,gain', _GAIN_MODIFIER_CASES)
@pytest.mark.parametrize('verdict', list(TriggerVerdict))
def test_gain_modifier_preserves_current_budget_but_uncertifies_later_matching_gain(monkeypatch, kind, resource, gain, verdict):
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *_: verdict)
    initial = _all_known_resource_state()
    setup = _card(green=0, effects=(_output(getattr(ProduceExamEffectType, kind), 500),))
    after_setup = simulate_play(initial, setup)
    assert after_setup.effect_values_for_planning == initial.effect_values_for_planning
    possible = verdict != TriggerVerdict.INACTIVE
    assert getattr(after_setup, 'projected_resource_gain_modifiers', frozenset()) == (
        frozenset({resource}) if possible else frozenset())
    fee = _card(green=0)
    fee.cost_type = next(key for key, value in costs.SPECIAL_COST_EFFECTS.items() if value == resource)
    fee.cost_value = 1
    assert can_afford_play(after_setup, fee) is True
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *_: TriggerVerdict.ACTIVE)
    after_gain = simulate_play(after_setup, _card(green=0, effects=(
        _output(getattr(ProduceExamEffectType, gain), 8),)))
    values = dict(after_gain.effect_values_for_planning)
    assert values[resource] == (None if possible else (8 if resource == 'parameter_buff' else 13))
    assert all(value == 5 for key, value in values.items() if key != resource)


@pytest.mark.parametrize('before_gain', [False, True])
def test_in_card_gain_modifier_respects_segment_order(before_gain):
    grant = _output(ProduceExamEffectType.ExamReview, 3)
    modifier = _output(ProduceExamEffectType.ExamReviewAdditive, 500)
    effects = (modifier, grant) if before_gain else (grant, modifier)
    after = simulate_play(_all_known_resource_state(), _card(green=0, effects=effects))
    assert dict(after.effect_values_for_planning)['review'] == (None if before_gain else 8)
    later = simulate_play(after, _card(green=0, effects=(grant,)))
    assert dict(later.effect_values_for_planning)['review'] is None


@pytest.mark.parametrize('modifier,resource,reduction', [
    ('ExamLessonBuffAdditive', 'lesson_buff', 'ExamLessonBuffReduce'),
    ('ExamReviewAdditive', 'review', 'ExamReviewReduce'),
    ('ExamAggressiveAdditive', 'aggressive', 'ExamAggressiveReduce'),
    ('ExamFullPowerPointAdditive', 'full_power_point', 'ExamFullPowerPointReduce'),
])
def test_gain_modifier_does_not_change_known_resource_reduction(modifier, resource, reduction):
    setup = simulate_play(_all_known_resource_state(), _card(green=0, effects=(
        _output(getattr(ProduceExamEffectType, modifier), 500),)))
    after = simulate_play(setup, _card(green=0, effects=(
        _output(getattr(ProduceExamEffectType, reduction), 2),)))
    assert dict(after.effect_values_for_planning)[resource] == 3


def test_matching_gain_after_modifier_does_not_certify_same_card_review_conversion():
    setup = simulate_play(_all_known_resource_state(), _card(green=0, effects=(
        _output(ProduceExamEffectType.ExamReviewAdditive, 500),)))
    follow = _card(green=0, effects=(
        _output(ProduceExamEffectType.ExamReview, 3),
        _output(ProduceExamEffectType.ExamLessonDependExamReview, 1000)))
    result = EffectHeuristicEvaluator().evaluate(follow, setup)
    assert result.projected_output > 0
    assert result.confirmed_output == 0
