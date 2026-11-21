from dataclasses import replace
from types import SimpleNamespace
import pytest
from kaa.idol_road.ranked_field_policy import RankedFieldPolicy
from kaa.idol_road.input_policy import InputBoundPolicy
from kaa.idol_road.hp_safety import HpSafetyPolicy
from kaa.tasks.produce.new.play_cards.planner.selection_policy import SelectionConstraint
from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluation, EvaluationConfidence
from tests.kaa.test_idol_road_hp_safety import state, card, budget, input_policy


class Scores:
    def __init__(self, values):
        self.values = values
        self.seen = []

    def evaluate(self, candidate, current):
        self.seen.append(candidate._id)
        value = self.values[candidate._id]
        return CardEvaluation(candidate._id, value, EvaluationConfidence.MEDIUM, 'estimate') if value is not None else None


def delegate(constraint=SelectionConstraint()):
    return SimpleNamespace(select=lambda *args: constraint, committed=lambda *args: None, feedback=lambda *args: None)


def test_field_preference_beats_larger_ordinary_estimate():
    evaluator = Scores({'field': 1., 'ordinary': 1000.})
    policy = RankedFieldPolicy(delegate(SelectionConstraint((0,), reason='last M card')), evaluator=evaluator)
    result = policy.select(state(), (card('field', 0), card('ordinary', 0)))
    assert result.preferred_indices == result.allowed_indices == (0,)
    assert evaluator.seen == ['field']
    assert policy.trace.frame_id == state().absolute_turn_observation.frame_id


def test_ordinary_estimate_ranks_only_current_allowed_subset():
    evaluator = Scores({'excluded': 1000., 'a': 2., 'b': 3.})
    policy = RankedFieldPolicy(delegate(SelectionConstraint(allowed_indices=(1, 2))), evaluator=evaluator)
    result = policy.select(state(), tuple(card(key, 0) for key in evaluator.values))
    assert result.preferred_indices == (2,) and result.allowed_indices == (1, 2)
    assert evaluator.seen == ['a', 'b']


def test_ties_keep_current_candidate_order_and_never_change_observation():
    current = state()
    policy = RankedFieldPolicy(delegate(), evaluator=Scores({'a': 2., 'b': 2.}))
    result = policy.select(current, (card('a', 0), card('b', 0)))
    assert result.preferred_indices == (0, 1)
    assert 'observation_unchanged' in result.reason
    assert current == state()


@pytest.mark.parametrize('constraint', [SelectionConstraint(stop=True, reason='field stop'),
    SelectionConstraint(end_turn=True, reason='preserve last card')])
def test_terminal_field_instruction_is_not_replaced_by_ranking(constraint):
    evaluator = Scores({})
    policy = RankedFieldPolicy(delegate(constraint), evaluator=evaluator)
    assert policy.select(state(), ()) == constraint and evaluator.seen == []


@pytest.mark.parametrize('constraint', [SelectionConstraint((1,), allowed_indices=(0,)),
    SelectionConstraint(allowed_indices=()), SelectionConstraint(allowed_indices=(True,)),
    SelectionConstraint((-1,)), SelectionConstraint(allowed_indices=(2,))])
def test_invalid_field_candidate_mapping_stops(constraint):
    policy = RankedFieldPolicy(delegate(constraint), evaluator=Scores({'a': 1.}))
    result = policy.select(state(), (card('a', 0),))
    assert result.stop and result.allowed_indices == ()
    assert policy.trace.reason == result.reason


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), float('-inf')])
def test_missing_or_nonfinite_evaluation_cannot_become_recommendation_fallback(value):
    policy = RankedFieldPolicy(delegate(), evaluator=Scores({'a': value}))
    result = policy.select(state(), (card('a', 0),))
    assert result.stop and result.allowed_indices == () and not result.preferred_indices


def test_hp_guard_filters_before_ranking_and_mapping_is_returned_to_current_hand():
    evaluator = Scores({'unsafe': 1000., 'safe': 1.})
    ranked = RankedFieldPolicy(delegate(), evaluator=evaluator)
    policy = input_policy(state(),ranked,HpSafetyPolicy(20,budget()))
    result = policy.select(state(), (card('unsafe', 8), card('safe', 1)))
    assert evaluator.seen == ['safe']
    assert result.preferred_indices == result.allowed_indices == (1,)


@pytest.mark.parametrize('change', [{'card_id': 'other'}, {'projected_output': float('nan')},
    {'confirmed_output': -1.}, {'projected_output': -1.}])
def test_evaluation_identity_and_output_sanity_are_checked(change):
    value = replace(CardEvaluation('a', 1., EvaluationConfidence.HIGH, 'estimate'), **change)
    policy = RankedFieldPolicy(delegate(), evaluator=SimpleNamespace(evaluate=lambda *args: value))
    assert policy.select(state(), (card('a', 0),)).stop


@pytest.mark.parametrize('kind',['concentration','good_condition'])
def test_unknown_active_buff_goal_cannot_be_hidden_by_score_setup_reason(kind):
    from kaa.idol_road.field_priority import FieldPriority
    from kaa.idol_road.concentration_priority import ConcentrationObjective
    from kaa.idol_road.concentration_rule import ConcentrationDoubleDeclaration
    from kaa.idol_road.good_condition_rule import GoodConditionRequirement
    from kaa.idol_road.score_priority import ScorePreparationObjective
    from kaa.battle_turn import AbsoluteTurnObservation
    current=replace(state(),remaining_turns=19,current_score=0,observed_effect_values=(),
        absolute_turn_observation=AbsoluteTurnObservation('battle','current',2,19))
    field=FieldPriority(20,(),
        (ConcentrationObjective(ConcentrationDoubleDeclaration(10,42),('frame',)),) if kind=='concentration' else (),
        (GoodConditionRequirement(6,4,('frame',)),) if kind=='good_condition' else (),
        (ScorePreparationObjective(16,11000,('frame',)),))
    evaluator=Scores({'a':1000.})
    result=RankedFieldPolicy(field,evaluator=evaluator).select(current,(card('a',0),))
    assert result.stop and result.allowed_indices==() and evaluator.seen==[]
    assert result.reason==f'field_{kind}:observation_unknown'


@pytest.mark.parametrize('kind',['concentration','good_condition'])
def test_expired_or_absent_buff_goals_do_not_require_unrelated_current_values(kind):
    from kaa.idol_road.concentration_priority import ConcentrationPriority,ConcentrationObjective
    from kaa.idol_road.concentration_rule import ConcentrationDoubleDeclaration
    from kaa.idol_road.good_condition_priority import GoodConditionPriority
    from kaa.idol_road.good_condition_rule import GoodConditionRequirement
    current=replace(state(),observed_effect_values=())
    cls=ConcentrationPriority if kind=='concentration' else GoodConditionPriority
    expired=(ConcentrationObjective(ConcentrationDoubleDeclaration(3,42),('frame',)) if kind=='concentration'
        else GoodConditionRequirement(3,4,('frame',)))
    assert not cls(20,()).select(current,()).stop
    assert not cls(20,(expired,)).select(current,()).stop
