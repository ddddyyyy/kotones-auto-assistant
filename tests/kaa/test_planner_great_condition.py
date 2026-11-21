from dataclasses import replace
from typing import Any,cast
import pytest
from kaa.db.constants import ProduceExamEffectType as Effect,ProducePlanType
from kaa.tasks.produce.new.play_cards.planner.evaluator import EffectHeuristicEvaluator
from kaa.tasks.produce.new.play_cards.planner.state import BattleState,ExamBuffState
from tests.kaa.test_planner_evaluator import _card,_effect


def state(good=30,great=3,turns=5,**changes):
    return replace(BattleState(turns,50,30,plan_type=ProducePlanType.Plan1,archetype=Effect.ExamLessonBuff,
        buffs=ExamBuffState(lesson_buff=10,parameter_buff_turns=good,parameter_buff_multiple_turns=great),
        observed_effect_values=(('lesson_buff',10),('parameter_buff',good),('parameter_buff_multiple',great))),**changes)


@pytest.mark.parametrize('good,great,multiplier',[(30,3,4.5),(30,1,4.5),(30,99,4.5),
    (10,2,2.5),(1,1,1.6),(30,0,1.5),(0,3,1.),(0,0,1.)])
def test_great_condition_scales_good_condition_and_has_no_independent_zero_good_bonus(good,great,multiplier):
    card=cast(Any,_card('score',_effect(Effect.ExamLesson,20)))
    before=state(good,great)
    result=EffectHeuristicEvaluator().evaluate(card,before)
    assert result.projected_output==pytest.approx(30*multiplier)
    assert before.buffs.parameter_buff_turns==good and before.buffs.parameter_buff_multiple_turns==great


def test_attribute_and_score_bonus_multiply_the_correct_great_condition_factor():
    before=state(is_exam=True,exam_score_percent=200,observed_effects=frozenset({'lesson_value_up'}),
        observed_effect_values=(('lesson_buff',10),('parameter_buff',30),('parameter_buff_multiple',3),('lesson_value_up',200)))
    card=cast(Any,_card('score',_effect(Effect.ExamLesson,20)))
    result=EffectHeuristicEvaluator().evaluate(card,before)
    assert result.projected_output==pytest.approx(30*4.5*2*3)
    assert result.confirmed_output==0


def setup_score(before):
    card=cast(Any,_card('great',_effect(Effect.ExamParameterBuffMultiplePerTurn,0,turn=4)))
    return EffectHeuristicEvaluator().evaluate(card,before)


def test_great_setup_option_value_grows_with_trusted_good_duration():
    high=setup_score(state(30,0));low=setup_score(state(4,0))
    assert high.score>low.score
    assert high.projected_output==high.confirmed_output==0
    assert '机会估计' in high.reason


def test_no_speculative_great_setup_bonus_on_last_play_or_when_already_covered():
    high=setup_score(state(30,0,turns=1));low=setup_score(state(4,0,turns=1))
    assert high.score==low.score
    assert setup_score(state(30,20)).score<setup_score(state(30,0)).score


@pytest.mark.parametrize('effect',['parameter_buff','lesson_buff'])
def test_unknown_good_or_concentration_does_not_supply_setup_opportunity_budget(effect):
    before=state(30,0,observed_effect_values=(('parameter_buff',None if effect=='parameter_buff' else 30),
        ('lesson_buff',None if effect=='lesson_buff' else 10),('parameter_buff_multiple',0)))
    unknown=setup_score(before)
    assert unknown.projected_output==unknown.confirmed_output==0
    assert unknown.score<setup_score(state(30,0)).score
