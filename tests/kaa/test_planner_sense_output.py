from dataclasses import replace
import pytest
from kaa.db.constants import ProduceExamEffectType
from kaa.tasks.produce.new.play_cards.planner.sense_output import estimate_sense_output
from kaa.tasks.produce.new.play_cards.planner.state import BattleState, ExamBuffState


def state(concentration=61, percent=1494):
    return BattleState(remaining_turns=2,hp=39,genki=0,is_exam=True,exam_score_percent=percent,
        archetype=ProduceExamEffectType.ExamLessonBuff,
        buffs=ExamBuffState(lesson_buff=concentration,parameter_buff_turns=22),
        observed_effect_values=(('lesson_buff',concentration),('parameter_buff',22)))


@pytest.mark.parametrize('concentration,percent,actual',[(51,1494,1285),(61,1494,1509),(62,2977,3037)])
def test_single_hit_good_condition_rounding_matches_separate_card_log_segments(concentration,percent,actual):
    result=estimate_sense_output(6.,1,state(concentration,percent),1.5*percent/100)
    assert result is not None and result.output==actual and result.observed_rounding_scope


def test_multi_hit_is_an_estimate_without_claiming_single_hit_rounding():
    result=estimate_sense_output(5.,2,state(7),1.5*1494/100)
    assert result is not None and result.output==pytest.approx(537.84)
    assert not result.observed_rounding_scope


def test_additional_multiplier_is_not_certified_by_plain_good_condition_examples():
    current=replace(state(),observed_effects=frozenset({'lesson_value_up'}))
    result=estimate_sense_output(6.,1,current,1.5*14.94*1.1)
    assert result is not None and not result.observed_rounding_scope


def test_unknown_attribute_and_non_exam_remain_estimates():
    for current in (replace(state(),exam_score_percent=None),replace(state(),is_exam=False)):
        result=estimate_sense_output(6.,1,current,1.5)
        assert result is not None and result.output==100.5 and not result.observed_rounding_scope


def test_other_archetypes_keep_existing_output_path():
    current=replace(state(),archetype=ProduceExamEffectType.ExamCardPlayAggressive)
    assert estimate_sense_output(6.,1,current,1.5) is None
