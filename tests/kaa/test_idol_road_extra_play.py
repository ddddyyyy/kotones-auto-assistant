from dataclasses import replace
import pytest
from kaa.idol_road.extra_play import confirm_extra_play,bind_sense_play_count
from kaa.idol_road.battle_profile import SenseBattleProfile
from kaa.db.constants import ProduceExamEffectType
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from tests.kaa.test_idol_road_hud_feedback import scenario,SESSION


def observe(score=.99,coverage=.99,unique=True):
    return confirm_extra_play('frame',(14,744,80,84),score,coverage,unique)


def test_complete_literal_one_is_an_observation_not_an_assumed_base_play():
    assert observe().value==1 and observe().scope=='absent_or_literal_one'


def two(**changes):
    args=dict(two_score=.99,two_stroke_coverage=.99,unique_two=True,two_blue_coverage=1.)
    return confirm_extra_play('frame',(14,744,80,84),.8,.8,True,1.,1.,**(args|changes))


def test_complete_literal_two_proves_three_total_plays_and_preserves_hp():
    from kaa.idol_road.extra_play import extra_play_valid
    value=two()
    assert value.value==2 and extra_play_valid(value,'frame')
    assert not extra_play_valid(replace(observe(),value=2),'frame')
    assert not extra_play_valid(value,'old')
    inputs,_=scenario();frame=inputs.pending.frame
    profile=SenseBattleProfile(SESSION,'formation',ProduceExamEffectType.ExamLessonBuff)
    state=BattleState(20,48,0,is_exam=True,absolute_turn_observation=frame.absolute_turn)
    current=bind_sense_play_count(state,frame,replace(value,frame_id=frame.values.frame_id),profile)
    assert current.playable_count==3 and current.hp==48 and state.playable_count==1


@pytest.mark.parametrize('changes',[{'two_score':.969},{'two_stroke_coverage':.94},{'unique_two':False},
    {'two_blue_coverage':.59},{'two_score':float('nan')},{'unique_two':None},{'two_score':True}])
def test_partial_or_invalid_two_is_unknown(changes):
    assert two(**changes).value is None


def test_conflicting_one_and_two_cannot_choose_a_count():
    value=confirm_extra_play('frame',(14,744,80,84),.99,.99,True,1.,1.,
        two_score=.99,two_stroke_coverage=.99,unique_two=True,two_blue_coverage=1.)
    assert value.value is None


def test_zero_requires_whole_control_strokes_to_be_absent():
    assert observe(score=0,coverage=0).value==0
    assert observe(score=.1,coverage=.3).value is None


def test_white_background_does_not_hide_absence_of_the_complete_blue_control():
    value=confirm_extra_play('frame',(14,744,80,84),.4,.5,True,0.,0.)
    assert value.value==0 and value.stroke_coverage==.5
    assert confirm_extra_play('frame',(14,744,80,84),.4,.5,True,.3,.3).value is None
    assert confirm_extra_play('frame',(14,744,80,84),0.,0.,True,.3,.3).value==0
    assert confirm_extra_play('frame',(14,744,80,84),.99,.99,True,1.,1.).value==1
    assert confirm_extra_play('frame',(14,744,80,84),.99,.99,True,.4,.4).value is None


@pytest.mark.parametrize('score,coverage,unique',[(.96,.99,True),(.99,.7,True),(.99,.99,False),
    (float('nan'),.99,True),(.99,float('nan'),True),(.1,.06,True)])
def test_unknown_or_occluded_control_is_not_silently_zero(score,coverage,unique):
    assert observe(score,coverage,unique).value is None


def test_geometry_and_frame_are_bound():
    assert confirm_extra_play('',(14,744,80,84),.99,.99,True).value is None
    assert confirm_extra_play('frame',(700,744,80,84),.99,.99,True).value is None


def test_adapter_replaces_the_default_with_actual_extra_plays_without_changing_hp():
    inputs,_=scenario();frame=inputs.pending.frame
    profile=SenseBattleProfile(SESSION,'formation',ProduceExamEffectType.ExamLessonBuff)
    state=BattleState(20,48,0,is_exam=True,playable_count=1,absolute_turn_observation=frame.absolute_turn)
    extra=confirm_extra_play(frame.values.frame_id,(14,744,80,84),.99,.99,True)
    current=bind_sense_play_count(state,frame,extra,profile)
    assert current.playable_count==2 and current.hp==48 and state.playable_count==1
    assert bind_sense_play_count(state,frame,replace(extra,frame_id='old'),profile) is None
