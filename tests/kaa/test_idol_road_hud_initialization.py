from dataclasses import replace
import pytest
from kaa.idol_road.hud_initialization import InitialHudRound
from kaa.idol_road.battle_observation import current_absolute_turn,bind_battle_frame
from kaa.idol_road.log_hand import LogHandCard,confirm_log_hand
from tests.kaa.test_idol_road_initialization import OWNER,setup


def raw(name,reference):
    result=bind_battle_frame(OWNER,replace(reference.values,frame_id=name))
    return replace(result,maximum_hp_parameter=replace(reference.maximum_hp_parameter,current_frame_id=name))


def hand(name):
    return confirm_log_hand(name,1,(LogHandCard('card',0,True,(100,900,50,100),False),))


def test_owned_start_idle_ledger_and_two_calibrated_huds_bind_first_round_without_log():
    inputs,_,_,ready=setup()
    window=InitialHudRound(OWNER,deadline=90)
    assert window.observe(inputs.snapshot(),raw('one',ready),hand('one'),now=1) is None
    bound=window.observe(inputs.snapshot(),raw('two',ready),hand('two'),now=2)
    assert current_absolute_turn(bound).turn==1
    assert bound.observation.turn_source=='owned_initial_hud'
    assert inputs.sequence==0 and inputs.pending is None


def test_same_frame_repeated_does_not_bind_first_round():
    inputs,_,_,ready=setup()
    window=InitialHudRound(OWNER,deadline=90)
    for _ in range(3):assert window.observe(inputs.snapshot(),raw('one',ready),hand('one'),now=1) is None


@pytest.mark.parametrize('change',['missing_maximum','stale_maximum','missing_hand','changed_remaining','partial'])
def test_uncertain_initial_observation_does_not_invent_turn(change):
    inputs,_,_,ready=setup()
    window=InitialHudRound(OWNER,deadline=90)
    sample=raw('one',ready);cards=hand('one')
    if change=='missing_maximum':sample=replace(sample,maximum_hp_parameter=None)
    elif change=='stale_maximum':sample=replace(sample,maximum_hp_parameter=replace(sample.maximum_hp_parameter,current_frame_id='old'))
    elif change=='missing_hand':cards=None
    elif change=='changed_remaining':sample=replace(sample,values=replace(sample.values,remaining_turns=19))
    elif change=='partial':sample=replace(sample,values=replace(sample.values,detail_mode='feedback_only'))
    assert window.observe(inputs.snapshot(),sample,cards,now=1) is None


@pytest.mark.parametrize('change',['sequence','owner','expired','deadline'])
def test_start_ownership_and_initial_deadline_are_required(change):
    inputs,_,_,ready=setup()
    state=inputs.snapshot()
    if change=='sequence':state=replace(state,sequence=1)
    elif change=='owner':state=replace(state,battle_id='other')
    elif change=='expired':state=replace(state,expired=True)
    with pytest.raises(ValueError):
        InitialHudRound(OWNER,deadline=90).observe(state,raw('one',ready),hand('one'),now=90 if change=='deadline' else 1)
