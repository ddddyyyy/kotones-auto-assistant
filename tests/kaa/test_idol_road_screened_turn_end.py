from dataclasses import replace
import pytest
from kaa.idol_road.screened_turn_end import screened_turn_end
from kaa.idol_road.hp_safety import HpSafetyPolicy
from kaa.idol_road.log_hand import LogHandCard,confirm_log_hand
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from tests.kaa.test_idol_road_skip_probe import seeded
from tests.kaa.test_idol_road_hp_safety import card,budget


def scenario():
    _,frame,inputs=seeded()
    state=BattleState(20,50,0,max_hp=53,hand_size=1,hand_card_ids=('costly',),
        absolute_turn_observation=frame.absolute_turn)
    cards=(card('costly',8),)
    guard=HpSafetyPolicy(20,budget())
    choice=guard.select(state,cards)
    hand=confirm_log_hand(frame.values.frame_id,1,(LogHandCard('costly',0,True,(10,700,100,150),False),))
    assert choice.stop and choice.allowed_indices==()
    return inputs,frame,state,cards,choice,guard,hand


def test_no_hp_eligible_card_proposes_separately_guarded_skip_without_releasing_a_card():
    inputs,frame,state,cards,choice,guard,hand=scenario()
    result=screened_turn_end(inputs,frame,state,cards,choice,guard.review,hand)
    assert result is not None and result.end_turn and result.allowed_indices==()
    assert inputs.sequence==0 and inputs.pending is None
    assert not guard.end_turn(state).stop


@pytest.mark.parametrize('change',['missing','old','partial','fingerprint'])
def test_incomplete_or_foreign_hand_cannot_authorize_skip(change):
    inputs,frame,state,cards,choice,guard,hand=scenario()
    assert hand is not None
    bad={'missing':None,'old':replace(hand,frame_id='old'),
         'partial':replace(hand,cards=()),'fingerprint':replace(hand,fingerprint='fake')}[change]
    assert screened_turn_end(inputs,frame,state,cards,choice,guard.review,bad) is None


def test_pending_input_or_unbound_review_cannot_schedule_another_skip():
    inputs,frame,state,cards,choice,guard,hand=scenario()
    assert guard.review is not None
    assert screened_turn_end(inputs,frame,state,cards,choice,replace(guard.review,frame_id='old'),hand) is None
    inputs.begin('skip',frame)
    assert screened_turn_end(inputs,frame,state,cards,choice,guard.review,hand) is None


def test_existing_imminent_hp_threshold_is_still_checked_separately():
    inputs,frame,state,cards,choice,guard,hand=scenario()
    low=replace(state,hp=20,remaining_turns=4,
        absolute_turn_observation=replace(state.absolute_turn_observation,turn=17,remaining_turns=4))
    assert guard.end_turn(low).stop
