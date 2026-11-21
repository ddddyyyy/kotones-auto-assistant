from dataclasses import replace
import pytest
from kaa.idol_road.queued_turns import confirm_queued_turns,queued_turns_valid
from kaa.idol_road.battle_observation import bind_battle_frame
from tests.kaa.test_idol_road_hud_feedback import frame,SESSION


@pytest.mark.parametrize('base,queued,total',[(4,1,5),(3,2,5),(1,2,3),(2,0,2)])
def test_complete_budget_counts_visible_pending_additions_without_guessing_absolute_turn(base,queued,total):
    cue=confirm_queued_turns('now',base,.99 if queued==1 else 0.,.99 if queued==2 else 0.,
        .9 if queued else 0.,1. if queued else 0.)
    assert (cue.queued,cue.remaining)==(queued,total)
    assert queued_turns_valid(cue,'now',total)
    current=frame('now',total)
    bound=bind_battle_frame(SESSION,replace(current.values,queued_turns=cue))
    assert bound is not None and bound.observation.turn is None
    assert bind_battle_frame(SESSION,replace(current.values,queued_turns=replace(cue,remaining=base+3))) is None


@pytest.mark.parametrize('one,two,pill,ink',[
    (.98,.98,.9,1.),(.96,.1,.9,.9),(.99,.1,.2,1.),(.99,.1,.9,.7),
    (.1,.1,.8,.0),(.1,.1,0.,.3)])
def test_conflicting_partial_or_unknown_badge_never_becomes_no_additions(one,two,pill,ink):
    value=confirm_queued_turns('now',1,one,two,pill,ink)
    assert value.queued is None and value.remaining is None


@pytest.mark.parametrize('base',[None,0,31,True])
def test_visible_badge_cannot_supply_a_missing_or_invalid_main_number(base):
    assert confirm_queued_turns('now',base,.1,.99,.9,1.).remaining is None


def test_queue_evidence_cannot_move_to_another_capture_or_overflow_the_budget():
    value=confirm_queued_turns('now',29,.1,.99,.9,1.)
    assert value.queued==2 and value.remaining is None
    assert not queued_turns_valid(value,'other',None)
