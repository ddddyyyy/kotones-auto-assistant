from dataclasses import replace
import pytest
from kaa.idol_road.battle_observation import (BattleFrameValues, BattleObservationSession,
                                             bind_battle_frame, create_battle_session)
from kaa.idol_road.models import Observation, Scene


def formation(**changes):
    return replace(Observation(Scene.CONFIRM, 'ttmr', 16, stars=0, formation_ready=True, total_turns=10), **changes)


def test_each_confirmed_start_request_gets_a_distinct_battle_id():
    a,b=create_battle_session(formation()),create_battle_session(formation())
    assert a.battle_id != b.battle_id and a.total_turns==10
    assert a.identity_source=='confirmed_formation_start_request'


@pytest.mark.parametrize('changes',[{'scene':Scene.BATTLE},{'stars':1},{'stars':3},
    {'formation_ready':False},{'character_id':None},{'stage':None},{'stage':True}])
def test_unconfirmed_or_cleared_formation_cannot_create_session(changes):
    with pytest.raises(ValueError):
        create_battle_session(formation(**changes))


@pytest.mark.parametrize('total',[None,True,0,31])
def test_unknown_or_invalid_total_does_not_invent_current_turn(total):
    session=create_battle_session(formation(total_turns=total))
    frame=bind_battle_frame(session,BattleFrameValues('frame',True,10,48))
    assert frame is not None and frame.observation.turn is None


def test_current_turn_comes_from_same_session_total_and_remaining_with_no_value_backfill():
    session=create_battle_session(formation())
    first=bind_battle_frame(session,BattleFrameValues('a',True,10,48))
    third=bind_battle_frame(session,BattleFrameValues('b',True,8,40))
    unknown=bind_battle_frame(session,BattleFrameValues('c',True,None,None))
    assert first is not None and third is not None and unknown is not None
    assert first.observation.turn==1 and third.observation.turn==3
    assert unknown.observation.turn is None and unknown.observation.hp is None
    assert third.observation.concentration is None and third.observation.max_hp is None
    assert third.observation.source=='confirmed_current_battle_observation'


def test_remaining_exceeding_total_is_a_conflict_not_a_clamped_turn():
    frame=bind_battle_frame(create_battle_session(formation()),BattleFrameValues('a',True,18,48))
    assert frame is not None and frame.observation.turn is None and 'turn' in frame.observation.conflicted_fields


@pytest.mark.parametrize('hp',[True,-1])
def test_invalid_hp_is_unknown_and_flagged(hp):
    frame=bind_battle_frame(create_battle_session(formation()),BattleFrameValues('a',True,10,hp))
    assert frame is not None and frame.observation.hp is None and 'hp' in frame.observation.conflicted_fields


def test_historical_session_never_claims_current_battle_observation():
    session=BattleObservationSession('capture','hski',13,18,'historical_capture_metadata')
    frame=bind_battle_frame(session,BattleFrameValues('a',True,18,49))
    assert frame is not None and frame.observation.turn==1
    assert frame.observation.source=='historical_or_unconfirmed_battle_observation'


@pytest.mark.parametrize('changes',[{'hud_confirmed':False},{'frame_id':''}])
def test_missing_hud_or_frame_has_no_bound_observation(changes):
    values=replace(BattleFrameValues('a',True,10,48),**changes)
    assert bind_battle_frame(create_battle_session(formation()),values) is None
