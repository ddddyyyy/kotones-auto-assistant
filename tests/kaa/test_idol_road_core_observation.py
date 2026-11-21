"""Partial-observation protocol and input gates; no device actions or fabricated gameplay tests."""
from dataclasses import replace
import pytest
from kaa.battle_turn import AbsoluteTurnObservation
from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame,current_absolute_turn
from kaa.idol_road.battle_inputs import BattleInputs
from kaa.idol_road.battle_log import BattleLogBinding,LogEntryEvidence
from kaa.idol_road.page import PageSnapshot
from kaa.idol_road.models import Observation,Scene
from kaa.idol_road.planner_context import IdolRoadBattleContext
from kaa.idol_road.planner_adapter import TrialFrameGuard
from kaa.idol_road.battle_trace import BattleTrace

SESSION=BattleObservationSession('core-protocol','ttmr',20,20,'confirmed_formation_start_request')


def frame(name='frame',remaining=20,mode='core_only'):
    result=bind_battle_frame(SESSION,BattleFrameValues(name,True,remaining,50,detail_mode=mode))
    assert result is not None
    return result


def test_core_has_only_observed_resources_and_is_not_empty_or_complete_state():
    core=frame()
    assert core.observation.hp==50 and core.values.remaining_turns==20
    assert core.observation.max_hp is None and core.observation.concentration is None
    assert core.observation.good_condition_turns is None and core.values.detail_mode=='core_only'
    assert frame(mode='complete').values.detail_mode=='complete'


def test_core_rejects_mislabelled_detailed_evidence():
    # Even malformed detail payloads cannot be accepted as core-only packet values.
    packet=BattleFrameValues('frame',True,20,50,detail_mode='core_only')
    assert bind_battle_frame(SESSION,replace(packet,hp_ratio=object())) is None
    assert bind_battle_frame(SESSION,replace(packet,good_condition=object())) is None
    assert bind_battle_frame(SESSION,replace(packet,concentration=object())) is None
    assert bind_battle_frame(SESSION,replace(packet,detail_mode='unverified')) is None


@pytest.mark.parametrize('kind,card_id',[('card','card'),('skip',None)])
def test_core_cannot_register_new_battle_input(kind,card_id):
    inputs=BattleInputs(SESSION,clock=lambda:0)
    with pytest.raises(ValueError):
        inputs.begin(kind,frame(),card_id)
    assert inputs.sequence==0 and inputs.pending is None


def test_core_fixed_clock_can_settle_skip_but_strategy_waits_for_next_full_frame():
    inputs=BattleInputs(SESSION,clock=lambda:0)
    inputs.begin('skip',frame('before',mode='complete'))
    core=frame('after',19)
    inputs.observe_hud(core)
    assert inputs.pending is None
    guard=TrialFrameGuard(SESSION,100,clock=lambda:0)
    assert guard.check(PageSnapshot(Observation(Scene.BATTLE),battle_frame=core))=='wait'
    with pytest.raises(ValueError,match='日志探针'):
        IdolRoadBattleContext(None,core).fetch_absolute_turn_observation()
    full=frame('next',19,mode='complete')
    assert guard.check(PageSnapshot(Observation(Scene.BATTLE),battle_frame=full))=='continue'


def test_core_never_matches_a_planner_state_or_settles_a_card_feedback():
    from kaa.tasks.produce.new.play_cards.planner.state import BattleState
    from kaa.tasks.produce.new.play_cards.planner.feedback import PendingPlay,PlayFeedback,PlayFeedbackStatus
    inputs=BattleInputs(SESSION,clock=lambda:0)
    before=BattleState(20,50,0)
    inputs.begin('card',frame('before',mode='complete'),'card')
    inputs.bind_card_state(before,'card')
    feedback=PlayFeedback(PlayFeedbackStatus.CONFIRMED,'card_left_hand')
    assert not inputs.finish_card(PendingPlay('card',before,0,0),feedback,BattleState(19,50,0),frame('after',19))
    assert inputs.pending is not None


def test_core_hud_can_stabilize_log_target_without_granting_strategy_access():
    binding=BattleLogBinding(clock=lambda:0)
    for name in ('a','b'):
        entry=LogEntryEvidence(name,(449,1179,30,39),.99,'d'*64)
        binding.observe_hud(frame(name),entry)
    assert binding.can_open(frame('b'),entry)
    with pytest.raises(ValueError,match='日志探针'):
        IdolRoadBattleContext(None,frame('b')).fetch_absolute_turn_observation()


def test_core_cannot_borrow_an_absolute_contract_even_with_matching_identity():
    session=replace(SESSION,turn_budget_mode='variable_budget')
    core=bind_battle_frame(session,BattleFrameValues('core',True,20,50,detail_mode='core_only'))
    forged=replace(core,absolute_turn=AbsoluteTurnObservation(session.battle_id,'core',1,20),
        observation=replace(core.observation,turn=1,turn_source='owned_initial_log_return'))
    assert current_absolute_turn(forged) is None
    with pytest.raises(ValueError):
        BattleInputs(session).begin('card',forged,'card')


def test_core_origin_cannot_be_mislabelled_as_a_pending_skip_read_permit():
    from kaa.idol_road.skip_probe import pending_skip_permit
    inputs=BattleInputs(SESSION,clock=lambda:0)
    inputs.begin('skip',frame('before',mode='complete'))
    state=inputs.snapshot()
    assert pending_skip_permit(SESSION,state,now=0) is not None
    malformed=replace(state,pending=replace(state.pending,frame=frame('before')))
    assert pending_skip_permit(SESSION,malformed,now=0) is None


def test_trace_does_not_merge_core_and_complete_packets_when_other_fields_equal():
    trace=BattleTrace()
    observation=Observation(Scene.BATTLE,'ttmr',20)
    trace.observe(observation,None,frame('a'))
    trace.observe(observation,None,frame('b',mode='complete'))
    assert len(trace.snapshot().steps)==2
