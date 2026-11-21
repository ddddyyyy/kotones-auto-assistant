"""Initial observation scheduling contracts; no device operations."""
from dataclasses import replace
import pytest
from kaa.idol_road.initialization import InitialBattleObservation,BattleInitializationError
from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame
from kaa.idol_road.battle_inputs import BattleInputs
from kaa.idol_road.maximum_hp import MaximumHpTracker
from kaa.idol_road.hp_ratio import HpRatioRead,confirm_hp_ratio
from kaa.idol_road.page import PageSnapshot
from kaa.idol_road.models import Observation,Scene,Action
from kaa.battle_turn import AbsoluteTurnObservation

OWNER=BattleObservationSession('initial','ttmr',20,20,'confirmed_formation_start_request','variable_budget')


def setup():
    inputs=BattleInputs(OWNER,clock=lambda:0)
    tracker=MaximumHpTracker(OWNER)
    def frame(name,ratio=False):
        pair=confirm_hp_ratio(name,(329,193,47,25),.99,tuple(HpRatioRead(name,kind,'50/53',.99)
                    for kind in ('original','grayscale'))) if ratio else None
        result=bind_battle_frame(OWNER,BattleFrameValues(name,True,20,50,hp_ratio=pair))
        assert result is not None
        return result
    before=frame('before')
    tracker.request(before,inputs.snapshot())
    tooltip=frame('tooltip',True)
    tracker.observe(tooltip,inputs.snapshot())
    calibrated=tracker.bind(tooltip,inputs.snapshot())
    raw=tracker.bind(frame('back'),inputs.snapshot())
    ready=replace(raw,absolute_turn=AbsoluteTurnObservation(OWNER.battle_id,'back',1,20),
                  observation=replace(raw.observation,turn=1,turn_source='owned_initial_log_return'))
    return inputs,before,calibrated,ready


def snapshot(frame=None,**kwargs):
    return PageSnapshot(Observation(Scene.BATTLE,battle_state=frame.observation if frame else None,**kwargs),battle_frame=frame)


def test_order_is_one_hp_request_then_one_log_then_exact_ready_frame():
    inputs,before,calibrated,ready=setup()
    policy=InitialBattleObservation(OWNER,deadline=100,clock=lambda:0)
    assert policy.step(snapshot(before,battle_hp_info_available=True),inputs.snapshot()).action==Action.INSPECT_BATTLE_HP
    assert policy.step(snapshot(before,battle_hp_info_available=True),inputs.snapshot()).action==Action.WAIT
    assert policy.step(snapshot(calibrated,battle_log_available=True),inputs.snapshot()).action==Action.INSPECT_BATTLE_LOG
    assert policy.step(PageSnapshot(Observation(Scene.UNKNOWN,battle_log_pending=True)),inputs.snapshot()).action==Action.WAIT
    decision=policy.step(snapshot(ready),inputs.snapshot())
    assert decision.action==Action.READY
    assert policy.step(snapshot(before),inputs.snapshot())==decision
    assert inputs.snapshot().sequence==0 and inputs.pending is None


@pytest.mark.parametrize('change',[{'sequence':1},{'expired':True},{'battle_id':'other'},{'sequence':False}])
def test_intervening_or_foreign_input_stops_before_any_inspection(change):
    inputs,before,_,_=setup()
    policy=InitialBattleObservation(OWNER,deadline=100,clock=lambda:0)
    result=policy.step(snapshot(before,battle_hp_info_available=True),replace(inputs.snapshot(),**change))
    assert result.action==Action.STOP and result.reason=='initial_battle_input_changed'


@pytest.mark.parametrize('kind',['preexisting_calibration','foreign_session','resume_prompt','unowned_pending_log'])
def test_unowned_state_is_not_adopted(kind):
    inputs,before,calibrated,_=setup()
    current=snapshot(before,battle_hp_info_available=True)
    if kind=='preexisting_calibration':current=snapshot(calibrated)
    elif kind=='foreign_session':current=snapshot(replace(before,session=replace(OWNER,battle_id='other')))
    elif kind=='resume_prompt':current=PageSnapshot(Observation(Scene.RESUME_PROMPT))
    else:current=snapshot(before,battle_log_pending=True)
    assert InitialBattleObservation(OWNER,deadline=100,clock=lambda:0).step(current,inputs.snapshot()).action==Action.STOP


@pytest.mark.parametrize('kind',['no_round','later_frame','changed_calibration'])
def test_log_return_cannot_be_ready_without_exact_bound_first_round_and_original_calibration(kind):
    inputs,before,calibrated,ready=setup()
    policy=InitialBattleObservation(OWNER,deadline=100,clock=lambda:0)
    policy.step(snapshot(before,battle_hp_info_available=True),inputs.snapshot())
    policy.step(snapshot(calibrated,battle_log_available=True),inputs.snapshot())
    if kind=='no_round':ready=replace(ready,absolute_turn=None)
    elif kind=='later_frame':ready=replace(ready,values=replace(ready.values,frame_id='later'))
    else:
        bound=ready.maximum_hp_parameter
        ready=replace(ready,maximum_hp_parameter=replace(bound,calibration=replace(bound.calibration,request_id=999)))
    assert policy.step(snapshot(ready),inputs.snapshot()).action==Action.STOP


def test_unknown_scene_waits_but_deadline_stops_permanently():
    inputs,_,_,_=setup()
    now=[0.]
    policy=InitialBattleObservation(OWNER,deadline=5,clock=lambda:now[0])
    current=PageSnapshot(Observation(Scene.UNKNOWN))
    assert policy.step(current,inputs.snapshot()).action==Action.WAIT
    now[0]=5
    stopped=policy.step(current,inputs.snapshot())
    assert stopped.reason=='initial_battle_observation_timeout'
    now[0]=0
    assert policy.step(current,inputs.snapshot())==stopped


def test_foreground_without_program_start_is_rejected_at_construction():
    with pytest.raises(BattleInitializationError):
        InitialBattleObservation(replace(OWNER,identity_source='diagnostic_foreground_unconfirmed'),deadline=100)
