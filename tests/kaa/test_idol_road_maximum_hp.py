from dataclasses import replace
import pytest
from kaa.idol_road.maximum_hp import MaximumHpTracker,confirmed_maximum_hp
from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame
from kaa.idol_road.battle_inputs import BattleInputs
from kaa.idol_road.hp_ratio import HpRatioRead,confirm_hp_ratio


def session(key='battle'):
    return BattleObservationSession(key,'ttmr',20,20,'confirmed_formation_start_request')


def frame(owner,name='frame',hp=50,maximum=None,mode='complete'):
    ratio=confirm_hp_ratio(name,(10,20,50,20),.99,tuple(HpRatioRead(name,kind,f'{hp}/{maximum}',.99)
        for kind in ('original','grayscale'))) if maximum is not None else None
    return bind_battle_frame(owner,BattleFrameValues(name,True,20,hp,hp_ratio=ratio,detail_mode=mode))


def calibrated():
    owner=session()
    ledger=BattleInputs(owner,clock=lambda:0)
    tracker=MaximumHpTracker(owner)
    tracker.request(frame(owner,'opening'),ledger.snapshot())
    tracker.observe(frame(owner,'tooltip',maximum=53),ledger.snapshot())
    assert tracker.calibration is not None
    return owner,ledger,tracker


def test_mode_constant_parameter_does_not_copy_old_current_hp_or_raw_observation():
    owner,ledger,tracker=calibrated()
    current=frame(owner,'later',hp=41)
    bound=tracker.bind(current,ledger.snapshot())
    assert confirmed_maximum_hp(bound)==53
    assert bound.observation.hp==41 and bound.observation.max_hp is None and bound.values.hp_ratio is None
    assert bound.maximum_hp_parameter.calibration.source_frame_id=='tooltip'
    assert bound.maximum_hp_parameter.current_frame_id=='later'


def test_unsolicited_old_or_same_frame_tooltip_cannot_calibrate():
    owner=session()
    ledger=BattleInputs(owner,clock=lambda:0)
    tracker=MaximumHpTracker(owner)
    tracker.observe(frame(owner,'external',maximum=53),ledger.snapshot())
    assert tracker.calibration is None
    current=frame(owner,'same',maximum=53)
    tracker.request(current,ledger.snapshot())
    tracker.observe(current,ledger.snapshot())
    assert tracker.calibration is None


def test_pending_input_cannot_request_and_intervening_input_cancels_capture():
    owner=session()
    ledger=BattleInputs(owner,clock=lambda:0)
    tracker=MaximumHpTracker(owner)
    current=frame(owner)
    tracker.request(current,ledger.snapshot())
    ledger.begin('skip',current)
    tracker.observe(frame(owner,'late-tooltip',maximum=53),ledger.snapshot())
    assert tracker.calibration is None
    with pytest.raises(ValueError):
        tracker.request(current,ledger.snapshot())


def test_same_battle_inputs_keep_constant_parameter_but_sequence_rollback_revokes_it():
    owner,ledger,tracker=calibrated()
    inputs=replace(ledger.snapshot(),sequence=3)
    current=frame(owner,'after-input',hp=39)
    tracker.observe(current,inputs)
    assert confirmed_maximum_hp(tracker.bind(current,inputs))==53
    tracker.observe(current,ledger.snapshot())
    assert tracker.invalid_reason and tracker.bind(current,inputs).maximum_hp_parameter is None


@pytest.mark.parametrize('kind',['changed_maximum','hp_exceeds_maximum','different_session','conflicting_ratio'])
def test_contradiction_or_wrong_battle_permanently_revokes_parameter(kind):
    owner,ledger,tracker=calibrated()
    current=frame(owner,'later')
    if kind=='changed_maximum':
        current=frame(owner,'later',maximum=60)
    elif kind=='hp_exceeds_maximum':
        current=frame(owner,'later',hp=54)
    elif kind=='different_session':
        current=frame(session('other'),'later')
    else:
        current=frame(owner,'later',maximum=53)
        bad=replace(current.values.hp_ratio,reads=(current.values.hp_ratio.reads[0],replace(current.values.hp_ratio.reads[1],text='50/60')))
        current=replace(current,values=replace(current.values,hp_ratio=bad))
    tracker.observe(current,ledger.snapshot())
    assert tracker.invalid_reason and tracker.calibration is None
    assert tracker.bind(frame(owner,'restored'),ledger.snapshot()).maximum_hp_parameter is None


@pytest.mark.parametrize('change',['value','source','source_frame','current_frame','sequence'])
def test_parameter_delivery_revalidates_all_provenance(change):
    owner,ledger,tracker=calibrated()
    current=tracker.bind(frame(owner,'now'),ledger.snapshot())
    bound=current.maximum_hp_parameter
    cal=bound.calibration
    if change=='value':cal=replace(cal,max_hp=99)
    elif change=='source':cal=replace(cal,source='inferred_full_bar')
    elif change=='source_frame':cal=replace(cal,source_frame_id='old')
    elif change=='current_frame':bound=replace(bound,current_frame_id='other')
    else:bound=replace(bound,input_sequence=-1)
    current=replace(current,maximum_hp_parameter=replace(bound,calibration=cal))
    assert confirmed_maximum_hp(current) is None


def test_probe_only_frame_never_receives_a_maximum_parameter():
    owner,ledger,tracker=calibrated()
    current=frame(owner,'probe',mode='core_only')
    assert tracker.bind(current,ledger.snapshot()).maximum_hp_parameter is None


def test_context_consumes_parameter_but_still_requires_current_battle_frame():
    import numpy as np
    from kaa.idol_road.planner_context import IdolRoadBattleContext
    owner,ledger,tracker=calibrated()
    bound=tracker.bind(frame(owner,'now',hp=41),ledger.snapshot())
    context=IdolRoadBattleContext(np.zeros((1280,720,3),dtype='uint8'),bound)
    assert context.fetch_exam_max_hp()==53
    assert context._road_frame.observation.hp==41


def test_maximum_parameter_does_not_grant_a_missing_variable_round_anchor():
    import numpy as np
    from kaa.idol_road.battle_observation import current_absolute_turn
    from kaa.idol_road.planner_context import IdolRoadBattleContext
    owner=replace(session(),turn_budget_mode='variable_budget')
    ledger=BattleInputs(owner,clock=lambda:0)
    tracker=MaximumHpTracker(owner)
    tracker.request(frame(owner,'opening'),ledger.snapshot())
    tracker.observe(frame(owner,'tooltip',maximum=53),ledger.snapshot())
    bound=tracker.bind(frame(owner,'fresh-hud'),ledger.snapshot())
    assert confirmed_maximum_hp(bound)==53 and current_absolute_turn(bound) is None
    context=IdolRoadBattleContext(np.zeros((1280,720,3),dtype='uint8'),bound)
    with pytest.raises(ValueError,match='当前回合'):
        context.fetch_exam_max_hp()
