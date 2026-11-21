from types import SimpleNamespace
import numpy as np
from kaa.tasks import R
from kaa.idol_road.initialization import InitialBattleObservation,initialize_variable_battle
from kaa.idol_road.models import Action
from tests.kaa.test_idol_road_initialization import OWNER,setup,snapshot
from tests.kaa.test_idol_road_hud_initialization import raw,hand


def packets():
    inputs,before,calibrated,ready=setup()
    tooltip=snapshot(calibrated)
    tooltip.targets[Action.CLOSE_BATTLE_HP]=R.IdolRoad.BoxBattleHpInspect
    first=snapshot(raw('one',ready));first.log_hand=hand('one')
    second=snapshot(raw('two',ready));second.log_hand=hand('two')
    return inputs,[snapshot(before,battle_hp_info_available=True),tooltip,first,second]


def test_hud_initialization_schedules_calibration_close_and_stable_hud_without_log():
    inputs,frames=packets()
    policy=InitialBattleObservation(OWNER,deadline=100,clock=lambda:0,feedback_mode='hud')
    actions=[policy.step(packet,inputs.snapshot()).action for packet in frames]
    assert actions==[Action.INSPECT_BATTLE_HP,Action.CLOSE_BATTLE_HP,Action.WAIT,Action.READY]
    assert policy.log_requested is False and policy.hud_frame.observation.turn_source=='owned_initial_hud'


def test_tooltip_that_has_not_closed_does_not_toggle_again_or_authorize_cards():
    inputs,frames=packets()
    policy=InitialBattleObservation(OWNER,deadline=100,clock=lambda:0,feedback_mode='hud')
    policy.step(frames[0],inputs.snapshot())
    assert policy.step(frames[1],inputs.snapshot()).action==Action.CLOSE_BATTLE_HP
    for _ in range(3):assert policy.step(frames[1],inputs.snapshot()).action==Action.WAIT
    assert policy.hud_frame is None


def test_initialization_callback_gets_the_bound_hud_source_before_ready_notification():
    inputs,packets_list=packets();stream=iter(packets_list);actions=[];ready=[]
    page=SimpleNamespace(battle_inputs=inputs,battle_session=OWNER,observe=lambda *_:next(stream),
        begin_hud_feedback=lambda _s:None,accept_initial_hud=lambda producer:producer.bound,
        execute=lambda _p,d:actions.append(d.action) if d.action!=Action.WAIT else None)
    def observed(_s,packet,decision):
        if decision.action==Action.READY:ready.append(packet.observation.battle_state.turn_source)
    _,result=initialize_variable_battle(page,OWNER,[np.zeros((1,1)) for _ in packets_list],
        deadline=100,observe=observed,clock=lambda:0,feedback_mode='hud')
    assert ready==['owned_initial_hud'] and result.battle_frame.observation.turn==1
    assert actions==[Action.INSPECT_BATTLE_HP,Action.CLOSE_BATTLE_HP]


def test_initial_round_comes_from_owned_start_without_requiring_a_readable_preview(monkeypatch):
    inputs,packet_list=packets()
    stream=iter(packet_list);decisions=[]
    def preview(*_):raise AssertionError('Initial turn is owned START, not preview OCR')
    monkeypatch.setattr('kaa.idol_road.schedule_countdown_page.read_schedule_countdown',preview)
    page=SimpleNamespace(battle_inputs=inputs,battle_session=OWNER,observe=lambda *_:next(stream),
        begin_hud_feedback=lambda _:None,accept_initial_hud=lambda producer:producer.bound,
        execute=lambda *_:None)
    initialize_variable_battle(page,OWNER,[np.zeros((1,1)) for _ in packet_list],deadline=100,
        observe=lambda _s,_p,d:decisions.append(d),clock=lambda:0,feedback_mode='hud')
    assert [d.action for d in decisions]==[Action.INSPECT_BATTLE_HP,Action.CLOSE_BATTLE_HP,Action.WAIT,Action.READY]
    assert inputs.sequence==0 and inputs.pending is None
