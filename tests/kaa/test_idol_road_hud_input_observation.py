from types import SimpleNamespace
import numpy as np
import pytest
from kaa.idol_road.hud_input_observation import observe_pending_hud
from kaa.idol_road.input_observation import InputObservationError
from kaa.idol_road.models import Observation,Scene,Action
from kaa.idol_road.page import PageSnapshot
from tests.kaa.test_idol_road_hud_feedback import SESSION,scenario,frame,hand


def page_for(inputs,snapshots):
    stream=iter(snapshots)
    ledger=SimpleNamespace(session=SESSION,snapshot=lambda:inputs)
    # There is deliberately no execute/device interface on this test double.
    return SimpleNamespace(battle_inputs=ledger,battle_session=SESSION,observe=lambda *_:next(stream))


def packet(name):
    return PageSnapshot(Observation(Scene.BATTLE,character_id='ttmr',stage=20),{},
        battle_frame=frame(name),log_hand=hand(name,('other',)))


def test_hud_loop_returns_evidence_without_any_log_or_device_actions():
    inputs,_=scenario()
    page=page_for(inputs,[packet('one'),packet('two')])
    decisions=[]
    result=observe_pending_hud(page,SESSION,hand('before',('card','other')),
        [np.zeros((1,1)),np.zeros((1,1))],deadline=100,clock=lambda:11,
        observe=lambda _s,_p,d:decisions.append(d))
    assert result.evidence.change=='requested_card_removed'
    assert [d.action for d in decisions]==[Action.WAIT,Action.READY]
    assert inputs.pending is not None and not result.battle_finished


def test_log_panel_is_rejected_instead_of_becoming_automatic_fallback():
    inputs,_=scenario()
    page=page_for(inputs,[PageSnapshot(Observation(Scene.BATTLE_LOG),{})])
    with pytest.raises(InputObservationError,match='unexpected_log'):
        observe_pending_hud(page,SESSION,hand('before',('card','other')),[np.zeros((1,1))],
            deadline=100,clock=lambda:11,observe=lambda *_:None)


def test_owned_result_returns_without_claiming_pending_card_success():
    inputs,_=scenario()
    page=page_for(inputs,[PageSnapshot(Observation(Scene.RESULT,character_id='ttmr',stage=20),{})])
    result=observe_pending_hud(page,SESSION,hand('before',('card','other')),[np.zeros((1,1))],
        deadline=100,clock=lambda:11,observe=lambda *_:None)
    assert result.battle_finished and result.evidence is None and inputs.pending is not None


def test_consumption_diagnostic_can_require_two_preview_cues_when_supplied(monkeypatch):
    from kaa.idol_road.schedule_countdown import confirm_schedule_countdown
    from tests.kaa.test_idol_road_schedule_countdown import reads,BOX
    def cue(name):return confirm_schedule_countdown(name,BOX,reads(frame=name))
    inputs,_=scenario()
    names=('unclear_one','unclear_two','clear_one','clear_two')
    page=page_for(inputs,[packet(n) for n in names])
    monkeypatch.setattr('kaa.idol_road.hud_input_observation.read_schedule_countdown',
        lambda _s,f:None if f.startswith('unclear') else cue(f))
    decisions=[]
    result=observe_pending_hud(page,SESSION,hand('before',('card','other')),
        [np.zeros((1,1)) for _ in names],deadline=100,clock=lambda:11,
        origin_countdown=cue('before'),observe=lambda _s,_p,d:decisions.append(d))
    assert [d.action for d in decisions]==[Action.WAIT]*3+[Action.READY]
    assert result.evidence.first_countdown.frame_id=='clear_one'
    assert result.evidence.countdown.frame_id=='clear_two'


def test_continuous_runner_waits_for_observed_play_controls_without_requiring_preview(monkeypatch):
    from tests.kaa.test_idol_road_hud_round import prepared,mock_root,extra
    inputs,_,_=prepared();state=inputs.snapshot();card=mock_root(monkeypatch)
    names=('unclear_one','unclear_two','clear_one','clear_two')
    page=page_for(state,[packet(n) for n in names])
    monkeypatch.setattr('kaa.idol_road.hud_input_observation.read_schedule_countdown',lambda *_:None)
    monkeypatch.setattr('kaa.idol_road.hud_input_observation.read_extra_play',
        lambda _s,f:None if f.startswith('unclear') else extra(f,0))
    decisions=[]
    result=observe_pending_hud(page,SESSION,hand('before',('card','other')),
        [np.zeros((1,1)) for _ in names],deadline=100,clock=lambda:12,
        require_round=True,card=card,observe=lambda _s,_p,d:decisions.append(d))
    assert [d.action for d in decisions]==[Action.WAIT]*3+[Action.READY]
    assert result.evidence.frame.values.frame_id=='clear_two'
    assert result.evidence.countdown is None and state.pending is not None


def test_confirmed_natural_round_waits_for_usable_next_play_count_under_original_deadline(monkeypatch):
    from dataclasses import replace
    from tests.kaa.test_idol_road_hud_round import prepared,mock_root,extra
    inputs,_,_=prepared(before_extra=0,remaining=19);state=inputs.snapshot();card=mock_root(monkeypatch)
    names=('unclear_one','unclear_two','clear_one','clear_two')
    page=page_for(state,[replace(packet(name),battle_frame=frame(name,19)) for name in names])
    monkeypatch.setattr('kaa.idol_road.hud_input_observation.read_schedule_countdown',lambda *_:None)
    monkeypatch.setattr('kaa.idol_road.hud_input_observation.read_extra_play',
        lambda _s,name:None if name.startswith('unclear') else extra(name,0))
    decisions=[]
    result=observe_pending_hud(page,SESSION,hand('before',('card','other')),
        [np.zeros((1,1)) for _ in names],deadline=100,clock=lambda:12,require_round=True,
        card=card,observe=lambda _s,_p,decision:decisions.append(decision))
    assert [decision.action for decision in decisions]==[Action.WAIT]*3+[Action.READY]
    assert result.evidence.first_frame.values.frame_id=='clear_one'
    assert result.evidence.extra.value==0 and state.pending is not None
