from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
import numpy as np
import pytest
from kaa.idol_road import hud_completion
from kaa.idol_road.page import IdolRoadPage,PageSnapshot
from kaa.idol_road.battle_log import BattleLogBinding
from kaa.idol_road.hand_context import CapturedHandContext
from kaa.idol_road.models import Observation,Scene
from kaa.idol_road.battle_observation import current_absolute_turn
from tests.kaa.test_idol_road_hud_round import prepared,mock_root,SESSION


def packet(monkeypatch,kind='card'):
    card=mock_root(monkeypatch);inputs,now,evidence=prepared(kind)
    screen=np.zeros((1280,720,3),dtype='uint8')
    frame=evidence.frame
    snapshot=PageSnapshot(Observation(Scene.BATTLE,character_id='ttmr',stage=20,battle_input_state=inputs.snapshot()),{},
        battle_frame=frame,log_hand=evidence.hand,hand_context=CapturedHandContext(SESSION.battle_id,frame.values.frame_id,
            SimpleNamespace(fetch_screen=lambda:screen)),capture_digest=sha256(screen.tobytes()).hexdigest())
    page=IdolRoadPage.__new__(IdolRoadPage);page._battle_session=SESSION;page._battle_inputs=inputs
    page._feedback_mode='hud';page._hud_capture_frame=frame;page._hud_capture_hand=evidence.hand
    page._log_binding=BattleLogBinding(clock=lambda:now[0]);page._maximum_hp=None
    def details(_screen,current):
        return replace(current,values=replace(current.values,detail_mode='complete'))
    monkeypatch.setattr(hud_completion,'read_planning_details',details)
    return page,inputs,now,screen,snapshot,evidence,card,details


def test_timely_hud_consumption_and_same_capture_details_authorize_next_frame(monkeypatch):
    page,inputs,_,screen,snapshot,evidence,card,_=packet(monkeypatch)
    completed=hud_completion.complete_hud_return(page,screen,snapshot,evidence,deadline=100,card=card)
    assert current_absolute_turn(completed.snapshot.battle_frame).turn==1
    assert not inputs.hud_round_pending and inputs.pending is None
    assert completed.snapshot.battle_frame.values.detail_mode=='complete'
    assert page._hud_authorization==(completed.snapshot.battle_frame,inputs.sequence)
    assert snapshot.battle_frame.values.detail_mode=='feedback_only'
    with pytest.raises(ValueError):hud_completion.complete_hud_return(page,screen,snapshot,evidence,deadline=100,card=card)


def test_planning_may_finish_after_input_deadline_only_if_settled_before_it(monkeypatch):
    page,inputs,now,screen,snapshot,evidence,card,details=packet(monkeypatch)
    def slow(*args):
        now[0]=35.
        return details(*args)
    monkeypatch.setattr(hud_completion,'read_planning_details',slow)
    completed=hud_completion.complete_hud_return(page,screen,snapshot,evidence,deadline=100,card=card)
    assert completed.settled_at==12 and not inputs.hud_round_pending


def test_changed_pixels_cannot_complete_or_authorize_hud_frame(monkeypatch):
    page,inputs,_,screen,snapshot,evidence,card,_=packet(monkeypatch);screen[0,0]=1
    with pytest.raises(ValueError,match='capture_changed'):
        hud_completion.complete_hud_return(page,screen,snapshot,evidence,deadline=100,card=card)
    assert inputs.pending is not None


def test_failed_planning_keeps_round_pending_and_history_hidden(monkeypatch):
    page,inputs,_,screen,snapshot,evidence,card,_=packet(monkeypatch)
    monkeypatch.setattr(hud_completion,'read_planning_details',lambda *_:None)
    with pytest.raises(ValueError):hud_completion.complete_hud_return(page,screen,snapshot,evidence,deadline=100,card=card)
    assert inputs.pending is None and inputs.hud_round_pending and inputs.last_owned_card_play is None
