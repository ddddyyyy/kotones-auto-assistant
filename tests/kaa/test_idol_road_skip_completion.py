"""Timely SKIP settlement and same-capture planning handoff, without OCR or device IO."""
from dataclasses import replace
from hashlib import sha256
import numpy as np
import pytest
from kaa.idol_road.skip_completion import complete_skip_return,SkipPlanningCompletionError
from kaa.idol_road.battle_observation import bind_battle_frame,current_absolute_turn,current_feedback_turn
from kaa.idol_road.hand_context import CapturedHandContext
from kaa.idol_road.log_hand import LogHandCard,confirm_log_hand
from kaa.idol_road.models import Action
from kaa.idol_road.page import IdolRoadPage
from kaa.idol_road.round_anchor import InitialLogAnchor,InitialRoundTracker,RoundAnchorState
from kaa.tasks.produce.new.play_cards.page import LessonBattleContext
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from tests.kaa.test_idol_road_input_observation import scenario


def fixture(monkeypatch):
    import kaa.idol_road.skip_completion as module
    policy,inputs,now,snapshot,closed=scenario('skip',feedback_only=True)
    assert policy.step(snapshot,inputs.snapshot()).action==Action.READY
    frame=snapshot.battle_frame
    resolution=snapshot.observation.battle_skip_resolution
    assert resolution.settled_at==0 and inputs.pending is None
    hand=confirm_log_hand(frame.values.frame_id,1,(LogHandCard('card',0,True,(10,700,100,150),False),))
    returned=snapshot.observation.battle_log_return
    returned=replace(returned,request=replace(returned.request,hand_fingerprint=hand.fingerprint))
    raw=replace(frame,absolute_turn=None,observation=replace(frame.observation,turn=None,turn_source=None))
    rounds=InitialRoundTracker(frame.session)
    anchor=InitialLogAnchor(frame.session.battle_id,frame.values.frame_id,resolution.request_id,
        resolution.log_frame_id,inputs.sequence,frame.absolute_turn,raw,hand.fingerprint,'owned_skip_log_return')
    rounds.state=RoundAnchorState(anchor,'owned_skip_return_confirmed')
    pixels=np.zeros((1280,720,3),dtype='uint8')
    snapshot.observation=replace(snapshot.observation,battle_input_state=inputs.snapshot(),
        battle_round_anchor=rounds.state,battle_log_hand=hand,battle_log_return=returned)
    snapshot.log_hand=hand
    snapshot.log_view=closed.observation.battle_log_inspection.viewport
    snapshot.capture_digest=sha256(pixels.tobytes()).hexdigest()
    snapshot.hand_context=CapturedHandContext(frame.session.battle_id,frame.values.frame_id,LessonBattleContext(pixels))
    page=object.__new__(IdolRoadPage)
    page._battle_session=frame.session
    page._battle_inputs=inputs
    page._round_tracker=rounds
    page._maximum_hp=None
    reads=[]
    def details(image,current):
        reads.append(image)
        return bind_battle_frame(current.session,replace(current.values,detail_mode='complete'))
    monkeypatch.setattr(module,'_read_details',details)
    return page,pixels,snapshot,now,reads


def test_timely_skip_partial_state_never_selects_and_completes_same_capture_after_twenty_seconds(monkeypatch):
    page,pixels,source,now,reads=fixture(monkeypatch)
    partial=source.battle_frame
    state=BattleState(19,49,0,absolute_turn_observation=partial.absolute_turn)
    assert current_absolute_turn(partial) is None and current_feedback_turn(partial).turn==2
    assert not page.battle_inputs.can_select(partial,state)
    now[0]=21
    result=complete_skip_return(page,pixels,source,deadline=100)
    assert result.settled_at==0 and len(reads)==1 and reads[0] is pixels
    full=result.snapshot.battle_frame
    assert current_absolute_turn(full).turn==2 and full.values.frame_id==partial.values.frame_id
    assert page.battle_inputs.can_select(full,state)
    assert source.battle_frame.values.detail_mode=='feedback_only'
    assert result.snapshot.observation.battle_skip_resolution is None
    with pytest.raises(SkipPlanningCompletionError):complete_skip_return(page,pixels,source,deadline=100)


@pytest.mark.parametrize('change',['late_receipt','missing_receipt','foreign_request','missing_log','wrong_hand','capture'])
def test_invalid_receipt_or_capture_fails_before_reading_details(monkeypatch,change):
    page,pixels,source,now,reads=fixture(monkeypatch)
    obs=source.observation
    if change=='late_receipt':source.observation=replace(obs,battle_skip_resolution=replace(obs.battle_skip_resolution,settled_at=20))
    elif change=='missing_receipt':source.observation=replace(obs,battle_skip_resolution=None)
    elif change=='foreign_request':
        returned=obs.battle_log_return
        source.observation=replace(obs,battle_log_return=replace(returned,request=replace(returned.request,request_id='foreign')))
    elif change=='missing_log':source.log_view=None
    elif change=='wrong_hand':source.log_hand=replace(source.log_hand,fingerprint='fake')
    elif change=='capture':pixels[0,0]=255
    with pytest.raises(SkipPlanningCompletionError):complete_skip_return(page,pixels,source,deadline=100)
    assert reads==[] and page.battle_inputs.pending is None


def test_global_deadline_is_not_extended_by_a_timely_old_receipt(monkeypatch):
    page,pixels,source,now,reads=fixture(monkeypatch)
    now[0]=100
    with pytest.raises(SkipPlanningCompletionError):complete_skip_return(page,pixels,source,deadline=100)
    assert reads==[]


def test_pixels_changed_during_completion_cannot_be_used_for_planning(monkeypatch):
    import kaa.idol_road.skip_completion as module
    page,pixels,source,now,reads=fixture(monkeypatch)
    original=module._read_details
    def changed(image,frame):
        full=original(image,frame)
        image[0,0]=255
        return full
    monkeypatch.setattr(module,'_read_details',changed)
    with pytest.raises(SkipPlanningCompletionError):complete_skip_return(page,pixels,source,deadline=100)
    assert current_absolute_turn(source.battle_frame) is None
