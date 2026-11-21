"""Same-capture handoff protocol; synthetic packets, no OCR or device calls."""
from dataclasses import replace,asdict
from hashlib import sha256
import numpy as np
import pytest
from kaa.idol_road.card_completion import complete_card_return,CardPlanningCompletionError
from kaa.idol_road.card_feedback import settle_card_hand_feedback
from kaa.idol_road.battle_observation import bind_battle_frame,current_absolute_turn
from kaa.idol_road.hand_context import CapturedHandContext
from kaa.idol_road.page import IdolRoadPage
from kaa.idol_road.round_anchor import InitialRoundTracker,InitialLogAnchor,RoundAnchorState
from kaa.tasks.produce.new.play_cards.page import LessonBattleContext
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from tests.kaa.test_idol_road_card_feedback import current_feedback


def fixture(monkeypatch,*,advance=False):
    import kaa.idol_road.card_completion as module
    policy,now,snapshot,pending,_=current_feedback(feedback_only=True,advance=advance)
    frame=snapshot.battle_frame
    resolution=snapshot.observation.battle_card_resolution
    rounds=InitialRoundTracker(frame.session)
    anchor=InitialLogAnchor(frame.session.battle_id,frame.values.frame_id,resolution.request_id,
        resolution.log_frame_id,policy.inputs.sequence,frame.absolute_turn,resolution.observed_frame,
        snapshot.log_hand.fingerprint,'owned_card_log_return',feedback_pending=policy.inputs.pending)
    rounds.state=RoundAnchorState(anchor,'owned_card_return_for_feedback')
    image=np.zeros((1280,720,3),dtype='uint8')
    snapshot.capture_digest=sha256(image.tobytes()).hexdigest()
    snapshot.hand_context=CapturedHandContext(frame.session.battle_id,frame.values.frame_id,LessonBattleContext(image))
    snapshot.observation=replace(snapshot.observation,battle_input_state=policy.inputs.snapshot(),battle_round_anchor=rounds.state)
    receipt=settle_card_hand_feedback(policy,snapshot,pending,allow_round_advance=advance)
    assert receipt is not None
    page=object.__new__(IdolRoadPage)
    page._battle_session=frame.session
    page._battle_inputs=policy.inputs
    page._round_tracker=rounds
    page._maximum_hp=None
    reads=[]
    def details(screen,partial):
        reads.append(screen)
        return bind_battle_frame(partial.session,replace(partial.values,detail_mode='complete'))
    monkeypatch.setattr(module,'_read_details',details)
    return page,image,snapshot,receipt,now,reads


def test_completion_keeps_original_capture_metadata_and_creates_new_idle_packet(monkeypatch):
    page,image,source,receipt,now,reads=fixture(monkeypatch)
    original=asdict(source.observation)
    now[0]=21.  # The card was settled at time zero, not retrospectively here.
    result=complete_card_return(page,image,source,receipt,deadline=100)
    full=result.snapshot.battle_frame
    assert full.values.detail_mode=='complete' and current_absolute_turn(full).turn==1
    assert full.values.frame_id==source.battle_frame.values.frame_id
    assert asdict(source.observation)==original and source.battle_frame.values.detail_mode=='feedback_only'
    assert source.observation.battle_input_state.pending is not None
    assert result.snapshot.observation.battle_input_state.pending is None
    assert result.source_input_state==source.observation.battle_input_state and result.receipt==receipt
    observed=BattleState(20,49,0,absolute_turn_observation=full.absolute_turn)
    assert page.battle_inputs.can_select(full,observed) and len(reads)==1
    with pytest.raises(CardPlanningCompletionError):complete_card_return(page,image,source,receipt,deadline=100)
    assert len(reads)==1  # Already completed packets do not repeat OCR.


@pytest.mark.parametrize('kind',['pixels','copied_image','digest','late_receipt','sequence','pending',
    'missing_return','return_source','missing_context','anchor','deadline'])
def test_unrelated_stale_or_unsettled_source_stops_before_detail_reads(monkeypatch,kind):
    page,image,source,receipt,now,reads=fixture(monkeypatch)
    if kind=='pixels':image[0,0,0]=1
    elif kind=='copied_image':image=image.copy()
    elif kind=='digest':source.capture_digest='b'*64
    elif kind=='late_receipt':receipt=replace(receipt,settled_at=receipt.permit.deadline)
    elif kind=='sequence':page.battle_inputs.sequence+=1
    elif kind=='pending':page.battle_inputs.pending=receipt.permit.pending
    elif kind=='missing_return':source.observation=replace(source.observation,battle_log_return=None)
    elif kind=='return_source':source.observation=replace(source.observation,
        battle_log_return=replace(source.observation.battle_log_return,source='historical'))
    elif kind=='missing_context':source.hand_context=None
    elif kind=='anchor':page.battle_rounds.invalidate('new_observation')
    elif kind=='deadline':now[0]=100
    before=page.battle_inputs.snapshot()
    with pytest.raises(CardPlanningCompletionError):complete_card_return(page,image,source,receipt,deadline=100)
    assert reads==[] and page.battle_inputs.snapshot()==before


@pytest.mark.parametrize('kind',['pixels','input','timeout','hp'])
def test_changes_during_detail_reads_cannot_publish_a_planning_anchor(monkeypatch,kind):
    import kaa.idol_road.card_completion as module
    page,image,source,receipt,now,_=fixture(monkeypatch)
    old_anchor=page.battle_rounds.state.anchor
    def changed(screen,partial):
        if kind=='pixels':screen[0,0,0]=1
        elif kind=='input':page.battle_inputs.sequence+=1
        elif kind=='timeout':now[0]=100
        values=replace(partial.values,detail_mode='complete',hp=48 if kind=='hp' else partial.values.hp)
        return bind_battle_frame(partial.session,values)
    monkeypatch.setattr(module,'_read_details',changed)
    with pytest.raises(CardPlanningCompletionError):complete_card_return(page,image,source,receipt,deadline=100)
    assert page.battle_rounds.state.anchor==old_anchor


def test_round_advance_completion_keeps_card_effect_unconfirmed(monkeypatch):
    page,image,source,receipt,now,reads=fixture(monkeypatch,advance=True)
    now[0]=21.
    completed=complete_card_return(page,image,source,receipt,deadline=100)
    assert current_absolute_turn(completed.snapshot.battle_frame).turn==2
    assert completed.receipt.feedback.status.value=='unconfirmed'
    assert source.observation.battle_input_state.pending is not None
    assert completed.snapshot.observation.battle_input_state.pending is None
