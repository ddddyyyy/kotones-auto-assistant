"""Read planning details from a timely settled card's exact original capture."""
from dataclasses import dataclass,replace
from hashlib import sha256
from typing import Literal
from cv2.typing import MatLike
from .battle_observation import BattleFrameObservation,current_absolute_turn
from .battle_inputs import BattleInputState
from .card_feedback import CardHandFeedback
from .card_probe import card_feedback_ready
from .planning_details import read_planning_details as _read_details
from .page import IdolRoadPage,PageSnapshot
from .models import Scene
from kaa.tasks.produce.new.play_cards.planner.feedback import PlayFeedbackStatus


class CardPlanningCompletionError(ValueError):
    pass


@dataclass(frozen=True)
class CompletedCardReturn:
    snapshot: PageSnapshot
    receipt: CardHandFeedback
    source_frame: BattleFrameObservation
    source_input_state: BattleInputState
    capture_digest: str
    scope: Literal['planning_details_from_same_settled_capture'] = 'planning_details_from_same_settled_capture'


def complete_card_return(page: IdolRoadPage,screen: MatLike,snapshot: PageSnapshot,
                         receipt: CardHandFeedback,*,deadline: float) -> CompletedCardReturn:
    """Create a new full packet; preserve the captured pending-state metadata.

    This follows settlement, not its twenty-second deadline. It cannot settle
    an expired request, take another screenshot, or authorize a partial state.
    """
    inputs,rounds=page.battle_inputs,page.battle_rounds
    frame=snapshot.battle_frame
    captured=snapshot.observation.battle_input_state
    resolution=snapshot.observation.battle_card_resolution
    returned=snapshot.observation.battle_log_return
    permit=receipt.permit
    from .battle_observation import current_feedback_turn
    observed=current_feedback_turn(frame) if frame is not None else None
    advanced=(permit is not None and resolution is not None and resolution.status=='next_round_observed'
        and observed is not None and observed.turn==permit.pending.turn+1
        and receipt.feedback.status is PlayFeedbackStatus.UNCONFIRMED
        and receipt.feedback.evidence=='turn_advanced_without_play_observation')
    anchor=rounds.state.anchor if rounds is not None else None
    if (inputs is None or rounds is None or frame is None or captured is None or permit is None
        or inputs.clock()>=deadline or page.battle_session!=frame.session or inputs.session!=frame.session
        or snapshot.observation.scene!=Scene.BATTLE or snapshot.observation.battle_log_pending
        or anchor is None or anchor.source!='owned_card_log_return'
        or anchor.observed_frame.values.detail_mode!='feedback_only'
        or frame.values.detail_mode!='feedback_only' or snapshot.hand_context is None
        or inputs.pending is not None or inputs.sequence!=receipt.input_sequence
        or receipt.input_sequence!=permit.sequence or receipt.source!='owned_return_complete_hand'
        or receipt.frame_id!=frame.values.frame_id or receipt.settled_at is None
        or not permit.pending.started<=receipt.settled_at<permit.deadline
        or not (advanced or (receipt.feedback.evidence=='card_left_hand'
            and receipt.feedback.status is PlayFeedbackStatus.CONFIRMED))
        or snapshot.log_hand is None or snapshot.log_hand.frame_id!=frame.values.frame_id
        or snapshot.observation.battle_log_hand!=snapshot.log_hand
        or receipt.hand_fingerprint!=snapshot.log_hand.fingerprint
        or resolution is None or resolution.permit!=permit
        or anchor.request_id!=resolution.request_id or anchor.return_frame_id!=frame.values.frame_id
        or returned is None or returned.source!='owned_log_return_candidate'
        or returned.request.card_probe!=permit or returned.request.request_id!=resolution.request_id
        or returned.return_frame_id!=frame.values.frame_id or returned.request.session!=frame.session
        or not returned.hud_unchanged or not returned.remaining_consistent or not returned.input_sequence_consistent
        or returned.request.hand_fingerprint!=receipt.hand_fingerprint
        or not card_feedback_ready(frame.session,captured,frame,resolution,now=receipt.settled_at)
        or not snapshot.capture_digest or snapshot.capture_digest!=receipt.capture_digest):
        raise CardPlanningCompletionError('card_planning_source_unconfirmed')
    try:
        snapshot.hand_context.validate(screen,frame)
    except ValueError as error:
        raise CardPlanningCompletionError('card_planning_context_changed') from error
    digest=sha256(screen.tobytes()).hexdigest()
    if screen.shape!=(1280,720,3) or digest!=receipt.capture_digest:
        raise CardPlanningCompletionError('card_planning_capture_changed')
    idle=inputs.snapshot()
    full=_read_details(screen,frame)
    if (full is None or inputs.snapshot()!=idle or page.battle_inputs is not inputs
        or page.battle_rounds is not rounds or page.battle_session!=frame.session
        or inputs.clock()>=deadline or sha256(screen.tobytes()).hexdigest()!=digest):
        raise CardPlanningCompletionError('card_planning_completion_changed_or_timeout')
    maximum=page.battle_maximum_hp
    if maximum is not None:
        maximum.observe(full,idle)
        full=maximum.bind(full,idle)
    if not rounds.complete_card_details(frame,full,idle):
        raise CardPlanningCompletionError('card_planning_anchor_changed')
    full=rounds.bind(full,idle,entry=snapshot.log_entry,hand=snapshot.log_hand)
    if current_absolute_turn(full) is None:
        raise CardPlanningCompletionError('card_planning_round_unconfirmed')
    observation=replace(snapshot.observation,battle_state=full.observation,battle_input_state=idle,
        battle_round_anchor=rounds.state,battle_card_resolution=None)
    completed=PageSnapshot(observation,dict(snapshot.targets),full,snapshot.log_entry,snapshot.log_hand,
        snapshot.hand_context,capture_digest=digest)
    return CompletedCardReturn(completed,receipt,frame,captured,digest)
