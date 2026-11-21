"""Complete planning only after a timely owned SKIP receipt, using its exact capture."""
from dataclasses import dataclass,replace
from hashlib import sha256
from typing import TYPE_CHECKING,Literal
from cv2.typing import MatLike
from .battle_inputs import BattleInputState
from .battle_observation import BattleFrameObservation,current_absolute_turn,current_feedback_turn
from .models import Scene
from .planning_details import read_planning_details as _read_details
from .skip_probe import assess_skip_log,pending_skip_permit
if TYPE_CHECKING:
    from .page import IdolRoadPage,PageSnapshot


class SkipPlanningCompletionError(ValueError):
    pass


@dataclass(frozen=True)
class CompletedSkipReturn:
    snapshot: 'PageSnapshot'
    source_frame: BattleFrameObservation
    source_input_state: BattleInputState
    capture_digest: str
    settled_at: float
    scope: Literal['planning_details_from_same_settled_skip_capture']='planning_details_from_same_settled_skip_capture'


def complete_skip_return(page: 'IdolRoadPage',screen: MatLike,snapshot: 'PageSnapshot',*,
                          deadline: float) -> CompletedSkipReturn:
    from .page import PageSnapshot
    inputs,rounds=page.battle_inputs,page.battle_rounds
    frame=snapshot.battle_frame
    captured=snapshot.observation.battle_input_state
    resolution=snapshot.observation.battle_skip_resolution
    returned=snapshot.observation.battle_log_return
    view=snapshot.log_view
    anchor=rounds.state.anchor if rounds is not None else None
    absolute=current_feedback_turn(frame) if frame is not None else None
    if (inputs is None or rounds is None or frame is None or captured is None or resolution is None
        or returned is None or view is None or anchor is None or absolute is None
        or page.battle_session!=frame.session or inputs.session!=frame.session
        or inputs.clock()>=deadline or inputs.pending is not None or captured!=inputs.snapshot()
        or snapshot.observation.scene!=Scene.BATTLE or snapshot.observation.battle_log_pending
        or frame.values.detail_mode!='feedback_only' or snapshot.hand_context is None
        or resolution.status!='next_round_confirmed' or resolution.settled_at is None
        or not resolution.permit.pending.started<=resolution.settled_at<resolution.permit.deadline
        or inputs.sequence!=resolution.permit.sequence or anchor.input_sequence!=inputs.sequence
        or anchor.source!='owned_skip_log_return' or anchor.feedback_pending is not None
        or anchor.request_id!=resolution.request_id or anchor.return_frame_id!=frame.values.frame_id
        or absolute.turn!=resolution.turn or absolute.turn!=resolution.permit.pending.turn+1
        or absolute.remaining_turns!=resolution.remaining_turns
        or snapshot.log_hand is None or snapshot.observation.battle_log_hand!=snapshot.log_hand
        or snapshot.log_hand.frame_id!=frame.values.frame_id
        or snapshot.log_hand.fingerprint!=returned.request.hand_fingerprint
        or not snapshot.capture_digest):
        raise SkipPlanningCompletionError('skip_planning_source_unconfirmed')
    raw=replace(frame,absolute_turn=None,observation=replace(frame.observation,turn=None,turn_source=None))
    origin=BattleInputState(inputs.session.battle_id,inputs.sequence,resolution.permit.pending,False)
    if (anchor.observed_frame!=raw
        or pending_skip_permit(inputs.session,origin,now=resolution.settled_at)!=resolution.permit
        or assess_skip_log(resolution.permit,returned,view,raw,origin,now=resolution.settled_at)
            !=replace(resolution,settled_at=None)):
        raise SkipPlanningCompletionError('skip_planning_receipt_unconfirmed')
    try:
        snapshot.hand_context.validate(screen,frame)
    except ValueError as error:
        raise SkipPlanningCompletionError('skip_planning_context_changed') from error
    digest=sha256(screen.tobytes()).hexdigest()
    if screen.shape!=(1280,720,3) or digest!=snapshot.capture_digest:
        raise SkipPlanningCompletionError('skip_planning_capture_changed')
    idle=inputs.snapshot()
    full=_read_details(screen,frame)
    if (full is None or inputs.snapshot()!=idle or page.battle_inputs is not inputs
        or page.battle_rounds is not rounds or page.battle_session!=frame.session
        or inputs.clock()>=deadline or sha256(screen.tobytes()).hexdigest()!=digest):
        raise SkipPlanningCompletionError('skip_planning_completion_changed_or_timeout')
    maximum=page.battle_maximum_hp
    if maximum is not None:
        maximum.observe(full,idle)
        full=maximum.bind(full,idle)
    if not rounds.complete_skip_details(frame,full,idle):
        raise SkipPlanningCompletionError('skip_planning_anchor_changed')
    full=rounds.bind(full,idle,entry=snapshot.log_entry,hand=snapshot.log_hand)
    if current_absolute_turn(full) is None:
        raise SkipPlanningCompletionError('skip_planning_round_unconfirmed')
    observation=replace(snapshot.observation,battle_state=full.observation,battle_input_state=idle,
        battle_round_anchor=rounds.state,battle_skip_resolution=None)
    completed=PageSnapshot(observation,dict(snapshot.targets),full,snapshot.log_entry,snapshot.log_hand,
        snapshot.hand_context,capture_digest=digest,log_view=view)
    return CompletedSkipReturn(completed,frame,captured,digest,resolution.settled_at)
