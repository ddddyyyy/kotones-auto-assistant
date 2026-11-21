"""Finish planning from one timely settled HUD capture, without another input."""
from dataclasses import dataclass,replace
from hashlib import sha256
from cv2.typing import MatLike
from kaa.battle_turn import AbsoluteTurnObservation
from kaa.db.skill_card import SkillCard
from kaa.tasks.produce.new.play_cards.planner.feedback import PlayFeedback,PlayFeedbackStatus
from .battle_observation import current_absolute_turn
from .hud_feedback import HudInputEvidence
from .hud_round import HudRoundResolution,resolve_hud_round
from .planning_details import read_planning_details
from .page import IdolRoadPage,PageSnapshot
from .models import Scene


@dataclass(frozen=True)
class CompletedHudReturn:
    snapshot: PageSnapshot
    resolution: HudRoundResolution
    settled_at: float
    feedback: PlayFeedback | None


def complete_hud_return(page: IdolRoadPage,screen: MatLike,snapshot: PageSnapshot,
                         evidence: HudInputEvidence,*,deadline: float,card: SkillCard | None=None,
                         allow_audited_redraw: bool=False) -> CompletedHudReturn:
    inputs=page.battle_inputs;frame=snapshot.battle_frame
    if (inputs is None or frame is None or page.battle_session!=frame.session
        or snapshot.observation.scene!=Scene.BATTLE or snapshot.observation.battle_log_pending
        or snapshot.battle_frame!=evidence.frame or snapshot.log_hand!=evidence.hand
        or snapshot.hand_context is None or frame.values.detail_mode!='feedback_only'
        or inputs.clock()>=deadline or not snapshot.capture_digest or screen.shape!=(1280,720,3)):
        raise ValueError('hud_planning_capture_unconfirmed')
    digest=sha256(screen.tobytes()).hexdigest()
    if digest!=snapshot.capture_digest:raise ValueError('hud_planning_capture_changed')
    snapshot.hand_context.validate(screen,frame)
    resolution=resolve_hud_round(inputs.snapshot(),evidence,now=inputs.clock(),card=card,
        allow_audited_redraw=allow_audited_redraw)
    if resolution is None:
        # Keep consumption and round uncertainty separate. This grants no next input.
        if evidence.pending.kind=='card':inputs.finish_hud_card(evidence)
        raise ValueError('hud_round_unconfirmed')
    if not inputs.finish_hud_resolution(resolution,card=card,allow_audited_redraw=allow_audited_redraw):
        raise ValueError('hud_input_settlement_unconfirmed')
    settled=inputs.hud_settled_at
    assert settled is not None
    idle=inputs.snapshot()
    full=read_planning_details(screen,frame)
    if (full is None or inputs.snapshot()!=idle or page.battle_inputs is not inputs
        or inputs.clock()>=deadline or sha256(screen.tobytes()).hexdigest()!=digest):
        raise ValueError('hud_planning_completion_changed_or_expired')
    maximum=page.battle_maximum_hp
    if maximum is not None:
        maximum.observe(full,idle);full=maximum.bind(full,idle)
    remaining=full.values.remaining_turns
    if type(remaining) is not int:raise ValueError('hud_remaining_unconfirmed')
    full=replace(full,absolute_turn=AbsoluteTurnObservation(full.session.battle_id,full.values.frame_id,resolution.turn,remaining),
        observation=replace(full.observation,turn=resolution.turn,turn_source=resolution.source))
    if current_absolute_turn(full) is None:raise ValueError('hud_completed_round_unconfirmed')
    page.accept_hud_return(snapshot,full,resolution)
    observation=replace(snapshot.observation,battle_state=full.observation,battle_input_state=inputs.snapshot(),
        battle_round_anchor=None,battle_card_resolution=None,battle_skip_resolution=None,battle_log_return=None)
    completed=replace(snapshot,observation=observation,battle_frame=full)
    feedback=None
    if evidence.pending.kind=='card':
        feedback=PlayFeedback(PlayFeedbackStatus.CONFIRMED,'card_left_hand') if resolution.play_confirmed else (
            PlayFeedback(PlayFeedbackStatus.UNCONFIRMED,'turn_advanced_without_play_observation'))
    return CompletedHudReturn(completed,resolution,settled,feedback)
