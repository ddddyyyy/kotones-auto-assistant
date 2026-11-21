"""Observe an owned input without opening the game's battle-log panel."""
from dataclasses import dataclass
from time import monotonic
from typing import Callable,Iterable
from cv2.typing import MatLike
from kaa.db.skill_card import SkillCard
from .battle_observation import BattleObservationSession
from .hud_feedback import HudFeedbackWindow,HudInputEvidence
from .input_observation import InputObservationError
from .log_hand import LogHandObservation
from .models import Action,Decision,Scene
from .page import IdolRoadPage,PageSnapshot
from .schedule_countdown import ScheduleCountdown
from .schedule_countdown_page import read_schedule_countdown
from .extra_play_page import read_extra_play
from .hud_round import resolve_hud_round


@dataclass(frozen=True)
class HudInputReturn:
    screen: MatLike
    snapshot: PageSnapshot
    evidence: HudInputEvidence | None
    battle_finished: bool = False


def observe_pending_hud(page: IdolRoadPage,session: BattleObservationSession,
                        origin_hand: LogHandObservation,frames: Iterable[MatLike],*,
                        deadline: float,observe: Callable[[MatLike,PageSnapshot,Decision],None],
                        clock: Callable[[],float]=monotonic,
                        origin_countdown: ScheduleCountdown | None=None,
                        require_round: bool=False,card: SkillCard | None=None,
                        allow_audited_redraw: bool=False) -> HudInputReturn:
    """Collect HUD evidence only; settlement and turn binding are separate.

    This loop makes no device calls and cannot fall back to opening a log.
    A completed battle returns control to results without settling the input.
    """
    inputs=page.battle_inputs
    if inputs is None or page.battle_session!=session or inputs.session!=session:
        raise InputObservationError('hud_input_owner_missing')
    window=HudFeedbackWindow(inputs.snapshot(),origin_hand,origin_countdown=origin_countdown)
    limit=min(deadline,window.pending.started+20)
    for screen in frames:
        if page.battle_session!=session or page.battle_inputs is not inputs:
            raise InputObservationError('hud_input_owner_changed')
        if clock()>=limit:
            raise InputObservationError('hud_input_observation_timeout')
        snapshot=page.observe(screen,session.character_id)
        obs=snapshot.observation
        now=clock()
        if now>=limit or inputs.snapshot()!=window.origin:
            raise InputObservationError('hud_input_changed_or_expired')
        if ((obs.character_id is not None and obs.character_id!=session.character_id)
            or (obs.stage is not None and obs.stage!=session.stage)):
            raise InputObservationError('hud_input_identity_changed')
        finished=(obs.scene in (Scene.JUDGING,Scene.RESULT)
            and (obs.character_id,obs.stage)==(session.character_id,session.stage))
        evidence=None
        if obs.battle_log_pending or obs.scene==Scene.BATTLE_LOG:
            raise InputObservationError('hud_input_unexpected_log')
        if finished:
            pass
        elif obs.scene==Scene.UNKNOWN:
            window.interrupt()
        elif obs.scene==Scene.BATTLE and snapshot.battle_frame is not None:
            current=snapshot.battle_frame
            extra=read_extra_play(screen,current.values.frame_id) if snapshot.log_hand is not None else None
            countdown=read_schedule_countdown(screen,current.values.frame_id) if snapshot.log_hand is not None else None
            now=clock()
            if now>=limit or inputs.snapshot()!=window.origin:
                raise InputObservationError('hud_input_changed_or_expired')
            if (not require_round and origin_countdown is not None and window.pending.kind=='card'
                and (countdown is None or countdown.turns_after is None)):
                # A consumption-only diagnostic may return without a round.
                # The continuous runner needs two readable current round cues;
                # keep waiting within the original deadline, never latch an
                # incomplete pair that prevents later readable frames.
                window.interrupt()
            else:
                evidence=window.observe(inputs.snapshot(),current,snapshot.log_hand,now=now,extra=extra,countdown=countdown)
            if evidence is not None and require_round:
                if (evidence.extra is None or evidence.extra.value is None
                    or resolve_hud_round(inputs.snapshot(),evidence,now=clock(),card=card,
                        allow_audited_redraw=allow_audited_redraw) is None):
                    # A round-only candidate is not ready for the next card.
                    # Keep observing under this input's original deadline;
                    # do not authorize an immediately unusable planning frame.
                    window.interrupt();evidence=None
                if clock()>=limit or inputs.snapshot()!=window.origin:
                    raise InputObservationError('hud_input_changed_or_expired')
        else:
            raise InputObservationError('hud_input_unexpected_scene')
        decision=Decision(Action.READY if evidence is not None or finished else Action.WAIT,
            'hud_input_battle_finished' if finished else 'hud_input_evidence_ready' if evidence is not None
            else 'hud_input_await_current_hud',session.character_id,session.stage)
        observe(screen,snapshot,decision)
        if evidence is not None or finished:
            return HudInputReturn(screen,snapshot,evidence,finished)
    raise InputObservationError('hud_input_frames_exhausted')
