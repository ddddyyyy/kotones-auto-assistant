"""Observe one already registered input; never chooses cards or requests SKIP."""
from time import monotonic
from typing import Callable, Iterable, TYPE_CHECKING
from cv2.typing import MatLike
from .battle_inputs import BattleInputState, PendingBattleInput
from .battle_observation import BattleObservationSession, current_absolute_turn, current_feedback_turn
from .battle_log import log_inspection_action
from .card_probe import card_feedback_ready, pending_card_permit
from .skip_probe import pending_skip_permit
from .models import Action, Decision, Scene
if TYPE_CHECKING:
    from .page import IdolRoadPage, PageSnapshot


class InputObservationError(ValueError):
    pass


class PendingInputObservation:
    """One input, one owned log window, original deadline and bounded samples."""
    def __init__(self, session: BattleObservationSession, inputs: BattleInputState, *,
                 deadline: float, clock: Callable[[], float] = monotonic):
        self.clock = clock
        now = clock()
        permit = pending_card_permit(session, inputs, now=now) or pending_skip_permit(session, inputs, now=now)
        if permit is None or session.turn_budget_mode != 'variable_budget':
            raise InputObservationError('pending_input_origin_unconfirmed')
        self.session = session
        self.pending: PendingBattleInput = permit.pending
        self.permit = permit
        self.deadline = min(deadline, permit.deadline)
        self.samples = 0
        self.log_requested = False
        self.opening_frame_id: str | None = None
        self.request_id: str | None = None
        self.terminal: Decision | None = None

    def _decision(self, action: Action, reason: str) -> Decision:
        decision = Decision(action, reason, self.session.character_id, self.session.stage)
        if action in (Action.READY, Action.STOP):
            self.terminal = decision
        return decision

    def step(self, snapshot: 'PageSnapshot', inputs: BattleInputState) -> Decision:
        if self.terminal is not None:
            return self.terminal
        self.samples += 1
        now = self.clock()
        if not self.pending.started <= now < self.deadline or self.samples > 50 or inputs.expired:
            return self._decision(Action.STOP, 'pending_input_observation_timeout')
        if (inputs.battle_id != self.session.battle_id or type(inputs.sequence) is not int
            or inputs.sequence != self.pending.sequence
            or (inputs.pending != self.pending and not (self.pending.kind == 'skip' and inputs.pending is None))):
            return self._decision(Action.STOP, 'pending_input_changed')
        obs, frame = snapshot.observation, snapshot.battle_frame
        if ((obs.character_id is not None and obs.character_id != self.session.character_id)
            or (obs.stage is not None and obs.stage != self.session.stage)):
            return self._decision(Action.STOP, 'pending_input_identity_changed')
        if frame is not None and frame.session != self.session:
            return self._decision(Action.STOP, 'pending_input_session_changed')
        if (obs.scene in (Scene.JUDGING,Scene.RESULT)
            and (obs.character_id,obs.stage)==(self.session.character_id,self.session.stage)):
            # Finishing a battle does not prove the pending card's effects or
            # release its ledger for another play. Return control to results.
            return self._decision(Action.READY,'pending_input_owned_battle_finished')
        if obs.scene == Scene.BATTLE_LOG:
            inspection = obs.battle_log_inspection
            if (not self.log_requested or inspection is None or inspection.request is None
                or not inspection.request.request_id
                or inspection.request.session != self.session
                or inspection.request.input_sequence != self.pending.sequence
                or inspection.request.pending_probe != self.permit
                or inspection.request.opening_frame_id != self.opening_frame_id
                or (self.request_id is not None and self.request_id != inspection.request.request_id)):
                return self._decision(Action.STOP, 'pending_input_log_unowned')
            self.request_id = inspection.request.request_id
            if inspection.expired or inspection.inputs_invalidated:
                return self._decision(Action.STOP, 'pending_input_log_invalid')
            action = log_inspection_action(inspection, self.session.battle_id)
            return self._decision({'wait': Action.WAIT, 'scroll': Action.SCROLL_BATTLE_LOG,
                'scroll_header': Action.SCROLL_BATTLE_LOG_HEADER, 'close': Action.CLOSE_BATTLE_LOG}[action],
                'pending_input_log_' + action)
        if obs.battle_log_pending:
            if not self.log_requested:
                return self._decision(Action.STOP, 'pending_input_log_unowned')
            return self._decision(Action.WAIT, 'pending_input_await_log_return')
        if obs.scene == Scene.UNKNOWN:
            return self._decision(Action.WAIT, 'pending_input_await_hud')
        if obs.scene != Scene.BATTLE:
            return self._decision(Action.STOP, 'pending_input_unexpected_scene')
        if self.log_requested and obs.battle_log_return is not None:
            returned = obs.battle_log_return
            if (returned.request.session != self.session or returned.request.pending_probe != self.permit
                or returned.source != 'owned_log_return_candidate'
                or returned.request.input_sequence != self.pending.sequence or self.request_id is None
                or returned.request.request_id != self.request_id
                or returned.request.opening_frame_id != self.opening_frame_id):
                return self._decision(Action.STOP, 'pending_input_return_unowned')
            if self.pending.kind == 'card':
                if card_feedback_ready(self.session, inputs, frame, obs.battle_card_resolution, now=now):
                    return self._decision(Action.READY, 'pending_card_feedback_ready')
            elif frame is not None:
                resolution = obs.battle_skip_resolution
                absolute = (current_feedback_turn(frame) if frame.values.detail_mode=='feedback_only'
                            else current_absolute_turn(frame))
                if (inputs.pending is None and resolution is not None and resolution.status == 'next_round_confirmed'
                    and resolution.permit == self.permit and resolution.request_id == returned.request.request_id
                    and resolution.return_frame_id == returned.return_frame_id == frame.values.frame_id
                    and returned.hud_unchanged and returned.remaining_consistent and returned.input_sequence_consistent
                    and frame.values.detail_mode in ('complete','feedback_only') and frame.values.resource_state == 'current'
                    and (frame.values.detail_mode=='complete' or (resolution.settled_at is not None
                        and self.pending.started<=resolution.settled_at<self.deadline))
                    and frame.observation.turn_source == 'owned_skip_log_return' and absolute is not None
                    and absolute.turn == resolution.turn == self.pending.turn + 1
                    and absolute.remaining_turns == resolution.remaining_turns):
                    return self._decision(Action.READY, 'pending_skip_feedback_ready')
            return self._decision(Action.STOP, 'pending_input_return_unconfirmed')
        if inputs.pending is None:
            return self._decision(Action.STOP, 'pending_input_cleared_without_owned_return')
        if not self.log_requested and obs.battle_log_available:
            if (frame is None or frame.values.resource_state != 'current' or not frame.values.hud_confirmed
                or not frame.values.frame_id or frame.values.frame_id != frame.observation.frame_id):
                return self._decision(Action.STOP, 'pending_input_log_target_unconfirmed')
            self.log_requested = True
            self.opening_frame_id = frame.values.frame_id
            return self._decision(Action.INSPECT_BATTLE_LOG, 'pending_input_inspect_log_once')
        return self._decision(Action.WAIT, 'pending_input_await_log_target')


def observe_pending_input(page: 'IdolRoadPage', session: BattleObservationSession,
                          frames: Iterable[MatLike], *, deadline: float,
                          observe: Callable[[MatLike, 'PageSnapshot', Decision], None],
                          clock: Callable[[], float] | None = None) -> tuple[MatLike, 'PageSnapshot']:
    inputs = page.battle_inputs
    if inputs is None or page.battle_session != session or inputs.session != session:
        raise InputObservationError('pending_input_owner_missing')
    policy = PendingInputObservation(session, inputs.snapshot(), deadline=deadline, clock=clock or inputs.clock)
    for screen in frames:
        if page.battle_session != session or page.battle_inputs is not inputs:
            raise InputObservationError('pending_input_owner_changed')
        snapshot = page.observe(screen, session.character_id)
        decision = policy.step(snapshot, inputs.snapshot())
        observe(screen, snapshot, decision)
        if decision.action == Action.READY:
            return screen, snapshot
        if decision.action == Action.STOP:
            raise InputObservationError(decision.reason)
        page.execute(snapshot, decision)
    raise InputObservationError('pending_input_frames_exhausted')
