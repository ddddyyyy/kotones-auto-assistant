"""Fresh owned variable-battle observation setup; never selects or plays cards."""
from time import monotonic
from typing import Callable,Iterable,TYPE_CHECKING,Literal
from dataclasses import replace
from cv2.typing import MatLike
from .battle_observation import BattleObservationSession,BattleFrameObservation,current_absolute_turn
from .battle_inputs import BattleInputState
from .maximum_hp import MaximumHpCalibration,confirmed_maximum_hp
from .battle_log import log_inspection_action
from .models import Action,Decision,Scene
from .hud_initialization import InitialHudRound
if TYPE_CHECKING:
    from .page import IdolRoadPage,PageSnapshot


class BattleInitializationError(ValueError):
    pass


class InitialBattleObservation:
    def __init__(self,session: BattleObservationSession,*,deadline: float,clock: Callable[[],float]=monotonic,
                 feedback_mode: Literal['log','hud']='log'):
        if feedback_mode not in ('log','hud'):
            raise BattleInitializationError('initial_battle_feedback_mode_unknown')
        if (session.identity_source!='confirmed_formation_start_request' or not session.battle_id
            or session.turn_budget_mode!='variable_budget' or type(session.total_turns) is not int
            or not 1<=session.total_turns<=30 or not session.character_id
            or type(session.stage) is not int or session.stage<1):
            raise BattleInitializationError('initial_battle_session_unconfirmed')
        self.session=session
        self.clock=clock
        self.deadline=min(deadline,clock()+90)
        self.feedback_mode=feedback_mode
        self.hud=InitialHudRound(session,deadline=self.deadline) if feedback_mode=='hud' else None
        self.hud_frame: BattleFrameObservation | None=None
        self.hp_close_requested=False
        self.samples=0
        self.hp_requested=False
        self.log_requested=False
        self.seen_battle=False
        self.calibration: MaximumHpCalibration | None=None
        self.terminal: Decision | None=None

    def _decision(self,action: Action,reason: str) -> Decision:
        decision=Decision(action,reason,self.session.character_id,self.session.stage)
        if action in (Action.STOP,Action.READY):
            self.terminal=decision
        return decision

    def step(self,snapshot: 'PageSnapshot',inputs: BattleInputState) -> Decision:
        if self.terminal is not None:
            return self.terminal
        self.samples+=1
        if self.clock()>=self.deadline or self.samples>50:
            return self._decision(Action.STOP,'initial_battle_observation_timeout')
        if (inputs.battle_id!=self.session.battle_id or type(inputs.sequence) is not int or inputs.sequence!=0
            or inputs.pending is not None or inputs.expired):
            return self._decision(Action.STOP,'initial_battle_input_changed')
        obs,frame=snapshot.observation,snapshot.battle_frame
        if obs.battle_log_pending and not self.log_requested:
            return self._decision(Action.STOP,'initial_battle_log_unowned')
        if frame is not None and frame.session!=self.session:
            return self._decision(Action.STOP,'initial_battle_session_changed')
        if obs.scene==Scene.BATTLE_LOG:
            inspection=obs.battle_log_inspection
            if (not self.log_requested or inspection is None or inspection.request is None
                or inspection.request.session!=self.session or inspection.request.input_sequence!=0):
                return self._decision(Action.STOP,'initial_battle_log_unowned')
            if inspection.expired or inspection.inputs_invalidated:
                return self._decision(Action.STOP,'initial_battle_log_invalid')
            action=log_inspection_action(inspection,self.session.battle_id)
            return self._decision({'wait':Action.WAIT,'scroll':Action.SCROLL_BATTLE_LOG,'scroll_header':Action.SCROLL_BATTLE_LOG_HEADER,
                                   'close':Action.CLOSE_BATTLE_LOG}[action],'initial_battle_log_'+action)
        if obs.scene==Scene.UNKNOWN or (obs.scene==Scene.CONFIRM and not self.seen_battle):
            return self._decision(Action.WAIT,'initial_battle_await_hud')
        if obs.scene!=Scene.BATTLE:
            return self._decision(Action.STOP,'initial_battle_unexpected_scene')
        if frame is None or frame.values.resource_state!='current' or frame.values.detail_mode!='complete':
            return self._decision(Action.WAIT,'initial_battle_await_complete_hud')
        state=frame.observation
        if (state.source!='confirmed_current_battle_observation' or not frame.values.hud_confirmed
            or state.frame_id!=frame.values.frame_id or not state.frame_id
            or (state.character_id,state.stage,state.battle_id)!=(self.session.character_id,self.session.stage,self.session.battle_id)
            or {'identity','hp','max_hp','turn'}&set(state.conflicted_fields)):
            return self._decision(Action.STOP,'initial_battle_frame_conflict')
        self.seen_battle=True
        if obs.battle_log_pending:
            return self._decision(Action.WAIT,'initial_battle_await_log_return')
        maximum=confirmed_maximum_hp(frame)
        if maximum is not None and frame.maximum_hp_parameter is not None and (
            frame.maximum_hp_parameter.input_sequence!=0 or frame.maximum_hp_parameter.calibration.input_sequence!=0):
            return self._decision(Action.STOP,'initial_battle_hp_sequence_changed')
        if not self.hp_requested:
            if maximum is not None or current_absolute_turn(frame) is not None:
                return self._decision(Action.STOP,'initial_battle_already_observed')
            if obs.battle_hp_info_available:
                self.hp_requested=True
                return self._decision(Action.INSPECT_BATTLE_HP,'initial_battle_calibrate_hp_once')
            return self._decision(Action.WAIT,'initial_battle_await_hp_target')
        if self.calibration is None:
            if maximum is None:
                return self._decision(Action.WAIT,'initial_battle_await_hp_calibration')
            assert frame.maximum_hp_parameter is not None
            self.calibration=frame.maximum_hp_parameter.calibration
        if self.hud is not None:
            if frame.values.hp_ratio is not None:
                if not self.hp_close_requested and Action.CLOSE_BATTLE_HP in snapshot.targets:
                    self.hp_close_requested=True
                    return self._decision(Action.CLOSE_BATTLE_HP,'initial_battle_close_hp_tooltip')
                return self._decision(Action.WAIT,'initial_battle_await_hp_tooltip_closed')
            bound=self.hud.observe(inputs,frame,snapshot.log_hand,now=self.clock())
            if bound is None:
                return self._decision(Action.WAIT,'initial_battle_await_stable_hud')
            self.hud_frame=bound
            return self._decision(Action.READY,'initial_battle_hud_ready')
        if self.log_requested:
            absolute=current_absolute_turn(frame)
            if (absolute is None or absolute.turn!=1 or absolute.remaining_turns!=self.session.total_turns
                or state.turn_source!='owned_initial_log_return' or maximum is None
                or frame.maximum_hp_parameter is None or frame.maximum_hp_parameter.calibration!=self.calibration):
                return self._decision(Action.STOP,'initial_battle_return_unconfirmed')
            return self._decision(Action.READY,'initial_battle_observation_ready')
        if obs.battle_log_available:
            self.log_requested=True
            return self._decision(Action.INSPECT_BATTLE_LOG,'initial_battle_inspect_log_once')
        return self._decision(Action.WAIT,'initial_battle_await_log_target')


def initialize_variable_battle(page: 'IdolRoadPage',session: BattleObservationSession,
                               frames: Iterable[MatLike],*,deadline: float,
                               observe: Callable[[MatLike,'PageSnapshot',Decision],None],
                               clock: Callable[[],float]=monotonic,
                               feedback_mode: Literal['log','hud']='log') -> tuple[MatLike,'PageSnapshot']:
    policy=InitialBattleObservation(session,deadline=deadline,clock=clock,feedback_mode=feedback_mode)
    inputs=page.battle_inputs
    if inputs is None or page.battle_session!=session or inputs.session!=session:
        raise BattleInitializationError('initial_battle_owner_missing')
    if feedback_mode=='hud':page.begin_hud_feedback(session)
    for screen in frames:
        snapshot=page.observe(screen,session.character_id)
        decision=policy.step(snapshot,inputs.snapshot())
        if decision.action==Action.READY and feedback_mode=='hud':
            if policy.hud_frame is None:raise BattleInitializationError('initial_hud_binding_missing')
            assert policy.hud is not None
            bound=page.accept_initial_hud(policy.hud)
            snapshot=replace(snapshot,battle_frame=bound,
                observation=replace(snapshot.observation,battle_state=bound.observation))
        observe(screen,snapshot,decision)
        if decision.action==Action.READY:
            return screen,snapshot
        if decision.action==Action.STOP:
            raise BattleInitializationError(decision.reason)
        page.execute(snapshot,decision)
    raise BattleInitializationError('initial_battle_frames_exhausted')
