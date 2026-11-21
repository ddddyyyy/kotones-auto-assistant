"""First-round binding from this program's START and stable initial HUD."""
from dataclasses import replace
from .battle_observation import BattleObservationSession,BattleFrameObservation
from .battle_inputs import BattleInputState
from .log_hand import LogHandObservation,confirm_log_hand
from .maximum_hp import confirmed_maximum_hp
from kaa.battle_turn import AbsoluteTurnObservation


class InitialHudRound:
    """Sequence zero is the first round; remaining turns are only a cross-check.

    Construction requires an owned START session. Two complete captures and
    this battle's maximum-HP calibration are required, without a log panel.
    """
    def __init__(self,session: BattleObservationSession,*,deadline: float):
        if (session.identity_source!='confirmed_formation_start_request' or not session.battle_id
            or session.turn_budget_mode!='variable_budget' or type(session.total_turns) is not int
            or not 1<=session.total_turns<=30):
            raise ValueError('initial_hud_session_unconfirmed')
        self.session=session
        self.deadline=deadline
        self.previous: tuple[tuple,str] | None=None
        self.bound: BattleFrameObservation | None=None
        self.bound_hand: LogHandObservation | None=None

    def observe(self,inputs: BattleInputState,frame: BattleFrameObservation,
                hand: LogHandObservation | None,*,now: float) -> BattleFrameObservation | None:
        if (now>=self.deadline or inputs.battle_id!=self.session.battle_id
            or type(inputs.sequence) is not int or inputs.sequence!=0
            or inputs.pending is not None or inputs.expired or frame.session!=self.session):
            raise ValueError('initial_hud_owner_changed_or_expired')
        if self.bound is not None:
            return self.bound if frame==self.bound else None
        values,state=frame.values,frame.observation
        maximum=confirmed_maximum_hp(frame)
        if (values.detail_mode!='complete' or values.resource_state!='current'
            or state.conflicted_fields or frame.absolute_turn is not None or state.turn is not None
            or type(values.remaining_turns) is not int or values.remaining_turns!=self.session.total_turns or maximum is None
            or frame.maximum_hp_parameter is None or frame.maximum_hp_parameter.input_sequence!=0
            or frame.maximum_hp_parameter.calibration.input_sequence!=0
            or hand is None or hand.frame_id!=values.frame_id
            or confirm_log_hand(hand.frame_id,hand.count,hand.cards,empty_confirmed=hand.empty_confirmed)!=hand):
            self.previous=None
            return None
        key=(values.remaining_turns,state.hp,maximum,hand.fingerprint,frame.maximum_hp_parameter.calibration)
        previous=self.previous
        self.previous=(key,values.frame_id)
        if previous is None or previous[0]!=key or previous[1]==values.frame_id:
            return None
        absolute=AbsoluteTurnObservation(self.session.battle_id,values.frame_id,1,values.remaining_turns)
        self.bound=replace(frame,absolute_turn=absolute,
            observation=replace(state,turn=1,turn_source='owned_initial_hud'))
        self.bound_hand=hand
        return self.bound
