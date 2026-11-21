"""待结SKIP的只读日志许可；窗口不授权再次输入。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal,TYPE_CHECKING
from .battle_inputs import BattleInputState,PendingBattleInput
from .battle_observation import BattleObservationSession,BattleFrameObservation,current_absolute_turn
if TYPE_CHECKING:
    from .battle_log import LogReturn,LogViewport


@dataclass(frozen=True)
class PendingSkipPermit:
    pending: PendingBattleInput
    deadline: float

    @property
    def sequence(self) -> int:
        return self.pending.sequence


def pending_skip_permit(session: BattleObservationSession,inputs: BattleInputState,*,now: float) -> PendingSkipPermit | None:
    request=inputs.pending
    if (request is None or request.kind!='skip' or request.card_id is not None or request.before is not None
        or not pending_input_origin_valid(session,inputs,now=now)):
        return None
    return PendingSkipPermit(request,request.started+20)


def pending_input_origin_valid(session: BattleObservationSession,inputs: BattleInputState,*,now: float) -> bool:
    """只核验输入来源与原期限，卡牌before绑定由各自许可单独核验。"""
    request=inputs.pending
    if (session.identity_source!='confirmed_formation_start_request' or inputs.battle_id!=session.battle_id
        or inputs.expired or request is None
        or type(inputs.sequence) is not int or inputs.sequence<1 or type(request.sequence) is not int or request.sequence!=inputs.sequence
        or not 0<=now-request.started<20 or request.frame.session!=session):
        return False
    frame=request.frame
    state,values=frame.observation,frame.values
    if (values.detail_mode!='complete' or values.resource_state!='current' or not values.hud_confirmed or not values.frame_id
        or state.source!='confirmed_current_battle_observation' or state.frame_id!=values.frame_id
        or (state.battle_id,state.character_id,state.stage)!=(session.battle_id,session.character_id,session.stage)
        or {'identity','turn'}&set(state.conflicted_fields) or type(request.turn) is not int
        or type(state.turn) is not int or request.turn!=state.turn):
        return False
    if session.turn_budget_mode=='variable_budget':
        if current_absolute_turn(frame) is None:
            return False
    elif session.turn_budget_mode=='fixed_budget':
        total,remaining=session.initial_total_turns,values.remaining_turns
        if (type(total) is not int or type(remaining) is not int or not 1<=remaining<=total<=30
            or state.turn_source!='fixed_budget_assumption' or state.turn!=total-remaining+1):
            return False
    else:
        return False
    return True


@dataclass(frozen=True)
class SkipLogResolution:
    status: Literal['unconfirmed','same_or_older_header','header_gap','next_round_confirmed']
    permit: PendingSkipPermit
    request_id: str
    log_frame_id: str | None
    return_frame_id: str
    turn: int | None = None
    remaining_turns: int | None = None
    settled_at: float | None = None


def assess_skip_log(permit: PendingSkipPermit,returned: LogReturn,view: LogViewport,
                    frame: BattleFrameObservation,inputs: BattleInputState,*,now: float) -> SkipLogResolution:
    from .battle_log import tail_log_header
    request=returned.request
    result=SkipLogResolution('unconfirmed',permit,request.request_id,returned.log_frame_id,returned.return_frame_id)
    session=permit.pending.frame.session
    state,values=frame.observation,frame.values
    if (pending_skip_permit(session,inputs,now=now)!=permit or request.skip_probe!=permit or request.card_probe is not None or request.session!=session
        or request.input_sequence!=permit.sequence or not returned.hud_unchanged or not returned.remaining_consistent
        or not returned.input_sequence_consistent or returned.source!='owned_log_return_candidate'
        or frame.session!=session or values.detail_mode not in ('complete','feedback_only') or values.resource_state!='current' or not values.hud_confirmed
        or state.source!='confirmed_current_battle_observation' or state.frame_id!=values.frame_id
        or (state.battle_id,state.character_id,state.stage)!=(session.battle_id,session.character_id,session.stage)
        or {'identity','turn','hp'}&set(state.conflicted_fields)
        or returned.return_frame_id!=values.frame_id or returned.log_frame_id!=view.frame_id
        or values.frame_id in (request.opening_frame_id,permit.pending.frame.values.frame_id)
        or request.hp!=state.hp or type(state.hp) is not int):
        return result
    header=tail_log_header(view)
    if (header is None or header!=returned.tail_header or header.absolute_turn is None
        or header.remaining_turns!=values.remaining_turns or header.remaining_turns!=request.remaining_turns):
        return result
    turn=header.absolute_turn
    status=('same_or_older_header' if turn<=permit.pending.turn else
            'next_round_confirmed' if turn==permit.pending.turn+1 else 'header_gap')
    return SkipLogResolution(status,permit,request.request_id,view.frame_id,values.frame_id,turn,header.remaining_turns)
