"""卡牌待结的只读日志证据；回合确认不等于出牌成功。"""
from __future__ import annotations
from dataclasses import dataclass,replace
from typing import Literal,TYPE_CHECKING
from .battle_inputs import BattleInputState,PendingBattleInput
from .battle_observation import BattleObservationSession,BattleFrameObservation,current_absolute_turn,current_feedback_turn
from .planning_turn import read_planning_turn
from .skip_probe import pending_input_origin_valid
if TYPE_CHECKING:
    from .battle_log import LogReturn,LogViewport


@dataclass(frozen=True)
class PendingCardPermit:
    pending: PendingBattleInput
    deadline: float

    @property
    def sequence(self) -> int:
        return self.pending.sequence


def pending_card_permit(session: BattleObservationSession,inputs: BattleInputState,*,now: float) -> PendingCardPermit | None:
    request=inputs.pending
    if (request is None or request.kind!='card' or not request.card_id or request.before is None
        or not pending_input_origin_valid(session,inputs,now=now)):
        return None
    before=request.before
    planning=read_planning_turn(before,session.initial_total_turns or 0)
    frame=request.frame
    if (planning is None or planning.turn!=request.turn or before.remaining_turns!=frame.values.remaining_turns
        or (before.hp_known and before.hp!=frame.observation.hp)):
        return None
    if session.turn_budget_mode=='variable_budget':
        if before.absolute_turn_observation!=current_absolute_turn(frame):
            return None
    elif planning.source!='fixed_budget':
        return None
    return PendingCardPermit(request,request.started+20)


@dataclass(frozen=True)
class CardLogResolution:
    status: Literal['unconfirmed','older_header','header_gap','same_round_observed','next_round_observed']
    permit: PendingCardPermit
    request_id: str
    log_frame_id: str | None
    return_frame_id: str
    turn: int | None = None
    remaining_turns: int | None = None
    observed_frame: BattleFrameObservation | None = None
    # 只提供后续反馈的回合候选；本模块没有释放ledger或授予输入的能力。
    play_confirmed: Literal[False] = False
    unconfirmed_reason: Literal['tail_header_unavailable'] | None = None
    round_evidence: Literal['visible_tail','owned_viewport_span'] = 'visible_tail'


def assess_card_log(permit: PendingCardPermit,returned: LogReturn,view: LogViewport,
                    frame: BattleFrameObservation,inputs: BattleInputState,*,now: float) -> CardLogResolution:
    from .battle_log import returned_log_header,log_at_bottom
    request=returned.request
    session=permit.pending.frame.session
    # Diagnostic only: a long turn may put its title above the visible tail.
    # This is not permission to reuse the title from an earlier log request.
    missing_tail=(request.card_probe==permit and request.session==frame.session==session
                  and request.input_sequence==permit.sequence and returned.log_frame_id==view.frame_id
                  and not view.headers and view.scrollbar is not None and view.scrollbar.frame_id==view.frame_id
                  and log_at_bottom(view.scrollbar) is True)
    result=CardLogResolution('unconfirmed',permit,request.request_id,returned.log_frame_id,returned.return_frame_id,
                            unconfirmed_reason='tail_header_unavailable' if missing_tail else None)
    state,values=frame.observation,frame.values
    if (pending_card_permit(session,inputs,now=now)!=permit or request.card_probe!=permit
        or request.skip_probe is not None or request.session!=session or not request.request_id
        or request.input_sequence!=permit.sequence or not returned.hud_unchanged or not returned.remaining_consistent
        or not returned.input_sequence_consistent or returned.source!='owned_log_return_candidate'
        or frame.session!=session or values.detail_mode not in ('complete','feedback_only')
        or (values.detail_mode=='feedback_only' and session.turn_budget_mode!='variable_budget')
        or values.resource_state!='current' or not values.hud_confirmed
        or state.source!='confirmed_current_battle_observation' or state.frame_id!=values.frame_id
        or (state.battle_id,state.character_id,state.stage)!=(session.battle_id,session.character_id,session.stage)
        or {'identity','turn','hp'}&set(state.conflicted_fields)
        or not values.frame_id or returned.return_frame_id!=values.frame_id or returned.log_frame_id!=view.frame_id
        or values.frame_id in (request.opening_frame_id,permit.pending.frame.values.frame_id)
        or request.hp!=state.hp or type(state.hp) is not int):
        return result
    header=returned_log_header(returned,view)
    if (header is None or header!=returned.tail_header or header.absolute_turn is None
        or header.remaining_turns!=values.remaining_turns or header.remaining_turns!=request.remaining_turns
        or (session.turn_budget_mode=='fixed_budget' and state.turn!=header.absolute_turn)):
        return result
    turn=header.absolute_turn
    status=('older_header' if turn<permit.pending.turn else 'same_round_observed' if turn==permit.pending.turn
            else 'next_round_observed' if turn==permit.pending.turn+1 else 'header_gap')
    return CardLogResolution(status,permit,request.request_id,view.frame_id,values.frame_id,turn,header.remaining_turns,
                             observed_frame=frame,round_evidence='owned_viewport_span' if returned.span is not None else 'visible_tail')


def card_feedback_ready(session: BattleObservationSession,inputs: BattleInputState,frame: BattleFrameObservation | None,
                        resolution: CardLogResolution | None,*,now: float) -> bool:
    """恢复原反馈判断；不结算输入，变量帧必须已有生产者绑定的同帧契约。"""
    if frame is None or resolution is None or frame.values.detail_mode not in ('complete','feedback_only'):
        return False
    raw=frame
    if session.turn_budget_mode=='variable_budget':
        absolute=current_feedback_turn(frame)
        if (absolute is None or frame.observation.turn_source!='owned_card_log_return'
            or absolute.turn!=resolution.turn or absolute.remaining_turns!=resolution.remaining_turns):
            return False
        raw=replace(frame,absolute_turn=None,observation=replace(frame.observation,turn=None,turn_source=None))
    elif session.turn_budget_mode!='fixed_budget':
        return False
    return bool(
        resolution.status in ('same_round_observed','next_round_observed')
        and pending_card_permit(session,inputs,now=now)==resolution.permit
        and raw==resolution.observed_frame and frame.session==session and frame.values.frame_id==resolution.return_frame_id
        and frame.observation.turn==resolution.turn and frame.values.remaining_turns==resolution.remaining_turns)
