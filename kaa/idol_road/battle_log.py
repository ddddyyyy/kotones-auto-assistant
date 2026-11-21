"""当前战斗日志请求生命周期；尾部标题只作为候选，不授予当前回合。"""
from dataclasses import dataclass
from time import monotonic
from typing import Callable,Literal
from uuid import uuid4
import re
from .battle_observation import BattleFrameObservation,BattleObservationSession
from .log_turns import VisibleLogTurn,confirm_visible_log_turn
from .log_hand import LogHandCard,LogHandObservation,confirm_log_hand
from .short_log import ShortLogEvidence,short_log_end_visible
from .skip_probe import PendingSkipPermit,pending_skip_permit
from .card_probe import PendingCardPermit,pending_card_permit
from .battle_inputs import BattleInputState
from .log_span import OwnedLogSpan,LogOverlap,span_header


def _digest(value: str) -> bool:
    return re.fullmatch('[0-9a-f]{64}',value) is not None


@dataclass(frozen=True)
class LogEntryEvidence:
    frame_id: str
    rect: tuple[int,int,int,int]
    score: float
    hand_fingerprint: str
    hand_cards: tuple[LogHandCard,...] = ()


@dataclass(frozen=True)
class LogScrollEvidence:
    frame_id: str
    track_rect: tuple[int,int,int,int]
    thumb_rect: tuple[int,int,int,int]
    covered_fraction: float
    light_rows: int


def log_at_bottom(evidence: LogScrollEvidence | None) -> bool | None:
    if evidence is None or not evidence.frame_id or not .9<=evidence.covered_fraction<=1 or evidence.light_rows<15:
        return None
    x,y,w,h=evidence.track_rect
    xx,yy,ww,hh=evidence.thumb_rect
    if (min(x,y,xx,yy)<0 or min(w,h,ww,hh)<=0 or hh<12 or hh>=h
        or x+w>720 or y+h>1280 or not (x<=xx and xx+ww<=x+w and y<=yy and yy+hh<=y+h)):
        return None
    return y+h-yy-hh<=2


@dataclass(frozen=True)
class LogViewport:
    frame_id: str
    body_fingerprint: str
    close_rect: tuple[int,int,int,int]
    headers: tuple[VisibleLogTurn,...]
    scrollbar: LogScrollEvidence | None
    tail_clear_fractions: tuple[float,...]
    content_confirmed: bool
    short_list: ShortLogEvidence | None = None


def tail_log_header(view: LogViewport) -> VisibleLogTurn | None:
    """只认位置最靠后的完整标题；未知／反序／未到底不能取最大值补结论。"""
    if (not view.content_confirmed or not _digest(view.body_fingerprint)
        or len(view.tail_clear_fractions)!=2
        or any(not .995<=value<=1 for value in view.tail_clear_fractions) or not view.headers):
        return None
    headers=sorted(view.headers,key=lambda row:row.evidence.suffix_rect[1])
    if view.scrollbar is not None:
        if view.scrollbar.frame_id!=view.frame_id or log_at_bottom(view.scrollbar) is not True:
            return None
    elif not short_log_end_visible(view.short_list,view.frame_id,headers[0].evidence.suffix_rect):
        return None
    previous=0
    for header in headers:
        if (header.evidence.frame_id!=view.frame_id or confirm_visible_log_turn(header.evidence)!=header
            or header.status!='resolved' or header.absolute_turn is None or header.remaining_turns is None
            or header.absolute_turn<=previous):
            return None
        previous=header.absolute_turn
    return headers[-1]


@dataclass(frozen=True)
class LogRequest:
    request_id: str
    session: BattleObservationSession
    opening_frame_id: str
    remaining_turns: int
    hp: int
    hand_fingerprint: str
    input_sequence: int = 0
    skip_probe: PendingSkipPermit | None = None
    card_probe: PendingCardPermit | None = None

    @property
    def pending_probe(self) -> PendingSkipPermit | PendingCardPermit | None:
        return self.skip_probe or self.card_probe


@dataclass(frozen=True)
class LogInspection:
    viewport: LogViewport
    request: LogRequest | None
    closing_requested: bool
    scroll_pending: bool
    scroll_attempts: int
    expired: bool
    tail_header: VisibleLogTurn | None
    currentness_confirmed: Literal[False] = False
    inputs_invalidated: bool = False
    short_list_stable: bool = False
    header_scroll_needed: bool = False
    span: OwnedLogSpan | None = None


@dataclass(frozen=True)
class LogReturn:
    request: LogRequest
    log_frame_id: str | None
    return_frame_id: str
    hud_unchanged: bool
    remaining_consistent: bool
    tail_header: VisibleLogTurn | None
    source: Literal['owned_log_return_candidate'] = 'owned_log_return_candidate'
    currentness_confirmed: Literal[False] = False
    input_sequence_consistent: bool = False
    span: OwnedLogSpan | None = None


def returned_log_header(returned: LogReturn,view: LogViewport) -> VisibleLogTurn | None:
    return (span_header(returned.span,returned.request,view) if returned.span is not None else tail_log_header(view))


def log_inspection_action(inspection: LogInspection, expected_battle_id: str | None) -> Literal['wait','scroll','scroll_header','close']:
    if inspection.closing_requested or (inspection.scroll_pending and not inspection.expired):
        return 'wait'
    owned=(inspection.request is not None and expected_battle_id==inspection.request.session.battle_id)
    if owned and not inspection.expired and inspection.span is not None and inspection.tail_header is not None:
        return 'close'
    if owned and not inspection.expired and inspection.header_scroll_needed:
        return 'scroll_header'
    if (owned and not inspection.expired and inspection.viewport.short_list is not None
        and tail_log_header(inspection.viewport) is not None and not inspection.short_list_stable):
        return 'wait'
    if (owned and not inspection.expired and inspection.scroll_attempts<2
        and inspection.viewport.content_confirmed and log_at_bottom(inspection.viewport.scrollbar) is False):
        return 'scroll'
    return 'close'


class BattleLogBinding:
    def __init__(self, *, clock: Callable[[],float]=monotonic):
        self.clock=clock
        self.reset()

    def reset(self) -> None:
        self.request: LogRequest | None=None
        self.closing=False
        self.started=0.
        self.scroll_attempts=0
        self.scroll_pending: LogViewport | None=None
        self.scroll_waits=0
        self.last_view: LogViewport | None=None
        self._last_key: tuple | None=None
        self._last_frame: str | None=None
        self._last_hand: LogHandObservation | None=None
        self._last_entry: LogEntryEvidence | None=None
        self._ready=False
        self._scroll_last_frame: str | None=None
        self._input_sequence=0
        self._input_pending=False
        self._inputs_invalid=False
        self._request_inputs_invalid=False
        self._stable_short_key: tuple | None=None
        self._skip_probe: PendingSkipPermit | None=None
        self._card_probe: PendingCardPermit | None=None
        self._return_recheck_frame: str | None=None
        self._span: OwnedLogSpan | None=None
        self._scroll_direction: Literal['down','header']='down'

    @property
    def pending_header_view(self) -> LogViewport | None:
        return self.scroll_pending if self._scroll_direction=='header' else None

    @property
    def _probe(self) -> PendingSkipPermit | PendingCardPermit | None:
        return self._skip_probe or self._card_probe

    @staticmethod
    def _short_key(view: LogViewport) -> tuple | None:
        evidence=view.short_list
        if view.scrollbar is not None or evidence is None or tail_log_header(view) is None:
            return None
        return (evidence.text_fingerprint,evidence.text_rect,evidence.last_content_y,
                tuple((row.absolute_turn,row.remaining_turns,row.evidence.suffix_rect) for row in view.headers))

    def _tail_candidate(self, view: LogViewport | None) -> VisibleLogTurn | None:
        if self._span is not None and self.request is not None and view is not None:
            return span_header(self._span,self.request,view)
        if view is None or (view.scrollbar is None and (self._stable_short_key is None or self._short_key(view)!=self._stable_short_key)):
            return None
        return tail_log_header(view)

    def update_inputs(self, sequence: int, pending: bool, *, skip_probe: PendingSkipPermit | None=None,
                      card_probe: PendingCardPermit | None=None) -> None:
        """输入变化撤销稳定性，但保留面板身份用于有界关闭；不能恢复旧候选。"""
        if type(sequence) is not int or sequence<0 or type(pending) is not bool:
            raise ValueError('日志缺少有效的当前输入状态')
        if skip_probe is not None and card_probe is not None:
            raise ValueError('日志不能同时绑定两种待结输入')
        if skip_probe is not None:
            state=BattleInputState(skip_probe.pending.frame.session.battle_id,sequence,skip_probe.pending,False)
            if not pending or pending_skip_permit(skip_probe.pending.frame.session,state,now=self.clock())!=skip_probe:
                skip_probe=None
        if card_probe is not None:
            state=BattleInputState(card_probe.pending.frame.session.battle_id,sequence,card_probe.pending,False)
            if not pending or pending_card_permit(card_probe.pending.frame.session,state,now=self.clock())!=card_probe:
                card_probe=None
        changed=(sequence!=self._input_sequence or pending!=self._input_pending
                 or skip_probe!=self._skip_probe or card_probe!=self._card_probe)
        if changed or (pending and skip_probe is None and card_probe is None):
            self._span=None
            self._ready=False
            self._last_key=None
            self._last_frame=None
            self._last_hand=None
            self._last_entry=None
            if self.request is not None:
                self._request_inputs_invalid=True
        if sequence<self._input_sequence:
            self._inputs_invalid=True
        self._input_sequence=max(sequence,self._input_sequence)
        self._input_pending=pending
        self._skip_probe=skip_probe
        self._card_probe=card_probe

    @property
    def pending(self) -> bool:
        return self.request is not None or self.closing

    @property
    def expired(self) -> bool:
        return bool(self.request is not None and (self.clock()-self.started>=30 or self.scroll_waits>=3
                    or self._request_inputs_invalid or self._inputs_invalid
                    or (self.request.pending_probe is not None and self.clock()>=self.request.pending_probe.deadline)))

    @staticmethod
    def _key(frame: BattleFrameObservation, entry: LogEntryEvidence | None,hand: LogHandObservation | None=None) -> tuple | None:
        values,state,session=frame.values,frame.observation,frame.session
        if (session.identity_source!='confirmed_formation_start_request'
            or values.resource_state!='current' or state.source!='confirmed_current_battle_observation'
            or not values.hud_confirmed or not values.frame_id or state.frame_id!=values.frame_id
            or (state.battle_id,state.character_id,state.stage)!=(session.battle_id,session.character_id,session.stage)
            or not session.battle_id
            or type(values.remaining_turns) is not int or not 1<=values.remaining_turns<=30
            or type(state.hp) is not int or state.hp<0 or {'identity','hp'}&set(state.conflicted_fields)):
            return None
        if hand is not None and (hand.frame_id!=values.frame_id
            or confirm_log_hand(hand.frame_id,hand.count,hand.cards,empty_confirmed=hand.empty_confirmed)!=hand):
            return None
        if entry is not None and (entry.frame_id!=values.frame_id or not .97<=entry.score<=1
            or not _digest(entry.hand_fingerprint) or min(entry.rect)<0 or min(entry.rect[2:])<=0
            or entry.rect[0]+entry.rect[2]>720 or entry.rect[1]+entry.rect[3]>1280
            or (hand is not None and entry.hand_fingerprint!=hand.fingerprint)):
            return None
        fingerprint=hand.fingerprint if hand is not None else entry.hand_fingerprint if entry is not None else None
        return (session,values.remaining_turns,state.hp,fingerprint) if fingerprint is not None else None

    def observe_hud(self, frame: BattleFrameObservation, entry: LogEntryEvidence | None,*,hand: LogHandObservation | None=None) -> LogReturn | None:
        readable=not self._input_pending or (self._probe is not None and self.clock()<self._probe.deadline)
        if self._probe is not None and frame.session!=self._probe.pending.frame.session:
            readable=False
        key=self._key(frame,entry,hand) if readable and not self._inputs_invalid else None
        if self.request is not None:
            if frame.session!=self.request.session:
                self.reset()
                return None
            if self.closing and key is not None:
                request=self.request
                candidate=self._tail_candidate(self.last_view) if not self.expired else None
                unchanged=(key[1:4]==(request.remaining_turns,request.hp,request.hand_fingerprint))
                consistent=(candidate is not None and candidate.remaining_turns==frame.values.remaining_turns==request.remaining_turns)
                sequence_consistent=(not self._request_inputs_invalid
                                     and self._input_sequence==request.input_sequence)
                # A closing animation can obscure an upgrade glyph in one HUD
                # frame. Keep this request for at most one distinct fresh read;
                # only that new frame's complete hand may confirm the return.
                # The original card deadline and log evidence are never renewed.
                if self._return_recheck_frame==frame.values.frame_id:
                    return None
                if (not unchanged and key[1:3]==(request.remaining_turns,request.hp)
                    and request.card_probe is not None and hand is not None
                    and consistent and sequence_consistent
                    and not self.expired and self._return_recheck_frame is None
                    and frame.values.frame_id not in (request.opening_frame_id,
                                                      request.card_probe.pending.frame.values.frame_id)):
                    self._return_recheck_frame=frame.values.frame_id
                    return None
                result=LogReturn(request,self.last_view.frame_id if self.last_view else None,frame.values.frame_id,
                                 unchanged,consistent,candidate if unchanged and consistent and (
                                     frame.values.detail_mode=='complete' or
                                     frame.values.detail_mode=='feedback_only' and request.pending_probe is not None) else None,
                                 input_sequence_consistent=sequence_consistent,
                                 span=self._span if unchanged and consistent and not self.expired else None)
                sequence,pending,skip,card=self._input_sequence,self._input_pending,self._skip_probe,self._card_probe
                self.reset()
                self.update_inputs(sequence,pending,skip_probe=skip,card_probe=card)
                return result
            return None
        if self.closing:
            sequence,pending,invalid,skip,card=self._input_sequence,self._input_pending,self._inputs_invalid,self._skip_probe,self._card_probe
            self.reset()
            self.update_inputs(sequence,pending,skip_probe=skip,card_probe=card)
            self._inputs_invalid=invalid
        self._ready=key is not None and key==self._last_key and self._last_frame!=frame.values.frame_id
        self._last_key=key
        self._last_frame=frame.values.frame_id
        self._last_hand=hand
        self._last_entry=entry
        return None

    def can_open(self, frame: BattleFrameObservation, entry: LogEntryEvidence | None) -> bool:
        return bool(not self.pending and self._ready and self._last_frame==frame.values.frame_id
                    and entry is not None and self._last_entry==entry
                    and (self._probe is None or frame.session==self._probe.pending.frame.session)
                    and (not self._input_pending or (self._probe is not None and self.clock()<self._probe.deadline))
                    and not self._inputs_invalid
                    and self._key(frame,entry,self._last_hand)==self._last_key and self._last_key is not None)

    def open(self, frame: BattleFrameObservation, entry: LogEntryEvidence, *, input_sequence: int=0,
             skip_probe: PendingSkipPermit | None=None,card_probe: PendingCardPermit | None=None) -> None:
        self.update_inputs(input_sequence,self._input_pending,skip_probe=skip_probe,card_probe=card_probe)
        if ((skip_probe is not None and self._skip_probe!=skip_probe)
            or (card_probe is not None and self._card_probe!=card_probe)):
            raise ValueError('待结输入日志许可失效')
        if not self.can_open(frame,entry):
            raise ValueError('当前日志入口缺少同会话两帧稳定证据')
        remaining,hp=frame.values.remaining_turns,frame.observation.hp
        if remaining is None or hp is None:
            raise ValueError('日志请求缺少当前体力或剩余回合')
        if type(input_sequence) is not int or input_sequence<0:
            raise ValueError('日志请求输入序号无效')
        if self._probe is not None and self._probe.pending.frame.session!=frame.session:
            raise ValueError('待结输入日志许可不属于当前战斗')
        self.request=LogRequest(uuid4().hex,frame.session,frame.values.frame_id,remaining,hp,entry.hand_fingerprint,
                                input_sequence,self._skip_probe,self._card_probe)
        self.started=self.clock()
        self._ready=False
        self.last_view=None
        self._stable_short_key=None

    def inspect(self, view: LogViewport, session: BattleObservationSession | None,*,overlap: LogOverlap | None=None) -> LogInspection:
        if self.request is not None and self.request.session!=session:
            self.reset()
        request=self.request
        if self.closing and self._span is not None:
            # Keep the exact proven view used to request closing, rather than
            # replacing it with a closing-animation frame.
            return LogInspection(view,request,True,False,self.scroll_attempts,self.expired,None,span=self._span)
        if request is not None and not self.closing:
            old_view=self.last_view
            short_key=self._short_key(view)
            if (short_key is not None and old_view is not None and old_view.frame_id!=view.frame_id
                and self._short_key(old_view)==short_key):
                self._stable_short_key=short_key
            elif short_key!=self._stable_short_key:
                self._stable_short_key=None
            previous=self.scroll_pending
            if previous is not None and self._scroll_last_frame!=view.frame_id:
                self._scroll_last_frame=view.frame_id
                old,new=previous.scrollbar,view.scrollbar
                moved=(old is not None and new is not None and new.frame_id==view.frame_id
                       and log_at_bottom(new) is not None and new.thumb_rect[1]>old.thumb_rect[1]+3
                       and view.body_fingerprint!=previous.body_fingerprint)
                if self._scroll_direction=='header':
                    moved=False
                    if overlap is not None and not self.expired:
                        span=OwnedLogSpan(request.request_id,request.session.battle_id,request.input_sequence,view,previous,overlap)
                        if span_header(span,request,view) is not None:
                            self._span=span
                            moved=True
                if moved:
                    self.scroll_pending=None
                    self.scroll_waits=0
                else:
                    self.scroll_waits+=1
        self.last_view=view
        header_needed=(request is not None and request.card_probe is not None and not self.expired
                       and not self.closing and self.scroll_pending is None and self._span is None
                       and self.scroll_attempts<2 and not view.headers and log_at_bottom(view.scrollbar) is True
                       and view.scrollbar is not None and view.scrollbar.frame_id==view.frame_id)
        return LogInspection(view,request,self.closing,self.scroll_pending is not None,self.scroll_attempts,self.expired,
                             self._tail_candidate(view) if request is not None and not self.expired and not self.closing else None,
                             inputs_invalidated=self._request_inputs_invalid or self._inputs_invalid,
                             short_list_stable=self._stable_short_key is not None and self._stable_short_key==self._short_key(view),
                             header_scroll_needed=header_needed,span=self._span)

    def scroll(self, inspection: LogInspection) -> None:
        if (self.request is None or inspection.request!=self.request or self.last_view!=inspection.viewport
            or inspection.scroll_attempts!=self.scroll_attempts or self.scroll_attempts>=2
            or inspection.closing_requested!=self.closing or inspection.scroll_pending!=(self.scroll_pending is not None)
            or self.expired or self.closing or self.scroll_pending is not None
            or log_inspection_action(inspection,self.request.session.battle_id)!='scroll'):
            raise ValueError('没有当前请求绑定的日志滚动证据')
        self.scroll_attempts+=1
        self.scroll_pending=inspection.viewport
        self._scroll_last_frame=inspection.viewport.frame_id
        self.scroll_waits=0
        self._scroll_direction='down'

    def scroll_to_header(self,inspection: LogInspection) -> None:
        if (self.request is None or inspection.request!=self.request or self.last_view!=inspection.viewport
            or inspection.scroll_attempts!=self.scroll_attempts or self.scroll_attempts>=2
            or self.expired or self.closing or self.scroll_pending is not None
            or self.request.card_probe is None or self._span is not None or inspection.viewport.headers
            or log_at_bottom(inspection.viewport.scrollbar) is not True
            or log_inspection_action(inspection,self.request.session.battle_id)!='scroll_header'):
            raise ValueError('没有当前请求的日志标题回查授权')
        self.scroll_attempts+=1
        self.scroll_pending=inspection.viewport
        self._scroll_last_frame=inspection.viewport.frame_id
        self.scroll_waits=0
        self._scroll_direction='header'

    def close(self, inspection: LogInspection) -> None:
        if self.closing or inspection.request!=self.request or inspection.viewport!=self.last_view:
            raise ValueError('日志关闭已请求，需等待新画面')
        self.closing=True
