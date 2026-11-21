"""本程序的战斗输入序号与待结请求；不从画面变化猜出牌成功。"""
from __future__ import annotations
from dataclasses import dataclass,replace
from collections import Counter
from time import monotonic
from typing import Callable,Literal,TYPE_CHECKING
if TYPE_CHECKING:
    from .hud_round import HudRoundResolution
    from kaa.db.skill_card import SkillCard
    from .log_hand import LogHandObservation
    from .extra_play import ExtraPlayObservation
    from .hud_feedback import HudInputEvidence
    from kaa.tasks.produce.new.play_cards.planner.feedback import PendingPlay,PlayFeedback
    from kaa.tasks.produce.new.play_cards.planner.state import BattleState
    from .battle_log import LogReturn,LogViewport
    from .skip_probe import SkipLogResolution
from .battle_observation import BattleFrameObservation,BattleObservationSession,current_absolute_turn,current_feedback_turn
from .planning_turn import read_planning_turn
from .card_history import OwnedCardPlay,PreviousRoundCard,freeze_previous_round


@dataclass(frozen=True)
class PendingBattleInput:
    sequence: int
    kind: Literal['card','skip']
    frame: BattleFrameObservation
    turn: int
    card_id: str | None
    started: float
    before: BattleState | None=None
    card_category: str | None=None
    card_upgrade: int | None=None
    extra_play: 'ExtraPlayObservation | None'=None
    origin_hand: 'LogHandObservation | None'=None


@dataclass(frozen=True)
class BattleInputState:
    battle_id: str
    sequence: int
    pending: PendingBattleInput | None
    expired: bool


class BattleInputs:
    def __init__(self,session: BattleObservationSession,*,clock: Callable[[],float]=monotonic):
        if session.identity_source!='confirmed_formation_start_request' or not session.battle_id:
            raise ValueError('输入记录缺少本程序的开战会话')
        self.session=session
        self.clock=clock
        self.sequence=0
        self.pending: PendingBattleInput | None=None
        self._last_owned_card_play: OwnedCardPlay | None=None
        self._previous_round_card: PreviousRoundCard | None=None
        self._requested_route: list[str]=[]
        self._hud_round_pending=False
        self._hud_settled_evidence: HudInputEvidence | None=None
        self._hud_resolution: HudRoundResolution | None=None
        self._hud_settled_at: float | None=None

    @property
    def hud_round_pending(self) -> bool:
        """A settled HUD play still needs a new, producer-bound round frame."""
        return self._hud_round_pending

    @property
    def requested_route(self) -> tuple[str,...]:
        """Accepted owned input requests, not evidence of observed card effects."""
        return tuple(self._requested_route)

    @property
    def previous_round_card(self) -> PreviousRoundCard | None:
        """Frozen previous-round evidence, independent of this round's plays."""
        return self._previous_round_card if self.pending is None and not self._hud_round_pending else None

    @property
    def last_owned_card_play(self) -> OwnedCardPlay | None:
        """Last confirmed owned request, hidden during any unresolved input.

        A settled SKIP preserves historical evidence. It does not prove the
        game's previous-card condition survives a turn boundary.
        """
        return self._last_owned_card_play if self.pending is None and not self._hud_round_pending else None

    def snapshot(self) -> BattleInputState:
        return BattleInputState(self.session.battle_id,self.sequence,self.pending,
            self.pending is not None and self.clock()-self.pending.started>=20)

    def _turn(self, frame: BattleFrameObservation,*,hand_feedback: bool=False) -> int | None:
        state,values=frame.observation,frame.values
        if (frame.session!=self.session or values.resource_state!='current' or not values.hud_confirmed
            or state.source!='confirmed_current_battle_observation' or not values.frame_id
            or state.frame_id!=values.frame_id or {'identity','turn'}&set(state.conflicted_fields)
            or (state.battle_id,state.character_id,state.stage)!=(self.session.battle_id,self.session.character_id,self.session.stage)
            or type(state.turn) is not int or state.turn<1):
            return None
        if self.session.turn_budget_mode=='variable_budget':
            absolute=current_feedback_turn(frame) if hand_feedback else current_absolute_turn(frame)
            return absolute.turn if absolute is not None else None
        if self.session.turn_budget_mode!='fixed_budget' or state.turn_source!='fixed_budget_assumption':
            return None
        total,remaining=self.session.initial_total_turns,values.remaining_turns
        if type(total) is not int or type(remaining) is not int or not 1<=remaining<=total<=30:
            return None
        return state.turn if state.turn==total-remaining+1 else None

    def begin(self, kind: Literal['card','skip'], frame: BattleFrameObservation, card_id: str | None=None,*,
              extra_play: 'ExtraPlayObservation | None'=None,origin_hand: 'LogHandObservation | None'=None) -> None:
        turn=self._turn(frame)
        if (self.pending is not None or self._hud_round_pending or turn is None or frame.values.detail_mode!='complete' or kind not in ('card','skip')
            or (kind=='card' and not card_id) or (kind=='skip' and card_id is not None)):
            raise ValueError('上一输入尚未结算或当前输入缺少可靠同场回合')
        if extra_play is not None:
            from .extra_play import extra_play_valid
            if type(extra_play.value) is not int or not extra_play_valid(extra_play,frame.values.frame_id):
                raise ValueError('额外出牌次数不属于当前输入帧')
        if origin_hand is not None:
            from .log_hand import confirm_log_hand
            if (origin_hand.frame_id!=frame.values.frame_id or confirm_log_hand(origin_hand.frame_id,
                origin_hand.count,origin_hand.cards,empty_confirmed=origin_hand.empty_confirmed)!=origin_hand):
                raise ValueError('原手牌不属于当前输入帧')
            if kind=='card' and not any(card.card_id==card_id for card in origin_hand.cards):
                raise ValueError('所选卡牌不在当前完整手牌中')
        self.sequence+=1
        self.pending=PendingBattleInput(self.sequence,kind,frame,turn,card_id,self.clock(),extra_play=extra_play,origin_hand=origin_hand)
        self._requested_route.append(f'{turn}:{kind}:{card_id or ""}')
        if kind=='card':
            self._last_owned_card_play=None

    def _matches(self, frame: BattleFrameObservation, state: BattleState,*,hand_feedback: bool=False) -> bool:
        if frame.values.detail_mode!='complete' and not (hand_feedback and frame.values.detail_mode=='feedback_only'
            and self.session.turn_budget_mode=='variable_budget'):
            return False
        turn=self._turn(frame,hand_feedback=hand_feedback)
        planning=read_planning_turn(state,self.session.initial_total_turns or 0)
        if turn is None or planning is None or planning.turn!=turn or state.remaining_turns!=frame.values.remaining_turns:
            return False
        if self.session.turn_budget_mode=='variable_budget':
            return planning.source=='absolute' and state.absolute_turn_observation==(
                current_feedback_turn(frame) if hand_feedback else current_absolute_turn(frame))
        return planning.source=='fixed_budget'

    def can_select(self, frame: BattleFrameObservation | None, state: BattleState) -> bool:
        """Check selection before evaluation or device input; never settle a request.

        Variable battles require the same observed absolute-turn frame. Fixed
        battles keep their existing round-budget contract. Resource simulation
        and incomplete probe frames cannot authorize another selection.
        """
        return bool(self.pending is None and not self._hud_round_pending and frame is not None and self._matches(frame,state)
            and (not state.hp_known or (type(state.hp) is int and state.hp==frame.observation.hp)))

    def bind_card_state(self, state: BattleState, card_id: str, *, category: str | None=None,
                        upgrade_count: int | None=None) -> None:
        request=self.pending
        if (request is None or request.kind!='card' or request.card_id!=card_id or request.before is not None
            or self.snapshot().expired or not self._matches(request.frame,state)):
            raise ValueError('出牌前状态不能绑定到本次输入请求')
        if upgrade_count is not None and (type(upgrade_count) is not int or upgrade_count<0):
            raise ValueError('所选卡牌强化版本未知或无效')
        if request.extra_play is not None:
            extra=request.extra_play.value
            if extra is None or state.playable_count!=1+extra:
                raise ValueError('出牌前次数与当前画面不一致')
        if request.origin_hand is not None and (state.hand_size!=request.origin_hand.count
            or Counter(state.hand_card_ids)!=Counter(card.card_id for card in request.origin_hand.cards)):
            raise ValueError('出牌前状态与绑定原手牌不一致')
        known_category=category if category in ('ProduceCardCategory_ActiveSkill','ProduceCardCategory_MentalSkill') else None
        self.pending=replace(request,before=state,card_category=known_category,card_upgrade=upgrade_count)

    def finish_card(self, pending: PendingPlay, feedback: PlayFeedback, state: BattleState,
                    frame: BattleFrameObservation) -> bool:
        from kaa.tasks.produce.new.play_cards.planner.feedback import PlayFeedbackStatus
        request=self.pending
        feedback_turn=current_feedback_turn(frame)
        feedback_only_safe=(feedback.status is PlayFeedbackStatus.CONFIRMED and feedback.evidence=='card_left_hand') or (
            feedback.status is PlayFeedbackStatus.UNCONFIRMED
            and feedback.evidence=='turn_advanced_without_play_observation'
            and request is not None and feedback_turn is not None
            and feedback_turn.turn==request.turn+1)
        if (request is None or request.kind!='card' or request.before is None
            or request.before!=pending.before or request.card_id!=pending.card_id or self.snapshot().expired
            or frame.values.frame_id==request.frame.values.frame_id or not self._matches(frame,state,
                hand_feedback=feedback_only_safe)):
            return False
        turn=self._turn(frame,hand_feedback=feedback_only_safe)
        if turn is None or turn<request.turn:
            return False
        # 换回合只结束旧输入的等待，不代表那张牌成功；原反馈仍保持unknown。
        if turn>request.turn or feedback.status is PlayFeedbackStatus.CONFIRMED:
            if turn>request.turn:
                # Advancing without a confirmed final play leaves that round's
                # last category unknown, even if an earlier card was confirmed.
                self._previous_round_card=None
            if (turn==request.turn and feedback.status is PlayFeedbackStatus.CONFIRMED
                and feedback.evidence=='card_left_hand' and request.card_category is not None
                and request.card_id is not None):
                self._last_owned_card_play=OwnedCardPlay(self.session.battle_id,request.sequence,
                    request.card_id,request.card_category,turn,request.frame.values.frame_id,frame.values.frame_id)
            self.pending=None
            return True
        return False

    def observe_hud(self, frame: BattleFrameObservation) -> None:
        request=self.pending
        if request is None or request.kind!='skip' or self.snapshot().expired:
            return
        turn=self._turn(frame)
        if turn is not None and turn>request.turn and frame.values.frame_id!=request.frame.values.frame_id:
            self._previous_round_card=None
            self.pending=None

    def finish_hud_card(self,evidence: 'HudInputEvidence') -> bool:
        """Confirm only the requested card's observed removal.

        This does not assign a round to the returned HUD or authorize another
        selection. The mode's round producer must independently bind it.
        """
        from .hud_settlement import confirmed_hud_card_removal
        if not confirmed_hud_card_removal(self.session,self.snapshot(),evidence,now=self.clock()):
            return False
        settled=self.clock()
        if not evidence.pending.started<=settled<evidence.pending.started+20 or self.snapshot().expired:return False
        request=self.pending
        assert request is not None and request.card_id is not None
        if request.card_category is not None:
            self._last_owned_card_play=OwnedCardPlay(self.session.battle_id,request.sequence,
                request.card_id,request.card_category,request.turn,request.frame.values.frame_id,evidence.frame.values.frame_id,
                source='owned_hud_card_left_hand')
        self.pending=None
        self._hud_round_pending=True
        self._hud_settled_evidence=evidence
        self._hud_settled_at=settled
        return True

    @property
    def hud_settled_at(self) -> float | None:
        return self._hud_settled_at

    def finish_hud_resolution(self,resolution: 'HudRoundResolution',*,card: 'SkillCard | None'=None,
                               allow_audited_redraw: bool=False) -> bool:
        from .hud_round import resolve_hud_round
        current=resolve_hud_round(self.snapshot(),resolution.evidence,now=self.clock(),card=card,
            allow_audited_redraw=allow_audited_redraw)
        if current!=resolution or self.pending is None:return False
        request=self.pending;settled=self.clock()
        if not request.started<=settled<request.started+20 or self.snapshot().expired:return False
        if request.kind=='card':
            self._last_owned_card_play=None
            if resolution.play_confirmed and request.card_id and request.card_category:
                self._last_owned_card_play=OwnedCardPlay(self.session.battle_id,request.sequence,request.card_id,
                    request.card_category,request.turn,request.frame.values.frame_id,resolution.evidence.frame.values.frame_id,
                    source='owned_hud_card_left_hand')
        self.pending=None
        self._hud_round_pending=True
        self._hud_settled_evidence=resolution.evidence
        self._hud_resolution=resolution
        self._hud_settled_at=settled
        return True

    def complete_hud_round(self,frame: BattleFrameObservation,resolution: 'HudRoundResolution') -> bool:
        pending=resolution.evidence.pending
        absolute=current_absolute_turn(frame)
        if (not self._hud_round_pending or self.pending is not None or self._hud_resolution!=resolution
            or self._hud_settled_evidence!=resolution.evidence or self._hud_settled_at is None
            or not pending.started<=self._hud_settled_at<pending.started+20
            or self.sequence!=pending.sequence or frame.session!=self.session or absolute is None
            or absolute.turn!=resolution.turn or frame.observation.turn_source!=resolution.source
            or frame.values.frame_id!=resolution.evidence.frame.values.frame_id
            or frame.values.remaining_turns!=resolution.evidence.frame.values.remaining_turns
            or frame.observation.hp!=resolution.evidence.frame.observation.hp):
            return False
        if resolution.turn==pending.turn+1:
            self._previous_round_card=None
            last=self._last_owned_card_play
            if (last is not None and last.turn==pending.turn
                and ((pending.kind=='skip' and last.input_sequence<pending.sequence)
                    or (pending.kind=='card' and resolution.play_confirmed and last.input_sequence==pending.sequence))):
                self._previous_round_card=PreviousRoundCard(last,resolution.turn,pending.sequence,
                    frame.values.frame_id,source=resolution.source)
        self._hud_round_pending=False
        self._hud_settled_evidence=None
        self._hud_resolution=None
        return True

    def finish_skip_log(self,returned: LogReturn,view: LogViewport,frame: BattleFrameObservation) -> SkipLogResolution | None:
        from .skip_probe import pending_skip_permit,assess_skip_log
        permit=pending_skip_permit(self.session,self.snapshot(),now=self.clock())
        if permit is None:
            return None
        settled_at=self.clock()
        resolution=assess_skip_log(permit,returned,view,frame,self.snapshot(),now=settled_at)
        if resolution.status=='next_round_confirmed':
            resolution=replace(resolution,settled_at=settled_at)
            self._previous_round_card=freeze_previous_round(self._last_owned_card_play,
                battle_id=self.session.battle_id,before_turn=permit.pending.turn,
                current_turn=resolution.turn,input_sequence=permit.pending.sequence,frame_id=frame.values.frame_id
                ) if resolution.turn is not None else None
            self.pending=None
        return resolution
