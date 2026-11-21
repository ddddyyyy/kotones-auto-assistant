"""自有日志返回帧的回合锚点；不跨帧推算或接管前景战斗。"""
from dataclasses import dataclass,replace
from typing import Literal
import re
from kaa.battle_turn import AbsoluteTurnObservation
from .battle_observation import BattleObservationSession,BattleFrameObservation
from .battle_inputs import BattleInputState,PendingBattleInput
from .battle_log import LogReturn,LogViewport,LogEntryEvidence,tail_log_header
from .skip_probe import SkipLogResolution
from .card_probe import CardLogResolution,assess_card_log
from .log_hand import LogHandObservation,confirm_log_hand


@dataclass(frozen=True)
class InitialLogAnchor:
    battle_id: str
    return_frame_id: str
    request_id: str
    log_frame_id: str
    input_sequence: int
    absolute_turn: AbsoluteTurnObservation
    observed_frame: BattleFrameObservation
    hand_fingerprint: str
    source: Literal['owned_initial_log_return','owned_skip_log_return','owned_card_log_return'] = 'owned_initial_log_return'
    scope: Literal['exact_return_frame_only'] = 'exact_return_frame_only'
    feedback_pending: PendingBattleInput | None = None


@dataclass(frozen=True)
class InitialHudRecheck:
    frame_id: str
    source_request_id: str
    verdict: Literal['input_not_idle','context_unconfirmed','visible_fields_changed',
                     'insufficient_visible_fields','visible_fields_unchanged']
    hp_matches: bool | None = None
    remaining_matches: bool | None = None
    hand_matches: bool | None = None
    requires_fresh_owned_log: Literal[True] = True
    currentness_confirmed: Literal[False] = False


@dataclass(frozen=True)
class RoundAnchorState:
    anchor: InitialLogAnchor | None
    reason: str
    recheck: InitialHudRecheck | None = None


def recheck_initial_hud(anchor: InitialLogAnchor,frame: BattleFrameObservation,inputs: BattleInputState,
                        entry: LogEntryEvidence | None,hand: LogHandObservation | None=None) -> InitialHudRecheck:
    if (inputs.battle_id!=anchor.battle_id or type(inputs.sequence) is not int or inputs.sequence!=0
        or inputs.pending is not None or inputs.expired):
        return InitialHudRecheck(frame.values.frame_id,anchor.request_id,'input_not_idle')
    values,state=frame.values,frame.observation
    if (frame.session!=anchor.observed_frame.session or values.resource_state!='current' or not values.hud_confirmed
        or values.reader_source!='ranked_hud_same_frame'
        or not values.frame_id or state.frame_id!=values.frame_id or state.source!='confirmed_current_battle_observation'
        or (state.battle_id,state.character_id,state.stage)!=(frame.session.battle_id,frame.session.character_id,frame.session.stage)
        or {'identity','turn'}&set(state.conflicted_fields)):
        return InitialHudRecheck(values.frame_id,anchor.request_id,'context_unconfirmed')
    hp=(state.hp==anchor.observed_frame.observation.hp if type(state.hp) is int and state.hp>=0
        and 'hp' not in state.conflicted_fields else None)
    remaining=(values.remaining_turns==anchor.absolute_turn.remaining_turns
               if type(values.remaining_turns) is int and 1<=values.remaining_turns<=30 else None)
    hand_matches=None
    if (hand is not None and hand.frame_id==values.frame_id
        and confirm_log_hand(hand.frame_id,hand.count,hand.cards,empty_confirmed=hand.empty_confirmed)==hand):
        hand_matches=hand.fingerprint==anchor.hand_fingerprint
    if (entry is not None and entry.frame_id==values.frame_id and .97<=entry.score<=1
        and re.fullmatch('[0-9a-f]{64}',entry.hand_fingerprint) is not None):
        if hand is not None and entry.hand_fingerprint!=hand.fingerprint:
            hand_matches=None
        elif hand is None:
            hand_matches=entry.hand_fingerprint==anchor.hand_fingerprint
    checks=(hp,remaining,hand_matches)
    verdict=('visible_fields_changed' if any(value is False for value in checks) else
             'insufficient_visible_fields' if any(value is None for value in checks) else 'visible_fields_unchanged')
    return InitialHudRecheck(values.frame_id,anchor.request_id,verdict,hp,remaining,hand_matches)


def confirm_initial_log_anchor(session: BattleObservationSession,frame: BattleFrameObservation,
                               returned: LogReturn,view: LogViewport,inputs: BattleInputState) -> InitialLogAnchor | None:
    values,state,request=frame.values,frame.observation,returned.request
    if (session.identity_source!='confirmed_formation_start_request' or session.turn_budget_mode!='variable_budget'
        or frame.session!=session or request.session!=session or not session.battle_id or values.detail_mode!='complete'
        or type(session.initial_total_turns) is not int or not 1<=session.initial_total_turns<=30
        or values.resource_state!='current' or not values.hud_confirmed or not values.frame_id
        or state.source!='confirmed_current_battle_observation' or state.frame_id!=values.frame_id
        or (state.battle_id,state.character_id,state.stage)!=(session.battle_id,session.character_id,session.stage)
        or {'identity','turn','hp'}&set(state.conflicted_fields)
        or frame.absolute_turn is not None or state.turn is not None or state.turn_source is not None
        or returned.source!='owned_log_return_candidate' or not returned.hud_unchanged
        or not returned.remaining_consistent or not returned.input_sequence_consistent
        or returned.return_frame_id!=values.frame_id or returned.log_frame_id!=view.frame_id
        or not request.request_id or not request.opening_frame_id or request.opening_frame_id==values.frame_id
        or not isinstance(request.hand_fingerprint,str) or re.fullmatch('[0-9a-f]{64}',request.hand_fingerprint) is None
        or inputs.battle_id!=session.battle_id or inputs.pending is not None or inputs.expired
        or type(inputs.sequence) is not int or inputs.sequence!=0
        or type(request.input_sequence) is not int or request.input_sequence!=inputs.sequence
        or values.remaining_turns!=session.initial_total_turns or request.remaining_turns!=values.remaining_turns
        or state.hp!=request.hp or type(state.hp) is not int or state.hp<0):
        return None
    header=tail_log_header(view)
    if (header is None or header!=returned.tail_header or header.absolute_turn!=1
        or header.remaining_turns!=values.remaining_turns):
        return None
    remaining=values.remaining_turns
    if type(remaining) is not int:
        return None
    absolute=AbsoluteTurnObservation(session.battle_id,values.frame_id,1,remaining)
    return InitialLogAnchor(session.battle_id,values.frame_id,request.request_id,view.frame_id,0,absolute,frame,request.hand_fingerprint)


class InitialRoundTracker:
    def __init__(self,session: BattleObservationSession):
        self.session=session
        self.state=RoundAnchorState(None,'await_owned_initial_log')
        self._seen_requests: set[str]=set()
        self._last_initial: InitialLogAnchor | None=None

    def invalidate(self,reason: str) -> None:
        self.state=RoundAnchorState(None,reason)
        if reason in ('battle_input_requested','page_action_requested'):
            self._last_initial=None

    def accept(self,frame: BattleFrameObservation,returned: LogReturn,view: LogViewport,
               inputs: BattleInputState) -> bool:
        if returned.request.request_id in self._seen_requests or len(self._seen_requests)>=128:
            self.invalidate('log_request_already_consumed_or_limit')
            return False
        self._seen_requests.add(returned.request.request_id)
        anchor=confirm_initial_log_anchor(self.session,frame,returned,view,inputs)
        self.state=RoundAnchorState(anchor,'owned_initial_return_confirmed' if anchor else 'initial_return_unconfirmed')
        if anchor is not None:
            self._last_initial=anchor
        return anchor is not None

    def accept_skip(self,frame: BattleFrameObservation,returned: LogReturn,resolution: SkipLogResolution,
                    inputs: BattleInputState) -> bool:
        permit=resolution.permit
        if (resolution.status!='next_round_confirmed' or resolution.turn is None or resolution.remaining_turns is None
            or self.session.identity_source!='confirmed_formation_start_request'
            or returned.request.skip_probe!=permit or permit.pending.frame.session!=self.session
            or returned.request.request_id!=resolution.request_id or returned.log_frame_id!=resolution.log_frame_id
            or returned.return_frame_id!=resolution.return_frame_id or not returned.hud_unchanged
            or not returned.remaining_consistent or not returned.input_sequence_consistent
            or returned.tail_header is None or returned.tail_header.absolute_turn!=resolution.turn
            or returned.tail_header.remaining_turns!=resolution.remaining_turns
            or frame.session!=self.session or self.session.turn_budget_mode!='variable_budget'
            or inputs.battle_id!=self.session.battle_id or inputs.pending is not None or inputs.expired
            or type(inputs.sequence) is not int or inputs.sequence!=permit.sequence or resolution.request_id in self._seen_requests
            or len(self._seen_requests)>=128 or resolution.return_frame_id!=frame.values.frame_id
            or resolution.turn!=permit.pending.turn+1 or frame.values.remaining_turns!=resolution.remaining_turns
            or frame.values.detail_mode not in ('complete','feedback_only') or frame.values.resource_state!='current' or not frame.values.hud_confirmed
            or (frame.values.detail_mode=='feedback_only' and (resolution.settled_at is None
                or not permit.pending.started<=resolution.settled_at<permit.deadline))
            or frame.observation.source!='confirmed_current_battle_observation' or frame.observation.frame_id!=frame.values.frame_id
            or frame.absolute_turn is not None or frame.observation.turn is not None):
            self.invalidate('skip_return_unconfirmed')
            return False
        self._seen_requests.add(resolution.request_id)
        absolute=AbsoluteTurnObservation(self.session.battle_id,frame.values.frame_id,resolution.turn,resolution.remaining_turns)
        anchor=InitialLogAnchor(self.session.battle_id,frame.values.frame_id,resolution.request_id,
            resolution.log_frame_id or '',inputs.sequence,absolute,frame,returned.request.hand_fingerprint,'owned_skip_log_return')
        self._last_initial=None
        self.state=RoundAnchorState(anchor,'owned_skip_return_confirmed')
        return True

    def accept_card(self,frame: BattleFrameObservation,returned: LogReturn,view: LogViewport,
                    resolution: CardLogResolution,inputs: BattleInputState,*,now: float) -> bool:
        """待结阶段只产出反馈回合；输入仍由原ledger阻挡，不能借日志先行释放。"""
        if (self.session.turn_budget_mode!='variable_budget' or frame.session!=self.session
            or resolution.status not in ('same_round_observed','next_round_observed')
            or resolution.observed_frame!=frame or resolution.turn is None or resolution.remaining_turns is None
            or frame.values.detail_mode not in ('complete','feedback_only') or frame.absolute_turn is not None or frame.observation.turn is not None
            or assess_card_log(resolution.permit,returned,view,frame,inputs,now=now)!=resolution
            or resolution.request_id in self._seen_requests or len(self._seen_requests)>=128):
            self.invalidate('card_return_unconfirmed')
            return False
        self._seen_requests.add(resolution.request_id)
        absolute=AbsoluteTurnObservation(self.session.battle_id,frame.values.frame_id,resolution.turn,resolution.remaining_turns)
        anchor=InitialLogAnchor(self.session.battle_id,frame.values.frame_id,resolution.request_id,
            resolution.log_frame_id or '',inputs.sequence,absolute,frame,returned.request.hand_fingerprint,
            'owned_card_log_return',feedback_pending=inputs.pending)
        self._last_initial=None
        self.state=RoundAnchorState(anchor,'owned_card_return_for_feedback')
        return True

    def bind(self,frame: BattleFrameObservation,inputs: BattleInputState,*,entry: LogEntryEvidence | None=None,
             hand: LogHandObservation | None=None) -> BattleFrameObservation:
        def strip_anchor(value: BattleFrameObservation) -> BattleFrameObservation:
            if value.absolute_turn is None and value.observation.turn_source not in ('owned_initial_log_return','owned_skip_log_return','owned_card_log_return'):
                return value
            return replace(value,absolute_turn=None,observation=replace(value.observation,turn=None,turn_source=None))
        anchor=self.state.anchor
        if anchor is None:
            if self._last_initial is not None:
                self.state=replace(self.state,recheck=recheck_initial_hud(self._last_initial,frame,inputs,entry,hand))
            return strip_anchor(frame)
        if (frame.session!=self.session or inputs.battle_id!=anchor.battle_id or type(inputs.sequence) is not int
            or inputs.sequence!=anchor.input_sequence
            or (inputs.pending is not None and inputs.pending!=anchor.feedback_pending)
            or inputs.expired or frame.values.frame_id!=anchor.return_frame_id
            or (frame.values.detail_mode!='complete' and not (
                frame.values.detail_mode=='feedback_only' and anchor.source in ('owned_card_log_return','owned_skip_log_return')))
            or frame.values.resource_state!='current' or not frame.values.hud_confirmed
            or frame.observation.source!='confirmed_current_battle_observation'
            or frame.observation.frame_id!=frame.values.frame_id
            or frame.values.remaining_turns!=anchor.absolute_turn.remaining_turns
            or {'identity','turn','hp'}&set(frame.observation.conflicted_fields)
            or strip_anchor(frame)!=anchor.observed_frame):
            self.invalidate('return_frame_or_input_changed')
            self.state=replace(self.state,recheck=recheck_initial_hud(anchor,frame,inputs,entry,hand))
            return strip_anchor(frame)
        return replace(frame,absolute_turn=anchor.absolute_turn,
            observation=replace(frame.observation,turn=anchor.absolute_turn.turn,turn_source=anchor.source))

    def complete_card_details(self,partial: BattleFrameObservation,full: BattleFrameObservation,
                              inputs: BattleInputState) -> bool:
        """Advance only the already settled card's same-capture observation.

        The caller verifies the capture pixels and settlement receipt before
        reading details. This method grants no input and consumes no new log.
        """
        return self._complete_input_details(partial,full,inputs,source='owned_card_log_return')

    def complete_skip_details(self,partial: BattleFrameObservation,full: BattleFrameObservation,
                              inputs: BattleInputState) -> bool:
        """Same-capture details only after the caller checks a timely SKIP receipt."""
        return self._complete_input_details(partial,full,inputs,source='owned_skip_log_return')

    def _complete_input_details(self,partial: BattleFrameObservation,full: BattleFrameObservation,
                                inputs: BattleInputState,*,source: Literal['owned_card_log_return','owned_skip_log_return']) -> bool:
        anchor=self.state.anchor
        raw=replace(partial,absolute_turn=None,
            observation=replace(partial.observation,turn=None,turn_source=None))
        if (anchor is None or anchor.source!=source
            or (source=='owned_card_log_return' and anchor.feedback_pending is None)
            or (source=='owned_skip_log_return' and anchor.feedback_pending is not None)
            or anchor.observed_frame!=raw or partial.values.detail_mode!='feedback_only'
            or inputs.battle_id!=self.session.battle_id or inputs.pending is not None or inputs.expired
            or inputs.sequence!=anchor.input_sequence or full.session!=self.session
            or full.values.frame_id!=partial.values.frame_id or full.values.detail_mode!='complete'
            or full.values.resource_state!='current' or not full.values.hud_confirmed
            or full.values.reader_source!=partial.values.reader_source
            or full.values.remaining_turns!=partial.values.remaining_turns or full.values.hp!=partial.values.hp
            or full.observation.hp!=partial.observation.hp
            or full.observation.source!='confirmed_current_battle_observation'
            or (full.observation.battle_id,full.observation.character_id,full.observation.stage,full.observation.frame_id)
                !=(self.session.battle_id,self.session.character_id,self.session.stage,partial.values.frame_id)
            or {'identity','turn','hp','max_hp'}&set(full.observation.conflicted_fields)
            or full.absolute_turn is not None or full.observation.turn is not None
            or full.observation.turn_source is not None):
            return False
        self.state=RoundAnchorState(replace(anchor,observed_frame=full),
            'owned_card_details_completed' if source=='owned_card_log_return' else 'owned_skip_details_completed')
        return True
