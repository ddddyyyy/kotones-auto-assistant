"""Owned input HUD evidence, independent of the game's battle-log panel.

This collector records visible changes. It neither invents an absolute turn
nor clears the input ledger; mode-specific settlement consumes its evidence.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Literal
from .battle_inputs import BattleInputState,PendingBattleInput
from .battle_observation import BattleFrameObservation,current_absolute_turn
from .log_hand import LogHandCard,LogHandObservation,confirm_log_hand
from .extra_play import ExtraPlayObservation,extra_play_valid
from .schedule_countdown import ScheduleCountdown,confirm_schedule_countdown
from .queued_turns import queued_turns_valid


@dataclass(frozen=True)
class HudInputEvidence:
    pending: PendingBattleInput
    origin_hand: LogHandObservation
    first_frame: BattleFrameObservation
    frame: BattleFrameObservation
    hand: LogHandObservation
    observed_at: float
    change: Literal['requested_card_removed','hand_changed','skip_remaining_changed','skip_hand_changed','skip_control_changed']
    source: Literal['owned_input_stable_hud'] = 'owned_input_stable_hud'
    origin_countdown: ScheduleCountdown | None=None
    first_extra: ExtraPlayObservation | None=None
    extra: ExtraPlayObservation | None=None
    first_countdown: ScheduleCountdown | None=None
    countdown: ScheduleCountdown | None=None


def _hand_valid(frame: BattleFrameObservation,hand: LogHandObservation) -> bool:
    return bool(hand.frame_id==frame.values.frame_id and confirm_log_hand(
        hand.frame_id,hand.count,hand.cards,empty_confirmed=hand.empty_confirmed)==hand)


def _signature(card: LogHandCard) -> tuple[str,int | None,bool]:
    return card.card_id,card.upgrade_count,card.growth_marker


def _versions_exposed(origin: LogHandObservation,current: LogHandObservation,expected: Counter) -> bool:
    actual=Counter(_signature(card) for card in current.cards)
    removed,added=expected-actual,actual-expected
    def obscured(card: LogHandCard,hand: LogHandObservation) -> bool:
        center=card.rect[0]+card.rect[2]/2
        # The native cards are 200 px wide. A closer right neighbour covers
        # the suffix/upgrade area, even when the card itself is playable.
        return any(0<other.rect[0]+other.rect[2]/2-center<200 for other in hand.cards)
    unavailable=Counter(_signature(card) for card in origin.cards if not card.available or obscured(card,origin))
    available=Counter(_signature(card) for card in current.cards if card.available)
    for signature in added:
        # A different available copy cannot vouch for a disabled copy's
        # newly read version when both now share the same signature.
        if any(_signature(card)==signature and not card.available for card in current.cards):
            available[signature]=0
    def kinds(versions: Counter) -> Counter:
        result=Counter()
        for (card_id,_upgrade,growth),count in versions.items():result[(card_id,growth)]+=count
        return result
    # Reflow can expose a suffix while its card remains disabled. Require
    # actual old overlap and a complete new suffix area for that same kind;
    # a moved or dim title alone does not authorize a version change.
    exposed_dim=Counter(_signature(card) for card in current.cards
        if not card.available and not obscured(card,current))
    for signature in added:
        if any(_signature(card)==signature and not card.available and obscured(card,current)
            for card in current.cards):
            exposed_dim[signature]=0
    previously_covered=Counter(_signature(card) for card in origin.cards if obscured(card,origin)) & removed
    needed_dim=added-available
    if (removed-unavailable or needed_dim-exposed_dim
        or kinds(needed_dim)-kinds(previously_covered)):
        return False
    return kinds(removed)==kinds(added)


class HudFeedbackWindow:
    """Two distinct current captures under the input's original deadline.

    No OCR, screenshots, clicks, log requests, resource prediction or retries.
    Stable unchanged hands never confirm a play, including redraw duplicates.
    """
    def __init__(self,inputs: BattleInputState,hand: LogHandObservation,*,origin_countdown: ScheduleCountdown | None=None):
        pending=inputs.pending
        if (pending is None or inputs.expired or inputs.battle_id!=pending.frame.session.battle_id
            or inputs.sequence!=pending.sequence or current_absolute_turn(pending.frame) is None
            or pending.frame.values.detail_mode!='complete' or not _hand_valid(pending.frame,hand)
            or (pending.origin_hand is not None and pending.origin_hand!=hand)):
            raise ValueError('hud_feedback_origin_unconfirmed')
        self.origin=inputs
        self.pending=pending
        self.hand=hand
        self.origin_countdown=origin_countdown
        self._check_cues(pending.frame.values.frame_id,None,origin_countdown)
        self.previous: tuple[tuple,BattleFrameObservation,ExtraPlayObservation | None,ScheduleCountdown | None] | None=None
        self.result: HudInputEvidence | None=None

    @staticmethod
    def _check_cues(frame_id: str,extra: ExtraPlayObservation | None,countdown: ScheduleCountdown | None) -> None:
        if extra is not None and not extra_play_valid(extra,frame_id):
            raise ValueError('hud_feedback_extra_play_unconfirmed')
        if countdown is not None and (countdown.frame_id!=frame_id or confirm_schedule_countdown(
            countdown.frame_id,countdown.rect,countdown.reads,countdown.scaled_reads,countdown.digits)!=countdown):
            raise ValueError('hud_feedback_countdown_unconfirmed')

    def interrupt(self) -> None:
        """An intervening unreadable/animation capture breaks stability."""
        self.previous=None
        self.result=None

    def observe(self,inputs: BattleInputState,frame: BattleFrameObservation,
                hand: LogHandObservation | None,*,now: float,extra: ExtraPlayObservation | None=None,
                countdown: ScheduleCountdown | None=None) -> HudInputEvidence | None:
        pending=self.pending
        if (inputs!=self.origin or not pending.started<=now<pending.started+20):
            raise ValueError('hud_feedback_input_changed_or_expired')
        if frame.session!=pending.frame.session:
            raise ValueError('hud_feedback_session_changed')
        self._check_cues(frame.values.frame_id,extra,countdown)
        if self.result is not None:
            return self.result if frame==self.result.frame and hand==self.result.hand else None
        values,state=frame.values,frame.observation
        if values.queued_turns is not None and not queued_turns_valid(values.queued_turns,values.frame_id,values.remaining_turns):
            self.interrupt()
            return None
        if (values.resource_state!='current' or not values.hud_confirmed
            or state.source!='confirmed_current_battle_observation'
            or state.frame_id!=values.frame_id or not values.frame_id
            or values.frame_id==pending.frame.values.frame_id or state.conflicted_fields
            or (state.battle_id,state.character_id,state.stage)!=(frame.session.battle_id,frame.session.character_id,frame.session.stage)
            or type(state.hp) is not int or state.hp<0
            or type(values.remaining_turns) is not int or not 1<=values.remaining_turns<=30
            or hand is None or not _hand_valid(frame,hand)):
            self.previous=None
            return None
        # Geometry and availability are included in the validated fingerprint.
        queued=values.queued_turns
        queue_key=(queued.base_remaining,queued.queued) if queued is not None else None
        key=(values.remaining_turns,state.hp,hand.fingerprint,extra.value if extra else None,queue_key)
        previous=self.previous
        self.previous=(key,frame,extra,countdown)
        if previous is None or previous[0]!=key or previous[1].values.frame_id==values.frame_id:
            return None
        change=None
        if pending.kind=='skip':
            if values.remaining_turns!=pending.frame.values.remaining_turns:
                change='skip_remaining_changed'
            elif Counter((c.card_id,c.upgrade_count,c.growth_marker) for c in hand.cards)!=Counter(
                (c.card_id,c.upgrade_count,c.growth_marker) for c in self.hand.cards):
                change='skip_hand_changed'
            elif (extra is not None and extra.value is not None and pending.extra_play is not None
                and pending.extra_play.value is not None and extra.value!=pending.extra_play.value) or (
                countdown is not None and countdown.turns_after is not None and self.origin_countdown is not None
                and self.origin_countdown.turns_after is not None and countdown.turns_after!=self.origin_countdown.turns_after):
                change='skip_control_changed'
            elif (queued is not None and pending.frame.values.queued_turns is not None
                and queued.remaining is not None and queue_key!=(pending.frame.values.queued_turns.base_remaining,
                    pending.frame.values.queued_turns.queued)):
                change='skip_control_changed'
        else:
            before=Counter(card.card_id for card in self.hand.cards)
            after=Counter(card.card_id for card in hand.cards)
            if pending.card_id and before[pending.card_id]>0:
                expected=before.copy();expected[pending.card_id]-=1
                variants={(card.upgrade_count,card.growth_marker) for card in self.hand.cards
                          if card.card_id==pending.card_id}
                if pending.card_upgrade is not None:
                    variants={variant for variant in variants if variant[0]==pending.card_upgrade}
                detailed=Counter(_signature(card) for card in self.hand.cards)
                if len(variants)==1:
                    upgrade,growth=next(iter(variants))
                    detailed[(pending.card_id,upgrade,growth)]-=1
                if (after==+expected and len(variants)==1
                    and Counter(_signature(card) for card in hand.cards)==+detailed):
                    change='requested_card_removed'
                elif (after==+expected and len(variants)==1 and before[pending.card_id]==1
                    and _versions_exposed(self.hand,hand,+detailed)):
                    # A formerly dimmed or overlapped neighbour may expose its
                    # upgrade glyph after this play. Its ID still matches; do
                    # not reuse its old version for the next planning frame.
                    # This proves only the unique requested card's removal.
                    change='requested_card_removed'
                elif before!=after:
                    change='hand_changed'
        if change is None:
            return None
        self.result=HudInputEvidence(pending,self.hand,previous[1],frame,hand,now,change,
            origin_countdown=self.origin_countdown,first_extra=previous[2],extra=extra,
            first_countdown=previous[3],countdown=countdown)
        return self.result
