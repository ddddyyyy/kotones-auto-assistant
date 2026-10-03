"""Conservative evidence for whether a requested card play took effect."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum

from .state import BattleState


class PlayFeedbackStatus(str, Enum):
    CONFIRMED = 'confirmed'
    UNCONFIRMED = 'unconfirmed'


@dataclass(frozen=True)
class PendingPlay:
    card_id: str
    before: BattleState
    projected_output: float
    confirmed_output: float
    move_dialog_seen: bool = False
    requires_resource_change: bool = False


@dataclass(frozen=True)
class PlayFeedback:
    status: PlayFeedbackStatus
    evidence: str
    observed_gap_reduction: int | None = None


def evaluate_play_feedback(pending: PendingPlay, after: BattleState) -> PlayFeedback:
    """Use only positive evidence; a redraw or animation can mimic a failed tap.

    The visible lesson gap can move for reasons besides the card, so its change
    is diagnostic rather than an exact measurement of that card's effect.
    """
    before = pending.before
    gap_reduction = (
        before.score_gap - after.score_gap
        if before.score_gap is not None
        and after.score_gap is not None
        and before.score_target_is_final is not None
        and before.score_target_is_final == after.score_target_is_final
        else None
    )
    if pending.move_dialog_seen:
        return PlayFeedback(
            PlayFeedbackStatus.CONFIRMED, 'card_move_dialog', gap_reduction
        )

    same_turn = (
        before.remaining_turns_known and after.remaining_turns_known
        and after.remaining_turns == before.remaining_turns
    )
    complete_hands = (
        before.hand_size == len(before.hand_card_ids)
        and after.hand_size == len(after.hand_card_ids)
    )
    if same_turn and complete_hands:
        before_count = Counter(before.hand_card_ids)[pending.card_id]
        after_count = Counter(after.hand_card_ids)[pending.card_id]
        if before_count > after_count:
            return PlayFeedback(
                PlayFeedbackStatus.CONFIRMED, 'card_left_hand', gap_reduction
            )

    if same_turn:
        if ((before.hp_known and after.hp_known and before.hp != after.hp)
                or (before.genki_known and after.genki_known and before.genki != after.genki)):
            return PlayFeedback(PlayFeedbackStatus.CONFIRMED, 'resource_changed', gap_reduction)
        before_values = dict(before.observed_effect_values)
        after_values = dict(after.observed_effect_values)
        if any(
            before_values[key] is not None and after_values[key] is not None
            and before_values[key] != after_values[key]
            for key in before_values.keys() & after_values.keys()
        ):
            return PlayFeedback(PlayFeedbackStatus.CONFIRMED, 'hud_value_changed', gap_reduction)
        if (before.is_exam and before.current_score is not None
                and after.current_score is not None and after.current_score > before.current_score):
            return PlayFeedback(PlayFeedbackStatus.CONFIRMED, 'exam_score_increased', gap_reduction)
        if gap_reduction is not None and gap_reduction > 0:
            return PlayFeedback(
                PlayFeedbackStatus.CONFIRMED,
                'lesson_gap_decreased',
                gap_reduction,
            )
        if before.score_target_is_final is False and after.score_target_is_final is True:
            return PlayFeedback(PlayFeedbackStatus.CONFIRMED, 'clear_target_advanced')

    unchanged_score = (
        gap_reduction == 0
        or (before.is_exam and before.current_score is not None
            and before.current_score == after.current_score)
    )
    if (
        pending.requires_resource_change
        and same_turn
        and complete_hands
        and Counter(before.hand_card_ids) == Counter(after.hand_card_ids)
        and before.hp_known and after.hp_known and before.hp == after.hp
        and before.genki_known and after.genki_known and before.genki == after.genki
        and unchanged_score
        and before.observed_effects == after.observed_effects
    ):
        # This is not proof of a failed tap: draws or refunds can cancel visible
        # changes. It is evidence that speculative effects must not accumulate
        # on repeated requests while the observable state remains identical.
        return PlayFeedback(
            PlayFeedbackStatus.UNCONFIRMED, 'paid_play_observably_unchanged', gap_reduction
        )

    return PlayFeedback(
        PlayFeedbackStatus.UNCONFIRMED, 'insufficient_evidence', gap_reduction
    )
