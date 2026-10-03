from dataclasses import replace

from kaa.tasks.produce.new.play_cards.planner.feedback import (
    PendingPlay,
    PlayFeedbackStatus,
    evaluate_play_feedback,
)
from kaa.tasks.produce.new.play_cards.planner.state import BattleState


def _state(
    *,
    hand: tuple[str, ...] = ('played', 'other'),
    turn: int = 3,
    gap: int = 20,
    final: bool | None = True,
):
    return BattleState(
        remaining_turns=turn,
        hp=10,
        genki=5,
        hand_card_ids=hand,
        hand_size=len(hand),
        score_gap=gap,
        score_target_is_final=final,
    )


def test_feedback_confirms_card_leaving_visible_hand_on_same_turn():
    pending = PendingPlay('played', _state(), 25.0, 25.0)

    feedback = evaluate_play_feedback(pending, _state(hand=('other',)))

    assert feedback.status is PlayFeedbackStatus.CONFIRMED
    assert feedback.evidence == 'card_left_hand'
    assert feedback.observed_gap_reduction == 0


def test_feedback_records_visible_gap_change_without_claiming_exact_card_output():
    pending = PendingPlay('played', _state(), 25.0, 25.0)

    feedback = evaluate_play_feedback(pending, _state(gap=13))

    assert feedback.status is PlayFeedbackStatus.CONFIRMED
    assert feedback.observed_gap_reduction == 7


def test_feedback_does_not_confirm_gap_change_without_known_target_stage():
    pending = PendingPlay('played', _state(final=None), 25.0, 25.0)

    feedback = evaluate_play_feedback(pending, _state(gap=13, final=None))

    assert feedback.status is PlayFeedbackStatus.UNCONFIRMED
    assert feedback.observed_gap_reduction is None


def test_feedback_accepts_clear_to_perfect_target_transition():
    pending = PendingPlay('played', _state(gap=2, final=False), 5.0, 5.0)

    feedback = evaluate_play_feedback(pending, _state(gap=80, final=True))

    assert feedback.status is PlayFeedbackStatus.CONFIRMED
    assert feedback.evidence == 'clear_target_advanced'
    assert feedback.observed_gap_reduction is None


def test_feedback_does_not_call_unchanged_or_next_turn_hand_a_failed_click():
    pending = PendingPlay('played', _state(), 0.0, 0.0)

    unchanged = evaluate_play_feedback(pending, _state())
    next_turn = evaluate_play_feedback(pending, _state(hand=('other',), turn=2))

    assert unchanged.status is PlayFeedbackStatus.UNCONFIRMED
    assert next_turn.status is PlayFeedbackStatus.UNCONFIRMED


def test_feedback_trusts_card_move_dialog_as_explicit_play_evidence():
    pending = PendingPlay('played', _state(), 0.0, 0.0, move_dialog_seen=True)

    feedback = evaluate_play_feedback(pending, _state())

    assert feedback.status is PlayFeedbackStatus.CONFIRMED
    assert feedback.evidence == 'card_move_dialog'


def test_feedback_identifies_unchanged_paid_request_without_claiming_failure():
    pending = PendingPlay('played', _state(), 36.0, 36.0, requires_resource_change=True)
    feedback = evaluate_play_feedback(pending, _state())
    assert feedback.status is PlayFeedbackStatus.UNCONFIRMED
    assert feedback.evidence == 'paid_play_observably_unchanged'


def test_unchanged_paid_detection_requires_complete_known_resources_and_score():
    pending = PendingPlay('played', _state(), 36.0, 36.0, requires_resource_change=True)
    for after in (
        replace(_state(), hp_known=False),
        replace(_state(), genki_known=False),
        replace(_state(), hand_size=3),
        replace(_state(), hp=9),
        replace(_state(), score_target_is_final=None),
        replace(_state(), observed_effects=frozenset({'full_power'})),
    ):
        assert evaluate_play_feedback(pending, after).evidence != 'paid_play_observably_unchanged'



def test_zero_cost_same_name_redraw_can_be_confirmed_by_raw_hud_value():
    before = replace(_state(), observed_effect_values=(('full_power_point', 3),))
    pending = PendingPlay('played', before, 0, 0)
    after = replace(before, observed_effect_values=(('full_power_point', 5),))
    feedback = evaluate_play_feedback(pending, after)
    assert feedback.status is PlayFeedbackStatus.CONFIRMED
    assert feedback.evidence == 'hud_value_changed'


def test_zero_cost_resource_recovery_is_positive_evidence():
    pending = PendingPlay('played', _state(), 0, 0)
    assert evaluate_play_feedback(pending, replace(_state(), hp=12)).evidence == 'resource_changed'


def test_unknown_turn_fallback_cannot_confirm_same_turn_hand_change():
    pending = PendingPlay('played', replace(_state(), remaining_turns_known=False), 0, 0)
    assert evaluate_play_feedback(pending, _state(hand=('other',))).status is PlayFeedbackStatus.UNCONFIRMED
