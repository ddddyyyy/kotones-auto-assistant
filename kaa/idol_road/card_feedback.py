"""Settle a proven hand change before reading the next planning state."""
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from kaa.tasks.produce.new.play_cards.planner.feedback import PendingPlay, PlayFeedback, evaluate_play_feedback
from .card_probe import card_feedback_ready,PendingCardPermit
from .log_hand import confirm_log_hand
from .input_policy import InputBoundPolicy
from .battle_observation import current_feedback_turn
from .models import Scene
if TYPE_CHECKING:
    from .page import PageSnapshot


@dataclass(frozen=True)
class CardHandFeedback:
    frame_id: str
    hand_fingerprint: str
    feedback: PlayFeedback
    source: Literal['owned_return_complete_hand'] = 'owned_return_complete_hand'
    input_sequence: int | None = None
    settled_at: float | None = None
    permit: PendingCardPermit | None = None
    capture_digest: str | None = None


def settle_card_hand_feedback(policy: InputBoundPolicy, snapshot: 'PageSnapshot',
                              pending: PendingPlay, *, allow_round_advance: bool=False) -> CardHandFeedback | None:
    """Settle owned return evidence without OCR or simulated resource gain.

    The small internal state is feedback-only and never returned for planning.
    By default only card-left-hand evidence settles here. An explicit opt-in
    also accepts an observed next round as UNCONFIRMED feedback, without
    attributing the transition to the card. The original deadline still applies.
    """
    frame, hand = snapshot.battle_frame, snapshot.log_hand
    obs = snapshot.observation
    resolution, returned = obs.battle_card_resolution, obs.battle_log_return
    inputs = policy.inputs
    if (obs.scene != Scene.BATTLE or obs.battle_log_pending
        or frame is None or hand is None or resolution is None or returned is None
        or policy.frame() != frame or obs.battle_log_hand != hand
        or not card_feedback_ready(inputs.session, inputs.snapshot(), frame, resolution, now=inputs.clock())
        or hand.frame_id != frame.values.frame_id
        or confirm_log_hand(hand.frame_id, hand.count, hand.cards, empty_confirmed=hand.empty_confirmed) != hand
        or returned.return_frame_id != frame.values.frame_id or returned.request.request_id != resolution.request_id
        or returned.source != 'owned_log_return_candidate'
        or returned.request.card_probe != resolution.permit or not returned.hud_unchanged
        or not returned.remaining_consistent or not returned.input_sequence_consistent
        or returned.request.hand_fingerprint != hand.fingerprint
        or resolution.permit.pending.before != pending.before or resolution.permit.pending.card_id != pending.card_id):
        return None
    absolute = current_feedback_turn(frame)
    if absolute is None or type(frame.values.remaining_turns) is not int:
        return None
    hp = frame.observation.hp
    state = BattleState(frame.values.remaining_turns, hp if type(hp) is int else 0, 0,
        hp_known=type(hp) is int, genki_known=False, is_exam=True, current_score=None,
        hand_size=hand.count, hand_card_ids=tuple(card.card_id for card in hand.cards),
        absolute_turn_observation=absolute)
    feedback = evaluate_play_feedback(pending, state)
    advanced=(allow_round_advance is True and resolution.status=='next_round_observed'
        and absolute.turn==resolution.permit.pending.turn+1
        and feedback.evidence=='turn_advanced_without_play_observation')
    if feedback.evidence != 'card_left_hand' and not advanced:
        return None
    settled_at=inputs.clock()
    policy.feedback(pending, feedback, state)
    return CardHandFeedback(frame.values.frame_id, hand.fingerprint, feedback,
        input_sequence=inputs.sequence,settled_at=settled_at,permit=resolution.permit,capture_digest=snapshot.capture_digest)
