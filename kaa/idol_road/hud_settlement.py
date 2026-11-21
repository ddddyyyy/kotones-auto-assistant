"""Validate current HUD card consumption without assigning the next round."""
from collections import Counter
from .battle_inputs import BattleInputState
from .battle_observation import BattleObservationSession
from .hud_feedback import HudInputEvidence,HudFeedbackWindow


def validated_hud_evidence(session: BattleObservationSession,inputs: BattleInputState,
                            evidence: HudInputEvidence,*,now: float) -> bool:
    pending=inputs.pending
    if (pending is None or pending!=evidence.pending or pending.frame.session!=session
        or evidence.source!='owned_input_stable_hud'
        or not pending.started<=evidence.observed_at<=now<pending.started+20):
        return False
    try:
        window=HudFeedbackWindow(inputs,evidence.origin_hand,origin_countdown=evidence.origin_countdown)
        first_hand=type(evidence.hand)(evidence.first_frame.values.frame_id,evidence.hand.count,
            evidence.hand.cards,evidence.hand.fingerprint,evidence.hand.empty_confirmed)
        if window.observe(inputs,evidence.first_frame,first_hand,now=evidence.observed_at,
            extra=evidence.first_extra,countdown=evidence.first_countdown) is not None:
            return False
        return window.observe(inputs,evidence.frame,evidence.hand,now=evidence.observed_at,
            extra=evidence.extra,countdown=evidence.countdown)==evidence
    except ValueError:
        return False


def confirmed_hud_card_removal(session: BattleObservationSession,inputs: BattleInputState,
                               evidence: HudInputEvidence,*,now: float) -> bool:
    pending=inputs.pending
    return bool(pending is not None and pending.kind=='card' and pending.before is not None and pending.card_id
        and evidence.change=='requested_card_removed'
        and Counter(pending.before.hand_card_ids)==Counter(card.card_id for card in evidence.origin_hand.cards)
        and pending.before.hand_size==evidence.origin_hand.count
        and validated_hud_evidence(session,inputs,evidence,now=now))
