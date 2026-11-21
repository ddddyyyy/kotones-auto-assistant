"""Mode-specific HUD round evidence, never initial-minus-remaining arithmetic."""
from dataclasses import dataclass
from typing import Literal
from kaa.db.skill_card import SkillCard
from kaa.db.constants import ProduceExamEffectType
from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play
from .battle_inputs import BattleInputState
from .hud_feedback import HudInputEvidence
from .hud_settlement import validated_hud_evidence
from .current_hp_projection import supports_current_hp_projection
from .current_action import assess_current_action
from .native_payment import supports_native_payment_conflict


@dataclass(frozen=True)
class HudRoundResolution:
    evidence: HudInputEvidence
    turn: int
    play_confirmed: bool
    source: Literal['owned_card_hud_return','owned_skip_hud_return']
    basis: Literal['owned_skip_transition','unchanged_preview_countdown',
        'observed_remaining_plays','same_preview_boundary','remaining_turn_decrement','queued_turn_increment']


def resolve_hud_round(inputs: BattleInputState,evidence: HudInputEvidence,*,now: float,
                       card: SkillCard | None=None,allow_audited_redraw: bool=False) -> HudRoundResolution | None:
    pending=evidence.pending;session=pending.frame.session
    if not validated_hud_evidence(session,inputs,evidence,now=now):return None
    if pending.kind=='skip':
        if evidence.change not in ('skip_remaining_changed','skip_hand_changed','skip_control_changed'):return None
        return HudRoundResolution(evidence,pending.turn+1,False,'owned_skip_hud_return','owned_skip_transition')
    before=pending.before
    if (before is not None and card is not None and card._id in before.native_payment_conflict_card_ids
        and not supports_native_payment_conflict(card)):
        return None
    if (card is None or before is None or pending.card_upgrade is None
        or card._id!=pending.card_id or card.upgrade_count!=pending.card_upgrade
        or pending.extra_play is None or pending.extra_play.value is None
        or before.playable_count!=1+pending.extra_play.value
        or not supports_current_hp_projection(card,before,allow_audited_redraw=allow_audited_redraw)):
        return None
    origin,first,last=evidence.origin_countdown,evidence.first_countdown,evidence.countdown
    projected=simulate_play(before,card)
    preview_turn=None
    if (origin is not None and first is not None and last is not None
        and origin.turns_after is not None and origin.turns_after>1
        and first.turns_after is not None and first.turns_after==last.turns_after):
        preview_turn=pending.turn+origin.turns_after-first.turns_after
        if preview_turn not in (pending.turn,pending.turn+1):return None
    remaining=pending.frame.values.remaining_turns;after=evidence.frame.values.remaining_turns
    if type(remaining) is not int or type(after) is not int:return None
    basis: Literal['unchanged_preview_countdown','observed_remaining_plays',
        'same_preview_boundary','remaining_turn_decrement','queued_turn_increment']
    actual_plays=(1+evidence.extra.value if evidence.first_extra is not None and evidence.extra is not None
        and evidence.extra.value is not None and evidence.first_extra.value==evidence.extra.value else None)
    if projected.playable_count>0 and actual_plays is not None and actual_plays<projected.playable_count:return None
    if projected.playable_count>0 and remaining==after:
        if actual_plays is not None and actual_plays>=projected.playable_count:
            turn=pending.turn;basis='observed_remaining_plays'
            if origin is not None and origin.turns_after==1:basis='same_preview_boundary'
        elif preview_turn==pending.turn:
            turn=pending.turn;basis='unchanged_preview_countdown'
        else:return None
    elif projected.playable_count==0:
        if remaining-after==1:
            turn=pending.turn+1;basis='remaining_turn_decrement'
        elif remaining==after and _queue_increment(evidence):
            turn=pending.turn+1;basis='queued_turn_increment'
        elif remaining==after and preview_turn==pending.turn+1:
            turn=pending.turn+1;basis='unchanged_preview_countdown'
        else:
            # An added turn may keep the budget unchanged at a preview rollover.
            # The root prediction alone cannot prove that transition.
            return None
    else:return None
    if preview_turn is not None and preview_turn!=turn:return None
    confirmed=evidence.change=='requested_card_removed'
    if not confirmed and evidence.change=='hand_changed' and turn==pending.turn:
        direct_draw=any(segment.produce_exam_effect is not None
            and segment.produce_exam_effect.effect_type==ProduceExamEffectType.ExamCardDraw
            for segment in getattr(card,'play_effects',()))
        current_redraw=allow_audited_redraw and assess_current_action(before,card).root_effects_supported
        confirmed=((direct_draw or current_redraw)
            and sum(c.card_id==pending.card_id for c in evidence.origin_hand.cards)==1
            and all(c.card_id!=pending.card_id for c in evidence.hand.cards))
    if not confirmed and turn==pending.turn:return None
    return HudRoundResolution(evidence,turn,confirmed,'owned_card_hud_return',basis)


def _queue_increment(evidence: HudInputEvidence) -> bool:
    before=evidence.pending.frame.values.queued_turns
    first=evidence.first_frame.values.queued_turns
    last=evidence.frame.values.queued_turns
    return (before is not None and first is not None and last is not None
        and type(before.base_remaining) is int and type(before.queued) is int
        and type(last.base_remaining) is int and type(last.queued) is int
        and (first.base_remaining,first.queued)==(last.base_remaining,last.queued)
        and before.base_remaining-last.base_remaining==1 and last.queued-before.queued==1)
