"""Local card-root projection for bounded mechanism diagnostics, not action authorization."""
from dataclasses import dataclass, replace
from typing import Literal
from kaa.db.skill_card import SkillCard
from kaa.db.constants import ProduceExamEffectType as Effect
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from kaa.tasks.produce.new.play_cards.planner.simulator import can_afford_play, simulate_play
from kaa.tasks.produce.new.play_cards.planner.costs import play_cost
from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict, evaluate_trigger_id


@dataclass(frozen=True)
class CurrentActionAssessment:
    card_id: str
    upgrade: int | None
    root_projection_available: bool = False
    reason: str = 'unsupported_root_shape'
    green_cost: int | None = None
    red_cost: int | None = None
    hp_after: int | None = None
    genki_after: int | None = None
    good_condition_after: int | None = None
    playable_count_after: int | None = None
    hand_transition: str = 'unknown'
    requires_fresh_hand: bool = True
    unresolved_events: tuple[str, ...] = ('equipped_item_triggers_unmodeled',)
    scope: Literal['card_root_effects_only'] = 'card_root_effects_only'
    root_effects_supported: bool = False


def assess_current_action(state: BattleState, card: SkillCard) -> CurrentActionAssessment:
    result = CurrentActionAssessment(card._id, card.upgrade_count)
    # First family is deliberately exact: good condition, redraw remaining hand,
    # one additional play. Other families need their own audited contracts.
    if (state.projected_effect_values is not None or state.projected_card_growth
        or state.projected_resource_restrictions or state.projected_resource_gain_modifiers
        or state.projected_stamina_consumption is not None
        or card._id in state.unresolved_growth_card_ids or card.move_produce_exam_effects
        or card._produce_card_status_enchant_id or len(card.play_effects) != 3
        or card.play_move_position_type != 'ProduceCardMovePositionType_Lost'
        or card.stamina != 3 or card.force_stamina != 0 or card.cost_value != 0
        or card.cost_type != 'ExamCostType_Unknown'):
        return result
    good=card.play_effects[0].produce_exam_effect
    # Audited DB variants: base/+ /++ /+++ differ only in this duration.
    # Payment, redraw, extra play and Lost destination retain their full checks.
    if good is None or type(good.effect_turn) is not int or good.effect_turn not in (1,3,4,5):
        return result
    shapes = ((Effect.ExamParameterBuff, good.effect_turn, 0),
              (Effect.ExamHandGraveCountCardDraw, 0, 0),
              (Effect.ExamPlayableValueAdd, 0, 1))
    for segment, (kind, turns, count) in zip(card.play_effects, shapes):
        effect = segment.produce_exam_effect
        if (effect is None or any(type(v) is not int for v in
                (effect.effect_turn,effect.effect_count,effect.effect_value1,effect.effect_value2))
            or effect.effect_type != kind or effect.effect_turn != turns
            or effect.effect_count != count or effect.effect_value1 != 0 or effect.effect_value2 != 0
            or effect.chain_effect_ids or effect._produce_exam_status_enchant_id
            or effect._produce_card_status_enchant_id or effect._produce_card_search_id
            or effect._target_produce_card_id or effect._produce_card_grow_effect_ids not in (None, '', '[]')
            or evaluate_trigger_id(segment._produce_exam_trigger_id, state) is not TriggerVerdict.ACTIVE):
            return result
    if evaluate_trigger_id(card._play_produce_exam_trigger_id or '',state) is not TriggerVerdict.ACTIVE:
        return result
    result=replace(result,root_effects_supported=True)
    if (evaluate_trigger_id(card._play_produce_exam_trigger_id or '', state) is not TriggerVerdict.ACTIVE
        or not state.hp_known or not state.genki_known or type(state.hp) is not int or state.hp < 0
        or type(state.genki) is not int or state.genki < 0 or type(state.playable_count) is not int
        or state.playable_count < 1 or can_afford_play(state, card) is not True):
        return replace(result, reason='payment_or_current_resources_unknown')
    green, red = play_cost(state, card)
    after = simulate_play(state, card)
    if not after.hp_known or not after.genki_known:
        return replace(result, reason='root_resources_unknown')
    good = dict(state.observed_effect_values).get('parameter_buff')
    return replace(result, root_projection_available=True, reason='audited_redraw_roots_only',
        green_cost=green, red_cost=red, hp_after=after.hp, genki_after=after.genki,
        good_condition_after=after.buffs.parameter_buff_turns if type(good) is int else None,
        playable_count_after=after.playable_count,
        hand_transition='replace_remaining_with_unobserved_draws',
        unresolved_events=('equipped_item_triggers_unmodeled', 'remaining_hand_to_grave',
                           'drawn_card_identity_and_on_move', 'played_card_to_lost'))
