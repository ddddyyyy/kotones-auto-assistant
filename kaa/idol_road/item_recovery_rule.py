"""Visible previous-card recovery conditions, separate from predicted events."""
from dataclasses import dataclass
import re
from typing import Literal, TYPE_CHECKING

from .item_description import ItemDescriptionBinding, ItemDescriptionCandidate, bind_item_description
from .card_history import PreviousRoundCard
from .planning_turn import read_planning_turn
if TYPE_CHECKING:
    from kaa.db.skill_card import SkillCard
    from kaa.tasks.produce.new.play_cards.planner.state import BattleState
    from .battle_inputs import BattleInputs
    from .battle_observation import BattleFrameObservation


@dataclass(frozen=True)
class ItemRecoveryDeclaration:
    item_id: str
    current_category: str
    previous_category: str
    recovery: int
    timing: Literal['card_use'] = 'card_use'
    cross_round_semantics: Literal['unverified', 'previous_turn_last_card'] = 'unverified'
    timing_source: Literal['unverified', 'user_confirmed_previous_turn_last_card'] = 'unverified'
    effect_order: Literal['unverified'] = 'unverified'
    remaining_uses: None = None
    executable: Literal[False] = False


@dataclass(frozen=True)
class ItemRecoveryAnalysis:
    binding: ItemDescriptionBinding
    declaration: ItemRecoveryDeclaration | None
    status: Literal['confirmed_visible_declaration', 'identity_unconfirmed', 'unsupported_description']


@dataclass(frozen=True)
class ItemRecoveryCategoryCheck:
    verdict: Literal['satisfied', 'not_satisfied', 'unknown']
    reason: str
    previous_round: PreviousRoundCard | None = None
    scope: Literal['category_clause_only'] = 'category_clause_only'
    recovery_guaranteed: Literal[False] = False


def check_item_recovery_categories(rule: ItemRecoveryDeclaration, inputs: 'BattleInputs',
                                   frame: 'BattleFrameObservation', state: 'BattleState',
                                   card: 'SkillCard') -> ItemRecoveryCategoryCheck:
    """Check categories only; not equipment, remaining uses, order or HP gain."""
    if not inputs.can_select(frame,state) or card._id not in state.hand_card_ids:
        return ItemRecoveryCategoryCheck('unknown','current_card_unbound')
    if (rule.cross_round_semantics!='previous_turn_last_card'
        or rule.timing_source!='user_confirmed_previous_turn_last_card'):
        return ItemRecoveryCategoryCheck('unknown','timing_unconfirmed')
    kinds=('ProduceCardCategory_MentalSkill','ProduceCardCategory_ActiveSkill')
    if card.category not in kinds or rule.current_category not in kinds or rule.previous_category not in kinds:
        return ItemRecoveryCategoryCheck('unknown','category_unknown')
    if card.category!=rule.current_category:
        return ItemRecoveryCategoryCheck('not_satisfied','current_category_mismatch')
    turn=read_planning_turn(state,inputs.session.initial_total_turns or 0)
    previous=inputs.previous_round_card
    if (turn is None or previous is None or previous.current_turn!=turn.turn
        or previous.last_play.turn!=turn.turn-1 or previous.last_play.battle_id!=inputs.session.battle_id
        or previous.source not in ('owned_skip_log_return','owned_card_hud_return','owned_skip_hud_return')
        or not (previous.last_play.input_sequence==previous.boundary_input_sequence<=inputs.sequence
            if previous.source=='owned_card_hud_return'
            else previous.last_play.input_sequence<previous.boundary_input_sequence<=inputs.sequence)):
        return ItemRecoveryCategoryCheck('unknown','previous_round_last_unknown')
    return ItemRecoveryCategoryCheck('satisfied' if previous.last_play.category==rule.previous_category else 'not_satisfied',
        'previous_round_category_match' if previous.last_play.category==rule.previous_category else 'previous_round_category_mismatch',
        previous)


def interpret_item_recovery(binding: ItemDescriptionBinding,
                            candidates: tuple[ItemDescriptionCandidate, ...] | None, *,
                            previous_turn_timing_confirmed: bool=False) -> ItemRecoveryAnalysis:
    """Recheck identity evidence before interpreting the complete declaration.

    The caller may supply user-confirmed timing for its bound stage context;
    that knowledge must not be generalized to every similarly worded item.
    An owned latest-card record alone cannot establish the previous round's
    condition or recovery before another cost/effect.
    """
    checked = bind_item_description(binding.reference, binding.reads, candidates)
    if checked != binding or checked.status != 'resolved' or checked.item_id is None:
        return ItemRecoveryAnalysis(binding, None, 'identity_unconfirmed')
    match = re.fullmatch(
        r'(アクティブ|メンタル)スキルカード使用時、直前に(アクティブ|メンタル)'
        r'スキルカードを使用した状態の場合、体力回復([1-9][0-9]{0,2})',
        checked.confirmed_description or '')
    if match is None:
        return ItemRecoveryAnalysis(binding, None, 'unsupported_description')
    categories = {'アクティブ': 'ProduceCardCategory_ActiveSkill', 'メンタル': 'ProduceCardCategory_MentalSkill'}
    return ItemRecoveryAnalysis(binding, ItemRecoveryDeclaration(checked.item_id,
        categories[match[1]], categories[match[2]], int(match[3]),
        cross_round_semantics='previous_turn_last_card' if previous_turn_timing_confirmed is True else 'unverified',
        timing_source='user_confirmed_previous_turn_last_card' if previous_turn_timing_confirmed is True else 'unverified'),
        'confirmed_visible_declaration')
