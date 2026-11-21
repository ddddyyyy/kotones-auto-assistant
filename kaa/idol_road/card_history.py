"""Owned card request evidence; it does not assert a game's previous-card status."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class OwnedCardPlay:
    battle_id: str
    input_sequence: int
    card_id: str
    category: str
    turn: int
    request_frame_id: str
    return_frame_id: str
    source: Literal['owned_same_turn_card_left_hand','owned_hud_card_left_hand'] = 'owned_same_turn_card_left_hand'
    previous_card_condition_confirmed: Literal[False] = False


@dataclass(frozen=True)
class PreviousRoundCard:
    last_play: OwnedCardPlay
    current_turn: int
    boundary_input_sequence: int
    boundary_frame_id: str
    source: Literal['owned_skip_log_return','owned_card_hud_return','owned_skip_hud_return'] = 'owned_skip_log_return'
    scope: Literal['previous_turn_last_owned_card_only'] = 'previous_turn_last_owned_card_only'


def freeze_previous_round(last_play: OwnedCardPlay | None, *, battle_id: str,
                          before_turn: int, current_turn: int, input_sequence: int,
                          frame_id: str) -> PreviousRoundCard | None:
    """Caller must first settle its owned SKIP log; no latest-card substitution."""
    if (last_play is None or not battle_id or last_play.battle_id!=battle_id
        or any(type(value) is not int for value in (before_turn,current_turn,input_sequence,last_play.turn,last_play.input_sequence))
        or before_turn<1 or current_turn!=before_turn+1 or last_play.turn!=before_turn
        or not 1<=last_play.input_sequence<input_sequence or not frame_id
        or frame_id in (last_play.request_frame_id,last_play.return_frame_id)
        or last_play.source!='owned_same_turn_card_left_hand'
        or last_play.category not in ('ProduceCardCategory_MentalSkill','ProduceCardCategory_ActiveSkill')):
        return None
    return PreviousRoundCard(last_play,current_turn,input_sequence,frame_id)
