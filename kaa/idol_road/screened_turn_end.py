"""Recognized hand with no HP-eligible play may request a separately checked SKIP."""
from collections import Counter
from kaa.db.skill_card import SkillCard
from kaa.tasks.produce.new.play_cards.planner.selection_policy import SelectionConstraint
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from .battle_inputs import BattleInputs
from .battle_observation import BattleFrameObservation
from .hp_safety import HpSelectionReview
from .log_hand import LogHandObservation,confirm_log_hand


def screened_turn_end(inputs: BattleInputs,frame: BattleFrameObservation,state: BattleState,
                       cards: tuple[SkillCard,...],choice: SelectionConstraint,
                       review: HpSelectionReview | None,hand: LogHandObservation | None) -> SelectionConstraint | None:
    """A scheduling proposal, not input authorization or promised field rewards.

    The runner must independently check current SKIP availability and the HP
    end-turn guard. Unknown/partial hands and invalid screening cannot enter.
    """
    if (not inputs.can_select(frame,state) or not choice.stop or choice.allowed_indices!=()
        or not choice.reason.startswith('hp_safety:allowed=[]')
        or review is None or review.frame_id!=frame.values.frame_id
        or review.turn!=frame.observation.turn or review.allowed_indices!=()
        or review.current_route is None or review.current_route.status not in ('safe','below_reserve','not_applicable')
        or not cards or len(review.candidates)!=len(cards)
        or any(row.index!=i or row.card_id!=card._id or row.reason=='allowed'
               for i,(row,card) in enumerate(zip(review.candidates,cards)))
        or hand is None or hand.frame_id!=frame.values.frame_id
        or confirm_log_hand(hand.frame_id,hand.count,hand.cards,empty_confirmed=hand.empty_confirmed)!=hand
        or hand.count!=state.hand_size
        or tuple(row.card_id for row in hand.cards)!=state.hand_card_ids):
        return None
    full=Counter(state.hand_card_ids)
    if Counter(card._id for card in cards)-full:return None
    return SelectionConstraint(end_turn=True,allowed_indices=(),
        reason='screened_turn_end:no_hp_eligible_play,check_current_skip_and_next_turn_hp')
