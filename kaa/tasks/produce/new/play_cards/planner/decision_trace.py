"""Opt-in structured decision/feedback evidence; no extra device operations."""

from dataclasses import asdict
from enum import Enum
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from kotonebot import logging
from kaa.db.card_status_enchant import get_card_status_enchant

if TYPE_CHECKING:
    from ..ui import CardGameObject
    from .evaluator import CardEvaluation
    from .feedback import PlayFeedback
    from .state import BattleState

from .costs import play_cost, special_cost_affordability
from .simulator import can_afford_play

logger = logging.getLogger(__name__)
TRACE_ENV = 'KAA_PLANNER_TRACE_DIR'


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f'Unsupported trace value: {type(value).__name__}')


class DecisionTrace:
    """One bounded JSONL per battle. State is inferred, not ground truth."""

    def __init__(self, directory: Path):
        self.path = directory / f'decisions-{uuid4().hex}.jsonl'
        self.count = 0
        self.screening_count = 0
        self.pending: int | None = None

    @classmethod
    def from_environment(cls) -> 'DecisionTrace | None':
        directory = os.environ.get(TRACE_ENV)
        return cls(Path(directory)) if directory else None

    def _append(self, payload: dict) -> None:
        serialized = json.dumps(payload, ensure_ascii=False, default=_json_default)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(serialized + '\n')

    def screening(
        self,
        hands: list['CardGameObject | None'],
        state: 'BattleState | None',
        evaluated: list[tuple['CardGameObject', 'CardEvaluation']],
        admitted: list[tuple['CardGameObject', 'CardEvaluation']],
        final: list[tuple['CardGameObject', 'CardEvaluation']],
        outcome: str,
    ) -> None:
        """Record each visible slot and why it left selection, without capture.

        Object identity keeps duplicate card IDs distinct. This separate event
        neither creates a pending play nor clears an earlier feedback link.
        """
        if self.screening_count >= 256:
            return
        self.screening_count += 1
        try:
            evaluations = {id(hand): evaluation for hand, evaluation in evaluated}
            admitted_ids = {id(hand) for hand, _ in admitted}
            final_ids = {id(hand) for hand, _ in final}
            rows = []
            for slot, hand in enumerate(hands):
                card = hand.card if hand is not None else None
                evaluation = evaluations.get(id(hand))
                if card is None:
                    reason = 'unrecognized_card'
                elif hand is not None and not hand.available:
                    reason = 'unavailable_card'
                elif evaluation is None:
                    reason = 'no_evaluation'
                elif id(hand) not in admitted_ids:
                    reason = 'modeled_unaffordable'
                elif id(hand) not in final_ids:
                    reason = 'clear_priority_filter'
                else:
                    reason = 'eligible'
                rows.append({
                    'slot': slot,
                    'card_id': card._id if card is not None else None,
                    'reason': reason,
                    'evaluation': asdict(evaluation) if evaluation is not None else None,
                    'modeled_affordable': can_afford_play(state, card)
                    if state is not None and card is not None else None,
                    'modeled_green_red': play_cost(state, card)
                    if state is not None and card is not None else None,
                    'cost_model': 'stance_only_existing_rounding',
                })
            self._append({
                'schema_version': 1, 'event': 'candidate_screening',
                'attempt': self.screening_count, 'outcome': outcome,
                'decision_sequence': self.count + 1
                if outcome == 'selected' and self.count < 256 else None,
                'state': asdict(state) if state is not None else None,
                'hands': rows,
            })
        except Exception:
            logger.warning('Could not record planner candidate screening.', exc_info=True)

    def decision(
        self, state: 'BattleState',
        candidates: list[tuple['CardGameObject', 'CardEvaluation']],
        selected: 'CardEvaluation',
    ) -> None:
        self.pending = None
        if self.count >= 256:
            return
        self.count += 1
        try:
            self._append({
                'schema_version': 1, 'event': 'decision', 'sequence': self.count,
                'state': asdict(state), 'selected': asdict(selected),
                'candidates': [
                    {
                        'evaluation': asdict(evaluation),
                        'upgrade_count': getattr(hand.card, 'upgrade_count', None),
                        'stamina': getattr(hand.card, 'stamina', None),
                        'force_stamina': getattr(hand.card, 'force_stamina', None),
                        'cost_type': getattr(hand.card, 'cost_type', None),
                        'cost_value': getattr(hand.card, 'cost_value', None),
                        'modeled_payment': {
                            'green_red': play_cost(state, hand.card),
                            'special_affordable': special_cost_affordability(state, hand.card),
                            'model': 'stance_only_existing_rounding',
                        } if hand.card is not None else None,
                        'customize_ids': getattr(hand.card, 'customize_ids', None),
                        'card_status_enchant_id': getattr(hand.card, '_produce_card_status_enchant_id', None),
                        'card_status_enchant': _card_enchant_metadata(hand.card),
                        'effects': [
                            {
                                'trigger_id': getattr(effect, '_produce_exam_trigger_id', ''),
                                'effect_id': getattr(effect.produce_exam_effect, '_id', None),
                                'effect_type': getattr(effect.produce_exam_effect, 'effect_type', None),
                                'value1': getattr(effect.produce_exam_effect, 'effect_value1', None),
                                'value2': getattr(effect.produce_exam_effect, 'effect_value2', None),
                                'count': getattr(effect.produce_exam_effect, 'effect_count', None),
                                'turn': getattr(effect.produce_exam_effect, 'effect_turn', None),
                                'chain_effect_id': getattr(effect.produce_exam_effect, '_chain_produce_exam_effect_id', None),
                                'chain_effect_ids': getattr(effect.produce_exam_effect, 'chain_effect_ids', ()),
                                'grow_effect_ids': getattr(effect.produce_exam_effect, '_produce_card_grow_effect_ids', None),
                                'search_id': getattr(effect.produce_exam_effect, '_produce_card_search_id', None),
                                'status_enchant_id': getattr(effect.produce_exam_effect, '_produce_exam_status_enchant_id', None),
                            }
                            for effect in hand.card.play_effects
                        ] if hand.card is not None else [],
                    }
                    for hand, evaluation in candidates
                ],
            })
            self.pending = self.count
        except Exception:
            logger.warning('Could not record planner decision trace.', exc_info=True)

    def feedback(self, feedback: 'PlayFeedback', after: 'BattleState | None' = None) -> None:
        pending, self.pending = self.pending, None
        if pending is None:
            return
        try:
            self._append({
                'schema_version': 1, 'event': 'feedback', 'sequence': pending,
                'feedback': asdict(feedback),
                'after_state': asdict(after) if after is not None else None,
            })
        except Exception:
            logger.warning('Could not record planner feedback trace.', exc_info=True)


def _card_enchant_metadata(card) -> dict | None:
    enchant_id = getattr(card, '_produce_card_status_enchant_id', None)
    if not enchant_id:
        return None
    enchant = get_card_status_enchant(enchant_id)
    return {
        'definition': asdict(enchant) if enchant is not None else None,
        'remaining_trigger_count': None,
        'status': 'unobserved_per_instance',
    }
