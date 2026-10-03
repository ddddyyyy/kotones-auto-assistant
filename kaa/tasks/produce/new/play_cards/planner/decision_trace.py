"""Opt-in structured decision/feedback evidence; no extra device operations."""

from dataclasses import asdict
from enum import Enum
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from kotonebot import logging

if TYPE_CHECKING:
    from ..ui import CardGameObject
    from .evaluator import CardEvaluation
    from .feedback import PlayFeedback
    from .state import BattleState

from .costs import play_cost, special_cost_affordability

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
