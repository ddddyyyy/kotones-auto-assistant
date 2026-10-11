"""Opt-in structured decision/feedback evidence; no extra device operations."""

from dataclasses import asdict
from enum import Enum
import hashlib
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

from .costs import play_cost, special_cost_affordability, projected_cost_supported
from .simulator import can_afford_play
from .trace_provenance import configured_solution, local_code_provenance

logger = logging.getLogger(__name__)
TRACE_ENV = 'KAA_PLANNER_TRACE_DIR'
INPUT_EVIDENCE_ENV = 'KAA_PLANNER_INPUT_EVIDENCE'


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f'Unsupported trace value: {type(value).__name__}')


class DecisionTrace:
    """One bounded JSONL per battle. State is inferred, not ground truth."""

    def __init__(self, directory: Path, *, save_inputs: bool = False):
        self.path = directory / f'decisions-{uuid4().hex}.jsonl'
        self.save_inputs = save_inputs
        self.input_attempts = 0
        self._code_provenance: dict | None = None
        self.count = 0
        self.screening_count = 0
        self.pending: int | None = None

    @classmethod
    def from_environment(cls) -> 'DecisionTrace | None':
        directory = os.environ.get(TRACE_ENV)
        return cls(Path(directory), save_inputs=os.environ.get(INPUT_EVIDENCE_ENV) == '1') if directory else None

    def _append(self, payload: dict) -> None:
        serialized = json.dumps(payload, ensure_ascii=False, default=_json_default)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(serialized + '\n')

    def _input_evidence(self, hands: list['CardGameObject | None']) -> dict | None:
        if not self.save_inputs:
            return None
        if self.input_attempts >= 32:
            return {'status': 'limit_reached'}
        # Read stored inputs only: no device operation, lazy property or OCR.
        frames = [vars(hand).get('_screenshot') for hand in hands if hand is not None]
        if not frames or any(frame is None for frame in frames):
            return {'status': 'snapshot_missing'}
        frame = frames[0]
        if frame is None:
            return {'status': 'snapshot_missing'}
        if any(other is not frame for other in frames):
            return {'status': 'mixed_snapshots'}
        self.input_attempts += 1
        try:
            import cv2
            from kaa.kaa_context import config_name, produce_session
            session = produce_session()
            if self._code_provenance is None:
                self._code_provenance = local_code_provenance()
            ok, encoded = cv2.imencode('.png', frame)
            if not ok:
                raise OSError('PNG encoding failed')
            data = encoded.tobytes()
            relative = Path('inputs') / f'{self.path.stem}-{self.screening_count}.png'
            path = self.path.parent / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return {
                'status': 'saved', 'path': relative.as_posix(),
                'sha256': hashlib.sha256(data).hexdigest(),
                'source': 'card_recognition_snapshot',
                'shape': list(frame.shape),
                'profile': config_name(),
                'code_provenance': self._code_provenance,
                **configured_solution(),
                'idol_card': session.idol_card if session is not None else None,
                'scenario': session.scenario if session is not None else None,
                # Caller-supplied label, not an independently checked Git revision.
                'declared_revision': os.environ.get('KAA_PLANNER_VALIDATION_REVISION'),
            }
        except Exception:
            logger.warning('Could not save planner input evidence.', exc_info=True)
            return {'status': 'write_failed'}

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
                letter = vars(hand).get('_letter_rect') if hand is not None else None
                rows.append({
                    'slot': slot,
                    'letter_rect': list(letter.xywh) if letter is not None else None,
                    'recognition': {
                        'asset': getattr(hand, 'res_name', None),
                        'database_match_distance': vars(hand).get('_recognition_distance'),
                        'ocr_confidence': vars(hand).get('_recognition_title_confidence'),
                        'title': vars(hand).get('_recognition_title'),
                        'method': 'title' if vars(hand).get('_recognition_title') else 'image',
                    } if hand is not None else None,
                    'card_id': card._id if card is not None else None,
                    'reason': reason,
                    'evaluation': asdict(evaluation) if evaluation is not None else None,
                    'modeled_affordable': can_afford_play(state, card)
                    if state is not None and card is not None else None,
                    'modeled_green_red': play_cost(state, card)
                    if state is not None and card is not None else None,
                    'cost_model_supported': projected_cost_supported(state, card)
                    if state is not None and card is not None else None,
                    'cost_model': 'stance_only_existing_rounding',
                })
            self._append({
                'schema_version': 1, 'event': 'candidate_screening',
                'attempt': self.screening_count, 'outcome': outcome,
                'decision_sequence': self.count + 1
                if outcome == 'selected' and self.count < 256 else None,
                'input_evidence': self._input_evidence(hands),
                'state': asdict(state) if state is not None else None,
                'hands': rows,
            })
        except Exception:
            logger.warning('Could not record planner candidate screening.', exc_info=True)

    def decision(
        self, state: 'BattleState',
        candidates: list[tuple['CardGameObject', 'CardEvaluation']],
        selected: 'CardEvaluation',
        *, selected_hand: 'CardGameObject | None' = None,
        selected_slot: int | None = None,
        accepted_output_previews: dict[int, dict] | None = None,
    ) -> None:
        self.pending = None
        if self.count >= 256:
            return
        self.count += 1
        try:
            # Identity, not card ID or equal scores, distinguishes visible copies.
            selected_indices = [
                index for index, (hand, evaluation) in enumerate(candidates)
                if (hand is selected_hand if selected_hand is not None else evaluation is selected)
            ]
            selected_index = selected_indices[0] if len(selected_indices) == 1 else None
            self._append({
                'schema_version': 1, 'event': 'decision', 'sequence': self.count,
                'selected_candidate_index': selected_index,
                'selected_slot': selected_slot if selected_index is not None else None,
                'state': asdict(state), 'selected': asdict(selected),
                'candidates': [
                    {
                        'evaluation': asdict(evaluation),
                        'observed_output_preview': _card_preview_metadata(hand),
                        # Supplied from the exact state evaluated for this copy;
                        # tracing never triggers another lazy OCR read.
                        'accepted_output_preview': (accepted_output_previews or {}).get(id(hand)),
                        'observed_growth_marker': getattr(hand, 'has_growth_marker', False),
                        'upgrade_count': getattr(hand.card, 'upgrade_count', None),
                        'stamina': getattr(hand.card, 'stamina', None),
                        'force_stamina': getattr(hand.card, 'force_stamina', None),
                        'cost_type': getattr(hand.card, 'cost_type', None),
                        'cost_value': getattr(hand.card, 'cost_value', None),
                        'modeled_payment': {
                            'green_red': play_cost(state, hand.card),
                            'special_affordable': special_cost_affordability(state, hand.card),
                            'supported': projected_cost_supported(state, hand.card),
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


def _card_preview_metadata(hand) -> dict | None:
    preview = vars(hand).get('output_preview')
    return preview._asdict() if preview is not None else None
