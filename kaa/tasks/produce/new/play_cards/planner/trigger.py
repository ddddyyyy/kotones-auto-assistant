"""Evaluate card-effect trigger conditions against the planner state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import json

from kaa.db.sqlite import select

from .state import AnomalyStance, BattleState


class TriggerVerdict(str, Enum):
    ACTIVE = 'active'
    INACTIVE = 'inactive'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class ExamTrigger:
    phase_types: tuple[str, ...] = ()
    field_status_check_types: tuple[str, ...] = ()
    field_status_types: tuple[str, ...] = ()
    field_status_values: tuple[int, ...] = ()


def _json_tuple(raw: str | None) -> tuple:
    if not raw:
        return ()
    value = json.loads(raw)
    return tuple(value) if isinstance(value, list) else ()


@lru_cache(maxsize=1024)
def load_exam_trigger(trigger_id: str) -> ExamTrigger | None:
    if not trigger_id:
        return None
    row = select(
        'SELECT phaseTypes, fieldStatusCheckTypes, fieldStatusTypes, fieldStatusValues '
        'FROM ProduceExamTrigger WHERE id = ?',
        trigger_id,
    )
    if row is None:
        return None
    return ExamTrigger(
        phase_types=_json_tuple(row['phaseTypes']),
        field_status_check_types=_json_tuple(row['fieldStatusCheckTypes']),
        field_status_types=_json_tuple(row['fieldStatusTypes']),
        field_status_values=_json_tuple(row['fieldStatusValues']),
    )


def evaluate_exam_trigger(trigger: ExamTrigger | None, state: BattleState) -> TriggerVerdict:
    """Return whether all currently observable field conditions are satisfied.

    Conditions involving deck searches, previous-card categories, HP percentages,
    counters not exposed by the HUD, or debuff inventories remain UNKNOWN.  The
    caller can then discount them without pretending that they definitely fire.
    """
    if trigger is None:
        return TriggerVerdict.ACTIVE

    # Card play effects can resolve during the current play. Enchant triggers
    # tied to later battle events cannot be asserted from the current HUD.
    immediate_phases = {
        'ProduceExamPhaseType_None',
        'ProduceExamPhaseType_ExamCardPlay',
    }
    unknown = any(
        phase not in immediate_phases for phase in trigger.phase_types
    )
    for index, status_type in enumerate(trigger.field_status_types):
        value = (
            trigger.field_status_values[index]
            if index < len(trigger.field_status_values)
            else 0
        )
        result = _evaluate_field_status(status_type, value, state)
        check = (
            trigger.field_status_check_types[index]
            if index < len(trigger.field_status_check_types)
            else ''
        )
        if check == 'ProduceExamTriggerCheckType_Not' and result is not None:
            result = not result
        if result is False:
            return TriggerVerdict.INACTIVE
        if result is None:
            unknown = True
    return TriggerVerdict.UNKNOWN if unknown else TriggerVerdict.ACTIVE


def evaluate_trigger_id(trigger_id: str, state: BattleState) -> TriggerVerdict:
    return evaluate_exam_trigger(load_exam_trigger(trigger_id), state)


def _evaluate_field_status(
    status_type: str,
    value: int,
    state: BattleState,
) -> bool | None:
    threshold = value if value > 0 else 1
    buffs = state.buffs
    anomaly = state.anomaly

    values = {
        'ProduceExamFieldStatusType_ReviewUp': buffs.review,
        'ProduceExamFieldStatusType_LessonBuffUp': buffs.lesson_buff,
        'ProduceExamFieldStatusType_ParameterBuffUp': buffs.parameter_buff_turns,
        'ProduceExamFieldStatusType_ParameterBuffMultiplePerTurnUp': (
            buffs.parameter_buff_multiple_turns
        ),
        'ProduceExamFieldStatusType_CardPlayAggressiveUp': buffs.aggressive,
        'ProduceExamFieldStatusType_BlockUp': state.genki,
        'ProduceExamFieldStatusType_FullPowerPointUp': anomaly.full_power_points,
        'ProduceExamFieldStatusType_FullPowerPointGetSumUp': (
            anomaly.cumulative_full_power_points
        ),
        'ProduceExamFieldStatusType_StanceChangeCountUp': (
            anomaly.stance_change_count
        ),
        'ProduceExamFieldStatusType_PreservationChangeCountUp': (
            anomaly.preservation_change_count
        ),
        'ProduceExamFieldStatusType_ConcentrationChangeCountUp': (
            anomaly.concentration_change_count
        ),
        'ProduceExamFieldStatusType_FullPowerChangeCountUp': (
            anomaly.full_power_change_count
        ),
    }
    observed_names = {
        'ProduceExamFieldStatusType_ReviewUp': 'review',
        'ProduceExamFieldStatusType_LessonBuffUp': 'lesson_buff',
        'ProduceExamFieldStatusType_ParameterBuffUp': 'parameter_buff',
        'ProduceExamFieldStatusType_ParameterBuffMultiplePerTurnUp': (
            'parameter_buff_multiple'
        ),
        'ProduceExamFieldStatusType_CardPlayAggressiveUp': 'aggressive',
        'ProduceExamFieldStatusType_BlockUp': 'genki',
        'ProduceExamFieldStatusType_FullPowerPointUp': 'full_power_point',
    }
    if status_type in values:
        if values[status_type] >= threshold:
            return True
        observed_name = observed_names.get(status_type)
        if observed_name == 'full_power_point' and observed_name in state.observed_effects:
            # Point observations are emitted only with a readable number,
            # including zero; the icon alone does not prove a positive value.
            return values[status_type] >= threshold
        if observed_name in state.observed_effects:
            # Presence proves a positive value, but not a higher threshold.
            return True if threshold <= 1 else None
        if observed_name in state.confirmed_absent_effects:
            return False
        # Inferred zero is not evidence of absence when a battle is resumed or
        # an unobserved item/event can change the field independently of cards.
        return None
    if status_type == 'ProduceExamFieldStatusType_NoBlock':
        return state.genki <= 0 if state.genki_known else None
    if status_type == 'ProduceExamFieldStatusType_FullPowerUp':
        if anomaly.full_power_active or 'full_power' in state.observed_effects:
            return True
        return False if 'full_power' in state.confirmed_absent_effects else None
    if status_type == 'ProduceExamFieldStatusType_ParameterBuff':
        if buffs.parameter_buff_turns > 0 or 'parameter_buff' in state.observed_effects:
            return True
        return False if 'parameter_buff' in state.confirmed_absent_effects else None
    if status_type == 'ProduceExamFieldStatusType_ConcentrationUp':
        level = 2 if anomaly.stance == AnomalyStance.AGGRESSIVE_2 else (
            1 if anomaly.stance == AnomalyStance.AGGRESSIVE else 0
        )
        if level >= threshold:
            return True
        if 'concentration' in state.observed_effects:
            return True if threshold <= 1 else None
        return False if 'concentration' in state.confirmed_absent_effects else None
    if status_type == 'ProduceExamFieldStatusType_PreservationUp':
        level = 2 if anomaly.stance == AnomalyStance.PRESERVATION_2 else (
            1 if anomaly.stance == AnomalyStance.PRESERVATION else 0
        )
        if level >= threshold:
            return True
        if 'preservation' in state.observed_effects:
            return True if threshold <= 1 else None
        return False if 'preservation' in state.confirmed_absent_effects else None
    if status_type == 'ProduceExamFieldStatusType_NoStance':
        if anomaly.stance != AnomalyStance.NORMAL:
            return False
        if {'preservation', 'concentration'} & state.observed_effects:
            return False
        if {'preservation', 'concentration'} <= state.confirmed_absent_effects:
            return True
        return None
    if status_type == 'ProduceExamFieldStatusType_RemainingTurn':
        return state.remaining_turns <= threshold if state.remaining_turns_known else None
    return None
