from dataclasses import replace

import pytest

from kaa.tasks.produce.new.play_cards.planner.state import (
    AnomalyState,
    AnomalyStance,
    BattleState,
    ExamBuffState,
)
from kaa.tasks.produce.new.play_cards.planner.trigger import (
    ExamTrigger,
    TriggerVerdict,
    evaluate_exam_trigger,
)


def _trigger(status: str, value: int = 0, *, negated: bool = False) -> ExamTrigger:
    return ExamTrigger(
        field_status_check_types=(
            ('ProduceExamTriggerCheckType_Not',) if negated else ()
        ),
        field_status_types=(status,),
        field_status_values=(value,),
    )


def test_known_buff_threshold_can_enable_or_disable_effect():
    trigger = _trigger('ProduceExamFieldStatusType_LessonBuffUp', 5)

    low = BattleState(
        remaining_turns=4,
        hp=20,
        genki=0,
        confirmed_absent_effects=frozenset({'lesson_buff'}),
    )
    ready = BattleState(
        remaining_turns=4,
        hp=20,
        genki=0,
        buffs=ExamBuffState(lesson_buff=5),
    )

    assert evaluate_exam_trigger(trigger, low) == TriggerVerdict.INACTIVE
    assert evaluate_exam_trigger(trigger, ready) == TriggerVerdict.ACTIVE


def test_negated_condition_is_respected():
    trigger = _trigger(
        'ProduceExamFieldStatusType_FullPowerUp',
        negated=True,
    )
    inactive = BattleState(
        remaining_turns=4,
        hp=20,
        genki=0,
        anomaly=AnomalyState(full_power_active=True),
    )

    assert evaluate_exam_trigger(trigger, inactive) == TriggerVerdict.INACTIVE


def test_stance_level_and_remaining_turn_conditions():
    state = BattleState(
        remaining_turns=3,
        hp=20,
        genki=0,
        anomaly=AnomalyState(stance=AnomalyStance.PRESERVATION_2),
    )

    assert evaluate_exam_trigger(
        _trigger('ProduceExamFieldStatusType_PreservationUp', 2),
        state,
    ) == TriggerVerdict.ACTIVE
    assert evaluate_exam_trigger(
        _trigger('ProduceExamFieldStatusType_RemainingTurn', 3),
        state,
    ) == TriggerVerdict.ACTIVE


def test_unobservable_condition_stays_unknown():
    state = BattleState(remaining_turns=4, hp=20, genki=0)

    assert evaluate_exam_trigger(
        _trigger('ProduceExamFieldStatusType_DebuffCountUp', 1),
        state,
    ) == TriggerVerdict.UNKNOWN


def test_no_block_requires_known_genki_value():
    trigger = _trigger('ProduceExamFieldStatusType_NoBlock')
    unknown = BattleState(
        remaining_turns=4, hp=20, genki=0, genki_known=False,
    )
    zero = BattleState(remaining_turns=4, hp=20, genki=0)
    positive = BattleState(remaining_turns=4, hp=20, genki=3)

    assert evaluate_exam_trigger(trigger, unknown) == TriggerVerdict.UNKNOWN
    assert evaluate_exam_trigger(trigger, zero) == TriggerVerdict.ACTIVE
    assert evaluate_exam_trigger(trigger, positive) == TriggerVerdict.INACTIVE
    assert evaluate_exam_trigger(
        _trigger('ProduceExamFieldStatusType_NoBlock', negated=True), unknown,
    ) == TriggerVerdict.UNKNOWN


def test_observed_stance_change_counter_enables_condition():
    state = BattleState(
        remaining_turns=4,
        hp=20,
        genki=0,
        anomaly=AnomalyState(stance_change_count=3),
    )

    assert evaluate_exam_trigger(
        _trigger('ProduceExamFieldStatusType_StanceChangeCountUp', 3),
        state,
    ) == TriggerVerdict.ACTIVE


def test_hud_presence_resolves_positive_but_not_numeric_threshold():
    state = BattleState(
        remaining_turns=4,
        hp=20,
        genki=0,
        observed_effects=frozenset({'lesson_buff'}),
    )

    assert evaluate_exam_trigger(
        _trigger('ProduceExamFieldStatusType_LessonBuffUp', 1),
        state,
    ) == TriggerVerdict.ACTIVE
    assert evaluate_exam_trigger(
        _trigger('ProduceExamFieldStatusType_LessonBuffUp', 5),
        state,
    ) == TriggerVerdict.UNKNOWN


def test_unobserved_zero_does_not_disable_a_trigger_after_resuming():
    state = BattleState(remaining_turns=4, hp=20, genki=0)

    for status in (
        'ProduceExamFieldStatusType_LessonBuffUp',
        'ProduceExamFieldStatusType_FullPowerPointUp',
        'ProduceExamFieldStatusType_FullPowerUp',
        'ProduceExamFieldStatusType_PreservationUp',
        'ProduceExamFieldStatusType_NoStance',
    ):
        assert evaluate_exam_trigger(_trigger(status), state) == TriggerVerdict.UNKNOWN


def test_observed_point_number_can_disprove_a_threshold_including_zero():
    for points, threshold in ((0, 1), (8, 10)):
        state = BattleState(
            remaining_turns=4, hp=20, genki=0,
            observed_effects=frozenset({'full_power_point'}),
            anomaly=AnomalyState(full_power_points=points),
        )
        assert evaluate_exam_trigger(
            _trigger('ProduceExamFieldStatusType_FullPowerPointUp', threshold), state,
        ) == TriggerVerdict.INACTIVE


def test_confirmed_absence_can_resolve_negated_condition():
    state = BattleState(
        remaining_turns=4,
        hp=20,
        genki=0,
        confirmed_absent_effects=frozenset({'parameter_buff'}),
    )

    assert evaluate_exam_trigger(
        _trigger('ProduceExamFieldStatusType_ParameterBuff', negated=True),
        state,
    ) == TriggerVerdict.ACTIVE


def test_future_phase_is_not_marked_active_from_current_hud():
    trigger = ExamTrigger(
        phase_types=('ProduceExamPhaseType_ExamEndTurn',),
    )
    state = BattleState(remaining_turns=3, hp=20, genki=0)

    assert evaluate_exam_trigger(trigger, state) == TriggerVerdict.UNKNOWN



def test_remaining_turn_condition_is_unknown_when_hud_is_missing():
    state = BattleState(remaining_turns=1, hp=10, genki=0, remaining_turns_known=False)
    trigger = _trigger('ProduceExamFieldStatusType_RemainingTurn', 2)
    assert evaluate_exam_trigger(trigger, state) == TriggerVerdict.UNKNOWN


def test_database_update_invalidates_existing_and_missing_triggers(monkeypatch):
    from kaa.db import _util
    from kaa.tasks.produce.new.play_cards.planner import trigger

    rows = {'changed': {'phaseTypes': '["old"]'}}
    calls = []

    def select(sql, key):
        calls.append(key)
        row = rows.get(key)
        if row is None:
            return None
        return {'phaseTypes': row['phaseTypes'], 'fieldStatusCheckTypes': '[]',
                'fieldStatusTypes': '[]', 'fieldStatusValues': '[]',
                'phaseValues': '[]', 'fieldStatusProduceCardSearchIds': '[]',
                'produceCardSearchId': '', 'upperSearchCount': 0, 'lowerSearchCount': 0,
                'cardMovePositionType': 'ProduceCardMovePositionType_Unknown',
                'effectTypes': '[]', 'lessonType': 'ProduceStepLessonType_Unknown'}

    monkeypatch.setattr(trigger, 'select', select)
    monkeypatch.setattr(_util, 'invalidate_connections', lambda: None)
    trigger.load_exam_trigger.cache_clear()
    try:
        loaded = trigger.load_exam_trigger('changed')
        assert loaded is not None and loaded.phase_types == ('old',)
        assert trigger.load_exam_trigger('missing') is None
        rows['changed'] = {'phaseTypes': '["new"]'}
        rows['missing'] = {'phaseTypes': '["added"]'}
        _util.clear_db_caches()
        loaded = trigger.load_exam_trigger('changed')
        assert loaded is not None and loaded.phase_types == ('new',)
        loaded = trigger.load_exam_trigger('missing')
        assert loaded is not None and loaded.phase_types == ('added',)
        assert calls == ['changed', 'missing', 'changed', 'missing']
    finally:
        trigger.load_exam_trigger.cache_clear()


@pytest.mark.parametrize('value, threshold', [(0, 1), (2, 3), (3, 3), (4, 3)])
@pytest.mark.parametrize('negated', [False, True])
def test_reconciled_numeric_hud_resolves_threshold_both_ways(value, threshold, negated):
    state = BattleState(
        remaining_turns=4, hp=20, genki=0,
        buffs=ExamBuffState(lesson_buff=value),
        observed_effects=frozenset({'lesson_buff'}),
        observed_effect_values=(('lesson_buff', value),),
    )
    expected = (value >= threshold) != negated
    trigger = _trigger('ProduceExamFieldStatusType_LessonBuffUp', threshold, negated=negated)
    assert evaluate_exam_trigger(trigger, state) == (
        TriggerVerdict.ACTIVE if expected else TriggerVerdict.INACTIVE
    )
    conflict = replace(state, observed_effect_values=(('lesson_buff', value + 1),))
    assert evaluate_exam_trigger(trigger, conflict) == TriggerVerdict.UNKNOWN


@pytest.mark.parametrize('genki, known, expected', [
    (0, True, TriggerVerdict.INACTIVE),
    (2, True, TriggerVerdict.INACTIVE),
    (3, True, TriggerVerdict.ACTIVE),
    (10, False, TriggerVerdict.UNKNOWN),
])
def test_block_threshold_respects_numeric_resource_knowledge(genki, known, expected):
    state = BattleState(remaining_turns=4, hp=20, genki=genki, genki_known=known)
    assert evaluate_exam_trigger(
        _trigger('ProduceExamFieldStatusType_BlockUp', 3), state,
    ) == expected


def test_missing_named_trigger_is_not_an_unconditional_effect(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner import trigger

    monkeypatch.setattr(trigger, 'load_exam_trigger', lambda key: None)
    state = BattleState(remaining_turns=4, hp=20, genki=0)
    assert trigger.evaluate_trigger_id('', state) == TriggerVerdict.ACTIVE
    assert trigger.evaluate_trigger_id('missing-condition', state) == TriggerVerdict.UNKNOWN


@pytest.mark.parametrize('column,value', [
    ('phaseValues', '[1]'),
    ('fieldStatusProduceCardSearchIds', '["search"]'),
    ('produceCardSearchId', 'search'),
    ('upperSearchCount', 2),
    ('lowerSearchCount', 1),
    ('cardMovePositionType', 'ProduceCardMovePositionType_Hand'),
    ('effectTypes', '["ProduceExamEffectType_ExamLesson"]'),
    ('lessonType', 'ProduceStepLessonType_LessonVisual'),
])
def test_unmodeled_database_constraints_never_become_unconditional(monkeypatch, column, value):
    from kaa.tasks.produce.new.play_cards.planner import trigger
    row = {
        'phaseTypes': '["ProduceExamPhaseType_ExamCardPlay"]',
        'fieldStatusCheckTypes': '[]', 'fieldStatusTypes': '[]', 'fieldStatusValues': '[]',
        'phaseValues': '[]', 'fieldStatusProduceCardSearchIds': '[]',
        'produceCardSearchId': '', 'upperSearchCount': 0, 'lowerSearchCount': 0,
        'cardMovePositionType': 'ProduceCardMovePositionType_Unknown',
        'effectTypes': '[]', 'lessonType': 'ProduceStepLessonType_Unknown',
    }
    monkeypatch.setattr(trigger, 'select', lambda sql, key: row)
    trigger.load_exam_trigger.cache_clear()
    try:
        state = BattleState(remaining_turns=3, hp=20, genki=20)
        assert trigger.evaluate_trigger_id('plain', state) == TriggerVerdict.ACTIVE
        row[column] = value
        assert trigger.evaluate_trigger_id('constrained', state) == TriggerVerdict.UNKNOWN
    finally:
        trigger.load_exam_trigger.cache_clear()


def test_known_false_field_disproves_trigger_with_unknown_extra_constraints():
    condition = replace(_trigger('ProduceExamFieldStatusType_BlockUp', 15),
                        unmodeled_constraints=True)
    assert evaluate_exam_trigger(condition,
        BattleState(remaining_turns=3, hp=20, genki=5)) == TriggerVerdict.INACTIVE
    assert evaluate_exam_trigger(condition,
        BattleState(remaining_turns=3, hp=20, genki=20)) == TriggerVerdict.UNKNOWN


@pytest.mark.parametrize('balance,expected', [
    (None, TriggerVerdict.UNKNOWN), (0, TriggerVerdict.INACTIVE),
    (2, TriggerVerdict.ACTIVE),
])
@pytest.mark.parametrize('negated', [False, True])
def test_parameter_buff_presence_uses_projected_balance_before_raw_icons(balance, expected, negated):
    state = BattleState(remaining_turns=3, hp=20, genki=20,
        buffs=ExamBuffState(parameter_buff_turns=2 if balance is None else balance),
        observed_effects=frozenset({'parameter_buff'}),
        observed_effect_values=(('parameter_buff', 2),),
        projected_effect_values=(('parameter_buff', balance),))
    if negated and expected != TriggerVerdict.UNKNOWN:
        expected = TriggerVerdict.INACTIVE if expected == TriggerVerdict.ACTIVE else TriggerVerdict.ACTIVE
    assert evaluate_exam_trigger(
        _trigger('ProduceExamFieldStatusType_ParameterBuff', negated=negated), state) == expected


def test_empty_parameter_buff_projection_does_not_reuse_hud_presence():
    state = BattleState(remaining_turns=3, hp=20, genki=20,
        buffs=ExamBuffState(parameter_buff_turns=2),
        observed_effects=frozenset({'parameter_buff'}), projected_effect_values=())
    assert evaluate_exam_trigger(_trigger('ProduceExamFieldStatusType_ParameterBuff'), state) == TriggerVerdict.UNKNOWN


def test_conflicting_parameter_buff_number_does_not_confirm_presence():
    state = BattleState(remaining_turns=3, hp=20, genki=20,
        buffs=ExamBuffState(parameter_buff_turns=2),
        observed_effects=frozenset({'parameter_buff'}),
        observed_effect_values=(('parameter_buff', 0),))
    assert evaluate_exam_trigger(_trigger('ProduceExamFieldStatusType_ParameterBuff'), state) == TriggerVerdict.UNKNOWN


@pytest.mark.parametrize('total', [0, 50])
@pytest.mark.parametrize('negated', [False, True])
def test_cumulative_point_trigger_needs_complete_history_even_with_current_hud(total, negated):
    state = BattleState(remaining_turns=3, hp=100, genki=0,
        anomaly=AnomalyState(full_power_points=50, cumulative_full_power_points=total),
        observed_effect_values=(('full_power_point', 50),))
    trigger = _trigger('ProduceExamFieldStatusType_FullPowerPointGetSumUp', 10, negated=negated)
    assert evaluate_exam_trigger(trigger, state) == TriggerVerdict.UNKNOWN


_JSON_CONDITION_COLUMNS = [
    'phaseTypes', 'fieldStatusCheckTypes', 'fieldStatusTypes',
    'fieldStatusValues', 'phaseValues', 'fieldStatusProduceCardSearchIds', 'effectTypes',
]


def _plain_condition_row():
    row: dict[str, str | int] = {key: '[]' for key in _JSON_CONDITION_COLUMNS}
    row.update(produceCardSearchId='', upperSearchCount=0, lowerSearchCount=0,
        cardMovePositionType='ProduceCardMovePositionType_Unknown',
        lessonType='ProduceStepLessonType_Unknown')
    return row


@pytest.mark.parametrize('column', _JSON_CONDITION_COLUMNS)
@pytest.mark.parametrize('raw', ['invalid-json', '{}', 'null', '1', 'true', '"[]"', '[{}]'])
def test_invalid_named_condition_arrays_are_unknown_without_throwing(monkeypatch, column, raw):
    from kaa.tasks.produce.new.play_cards.planner import trigger
    row = _plain_condition_row()
    row[column] = raw
    monkeypatch.setattr(trigger, 'select', lambda *_: row)
    trigger.load_exam_trigger.cache_clear()
    try:
        state = BattleState(remaining_turns=3, hp=20, genki=20)
        assert trigger.evaluate_trigger_id('damaged', state) == TriggerVerdict.UNKNOWN
        assert trigger.evaluate_trigger_id('', state) == TriggerVerdict.ACTIVE
    finally:
        trigger.load_exam_trigger.cache_clear()


@pytest.mark.parametrize('column', _JSON_CONDITION_COLUMNS)
def test_named_condition_array_element_types_are_validated(monkeypatch, column):
    from kaa.tasks.produce.new.play_cards.planner import trigger
    row = _plain_condition_row()
    row[column] = '["1"]' if column in {'fieldStatusValues', 'phaseValues'} else '[1]'
    monkeypatch.setattr(trigger, 'select', lambda *_: row)
    trigger.load_exam_trigger.cache_clear()
    try:
        assert trigger.evaluate_trigger_id('wrong-type', BattleState(remaining_turns=3, hp=20, genki=20)) == TriggerVerdict.UNKNOWN
    finally:
        trigger.load_exam_trigger.cache_clear()


@pytest.mark.parametrize('column', ['fieldStatusValues', 'phaseValues'])
def test_boolean_condition_value_is_not_an_integer_threshold(monkeypatch, column):
    from kaa.tasks.produce.new.play_cards.planner import trigger
    row = _plain_condition_row()
    row[column] = '[true]'
    monkeypatch.setattr(trigger, 'select', lambda *_: row)
    trigger.load_exam_trigger.cache_clear()
    try:
        assert trigger.evaluate_trigger_id('boolean', BattleState(remaining_turns=3, hp=20, genki=20)) == TriggerVerdict.UNKNOWN
    finally:
        trigger.load_exam_trigger.cache_clear()


def test_repaired_condition_is_reloaded_after_database_cache_clear(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner import trigger
    from kaa.db import _util
    row = _plain_condition_row()
    row['fieldStatusTypes'] = '{}'
    monkeypatch.setattr(trigger, 'select', lambda *_: row)
    trigger.load_exam_trigger.cache_clear()
    try:
        state = BattleState(remaining_turns=3, hp=20, genki=20)
        assert trigger.evaluate_trigger_id('repair', state) == TriggerVerdict.UNKNOWN
        row['fieldStatusTypes'] = '[]'
        _util.clear_db_caches()
        assert trigger.evaluate_trigger_id('repair', state) == TriggerVerdict.ACTIVE
    finally:
        trigger.load_exam_trigger.cache_clear()


@pytest.mark.parametrize('status,stance', [
    ('ConcentrationUp', AnomalyStance.AGGRESSIVE),
    ('PreservationUp', AnomalyStance.PRESERVATION),
    ('NoStance', AnomalyStance.NORMAL),
])
@pytest.mark.parametrize('negated', [False, True])
@pytest.mark.parametrize('full_power', [False, True])
def test_unknown_projected_stance_ignores_old_stance_and_hud(status, stance, negated, full_power):
    state = BattleState(remaining_turns=4, hp=20, genki=20,
                        anomaly=AnomalyState(stance=stance, full_power_active=full_power),
                        observed_effects=frozenset({'concentration', 'preservation'}),
                        confirmed_absent_effects=frozenset({'concentration', 'preservation'}),
                        projected_effect_values=(), projected_stance_uncertain=True)
    assert evaluate_exam_trigger(_trigger('ProduceExamFieldStatusType_' + status, negated=negated), state) == TriggerVerdict.UNKNOWN
