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
