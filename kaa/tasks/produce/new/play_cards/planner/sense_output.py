"""普通Sense产分的集中增量与有限取整估算；不是实际收益证明。"""
from dataclasses import dataclass
from fractions import Fraction
import math
from kaa.db.constants import ProduceExamEffectType, ProducePlanType
from .state import BattleState


@dataclass(frozen=True)
class SenseOutputEstimate:
    output: float
    observed_rounding_scope: bool


def estimate_sense_output(base: float, count: int, state: BattleState, multiplier: float) -> SenseOutputEstimate | None:
    if (state.plan_type != ProducePlanType.Plan1 and state.archetype not in (
        ProduceExamEffectType.ExamLessonBuff, ProduceExamEffectType.ExamParameterBuff,
    )):
        return None
    value = base + max(0, state.buffs.lesson_buff) + state.anomaly.enthusiasm
    estimate = value * count * multiplier
    percent = state.exam_score_percent
    # 三段原局日志仅校准：单次普通产分、好调、已读整数属性；组合／多击仍是估算。
    expected_multiplier = 1.5 * percent / 100 if percent is not None else None
    simple = (state.is_exam and type(percent) is int and percent >= 0 and count == 1
              and float(base).is_integer() and state.buffs.parameter_buff_turns > 0
              and state.buffs.parameter_buff_multiple_turns == 0
              and state.anomaly.stance.value == 'normal' and not state.anomaly.full_power_active
              and state.anomaly.enthusiasm == 0
              and not state.projected_stance_uncertain and not state.projected_full_power_uncertain
              and not {'lesson_value_up', 'lesson_value_down', 'pride'} & (
                  state.observed_effects | dict(state.effect_values_for_planning).keys())
              and expected_multiplier is not None and math.isclose(multiplier, expected_multiplier))
    if simple:
        assert percent is not None
        parameter = math.ceil(Fraction(str(value)) * Fraction(3, 2))
        estimate = float(math.ceil(Fraction(parameter * percent, 100)))
    return SenseOutputEstimate(estimate, bool(simple))
