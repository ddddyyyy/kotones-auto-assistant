"""Per-archetype priorities used by the planner's shared evaluator."""

from __future__ import annotations

from dataclasses import dataclass

from kaa.db.constants import ProduceExamEffectType, ProducePlanType


@dataclass(frozen=True)
class ArchetypeProfile:
    name: str
    plan_type: ProducePlanType
    primary: frozenset[ProduceExamEffectType]
    secondary: frozenset[ProduceExamEffectType] = frozenset()

    def setup_multiplier(self, effect: ProduceExamEffectType) -> float:
        if effect in self.primary:
            return 1.8
        if effect in self.secondary:
            return 1.25
        return 0.65


PROFILES: dict[ProduceExamEffectType, ArchetypeProfile] = {
    ProduceExamEffectType.ExamParameterBuff: ArchetypeProfile(
        '好调', ProducePlanType.Plan1,
        frozenset({
            ProduceExamEffectType.ExamParameterBuff,
            ProduceExamEffectType.ExamParameterBuffMultiplePerTurn,
            ProduceExamEffectType.ExamLessonDependParameterBuff,
            ProduceExamEffectType.ExamLessonAddMultipleParameterBuff,
            ProduceExamEffectType.ExamParameterBuffDependLessonBuff,
        }),
        frozenset({ProduceExamEffectType.ExamLessonBuff}),
    ),
    ProduceExamEffectType.ExamLessonBuff: ArchetypeProfile(
        '集中', ProducePlanType.Plan1,
        frozenset({
            ProduceExamEffectType.ExamLessonBuff,
            ProduceExamEffectType.ExamLessonBuffAdditive,
            ProduceExamEffectType.ExamLessonBuffMultiple,
            ProduceExamEffectType.ExamLessonBuffDependParameterBuff,
            ProduceExamEffectType.ExamLessonAddMultipleLessonBuff,
            ProduceExamEffectType.ExamMultipleLessonBuffLesson,
        }),
        frozenset({ProduceExamEffectType.ExamParameterBuff}),
    ),
    ProduceExamEffectType.ExamReview: ArchetypeProfile(
        '好印象', ProducePlanType.Plan2,
        frozenset({
            ProduceExamEffectType.ExamReview,
            ProduceExamEffectType.ExamReviewAdditive,
            ProduceExamEffectType.ExamReviewMultiple,
            ProduceExamEffectType.ExamReviewValueMultiple,
            ProduceExamEffectType.ExamLessonDependExamReview,
            ProduceExamEffectType.ExamReviewDependExamBlock,
        }),
        frozenset({ProduceExamEffectType.ExamBlock}),
    ),
    ProduceExamEffectType.ExamCardPlayAggressive: ArchetypeProfile(
        '干劲', ProducePlanType.Plan2,
        frozenset({
            ProduceExamEffectType.ExamCardPlayAggressive,
            ProduceExamEffectType.ExamAggressiveAdditive,
            ProduceExamEffectType.ExamAggressiveValueMultiple,
            ProduceExamEffectType.ExamBlockAddMultipleAggressive,
            ProduceExamEffectType.ExamLessonDependExamCardPlayAggressive,
            ProduceExamEffectType.ExamLessonDependBlock,
        }),
        frozenset({ProduceExamEffectType.ExamBlock}),
    ),
    ProduceExamEffectType.ExamConcentration: ArchetypeProfile(
        '强气/温存', ProducePlanType.Plan3,
        frozenset({
            ProduceExamEffectType.ExamConcentration,
            ProduceExamEffectType.ExamPreservation,
            ProduceExamEffectType.ExamOverPreservation,
            ProduceExamEffectType.ExamEnthusiasticAdditive,
            ProduceExamEffectType.ExamEnthusiasticMultiple,
        }),
    ),
    ProduceExamEffectType.ExamFullPower: ArchetypeProfile(
        '全力', ProducePlanType.Plan3,
        frozenset({
            ProduceExamEffectType.ExamFullPowerPoint,
            ProduceExamEffectType.ExamFullPowerPointAdditive,
            ProduceExamEffectType.ExamFullPowerLessonMultipleAdditive,
            ProduceExamEffectType.ExamLessonFullPowerPoint,
        }),
        frozenset({
            ProduceExamEffectType.ExamPreservation,
            ProduceExamEffectType.ExamEnthusiasticAdditive,
        }),
    ),
}


def profile_for(archetype: ProduceExamEffectType | None) -> ArchetypeProfile | None:
    return PROFILES.get(archetype) if archetype is not None else None
