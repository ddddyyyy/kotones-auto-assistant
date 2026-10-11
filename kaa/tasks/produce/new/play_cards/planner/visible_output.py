"""Narrow use of current card previews for ranking, never actual gain labels."""
from kaa.db.constants import ProduceExamEffectType
from kaa.db.skill_card import SkillCard
from .state import BattleState
from .trigger import TriggerVerdict, evaluate_trigger_id


_ALLOWED = {
    ProduceExamEffectType.ExamLesson, ProduceExamEffectType.ExamConcentration,
    ProduceExamEffectType.ExamPreservation, ProduceExamEffectType.ExamStanceReset,
    ProduceExamEffectType.ExamParameterBuff,
    ProduceExamEffectType.ExamParameterBuffMultiplePerTurn,
    ProduceExamEffectType.ExamFullPowerPoint,
}


def is_single_genki_conversion(card: SkillCard) -> bool:
    """Static single-segment scope, never an observed hit-count claim."""
    effects = getattr(card, 'play_effects', ())
    if len(effects) != 1 or getattr(card, '_produce_card_status_enchant_id', ''):
        return False
    effect = effects[0].produce_exam_effect
    return bool(
        effect is not None
        and effect.effect_type == ProduceExamEffectType.ExamLessonDependBlock
        and (effect.effect_value1 or 0) > 0
        and effect.effect_value2 == 0
        and effect.effect_count in {0, 1}
        and getattr(effect, '_produce_card_grow_effect_ids', None) == '[]'
    )


def supports_current_preview(card: SkillCard, state: BattleState) -> bool:
    """Check the shared ranking rule before requesting any card image OCR."""
    if state.projected_effect_values is not None or state.projected_card_growth:
        return False
    conversion = is_single_genki_conversion(card)
    if conversion and card._id in state.unresolved_growth_card_ids:
        return False
    scoring_segments = 0
    for segment in getattr(card, 'play_effects', ()):
        effect = segment.produce_exam_effect
        if (effect is None or (effect.effect_type not in _ALLOWED and not conversion)
                or getattr(effect, '_chain_produce_exam_effect_id', '')
                or getattr(effect, 'chain_effect_ids', ())
                or evaluate_trigger_id(getattr(segment, '_produce_exam_trigger_id', ''), state)
                != TriggerVerdict.ACTIVE):
            return False
        if effect.effect_type in {
            ProduceExamEffectType.ExamLesson, ProduceExamEffectType.ExamLessonDependBlock,
        }:
            if effect.effect_value2:
                return False
            scoring_segments += 1
    return scoring_segments == 1


def current_preview_output(card: SkillCard, state: BattleState) -> int | None:
    observation = state.observed_card_output
    if observation is None or observation[0] != card._id:
        return None
    _, value, count = observation
    if not 0 <= value <= 9999 or not supports_current_preview(card, state):
        return None
    if is_single_genki_conversion(card):
        # An absent badge stays unobserved in state. This narrow static scope
        # ranks one displayed hit only; it never certifies total actual gain.
        return value if count in {None, 1} else None
    return value * count if count is not None and 1 <= count <= 9 else None
