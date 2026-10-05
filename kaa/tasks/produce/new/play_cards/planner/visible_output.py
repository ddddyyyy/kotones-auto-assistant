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


def supports_current_preview(card: SkillCard, state: BattleState) -> bool:
    """Check the shared ranking rule before requesting any card image OCR."""
    if state.projected_effect_values is not None or state.projected_card_growth:
        return False
    scoring_segments = 0
    for segment in getattr(card, 'play_effects', ()):
        effect = segment.produce_exam_effect
        if (effect is None or effect.effect_type not in _ALLOWED
                or getattr(effect, '_chain_produce_exam_effect_id', '')
                or getattr(effect, 'chain_effect_ids', ())
                or evaluate_trigger_id(getattr(segment, '_produce_exam_trigger_id', ''), state)
                != TriggerVerdict.ACTIVE):
            return False
        if effect.effect_type == ProduceExamEffectType.ExamLesson:
            if effect.effect_value2:
                return False
            scoring_segments += 1
    return scoring_segments == 1


def current_preview_output(card: SkillCard, state: BattleState) -> int | None:
    observation = state.observed_card_output
    if observation is None or observation[0] != card._id:
        return None
    _, value, count = observation
    if not 0 <= value <= 9999 or not 1 <= count <= 9:
        return None
    return value * count if supports_current_preview(card, state) else None
