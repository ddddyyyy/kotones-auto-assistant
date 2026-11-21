"""Current native score block: prepare utility and preserve pure scorers."""
from dataclasses import dataclass,replace
from typing import Literal
from kaa.db.constants import ProduceExamEffectType as Effect
from kaa.db.skill_card import SkillCard,ProduceExamEffect
from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluator,CardEvaluation,CompositeEvaluator
from kaa.tasks.produce.new.play_cards.planner.state import BattleState


@dataclass(frozen=True)
class CurrentScoreSuppression:
    frame_id: str
    rect: tuple[int,int,int,int]
    score: float
    red_fraction: float
    unique: bool
    source: Literal['current_native_slump_glyph']='current_native_slump_glyph'


def score_suppression_active(value: CurrentScoreSuppression | None,state: BattleState) -> bool:
    absolute=state.absolute_turn_observation
    if value is None or absolute is None or not state.is_exam:return False
    x,y,w,h=value.rect
    return bool(value.frame_id and value.frame_id==absolute.frame_id
        and state.remaining_turns==absolute.remaining_turns and value.source=='current_native_slump_glyph'
        and 20<=x and 237<=y and min(w,h)>0 and x+w<=66 and y+h<=730
        and .97<=value.score<=1 and .15<value.red_fraction<=1 and value.unique is True)


def _plain_score(effect: ProduceExamEffect | None) -> bool:
    return bool(effect is not None and effect.effect_type in {
        Effect.ExamLesson,Effect.ExamMultipleLessonBuffLesson,Effect.ExamLessonAddMultipleParameterBuff}
        and effect.effect_turn==0 and not effect.chain_effect_ids
        and not effect._produce_exam_status_enchant_id and not effect._produce_card_status_enchant_id
        and effect._produce_card_grow_effect_ids in (None,'','[]')
        and not effect._produce_card_search_id and not effect._target_produce_card_id)


def score_only_card(card: SkillCard) -> bool:
    # Only a family with no setup/move benefit is withheld. The original
    # card remains untouched; this promises no later draw or known deck order.
    if not card._asset_id:
        return False
    for upgrade in range(4):
        variant=SkillCard.from_asset_id(card._asset_id,upgrade)
        if variant is None or variant.upgrade_count!=upgrade:continue
        if (variant.play_move_position_type not in ('ProduceCardMovePositionType_Lost','ProduceCardMovePositionType_Grave')
            or variant.move_produce_exam_effects or variant._produce_card_status_enchant_id
            or not variant.play_effects or any(not _plain_score(row.produce_exam_effect) for row in variant.play_effects)):
            return False
    return bool(card.play_effects and all(_plain_score(row.produce_exam_effect) for row in card.play_effects)
        and card.play_move_position_type in ('ProduceCardMovePositionType_Lost','ProduceCardMovePositionType_Grave')
        and not card.move_produce_exam_effects and not card._produce_card_status_enchant_id)


class SuppressionAwareEvaluator:
    """Reuse shared utility/cost scoring while a proven current block is active."""
    def __init__(self,delegate: CardEvaluator | None=None):
        self.delegate=delegate or CompositeEvaluator()
        self.current: CurrentScoreSuppression | None=None

    def evaluate(self,card: SkillCard,state: BattleState) -> CardEvaluation | None:
        if not score_suppression_active(self.current,state):
            return self.delegate.evaluate(card,state)
        effects=[] if score_only_card(card) else [row for row in card.play_effects
            if not _plain_score(row.produce_exam_effect)]
        # Evaluation copy only; physical candidates, payment checks, input
        # metadata and actual feedback always retain the complete original.
        value=self.delegate.evaluate(replace(card,play_effects=effects),state)
        return replace(value,projected_output=0.,confirmed_output=0.,
            reason=value.reason+'；当前原生スランプ：仅评价非直接得分效果') if value is not None else None
