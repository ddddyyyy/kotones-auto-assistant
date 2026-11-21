"""爬塔目标协调：当前末手约束优先，其他时机提前准备集中阈值。"""
from kaa.db.skill_card import SkillCard
from kaa.tasks.produce.new.play_cards.planner.feedback import PendingPlay, PlayFeedback
from kaa.tasks.produce.new.play_cards.planner.selection_policy import SelectionConstraint
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluator
from .category_priority import CategoryPriority
from .category_rule import CategoryObjective
from .concentration_priority import ConcentrationObjective, ConcentrationPriority
from .good_condition_rule import GoodConditionRequirement
from .good_condition_priority import GoodConditionPriority
from .score_priority import ScorePreparationObjective,ScorePreparationPriority
from .hp_preparation import HpPreparationBudget


class FieldPriority:
    def __init__(self, total_turns: int, categories: tuple[CategoryObjective, ...],
                 concentration: tuple[ConcentrationObjective, ...],
                 good_condition: tuple[GoodConditionRequirement, ...] = (),
                 score: tuple[ScorePreparationObjective,...] = (),
                 hp_budget: HpPreparationBudget | None=None,*,strict_hp_reserve: bool=True,
                 allow_unknown_good_condition: bool=False,evaluator: CardEvaluator | None=None):
        self.category = CategoryPriority(total_turns,categories,evaluator=evaluator)
        self.concentration = ConcentrationPriority(total_turns,concentration,evaluator=evaluator)
        self.good_condition = GoodConditionPriority(total_turns,good_condition,
            allow_unknown_duration=allow_unknown_good_condition,allow_current_effects=allow_unknown_good_condition,evaluator=evaluator)
        self.score = ScorePreparationPriority(total_turns,score,hp_budget=hp_budget,strict_hp_reserve=strict_hp_reserve,evaluator=evaluator)

    def committed(self, state: BattleState, card: SkillCard) -> None:
        self.category.committed(state,card)

    def feedback(self, pending: PendingPlay, feedback: PlayFeedback, state: BattleState) -> None:
        self.category.feedback(pending,feedback,state)

    def select(self, state: BattleState, cards: tuple[SkillCard, ...]) -> SelectionConstraint:
        # SelectionPolicy's input contract is current recognized available
        # cards. Preserve that proof for the first play, not later simulation.
        first = self.category.select(state,cards,current_cards_available=True)
        if first.stop or first.preferred_indices or first.end_turn:
            return first
        score=self.score.select(state,cards,current_cards_available=True)
        if score.stop or score.preferred_indices:
            return score
        good = self.good_condition.select(state,cards,current_cards_available=True)
        if good.stop or good.preferred_indices:
            return good
        second = self.concentration.select(state,cards,current_cards_available=True)
        return second if second.stop or second.preferred_indices else score if score.reason else good if good.reason else first if first.reason else second
