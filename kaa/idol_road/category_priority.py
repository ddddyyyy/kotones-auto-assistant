"""塔场地末手目标的有限搜索；预测路线不证明奖励已触发。"""
from kaa.db.constants import ProduceExamEffectType as Effect
from kaa.db.skill_card import SkillCard
from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluator, CompositeEvaluator
from kaa.tasks.produce.new.play_cards.planner.feedback import PendingPlay, PlayFeedback, PlayFeedbackStatus
from kaa.tasks.produce.new.play_cards.planner.selection_policy import SelectionConstraint
from kaa.tasks.produce.new.play_cards.planner.simulator import can_afford_play, simulate_play
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict, evaluate_trigger_id
from .category_rule import CategoryObjective
from .planning_turn import PlanningTurnTracker,read_planning_turn


# Only these simple, modeled effects may precede a second visible card. Draw,
# exchange, stance transitions, chains and unknown effects cannot certify a route.
_SIMPLE_PREFIX_EFFECTS = frozenset({
    Effect.ExamLesson, Effect.ExamLessonBuff, Effect.ExamParameterBuff,
    Effect.ExamPlayableValueAdd, Effect.ExamBlock, Effect.ExamStaminaRecoverFix,
})


class CategoryPriority:
    def __init__(self, total_turns: int, objectives: tuple[CategoryObjective, ...],
                 *, evaluator: CardEvaluator | None = None):
        self.total_turns = total_turns
        self.objectives = objectives
        self.evaluator = evaluator or CompositeEvaluator()
        self._turns=PlanningTurnTracker()
        self._pending: tuple[tuple[str,str | int,int],BattleState,str,str] | None = None
        self._confirmed_last: tuple[tuple[str,str | int,int],str] | None = None

    def committed(self, state: BattleState, card: SkillCard) -> None:
        self._confirmed_last = None
        turn=self._turns.observe(state,self.total_turns)
        self._pending = (turn.key,state,card._id,card.category) if turn is not None and card.category else None

    def feedback(self, pending: PendingPlay, feedback: PlayFeedback, state: BattleState) -> None:
        requested = self._pending
        self._pending = None
        self._confirmed_last = None
        # Score/resources can also change due to stage/P-item events. Require
        # the requested card to leave a complete hand in the same known turn.
        before_turn=read_planning_turn(pending.before,self.total_turns)
        after_turn=self._turns.observe(state,self.total_turns)
        if (requested is not None and requested[1:3] == (pending.before,pending.card_id)
            and feedback.status is PlayFeedbackStatus.CONFIRMED and feedback.evidence == 'card_left_hand'
            and before_turn is not None and after_turn is not None
            and before_turn.key==after_turn.key==requested[0]):
            self._confirmed_last = (requested[0], requested[3])

    def select(self, state: BattleState, cards: tuple[SkillCard, ...], *,
               current_cards_available: bool = False) -> SelectionConstraint:
        current_turn=self._turns.observe(state,self.total_turns)
        if current_turn is None:
            self._pending=self._confirmed_last=None
            return SelectionConstraint(reason='field_goal:turn_unknown')
        turn = current_turn.turn
        if self._confirmed_last is not None and self._confirmed_last[0]!=current_turn.key:
            self._confirmed_last=None
        if self._pending is not None and self._pending[0]!=current_turn.key:
            self._pending=None
        goals = [goal for goal in self.objectives if goal.deadline_turn == turn
                 and 2 <= goal.trigger_turn <= min(99,current_turn.last_scheduled_turn)
                 and goal.timing_source == 'user_confirmed_previous_turn_last_card']
        if not goals:
            return SelectionConstraint()
        if len({goal.category for goal in goals}) != 1:
            return SelectionConstraint(reason='field_goal:conflicting_categories')
        category = goals[0].category
        prefix = f'field_goal:deadline={turn},trigger={turn+1},last={category}'
        # A matching confirmed last play can be preserved even when another
        # wrong-category card remains playable. Never use speculative memory.
        preserving = self._confirmed_last == (current_turn.key, category)
        paths: list[tuple[float, int, tuple[str, ...]]] = []
        for index, card in enumerate(cards):
            if (not supports_first_play(state,card,observed_available=current_cards_available)
                or evaluate_trigger_id(card._play_produce_exam_trigger_id or '', state) is not TriggerVerdict.ACTIVE):
                continue
            evaluation = self.evaluator.evaluate(card, state)
            if evaluation is None:
                continue
            if card.category == category:
                paths.append((evaluation.score, index, (card._id,)))
            if preserving:
                # Once the goal is observed, continue useful matching cards;
                # don't risk an already met goal on another estimated bridge.
                continue
            if not supports_visible_followup(card, state):
                continue
            future = simulate_play(state, card)
            if (future.playable_count < 1 or not future.remaining_turns_known
                or future.remaining_turns != state.remaining_turns):
                continue
            for second_index, second in enumerate(cards):
                if (second_index == index or second.category != category
                    or can_afford_play(future, second) is not True
                    or evaluate_trigger_id(second._play_produce_exam_trigger_id or '', future) is not TriggerVerdict.ACTIVE):
                    continue
                following = self.evaluator.evaluate(second, future)
                if following is not None:
                    paths.append((evaluation.score+following.score, index, (card._id, second._id)))
        if preserving and (not paths or max(path[0] for path in paths) <= 0):
            return SelectionConstraint(end_turn=True, reason=prefix+',preserve_confirmed_last')
        if not paths:
            return SelectionConstraint(reason=prefix+',no_supported_visible_route')
        score = max(path[0] for path in paths)
        best = [path for path in paths if path[0] == score]
        return SelectionConstraint(tuple(dict.fromkeys(path[1] for path in best)), reason=(
            prefix+f',estimated_sequences={[path[2] for path in best]},reward_unconfirmed'))


def supports_visible_followup(card: SkillCard, state: BattleState) -> bool:
    """当前可见第二张牌的保守搜索边界，不证明后续实际可点击。"""
    if card._id in state.native_payment_conflict_card_ids:
        return False
    if not card.play_effects:
        return False
    for segment in card.play_effects:
        effect = segment.produce_exam_effect
        if (effect is None or effect.effect_type not in _SIMPLE_PREFIX_EFFECTS
            or effect.chain_effect_ids
            or evaluate_trigger_id(segment._produce_exam_trigger_id, state) is not TriggerVerdict.ACTIVE):
            return False
    return True


def supports_first_play(state: BattleState, card: SkillCard, *, observed_available: bool) -> bool:
    """当前可用性只能证明首手，不把未知体力费用填为零或传给后续路线。"""
    affordable = can_afford_play(state,card)
    return affordable is True or (affordable is None and observed_available is True
                                  and (card.cost_value or 0) == 0)
