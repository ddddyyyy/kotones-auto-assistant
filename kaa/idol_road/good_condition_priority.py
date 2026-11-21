"""用已确认持续时间准备场地条件，不模拟场地收益或保证后续抽牌。"""
from kaa.db.skill_card import SkillCard
from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluator, CompositeEvaluator
from kaa.tasks.produce.new.play_cards.planner.selection_policy import SelectionConstraint
from kaa.tasks.produce.new.play_cards.planner.simulator import can_afford_play, simulate_play
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict, evaluate_trigger_id
from .category_priority import supports_visible_followup, supports_first_play
from .good_condition_rule import GoodConditionRequirement
from .planning_turn import PlanningTurnTracker
from .current_hp_projection import supports_current_hp_projection


class GoodConditionPriority:
    def __init__(self, total_turns: int, requirements: tuple[GoodConditionRequirement, ...],
                 *, evaluator: CardEvaluator | None = None,allow_unknown_duration: bool=False,
                 allow_current_effects: bool=False):
        self.total_turns = total_turns
        self.requirements = requirements
        self.evaluator = evaluator or CompositeEvaluator()
        self._turns=PlanningTurnTracker()
        self.allow_unknown_duration=allow_unknown_duration
        self.allow_current_effects=allow_current_effects

    @staticmethod
    def _balance(state: BattleState) -> int | None:
        value = dict(state.effect_values_for_planning).get('parameter_buff')
        return value if type(value) is int and value >= 0 and value == state.buffs.parameter_buff_turns else None

    def select(self, state: BattleState, cards: tuple[SkillCard, ...], *,
               current_cards_available: bool = False) -> SelectionConstraint:
        if not self.requirements:
            return SelectionConstraint()
        current_turn=self._turns.observe(state,self.total_turns)
        if current_turn is None:
            return SelectionConstraint(reason='field_good_condition:observation_unknown',allowed_indices=(),stop=True)
        turn = current_turn.turn
        goals = [goal for goal in self.requirements
                 if type(goal.start_turn) is int and turn < goal.start_turn <= min(99,current_turn.last_scheduled_turn)
                 and type(goal.minimum_turns) is int and goal.minimum_turns > 0
                 and goal.scope == 'condition_only' and goal.evidence_frames and all(goal.evidence_frames)
                 and not goal.executable and not goal.trigger_phase_confirmed and not goal.reward_confirmed]
        if not goals:
            return SelectionConstraint()
        grouped = {goal.start_turn for goal in goals
                   if len({other.minimum_turns for other in goals if other.start_turn == goal.start_turn}) > 1}
        goals = [goal for goal in goals if goal.start_turn not in grouped]
        if not goals:
            return SelectionConstraint(reason='field_good_condition:conflicting_requirements')
        current = self._balance(state)
        if current is None:
            if self.allow_unknown_duration:
                return SelectionConstraint(reason='field_good_condition:duration_unknown,condition_unconfirmed,no_numeric_route')
            return SelectionConstraint(reason='field_good_condition:observation_unknown',allowed_indices=(),stop=True)
        # 共享模型每跨一回合减少1；保守准备到声明回合，不假定在衰减前结算。
        reserve = lambda goal: goal.minimum_turns+goal.start_turn-turn
        unmet = [goal for goal in goals if current < reserve(goal)]
        goal = min(unmet or goals, key=lambda item: item.start_turn)
        target = reserve(goal)
        paths: list[tuple[int, float, int, int, tuple[str, ...]]] = []

        def record(index: int, future: BattleState, score: float, sequence: tuple[str, ...]) -> None:
            value = self._balance(future)
            if (value is not None and future.remaining_turns_known
                and future.remaining_turns == state.remaining_turns):
                paths.append((min(value,target)-min(current,target), score, index, value, sequence))

        for index, card in enumerate(cards):
            if (not supports_first_play(state, card, observed_available=current_cards_available)
                or not (supports_visible_followup(card,state) or (self.allow_current_effects
                    and supports_current_hp_projection(card,state,allow_audited_redraw=True)))
                or evaluate_trigger_id(card._play_produce_exam_trigger_id or '', state) is not TriggerVerdict.ACTIVE):
                continue
            evaluation = self.evaluator.evaluate(card, state)
            if evaluation is None:
                continue
            future = simulate_play(state, card)
            record(index, future, evaluation.score, (card._id,))
            if future.playable_count < 1 or not supports_visible_followup(card,state):
                continue
            for second_index, second in enumerate(cards):
                if (second_index == index or can_afford_play(future, second) is not True
                    or not supports_visible_followup(second, future)
                    or evaluate_trigger_id(second._play_produce_exam_trigger_id or '', future) is not TriggerVerdict.ACTIVE):
                    continue
                following = self.evaluator.evaluate(second, future)
                if following is not None:
                    record(index, simulate_play(future, second), evaluation.score+following.score, (card._id,second._id))
        prefix = (f'field_good_condition:minimum={goal.minimum_turns},declared_turn={goal.start_turn},'
                  f'prepare_by={goal.start_turn-1},current_reserve={target},phase_unconfirmed')
        gains = [path for path in paths if path[0] > 0]
        if gains:
            paths = gains
        else:
            floor = min(current, target)
            safe = [path for path in paths if path[3] >= floor]
            if not safe or not any(path[3] < floor for path in paths):
                return SelectionConstraint(reason=prefix+',no_supported_progress,reward_unconfirmed')
            paths = safe
            prefix += f',protect_duration={floor}'
        best = max((path[0],path[1]) for path in paths)
        selected = [path for path in paths if (path[0],path[1]) == best]
        return SelectionConstraint(tuple(dict.fromkeys(path[2] for path in selected)), reason=(
            prefix+f',estimated_progress={best[0]},estimated_sequences={[path[4] for path in selected]},'
            f'current_hand_available={current_cards_available},reward_unconfirmed'))
