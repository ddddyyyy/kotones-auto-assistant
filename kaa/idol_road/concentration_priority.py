"""场地阈值的提前准备目标；不模拟场地倍率或把推算值当作观察。"""
from dataclasses import dataclass
from kaa.db.skill_card import SkillCard
from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluator, CompositeEvaluator
from kaa.tasks.produce.new.play_cards.planner.selection_policy import SelectionConstraint
from kaa.tasks.produce.new.play_cards.planner.simulator import can_afford_play, simulate_play
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict, evaluate_trigger_id
from .category_priority import supports_visible_followup, supports_first_play
from .concentration_rule import ConcentrationDoubleDeclaration
from .threshold_turn_end import ThresholdTurnEndDeclaration
from .rule_catalog import build_rule_catalog
from .rules import StageRules
from .planning_turn import PlanningTurnTracker


@dataclass(frozen=True)
class ConcentrationObjective:
    declaration: ConcentrationDoubleDeclaration | ThresholdTurnEndDeclaration
    evidence_frames: tuple[str, ...]
    timing: str = 'prepare_before_declared_turn_phase_unconfirmed'

    @property
    def deadline_turn(self) -> int:
        return self.declaration.start_turn-1


def concentration_objectives(character_id: str, stage: int, rules: StageRules | None) -> tuple[ConcentrationObjective, ...]:
    if rules is None:
        return ()
    catalog = build_rule_catalog(character_id, stage, rules)
    blocked = {row.turn for viewport in rules.viewports for row in viewport.visible_rules or ()
               if row.concentration_rule is not None
               and row.concentration_rule.status in ('high_confidence_conflict', 'unsupported_text')}
    blocked.update(row.turn for viewport in rules.viewports for row in viewport.visible_rules or ()
                   if row.threshold_turn_end is not None and row.threshold_turn_end.status=='high_confidence_conflict')
    grouped: dict[int, list[ConcentrationObjective]] = {}
    for sighting in catalog.sightings:
        declaration = sighting.declaration
        if sighting.kind not in ('concentration_double','concentration_turn_end') or not isinstance(
            declaration,(ConcentrationDoubleDeclaration,ThresholdTurnEndDeclaration)):
            continue
        rows = rules.viewports[sighting.viewport_index].visible_rules
        assert rows is not None
        row=rows[sighting.rule_index]
        analysis = row.concentration_rule if sighting.kind=='concentration_double' else row.threshold_turn_end
        assert analysis is not None
        grouped.setdefault(declaration.start_turn, []).append(ConcentrationObjective(declaration, (analysis.evidence.frame_id,)))
    result = []
    for _, values in sorted(grouped.items()):
        if values[0].declaration.start_turn in blocked or len({value.declaration for value in values}) != 1:
            continue
        result.append(ConcentrationObjective(values[0].declaration,
                      tuple(dict.fromkeys(frame for value in values for frame in value.evidence_frames))))
    return tuple(result)


class ConcentrationPriority:
    def __init__(self, total_turns: int, objectives: tuple[ConcentrationObjective, ...],
                 *, evaluator: CardEvaluator | None = None):
        self.total_turns = total_turns
        self.objectives = objectives
        self.evaluator = evaluator or CompositeEvaluator()
        self._turns=PlanningTurnTracker()

    @staticmethod
    def _balance(state: BattleState) -> int | None:
        value = dict(state.effect_values_for_planning).get('lesson_buff')
        return value if type(value) is int and value >= 0 and value == state.buffs.lesson_buff else None

    def select(self, state: BattleState, cards: tuple[SkillCard, ...], *,
               current_cards_available: bool = False) -> SelectionConstraint:
        if not self.objectives:
            return SelectionConstraint()
        current_turn=self._turns.observe(state,self.total_turns)
        if current_turn is None:
            return SelectionConstraint(reason='field_concentration:observation_unknown',allowed_indices=(),stop=True)
        turn = current_turn.turn
        upcoming = [goal for goal in self.objectives if turn <= goal.deadline_turn < min(99,current_turn.last_scheduled_turn)]
        if not upcoming:
            return SelectionConstraint()
        current = self._balance(state)
        if current is None:
            return SelectionConstraint(reason='field_concentration:observation_unknown',allowed_indices=(),stop=True)
        unmet=[goal for goal in upcoming if current<goal.declaration.concentration_at_least]
        goal = min(unmet or upcoming, key=lambda value: value.deadline_turn)
        threshold = goal.declaration.concentration_at_least
        paths: list[tuple[int, float, int, tuple[str, ...], int]] = []
        def record(index: int, future: BattleState, score: float, sequence: tuple[str, ...]) -> None:
            value = self._balance(future)
            if value is not None:
                paths.append((min(value, threshold)-min(current,threshold), score, index, sequence, value))
        for index, card in enumerate(cards):
            if (not supports_first_play(state,card,observed_available=current_cards_available)
                or evaluate_trigger_id(card._play_produce_exam_trigger_id or '', state) is not TriggerVerdict.ACTIVE
                or not supports_visible_followup(card,state)):
                continue
            evaluation = self.evaluator.evaluate(card, state)
            if evaluation is None:
                continue
            future = simulate_play(state, card)
            if future.remaining_turns != state.remaining_turns or not future.remaining_turns_known:
                continue
            record(index, future, evaluation.score, (card._id,))
            if future.playable_count < 1:
                continue
            for second_index, second in enumerate(cards):
                if (second_index == index or can_afford_play(future, second) is not True
                    or evaluate_trigger_id(second._play_produce_exam_trigger_id or '', future) is not TriggerVerdict.ACTIVE):
                    continue
                following = self.evaluator.evaluate(second, future)
                if following is not None:
                    final = simulate_play(future, second)
                    if final.remaining_turns == state.remaining_turns and final.remaining_turns_known:
                        record(index, final, evaluation.score+following.score, (card._id,second._id))
        prefix = f'field_concentration:threshold={threshold},prepare_by={goal.deadline_turn},timing={goal.timing}'
        gains=[path for path in paths if path[0]>0]
        if gains:
            paths=gains
        else:
            floor=min(current,threshold)
            # Only override ordinary ranking when an observed route is known
            # to spend the stock and another visible route can preserve it.
            safe=[path for path in paths if path[4]>=floor]
            if not safe or not any(path[4]<floor for path in paths):
                return SelectionConstraint(reason=prefix+(',prepared_reward_unconfirmed' if current>=threshold
                                                         else ',no_supported_progress'))
            paths=safe
            prefix+=f',protect_concentration={floor}'
        if not paths:
            return SelectionConstraint(reason=prefix+',no_supported_progress')
        best_key = max((path[0],path[1]) for path in paths)
        best = [path for path in paths if (path[0],path[1]) == best_key]
        return SelectionConstraint(tuple(dict.fromkeys(path[2] for path in best)), reason=(
            prefix+f',estimated_progress={best_key[0]},estimated_sequences={[path[3] for path in best]},'
            f'current_hand_available={current_cards_available},future_costs_require_model_proof,reward_unconfirmed'))
