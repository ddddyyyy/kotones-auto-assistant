"""可见追加回合行的分数准备目标；预计得分不成为观察或场地奖励。"""
from dataclasses import dataclass,replace
import math
import re
from kaa.db.skill_card import SkillCard
from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluator,CompositeEvaluator,conservative_target_output
from kaa.tasks.produce.new.play_cards.planner.selection_policy import SelectionConstraint
from kaa.tasks.produce.new.play_cards.planner.simulator import can_afford_play,simulate_play
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict,evaluate_trigger_id
from .category_priority import supports_first_play,supports_visible_followup
from .concentration_rule import compact
from .planning_turn import PlanningTurnTracker
from .rules import StageRules,RuleViewport,confirmed_rule_turn_with_number
from .turn_budget import fixed_turn_budget_conflicts
from .hp_preparation import HpPreparationBudget,assess_hp_route
from .current_hp_projection import supports_current_hp_projection


@dataclass(frozen=True)
class ScorePreparationObjective:
    declared_turn: int
    score_at_least: int
    evidence_frames: tuple[str,...]
    timing: str = 'prepare_before_declared_turn_phase_unconfirmed'
    reward_confirmed: bool = False

    @property
    def deadline_turn(self) -> int:
        return self.declared_turn-1


def score_preparation_objectives(character_id: str,stage: int,rules: StageRules | None) -> tuple[ScorePreparationObjective,...]:
    conflicts=fixed_turn_budget_conflicts(character_id,stage,rules)
    if not conflicts or rules is None:
        return ()
    grouped: dict[int,list[tuple[int,str]]]={}
    blocked=set()
    for viewport in rules.viewports:
        for row in viewport.visible_rules or ():
            turn=confirmed_rule_turn_with_number(row.turn_reads,row.turn_number_evidence)
            symbol=row.turn_addition
            if (symbol is None or turn is None or turn<2
                or not fixed_turn_budget_conflicts(character_id,stage,replace(rules,viewports=(RuleViewport(None,(row,)),)))):
                continue
            values=[]
            for read in row.description_reads:
                # Only the complete first condition line is interpreted. The later icon-bearing
                # reward text stays unknown; no threshold or amount is inferred from the marker.
                if not read.lines or not .98<=read.lines[0].confidence<=1:
                    continue
                text=compact(read.lines[0].text)
                match=re.fullmatch(r'スコアが([1-9][0-9]{0,8})以上の場合、以降の([1-9][0-9]?)ターンの',text)
                if match is None:
                    blocked.add(turn)
                    continue
                values.append(int(match[1]))
            if len(set(values))>1:
                blocked.add(turn)
            elif len(values)==2:
                grouped.setdefault(turn,[]).append((values[0],symbol.frame_id))
    return tuple(ScorePreparationObjective(turn,values[0][0],tuple(dict.fromkeys(value[1] for value in values)))
                 for turn,values in sorted(grouped.items()) if turn not in blocked and len({value[0] for value in values})==1)


class ScorePreparationPriority:
    def __init__(self,total_turns: int,objectives: tuple[ScorePreparationObjective,...],*,evaluator: CardEvaluator | None=None,
                 hp_budget: HpPreparationBudget | None=None,strict_hp_reserve: bool=True):
        self.total_turns=total_turns
        self.objectives=objectives
        self.evaluator=evaluator or CompositeEvaluator()
        self._turns=PlanningTurnTracker()
        self.hp_budget=hp_budget
        self.strict_hp_reserve=strict_hp_reserve

    def select(self,state: BattleState,cards: tuple[SkillCard,...],*,current_cards_available: bool=False) -> SelectionConstraint:
        if not self.objectives:
            return SelectionConstraint()
        turn=self._turns.observe(state,self.total_turns)
        if turn is None:
            return SelectionConstraint(reason='field_score:observation_unknown',allowed_indices=(),stop=True)
        goals=[goal for goal in self.objectives if turn.turn<=goal.deadline_turn<turn.last_scheduled_turn]
        if not goals:
            return SelectionConstraint()
        score=state.current_score
        if type(score) is not int or score<0:
            return SelectionConstraint(reason='field_score:observation_unknown',allowed_indices=(),stop=True)
        goal=min(goals,key=lambda value:value.deadline_turn)
        gap=max(0,goal.score_at_least-score)
        prefix=f'field_score:threshold={goal.score_at_least},observed={score},prepare_by={goal.deadline_turn},timing={goal.timing}'
        if gap==0:
            return SelectionConstraint(reason=prefix+',prepared_reward_unconfirmed')
        if turn.turn<goal.deadline_turn:
            return SelectionConstraint(reason=prefix+f',gap={gap},setup_window')
        paths=[]
        hp_rejected=False
        for index,card in enumerate(cards):
            if (not supports_first_play(state,card,observed_available=current_cards_available)
                or evaluate_trigger_id(card._play_produce_exam_trigger_id or '',state) is not TriggerVerdict.ACTIVE
                or not supports_current_hp_projection(card,state)):
                continue
            evaluation=self.evaluator.evaluate(card,state)
            if evaluation is None or not math.isfinite(evaluation.confirmed_output) or not math.isfinite(evaluation.score):
                continue
            output=conservative_target_output(evaluation.confirmed_output)
            future=simulate_play(state,card)
            if self.strict_hp_reserve and self.hp_budget is not None and assess_hp_route(state,future,turn,self.hp_budget).status not in ('safe','not_applicable'):
                hp_rejected=True
                continue
            paths.append((min(gap,output),evaluation.score,index,(card._id,)))
            if (not supports_visible_followup(card,state) or future.playable_count<1
                or future.remaining_turns!=state.remaining_turns or not future.remaining_turns_known):
                continue
            for second_index,second in enumerate(cards):
                # The simulator does not update observed score. Score-dependent follow-up triggers
                # cannot be evaluated using that stale score, even when other effects are modeled.
                if (second_index==index or second._play_produce_exam_trigger_id
                    or any(segment._produce_exam_trigger_id for segment in second.play_effects)
                    or can_afford_play(future,second) is not True or not supports_visible_followup(second,future)):
                    continue
                following=self.evaluator.evaluate(second,future)
                if following is None or not math.isfinite(following.confirmed_output) or not math.isfinite(following.score):
                    continue
                final=simulate_play(future,second)
                if self.strict_hp_reserve and self.hp_budget is not None and assess_hp_route(state,final,turn,self.hp_budget).status not in ('safe','not_applicable'):
                    hp_rejected=True
                    continue
                gain=output+conservative_target_output(following.confirmed_output)
                paths.append((min(gap,gain),evaluation.score+following.score,index,(card._id,second._id)))
        paths=[path for path in paths if path[0]>0]
        if not paths:
            return SelectionConstraint(reason=prefix+(',hp_budget_unknown_or_breached' if hp_rejected else ',no_supported_score_progress'))
        key=max((path[0],path[1]) for path in paths)
        best=[path for path in paths if (path[0],path[1])==key]
        return SelectionConstraint(tuple(dict.fromkeys(path[2] for path in best)),reason=(
            prefix+f',estimated_progress={key[0]},estimated_sequences={[path[3] for path in best]},'
            'observed_score_unchanged,reward_unconfirmed'))
