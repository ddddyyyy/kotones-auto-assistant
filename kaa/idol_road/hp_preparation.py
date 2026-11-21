"""可靠场地声明的体力准备预算；不预支回血，也不修改实际体力。"""
from dataclasses import dataclass
import re
from typing import Literal
from .rules import StageRules,confirmed_rule_turn_with_number
from .concentration_rule import compact
from .planning_turn import PlanningTurn
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from .recurring_hp_cost import recurring_hp_amount


@dataclass(frozen=True)
class HpThresholdGoal:
    declared_turn: int
    at_least_percent: int
    evidence_frames: tuple[str,...]


@dataclass(frozen=True)
class RecurringHpCost:
    start_turn: int
    amount: int
    evidence_frames: tuple[str,...]


@dataclass(frozen=True)
class HpPreparationBudget:
    goals: tuple[HpThresholdGoal,...]
    recurring: RecurringHpCost | None
    recurring_known: bool
    timing: str = 'reserve_through_declared_turn_start_phase_unconfirmed'


def hp_preparation_budget(character_id: str,stage: int,rules: StageRules | None) -> HpPreparationBudget | None:
    if (rules is None or (rules.character_id,rules.stage)!=(character_id,stage)
        or rules.source!='visible_support_details' or rules.identity_source!='opened_from_confirmed_formation'):
        return None
    hp_rows: dict[int,list[tuple[int,str]]]={}
    costs: list[tuple[int,int,str]]=[]
    blocked=set()
    cost_unknown=False
    for viewport in rules.viewports:
        for row in viewport.visible_rules or ():
            turn=confirmed_rule_turn_with_number(row.turn_reads,row.turn_number_evidence)
            reads=row.description_reads
            if (turn is None or turn!=row.turn or len(reads)!=2
                or {read.representation for read in reads}!={'original','grayscale'}
                or len({read.frame_id for read in (*reads,*row.turn_reads)})!=1
                or any(not read.frame_id or not read.lines for read in reads)):
                continue
            x,y,w,h=row.card_rect
            if min(x,y)<0 or min(w,h)<=0 or x+w>720 or y+h>1280:
                continue
            first=[compact(read.lines[0].text) for read in reads]
            if any(text.startswith('体力が') for text in first):
                values=[]
                for read,text in zip(reads,first):
                    if not .98<=read.lines[0].confidence<=1:
                        continue
                    match=re.fullmatch(r'体力が([1-9][0-9]?|100)%以上の場合、?',text)
                    if match is None:
                        blocked.add(turn)
                    else:
                        values.append(int(match[1]))
                if len(set(values))>1:
                    blocked.add(turn)
                elif len(values)==2:
                    hp_rows.setdefault(turn,[]).append((values[0],reads[0].frame_id))
            if any('ターン開始時' in text for text in first) or row.recurring_hp_timing is not None:
                amount=recurring_hp_amount(reads,row.card_rect,row.recurring_hp_timing)
                if amount is None:
                    cost_unknown=True
                else:
                    costs.append((turn,amount,reads[0].frame_id))
    goals=tuple(HpThresholdGoal(turn,rows[0][0],tuple(dict.fromkeys(row[1] for row in rows)))
                for turn,rows in sorted(hp_rows.items()) if turn not in blocked and len({row[0] for row in rows})==1)
    known=bool(costs) and not cost_unknown and len({row[:2] for row in costs})==1
    recurring=RecurringHpCost(costs[0][0],costs[0][1],tuple(dict.fromkeys(row[2] for row in costs))) if known else None
    return HpPreparationBudget(goals,recurring,known)


@dataclass(frozen=True)
class HpRouteAssessment:
    status: Literal['safe','unknown','below_reserve','not_applicable']
    required_hp: int | None = None
    reserve_cost: int | None = None


def assess_hp_route(before: BattleState,after: BattleState,turn: PlanningTurn,budget: HpPreparationBudget) -> HpRouteAssessment:
    goals=[goal for goal in budget.goals if turn.turn<goal.declared_turn<=turn.last_scheduled_turn]
    if not goals:
        return HpRouteAssessment('not_applicable')
    maximum=before.max_hp
    if (not before.hp_known or not after.hp_known or type(before.hp) is not int or type(after.hp) is not int
        or type(maximum) is not int or maximum<=0 or not 0<=before.hp<=maximum or not 0<=after.hp<=maximum
        or after.max_hp!=maximum or not budget.recurring_known or budget.recurring is None):
        return HpRouteAssessment('unknown')
    cost=budget.recurring
    if (type(cost.amount) is not int or cost.amount<1 or type(cost.start_turn) is not int or cost.start_turn<1
        or not after.remaining_turns_known or after.remaining_turns!=before.remaining_turns
        or any(type(goal.at_least_percent) is not int or not 1<=goal.at_least_percent<=100 for goal in goals)):
        return HpRouteAssessment('unknown')
    required=[]
    for goal in goals:
        threshold=(maximum*goal.at_least_percent+99)//100
        future_starts=max(0,goal.declared_turn-max(turn.turn+1,cost.start_turn)+1)
        reserve=future_starts*cost.amount
        required.append((threshold+reserve,reserve))
    floor,reserve=max(required)
    return HpRouteAssessment('safe' if after.hp>=floor and after.hp>0 else 'below_reserve',floor,reserve)
