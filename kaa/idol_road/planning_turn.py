"""场地策略的当前回合身份；只消费观察，不预测追加回合。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal,TYPE_CHECKING
from kaa.battle_turn import AbsoluteTurnObservation
if TYPE_CHECKING:
    from kaa.tasks.produce.new.play_cards.planner.state import BattleState


@dataclass(frozen=True)
class PlanningTurn:
    source: Literal['fixed_budget','absolute']
    owner: str | int
    turn: int
    last_scheduled_turn: int

    @property
    def key(self) -> tuple[str,str | int,int]:
        return self.source,self.owner,self.turn


def read_planning_turn(state: BattleState, initial_total: int) -> PlanningTurn | None:
    if (not state.remaining_turns_known or type(state.remaining_turns) is not int
        or not 1<=state.remaining_turns<=30 or state.projected_effect_values is not None):
        return None
    observed=state.absolute_turn_observation
    if observed is not None:
        try:
            observed.validate()
        except ValueError:
            return None
        if observed.remaining_turns!=state.remaining_turns:
            return None
        return PlanningTurn('absolute',observed.battle_id,observed.turn,observed.turn+observed.remaining_turns-1)
    if type(initial_total) is not int or not state.remaining_turns<=initial_total<=30:
        return None
    return PlanningTurn('fixed_budget',initial_total,initial_total-state.remaining_turns+1,initial_total)


class PlanningTurnTracker:
    """同一策略实例不能换场、回退或丢绝对回合后退回固定预算。"""
    def __init__(self) -> None:
        self._last: PlanningTurn | None=None
        self._absolute: AbsoluteTurnObservation | None=None

    def observe(self, state: BattleState, initial_total: int) -> PlanningTurn | None:
        current=read_planning_turn(state,initial_total)
        if current is None:
            return None
        previous=self._last
        absolute=state.absolute_turn_observation
        if previous is not None and (current.key[:2]!=previous.key[:2] or current.turn<previous.turn):
            return None
        if (self._absolute is not None and absolute is not None
            and self._absolute.frame_id==absolute.frame_id and self._absolute!=absolute):
            return None
        self._last=current
        self._absolute=absolute
        return current
