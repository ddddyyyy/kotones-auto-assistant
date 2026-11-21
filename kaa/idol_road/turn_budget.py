"""当前可见追加回合声明与固定总回合预算的冲突；不预测触发或新增回合。"""
from dataclasses import dataclass
from typing import Literal
from .rules import StageRules, confirmed_rule_turn_with_number


@dataclass(frozen=True)
class TurnBudgetConflict:
    declared_turn: int
    frame_id: str
    marker_rect: tuple[int,int,int,int]
    marker_score: float
    reason: Literal['visible_turn_addition'] = 'visible_turn_addition'


def fixed_turn_budget_conflicts(character_id: str, stage: int,
                                rules: StageRules | None) -> tuple[TurnBudgetConflict,...]:
    if (not character_id or type(stage) is not int or stage<1
        or rules is None or (rules.character_id,rules.stage)!=(character_id,stage)
        or rules.source!='visible_support_details'
        or rules.identity_source!='opened_from_confirmed_formation'):
        return ()
    found=[]
    seen=set()
    for viewport in rules.viewports:
        for row in viewport.visible_rules or ():
            symbol=row.turn_addition
            turn=confirmed_rule_turn_with_number(row.turn_reads,row.turn_number_evidence)
            if (symbol is None or symbol.kind!='turn_addition' or turn!=row.turn or type(turn) is not int
                or not 1<=turn<=30 or not .97<=symbol.score<=1
                or not symbol.frame_id or any(read.frame_id!=symbol.frame_id for read in row.turn_reads)
                or len(row.description_reads)!=2
                or {read.representation for read in row.description_reads}!={'original','grayscale'}
                or any(read.frame_id!=symbol.frame_id for read in row.description_reads)):
                continue
            x,y,w,h=row.card_rect
            sx,sy,sw,sh=symbol.rect
            if (min(x,y,sx,sy)<0 or min(w,h,sw,sh)<=0
                or not (x<=sx and y<=sy and sx+sw<=x+w and sy+sh<=y+h)):
                continue
            key=(turn,symbol.frame_id,symbol.rect)
            if key not in seen:
                seen.add(key)
                found.append(TurnBudgetConflict(turn,symbol.frame_id,symbol.rect,symbol.score))
    return tuple(found)
