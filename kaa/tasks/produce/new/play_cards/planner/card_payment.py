"""Same-frame native payment evidence, never a future cost assumption."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CardPaymentObservation:
    frame_id: str
    card_id: str
    upgrade: int
    rect: tuple[int, int, int, int]
    green: int
    score: float
    coverage: float
    green_coverage: float
    unique: bool
    source: Literal['native_sleep_green_cost_literal'] = 'native_sleep_green_cost_literal'


def valid_card_payment(value: CardPaymentObservation, frame_id: str) -> bool:
    x,y,w,h=value.rect
    if any(type(number) not in (int,float) for number in (value.score,value.coverage,value.green_coverage)):
        return False
    return bool(frame_id and value.frame_id==frame_id and value.card_id=='p_card-00-acc-0_002'
        and type(value.upgrade) is int and value.upgrade==0 and value.source=='native_sleep_green_cost_literal'
        and type(value.green) is int and value.green in (0,2)
        and all(type(v) is int for v in value.rect) and min(x,y)>=0 and min(w,h)>0
        and x+w<=720 and y+h<=1280 and .97<=value.score<=1 and .95<=value.coverage<=1
        and .6<=value.green_coverage<=1 and value.unique is True)
