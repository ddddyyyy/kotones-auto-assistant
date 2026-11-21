"""明确体力提示中的当前值／最大值；不从满血或编成推断分母。"""
from dataclasses import dataclass
import re
from typing import Literal


@dataclass(frozen=True)
class HpRatioRead:
    frame_id: str
    representation: Literal['original','grayscale']
    text: str | None
    confidence: float | None


@dataclass(frozen=True)
class HpRatioObservation:
    frame_id: str
    label_rect: tuple[int,int,int,int]
    label_score: float
    reads: tuple[HpRatioRead,...]
    status: Literal['unknown','conflict','resolved']
    hp: int | None = None
    max_hp: int | None = None


def confirm_hp_ratio(frame_id: str,label_rect: tuple[int,int,int,int],label_score: float,
                      reads: tuple[HpRatioRead,...]) -> HpRatioObservation:
    def result(status,hp=None,max_hp=None):
        return HpRatioObservation(frame_id,label_rect,label_score,reads,status,hp,max_hp)
    if (not frame_id or not .97<=label_score<=1 or min(label_rect)<0 or min(label_rect[2:])<=0
        or len(reads)!=2 or {r.representation for r in reads}!={'original','grayscale'}
        or any(r.frame_id!=frame_id for r in reads)):
        return result('unknown')
    values=[]
    for read in reads:
        if read.confidence is None or not .98<=read.confidence<=1 or read.text is None:
            return result('unknown')
        match=re.fullmatch(r'(0|[1-9][0-9]*)/([1-9][0-9]*)',re.sub(r'\s+','',read.text))
        if match is None:
            return result('unknown')
        values.append((int(match[1]),int(match[2])))
    if values[0]!=values[1] or values[0][0]>values[0][1]:
        return result('conflict')
    return result('resolved',*values[0])
