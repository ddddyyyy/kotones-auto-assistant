"""Current SKIP button recovery display, independent of future HP guarantees."""
from dataclasses import dataclass
import math
import re
from typing import Literal
from .rules import RuleTextRead


@dataclass(frozen=True)
class SkipRecoveryEvidence:
    frame_id: str
    enabled: bool
    icon_rect: tuple[int,int,int,int]
    number_mask_rect: tuple[int,int,int,int]
    number_rect: tuple[int,int,int,int]
    icon_score: float | None
    edge_clipped: bool | None
    reads: tuple[RuleTextRead,...]


@dataclass(frozen=True)
class SkipRecoveryObservation:
    evidence: SkipRecoveryEvidence
    status: Literal['resolved','unknown']
    value: int | None = None
    scope: Literal['current_button_recovery_display_only'] = 'current_button_recovery_display_only'
    next_turn_hp_guaranteed: Literal[False] = False


def confirm_skip_recovery(evidence: SkipRecoveryEvidence) -> SkipRecoveryObservation:
    unknown=SkipRecoveryObservation(evidence,'unknown')
    score=evidence.icon_score
    rects=(evidence.icon_rect,evidence.number_mask_rect,evidence.number_rect)
    if (not evidence.frame_id or evidence.enabled is not True or evidence.edge_clipped is not False
        or score is None or type(score) not in (int,float) or not math.isfinite(score) or not .97<=score<=1
        or any(any(type(v) is not int for v in rect) or min(rect)<0 or min(rect[2:])<1
               or rect[0]+rect[2]>720 or rect[1]+rect[3]>1280 for rect in rects)):
        return unknown
    for outer,inner in zip(rects,rects[1:]):
        x,y,w,h=outer
        a,b,c,d=inner
        if not (x<=a and y<=b and a+c<=x+w and b+d<=y+h):
            return unknown
    reads=evidence.reads
    if (len(reads)!=2 or {r.representation for r in reads}!={'original','grayscale'}
        or any(r.frame_id!=evidence.frame_id or len(r.lines)!=1 for r in reads)):
        return unknown
    values=[]
    for read in reads:
        line=read.lines[0]
        if (type(line.confidence) not in (int,float) or not .98<=line.confidence<=1
            or not re.fullmatch(r'[0-9]',line.text)):
            return unknown
        values.append(int(line.text))
    return SkipRecoveryObservation(evidence,'resolved',values[0]) if values[0]==values[1] else unknown
