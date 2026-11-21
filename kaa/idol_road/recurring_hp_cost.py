"""完整回合开始短语的当前图样证据；数字仍由当前文字双读确认。"""
from dataclasses import dataclass
from typing import Literal
import re
from .rules import RuleTextRead
from .concentration_rule import compact


@dataclass(frozen=True)
class RecurringHpTiming:
    frame_id: str
    body_rect: tuple[int,int,int,int]
    phrase_rect: tuple[int,int,int,int]
    phrase_score: float
    unique: bool
    template_size: tuple[int,int]
    variant: Literal['complete_turn_start_phrase'] = 'complete_turn_start_phrase'
    phrase_reads: tuple[RuleTextRead,...] = ()


def recurring_hp_amount(reads: tuple[RuleTextRead,...],parent: tuple[int,int,int,int],
                        timing: RecurringHpTiming | None) -> int | None:
    if (len(reads)!=2 or {read.representation for read in reads}!={'original','grayscale'}
        or len({read.frame_id for read in reads})!=1 or any(not read.frame_id or len(read.lines)!=2 for read in reads)):
        return None
    amounts=[]
    for read in reads:
        if compact(read.lines[0].text)!='以降、ターン開始時、' or not .98<=read.lines[1].confidence<=1:
            return None
        match=re.fullmatch(r'体力消費([1-9][0-9]?)',compact(read.lines[1].text))
        if match is None:
            return None
        amounts.append(int(match[1]))
    if len(set(amounts))!=1:
        return None
    if all(.98<=read.lines[0].confidence<=1 for read in reads):
        return amounts[0]
    if (timing is None or timing.frame_id!=reads[0].frame_id or timing.unique is not True
        or timing.variant!='complete_turn_start_phrase' or not .97<=timing.phrase_score<=1):
        return None
    x,y,w,h=parent
    bx,by,bw,bh=timing.body_rect
    px,py,pw,ph=timing.phrase_rect
    if (min(x,y,bx,by,px,py)<0 or min(w,h,bw,bh,pw,ph)<=0 or x+w>720 or y+h>1280
        or (pw,ph)!=timing.template_size
        or not (x<=bx and y<=by and bx+bw<=x+w and by+bh<=y+h)
        or not (bx<=px and by<=py and px+pw<=bx+bw and py+ph<=by+bh)):
        return None
    if not any(.98<=read.lines[0].confidence<=1 for read in reads):
        targeted=timing.phrase_reads
        if (len(targeted)!=2 or {r.representation for r in targeted}!={'original','grayscale'}
            or any(r.frame_id!=timing.frame_id or len(r.lines)!=1
                or compact(r.lines[0].text)!='以降、ターン開始時、'
                or not .98<=r.lines[0].confidence<=1 for r in targeted)):
            return None
    return amounts[0]
