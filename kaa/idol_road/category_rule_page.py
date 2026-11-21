"""Isolate one complete native category icon before reading a previous-card clause."""
from typing import Literal
import cv2
from cv2.typing import MatLike
from kotonebot.backend.ocr import jp
from kaa.tasks import R
from .category_rule import CategoryConditionEvidence
from .rule_symbols_page import find_rule_symbols
from .rules import RuleTextRead,RuleTextLine


def read_category_condition_evidence(body: MatLike,frame_id: str,
                                      original: tuple[RuleTextRead,...],offset: tuple[int,int]) -> CategoryConditionEvidence | None:
    if not frame_id or not any('直前' in line.text for read in original for line in read.lines):
        return None
    symbols=find_rule_symbols(body,frame_id,(
        ('mental',(R.IdolRoad.RuleMentalIcon,R.IdolRoad.RuleMentalIconScrolled)),
        ('active',(R.IdolRoad.RulePreviousActiveIcon,)),))
    if len(symbols)!=1:return None
    icon=symbols[0];x,y,w,h=icon.rect
    masked=body.copy();masked[y:y+h,x:x+w]=255
    variants: tuple[tuple[Literal['original','grayscale'],MatLike],...]=(
        ('original',masked),('grayscale',cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY)))
    reads=tuple(RuleTextRead(frame_id,kind,tuple(RuleTextLine(row.text,row.confidence)
        for row in sorted(jp().ocr(image),key=lambda row:(round(row.original_rect.center.y/10),row.original_rect.x1))))
        for kind,image in variants)
    return CategoryConditionEvidence(frame_id,(*offset,body.shape[1],body.shape[0]),icon,reads)
