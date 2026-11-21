"""Read one screenshot's enabled SKIP heart border and complete single digit."""
from typing import Literal
from inspect import signature
from importlib import import_module
import cv2
import numpy as np
from cv2.typing import MatLike
from kotonebot.backend.ocr import en
from kaa.tasks import R
from .turn_end_page import read_enabled_skip
from .ocr_crop import neutral_white_glyphs,right_edge_digit
from .rules import RuleTextRead,RuleTextLine
from .skip_recovery import SkipRecoveryEvidence,SkipRecoveryObservation,confirm_skip_recovery


def read_skip_recovery(screen: MatLike,frame_id: str) -> SkipRecoveryObservation:
    road=R.IdolRoad
    icon,masked,number=road.BoxSkipRecoveryIcon,road.BoxSkipRecoveryNumberMask,road.BoxSkipRecoveryNumber
    enabled=screen.shape==(1280,720,3) and read_enabled_skip(screen)
    evidence=SkipRecoveryEvidence(frame_id,enabled,icon.xywh,masked.xywh,number.xywh,None,None,())
    if not enabled:
        return confirm_skip_recovery(evidence)
    template=road.SkipRecoveryIcon.template.pixels
    mask=np.full(template.shape[:2],255,dtype='uint8')
    mask[masked.y1-icon.y1:masked.y2-icon.y1,masked.x1-icon.x1:masked.x2-icon.x1]=0
    current=screen[icon.y1:icon.y2,icon.x1:icon.x2]
    score=float(cv2.matchTemplate(current,template,cv2.TM_CCORR_NORMED,mask=mask)[0,0])
    crop=screen[number.y1:number.y2,number.x1:number.x2]
    glyphs=neutral_white_glyphs(crop)
    clipped=right_edge_digit(glyphs) or right_edge_digit(np.fliplr(glyphs))
    _,_,stats,_=cv2.connectedComponentsWithStats(glyphs)
    height,width=glyphs.shape
    clipped=clipped or any(area>=4 and h>=height*.4
        and (x==0 or y==0 or x+w>=width or y+h>=height) for x,y,w,h,area in stats[1:])
    variants: tuple[tuple[Literal['original','grayscale'],MatLike],...]=(
        ('original',crop),('grayscale',cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)))
    reads=[]
    reader=en()
    try:
        supported='only_rec' in signature(reader.ocr).parameters
    except (TypeError,ValueError):
        supported=False
    if .97<=score<=1.000001 and not clipped and supported:
        for kind,image in variants:
            try:
                rows=reader.ocr(image,only_rec=True)
            except Exception as error:
                missing=getattr(import_module('kotonebot.backend.ocr'),'OnlyRecEngineMissingError',None)
                if missing is not None and isinstance(error,missing):
                    reads=[]
                    break
                raise
            reads.append(RuleTextRead(frame_id,kind,tuple(RuleTextLine(row.text,row.confidence) for row in rows)))
    return confirm_skip_recovery(SkipRecoveryEvidence(frame_id,enabled,icon.xywh,masked.xywh,number.xywh,
        1. if 1<score<=1.000001 else score,clipped,tuple(reads)))
