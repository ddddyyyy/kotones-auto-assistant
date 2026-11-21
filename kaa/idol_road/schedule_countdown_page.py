"""Read the complete current field countdown from this screenshot only."""
import cv2
import numpy as np
from typing import Literal
from dataclasses import replace
from inspect import signature
from importlib import import_module
from cv2.typing import MatLike
from kotonebot.backend.ocr import jp,en
from kaa.tasks import R
from .rules import RuleTextRead,RuleTextLine
from .schedule_countdown import ScheduleCountdown,CountdownDigits,confirm_schedule_countdown
from .ocr_crop import right_edge_digit


def _read_pair(pixels: MatLike,frame_id: str,*,digits: bool=False) -> tuple[RuleTextRead,...]:
    reader=en() if digits else jp()
    variants: tuple[tuple[Literal['original','grayscale'],MatLike],...]=(
        ('original',pixels),('grayscale',cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY)))
    reads=[]
    for kind,image in variants:
        rows=reader.ocr(image,only_rec=True) if digits else reader.ocr(image)
        reads.append(RuleTextRead(frame_id,kind,tuple(RuleTextLine(row.text,row.confidence) for row in rows)))
    return tuple(reads)


def _digits(crop: MatLike,frame_id: str,rect: tuple[int,int,int,int]) -> CountdownDigits | None:
    proposals=[]
    white=lambda p:(p.min(axis=2)>220).astype('uint8')*255
    for control in (R.IdolRoad.BattleScheduleCountdownSuffix,R.IdolRoad.BattleScheduleCountdownInitialSuffix):
        template=white(control.template.pixels);h,w=template.shape
        # Preserve every suffix stroke and its contour while excluding the
        # animated stage background surrounding the native label.
        mask=cv2.dilate((template>0).astype('uint8'),np.ones((3,3),dtype='uint8'))
        scores=cv2.matchTemplate(white(crop),template,cv2.TM_CCORR_NORMED,mask=mask)
        scores=np.nan_to_num(scores,nan=-1.,posinf=-1.,neginf=-1.)
        _,score,_,(x,y)=cv2.minMaxLoc(scores)
        if not .97<=score<=1.000001 or x<=0:continue
        others=scores.copy();others[max(0,y-h+1):y+h,max(0,x-w+1):x+w]=-1
        if cv2.minMaxLoc(others)[1]>=.97:return None
        proposals.append((x,y,w,h,min(1.,float(score))))
    if not proposals or len({p[:4] for p in proposals})!=1:return None
    x,y,w,h,score=proposals[0]
    prefix=crop[:,:x];px,py,_,height=rect
    clipped=right_edge_digit(white(prefix))
    if 'only_rec' not in signature(en().ocr).parameters:return None
    try:
        reads=() if clipped else _read_pair(prefix,frame_id,digits=True)
    except Exception as error:
        missing=getattr(import_module('kotonebot.backend.ocr'),'OnlyRecEngineMissingError',None)
        if missing is not None and isinstance(error,missing):return None
        raise
    proof=CountdownDigits(frame_id,(px,py,x,height),(px+x,py+y,w,h),score,reads,clipped)
    if not clipped and any(len(read.lines)!=1 or read.lines[0].confidence<.98 for read in reads):
        # Scale the entire same prefix, preserving its boundary and every raw
        # reading. Strong conflicting originals still veto the scaled pair.
        scaled=cv2.resize(prefix,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
        proof=replace(proof,scaled_reads=_read_pair(scaled,frame_id,digits=True))
    return proof


def read_schedule_countdown(screen: MatLike,frame_id: str) -> ScheduleCountdown | None:
    if screen.shape!=(1280,720,3) or not frame_id:return None
    box=R.IdolRoad.BoxBattleScheduleCountdown
    crop=screen[box.y1:box.y2,box.x1:box.x2]
    reads=_read_pair(crop,frame_id)
    original=confirm_schedule_countdown(frame_id,box.xywh,reads)
    if original.turns_after is not None:return original
    scaled=_read_pair(cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC),frame_id)
    expanded=confirm_schedule_countdown(frame_id,box.xywh,reads,scaled)
    if expanded.turns_after is not None:return expanded
    return confirm_schedule_countdown(frame_id,box.xywh,reads,scaled,_digits(crop,frame_id,box.xywh))
