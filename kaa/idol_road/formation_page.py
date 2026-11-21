"""Main/Sub锚点下的当前槽位；行数按可见区域计算，未知布局停止。"""
from hashlib import sha256
from uuid import uuid4
from typing import Literal
import re
import cv2
import numpy as np
from cv2.typing import MatLike
from kotonebot.backend.image import find
from kotonebot.backend.ocr import en,jp
from kotonebot.primitives import Rect
from kaa.tasks import R
from .formation import FormationSlot,FormationSlots


def read_memory_power(screen: MatLike,rect: Rect) -> int | None:
    crop=screen[rect.y1:rect.y2,rect.x1:rect.x2]
    variants=(crop,cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY))
    primary=[en().ocr(variant) for variant in variants]
    if any(len(rows)!=1 or re.fullmatch(r'[1-9][0-9]*',rows[0].text) is None for rows in primary):
        return None
    values=[int(rows[0].text) for rows in primary]
    if values[0]!=values[1]:
        return None
    if all(.98<=rows[0].confidence<=1 for rows in primary):
        return values[0]
    for variant in variants:
        rows=en().ocr(cv2.resize(variant,None,fx=2,fy=2))
        if len(rows)!=1 or not .98<=rows[0].confidence<=1 or rows[0].text!=str(values[0]):
            return None
    return values[0]


def read_recommended_effect(screen: MatLike,main: Rect) -> Literal['concentration','good_condition'] | None:
    road=R.IdolRoad
    box=road.BoxFormationRecommendedEffect
    dx,dy=main.x1-road.PointFormationSlotsReference.x,main.y1-road.PointFormationSlotsReference.y
    crop=screen[box.y1+dy:box.y2+dy,box.x1+dx:box.x2+dx]
    values=[]
    for variant in (crop,cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)):
        rows=jp().ocr(variant)
        if len(rows)!=1 or not .98<=rows[0].confidence<=1:
            return None
        values.append(rows[0].text)
    if values[0]!=values[1]:
        return None
    return 'concentration' if values[0]=='集中' else 'good_condition' if values[0]=='好調' else None


def read_formation_slots(screen: MatLike,main: Rect) -> FormationSlots:
    road=R.IdolRoad
    frame=uuid4().hex
    origin=road.PointFormationSlotsReference
    dx,dy=main.x1-origin.x,main.y1-origin.y
    sub=find(screen,road.FormationSubLabel.template,threshold=.97)
    sub_reference=road.BoxFormationSubLabelReference
    if (sub is None or abs(sub.rect.x1-(sub_reference.x1+dx))>3
        or abs(sub.rect.y1-(sub_reference.y1+dy))>3):
        return FormationSlots(frame,())
    first=road.BoxFormationSubTile
    pitch_x=road.BoxFormationSubTileTwo.x1-first.x1
    pitch_y=road.BoxFormationSubTileNextRow.y1-first.y1
    rows=max(0,(road.BoxStartButton.y1-(first.y2+dy))//pitch_y+1)
    boxes: list[tuple[Literal['main','sub'],Rect]]=[('main',Rect(road.BoxFormationMainTile.x1+dx,road.BoxFormationMainTile.y1+dy,
                        road.BoxFormationMainTile.w,road.BoxFormationMainTile.h))]
    boxes.extend(('sub',Rect(first.x1+dx+column*pitch_x,first.y1+dy+row*pitch_y,first.w,first.h))
                 for row in range(rows) for column in range(2))
    result=[]
    for role,box in boxes:
        if box.x1<0 or box.y1<0 or box.x2>screen.shape[1] or box.y2>road.BoxStartButton.y1:
            result.append(FormationSlot(role,box.xywh,'unknown'))
            continue
        crop=screen[box.y1:box.y2,box.x1:box.x2]
        offset_x=box.x1-(road.BoxFormationMainTile.x1+dx)
        offset_y=box.y1-(road.BoxFormationMainTile.y1+dy)
        power_box=road.BoxMainPower
        power=read_memory_power(screen,Rect(power_box.x1+(main.x1-road.PointMainLabel.x)+offset_x,
                                power_box.y1+(main.y1-road.PointMainLabel.y)+offset_y,power_box.w,power_box.h))
        if power is not None:
            result.append(FormationSlot(role,box.xywh,'occupied',power,sha256(crop.tobytes()).hexdigest()))
            continue
        plus=find(crop,road.FormationEmptyPlus.template,threshold=.97)
        border=find(crop,road.FormationEmptyBorder.template,threshold=.97)
        if plus is not None and border is not None and border.rect.y1<plus.rect.y1:
            result.append(FormationSlot(role,box.xywh,'empty'))
        elif role=='sub' and np.all((crop>=30)&(crop<=75)) and float(crop.std())<8:
            result.append(FormationSlot(role,box.xywh,'absent'))
        else:
            result.append(FormationSlot(role,box.xywh,'unknown'))
    return FormationSlots(frame,tuple(result))
