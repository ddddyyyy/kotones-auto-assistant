"""搜索当前状态列集中，不读取上方预告、详情或卡面内图标。"""
import cv2
from typing import Literal
import numpy as np
from cv2.typing import MatLike
from kotonebot.backend.ocr import en
from kaa.tasks import R
from .concentration import ConcentrationNumberRead, ConcentrationObservation, ConcentrationExpandedDigits,confirm_concentration,neutral_concentration_proposal,expanded_concentration_proposal
from .rules import RuleTextRead,RuleTextLine
from .hud_icon_page import find_current_blue_icon
from .ocr_crop import right_edge_digit as _edge_digit,neutral_white_glyphs


def read_concentration(screen: MatLike, frame_id: str) -> ConcentrationObservation:
    road=R.IdolRoad
    matched=find_current_blue_icon(screen,road.BoxBattleConcentrationStack,
        (road.BattleConcentrationIcon.template.pixels,road.BattleConcentrationLowerIcon.template.pixels,
         road.BattleConcentrationMiddleIcon.template.pixels))
    if matched is None:
        return confirm_concentration(frame_id,None,None,())
    reference, number=road.BoxBattleConcentrationReference,road.BoxBattleConcentrationDigits
    icon=matched.rect
    x,y=icon[0]+number.x1-reference.x1,icon[1]+number.y1-reference.y1
    crop=screen[y:y+number.h,x:x+number.w]
    color_mask=cv2.inRange(crop,np.array([200,200,200]),np.array([255,255,255]))
    gray_mask=cv2.threshold(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),200,255,cv2.THRESH_BINARY)[1]
    def read(scale: Literal[1,2]) -> tuple[ConcentrationNumberRead,...]:
        reads=[]
        for kind,image in [('original',crop),('color_white_mask',color_mask),('gray_white_mask',gray_mask)]:
            rows=en().ocr(image if scale==1 else cv2.resize(image,None,fx=scale,fy=scale))
            reads.append(ConcentrationNumberRead(frame_id,kind,rows[0].text if len(rows)==1 else None,
                                              rows[0].confidence if len(rows)==1 else None,(x,y,number.w,number.h),len(rows),
                                              _edge_digit(image) if kind!='original' else False,scale))
        return tuple(reads)
    reads=read(1)
    result=confirm_concentration(frame_id,icon,matched.score,reads)
    if result.status!='number_unknown':
        return result
    scaled=()
    if all(row.row_count==1 and row.text is not None for row in reads) and len({row.text for row in reads})==1:
        scaled=read(2)
        result=confirm_concentration(frame_id,icon,matched.score,reads,scaled)
    if result.status!='number_unknown':
        return result
    neutral=neutral_white_glyphs(crop)
    neutral_gray=cv2.threshold(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),245,255,cv2.THRESH_BINARY)[1]
    supplement=[]
    variants: tuple[tuple[Literal['neutral_white_mask','neutral_gray_white_mask'],MatLike],...]=(
        ('neutral_white_mask',neutral),('neutral_gray_white_mask',neutral_gray))
    if neutral_concentration_proposal(reads,scaled) is not None:
        for kind,image in variants:
            rows=en().ocr(cv2.resize(image,None,fx=2,fy=2))
            supplement.append(ConcentrationNumberRead(frame_id,kind,rows[0].text if len(rows)==1 else None,
                rows[0].confidence if len(rows)==1 else None,(x,y,number.w,number.h),len(rows),_edge_digit(image),2))
    result=confirm_concentration(frame_id,icon,matched.score,reads,scaled,tuple(supplement))
    if result.status!='number_unknown' or expanded_concentration_proposal(reads) is None:
        return result
    wide=road.BoxBattleConcentrationExpandedDigits
    xx,yy=icon[0]+wide.x1-reference.x1,icon[1]+wide.y1-reference.y1
    pixels=screen[yy:yy+wide.h,xx:xx+wide.w]
    masks=((pixels.min(axis=2)>=200).astype('uint8')*255,
        cv2.threshold(cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY),200,255,cv2.THRESH_BINARY)[1],
        neutral_white_glyphs(pixels),
        cv2.threshold(cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY),245,255,cv2.THRESH_BINARY)[1])
    clipped=any(_edge_digit(mask) for mask in masks)
    wide_variants: tuple[tuple[Literal['original','grayscale'],MatLike],...]=(
        ('original',pixels),('grayscale',cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY)))
    wide_reads=tuple(RuleTextRead(frame_id,kind,tuple(RuleTextLine(row.text,row.confidence)
        for row in en().ocr(image))) for kind,image in wide_variants)
    proof=ConcentrationExpandedDigits(frame_id,reads[0].rect,(xx,yy,wide.w,wide.h),wide_reads,clipped)
    return confirm_concentration(frame_id,icon,matched.score,reads,scaled,tuple(supplement),proof)
