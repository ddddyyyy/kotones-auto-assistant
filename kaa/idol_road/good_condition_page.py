"""只读当前状态列好调图标及完整Nターン，不从卡面或历史帧补值。"""
import cv2
import re
from dataclasses import replace
from inspect import signature
from importlib import import_module
from typing import Literal
from cv2.typing import MatLike
from kotonebot.backend.ocr import jp,en,Ocr
from kaa.tasks import R
from .good_condition import GoodConditionObservation,GoodConditionDigits,GoodConditionNeutralDigits,confirm_good_condition,duration_digit_proposals
from .rules import RuleTextRead, RuleTextLine
from .hud_icon_page import find_current_blue_icon
from .ocr_crop import neutral_white_glyphs,right_edge_digit


def _read(pixels: MatLike, frame_id: str) -> tuple[RuleTextRead, ...]:
    variants: tuple[tuple[Literal['original', 'grayscale'], MatLike], ...] = (
        ('original', pixels), ('grayscale', cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)))
    return tuple(RuleTextRead(frame_id, kind, tuple(RuleTextLine(row.text, row.confidence)
                 for row in jp().ocr(image))) for kind, image in variants)


def _supports_only_rec(reader: Ocr) -> bool:
    try:
        return 'only_rec' in signature(reader.ocr).parameters
    except (TypeError,ValueError):
        return False


def _detection_digits(prefix: MatLike,frame_id: str,digit_rect: tuple[int,int,int,int],
                       suffix_rect: tuple[int,int,int,int],score: float,clipped: bool) -> GoodConditionDigits:
    reads=[]
    variants: tuple[tuple[Literal['original','grayscale'],MatLike],...]=(
        ('original',prefix),('grayscale',cv2.cvtColor(prefix,cv2.COLOR_BGR2GRAY)))
    for kind,image in variants:
        rows=en().ocr(cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)) if not clipped else ()
        reads.append(RuleTextRead(frame_id,kind,tuple(RuleTextLine(row.text,row.confidence) for row in rows)))
    return GoodConditionDigits(frame_id,digit_rect,suffix_rect,score,tuple(reads),clipped,
        mode='whole_prefix_double_size')


def _read_digits(screen: MatLike,frame_id: str,number_rect: tuple[int,int,int,int]) -> GoodConditionDigits | None:
    x,y,w,h=number_rect
    pixels=screen[y:y+h,x:x+w]
    candidates=[]
    for resource in (R.IdolRoad.GoodConditionTurnsSuffix,R.IdolRoad.GoodConditionTurnsSuffixCurrent):
        template=neutral_white_glyphs(resource.template.pixels);th,tw=template.shape
        if pixels.shape[0]<th or pixels.shape[1]<tw:continue
        scores=cv2.matchTemplate(neutral_white_glyphs(pixels),template,cv2.TM_CCOEFF_NORMED)
        _,score,_,(xx,yy)=cv2.minMaxLoc(scores)
        if not .97<=score<=1.000001 or xx<=0:continue
        others=scores.copy();others[max(0,yy-th+1):yy+th,max(0,xx-tw+1):xx+tw]=-1
        if cv2.minMaxLoc(others)[1]<.97:candidates.append((score,xx,yy,tw,th))
    if not candidates:return None
    score,xx,yy,tw,th=max(candidates)
    reader=en()
    prefix=pixels[:,:xx]  # Complete original-height region, no digit filtering, shrinking or text slicing.
    neutral=neutral_white_glyphs(prefix)
    gray_mask=cv2.threshold(cv2.cvtColor(prefix,cv2.COLOR_BGR2GRAY),245,255,cv2.THRESH_BINARY)[1]
    clipped=right_edge_digit(neutral) or right_edge_digit(gray_mask)
    digit_rect=(x,y,xx,h);suffix_rect=(x+xx,y+yy,tw,th)
    if not _supports_only_rec(reader):
        return _detection_digits(prefix,frame_id,digit_rect,suffix_rect,min(1.,float(score)),clipped)
    reads=[]
    api=import_module('kotonebot.backend.ocr')
    for kind,image in [('original',prefix),('grayscale',cv2.cvtColor(prefix,cv2.COLOR_BGR2GRAY))]:
        try:
            rows=reader.ocr(image,only_rec=True) if not clipped else ()
        except Exception as error:
            missing=getattr(api,'OnlyRecEngineMissingError',None)
            if missing is not None and isinstance(error,missing):
                return _detection_digits(prefix,frame_id,digit_rect,suffix_rect,min(1.,float(score)),clipped)
            raise
        reads.append(RuleTextRead(frame_id,kind,tuple(RuleTextLine(row.text,row.confidence) for row in rows)))
    evidence=GoodConditionDigits(frame_id,(x,y,xx,h),(x+xx,y+yy,tw,th),min(1.,float(score)),tuple(reads),clipped)
    strong=[read.lines[0].text for read in reads if len(read.lines)==1 and .98<=read.lines[0].confidence<=1]
    if (not clipped and len(strong)<2 and len(set(strong))<=1
        and all(re.fullmatch(r'0|[1-9][0-9]*',text) for text in strong)):
        return _detection_digits(prefix,frame_id,digit_rect,suffix_rect,min(1.,float(score)),clipped)
    if (clipped or any(len(read.lines)!=1 for read in reads)
        or re.fullmatch(r'0|[1-9][0-9]*',reads[0].lines[0].text) is None
        or not .98<=reads[0].lines[0].confidence<=1
        or reads[0].lines[0].text!=reads[1].lines[0].text
        or .98<=reads[1].lines[0].confidence<=1):
        return evidence
    neutral_reads=[]
    neutral_variants: tuple[tuple[Literal['original','grayscale'],MatLike],...]=(
        ('original',neutral),('grayscale',gray_mask))
    for representation,neutral_image in neutral_variants:
        try:
            rows=reader.ocr(cv2.resize(neutral_image,None,fx=2,fy=2),only_rec=True)
        except Exception as error:
            missing=getattr(api,'OnlyRecEngineMissingError',None)
            if missing is not None and isinstance(error,missing):
                return evidence
            raise
        neutral_reads.append(RuleTextRead(frame_id,representation,tuple(RuleTextLine(row.text,row.confidence) for row in rows)))
    return replace(evidence,neutral=GoodConditionNeutralDigits(frame_id,evidence.digit_rect,tuple(neutral_reads),clipped))


def read_good_condition(screen: MatLike, frame_id: str) -> GoodConditionObservation:
    road = R.IdolRoad
    box, reference, number = road.BoxGoodConditionSearch, road.BoxGoodConditionReference, road.BoxGoodConditionTurns
    unknown = lambda: confirm_good_condition(frame_id, None, None, None, None, ())
    matched = find_current_blue_icon(screen,box,
        (road.BattleGoodConditionIcon.template.pixels,road.BattleGoodConditionFirstIcon.template.pixels))
    if matched is None:
        return unknown()
    x,y,w,h = matched.rect
    score,blue = matched.score,matched.active_blue_fraction
    xx, yy = x+number.x1-reference.x1, y+number.y1-reference.y1
    number_rect = (xx, yy, number.w, number.h)
    pixels = screen[yy:yy+number.h, xx:xx+number.w]
    reads = _read(pixels, frame_id)
    result = confirm_good_condition(frame_id, (x,y,w,h), float(score), blue, number_rect, reads)
    if result.status != 'number_unknown':
        return result
    scaled=_read(cv2.resize(pixels, None, fx=2, fy=2), frame_id)
    result=confirm_good_condition(frame_id,(x,y,w,h),float(score),blue,number_rect,reads,scaled)
    if result.status!='number_unknown' or not duration_digit_proposals(reads,scaled):
        return result
    digits=_read_digits(screen,frame_id,number_rect)
    return confirm_good_condition(frame_id,(x,y,w,h),float(score),blue,number_rect,reads,scaled,digits)
