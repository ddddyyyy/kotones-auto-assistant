"""只读取同一截图中唯一自己的分数框；无截图或输入操作。"""
from inspect import signature
from importlib import import_module
from typing import Literal
import cv2
import numpy as np
from cv2.typing import MatLike
from kotonebot.backend.ocr import en
from kaa.tasks.produce.new.play_cards.page import find_exam_own_score_box
from .rules import RuleTextRead,RuleTextLine
from .own_score import OwnScoreObservation,OwnScoreMaskDigits,OwnScoreMarginDigits,confirm_own_score,score_mask_proposal
from .ocr_crop import neutral_white_glyphs,right_edge_digit


def _score_masks(crop: MatLike) -> tuple[MatLike,MatLike] | None:
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    masks=(neutral_white_glyphs(crop),(gray>245).astype('uint8')*255)
    h,w=gray.shape
    yy,xx=np.indices(gray.shape)
    # The owner-frame crop includes its right/bottom lines. Remove a connected
    # component only when ALL its pixels lie in those thin border bands.
    border=(xx>=w-10)|(yy>=h-3)
    for mask in masks:
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        for index in range(1,count):
            x,y,cw,ch,_=stats[index]
            if cw>=w*.6 and ch==h and x+cw>=w-6:
                if not np.all(border[labels==index]):
                    return None  # A digit joined to the border must not be deleted.
                mask[labels==index]=0
        if right_edge_digit(mask) or right_edge_digit(np.fliplr(mask)):
            return None
    return masks


def _digits_from_masks(masks: tuple[MatLike,MatLike],rect: tuple[int,int,int,int],frame_id: str,reader) -> OwnScoreMaskDigits | None:
    h,w=masks[0].shape
    contents=[]
    for mask in masks:
        if right_edge_digit(mask) or right_edge_digit(np.fliplr(mask)):
            return None
        points=cv2.findNonZero(mask)
        if points is None:
            return None
        bounds=cv2.boundingRect(points)
        x,y,cw,ch=bounds
        if ch<h*.5 or cw<5 or cv2.countNonZero(mask)<40:
            return None
        contents.append((bounds,mask[y:y+ch,x:x+cw],cv2.countNonZero(mask)))
    if contents[0][0]!=contents[1][0]:
        return None
    reads=[]
    for kind,(_,pixels,_) in zip(('original','grayscale'),contents):
        rows=reader.ocr(pixels,only_rec=True)
        reads.append(RuleTextRead(frame_id,kind,tuple(RuleTextLine(row.text,row.confidence) for row in rows)))
    x,y,cw,ch=contents[0][0]
    return OwnScoreMaskDigits(frame_id,rect,(rect[0]+x,rect[1]+y,cw,ch),tuple(reads),
                              min(item[2] for item in contents))


def _foreground_digits(crop: MatLike,rect: tuple[int,int,int,int],frame_id: str,reader) -> OwnScoreMaskDigits | None:
    masks=_score_masks(crop)
    return _digits_from_masks(masks,rect,frame_id,reader) if masks is not None else None


def _margin_digits(crop: MatLike,rect: tuple[int,int,int,int],frame_id: str,reader,
                   parent: OwnScoreMaskDigits | None) -> OwnScoreMarginDigits | None:
    if parent is None or parent.frame_id!=frame_id or parent.score_rect!=rect or score_mask_proposal(parent) is None:
        return None
    masks=_score_masks(crop)
    if masks is None:
        return None
    h,w=masks[0].shape
    geometries=[]
    for mask in masks:
        _,_,stats,_=cv2.connectedComponentsWithStats(mask)
        components=tuple(tuple(int(v) for v in row) for row in stats[1:])
        # Include every non-speckle component, not the largest or most digit-like.
        significant=[c for c in components if c[2]>2 or c[3]>2 or c[4]>4]
        if not significant:
            return None
        margin=min(c[1] for c in significant)-1
        if not 1<=margin<=h//5 or any(c[1]<margin<c[1]+c[3] for c in components):
            return None
        excluded=tuple(sorted(c for c in components if c[1]+c[3]<=margin))
        if not excluded or not 1<=sum(c[4] for c in excluded)<=margin:
            return None
        geometries.append((margin,excluded))
    if geometries[0]!=geometries[1]:
        return None
    margin,excluded=geometries[0]
    line_rect=(rect[0],rect[1]+margin,w,h-margin)
    digits=_digits_from_masks((masks[0][margin:,:],masks[1][margin:,:]),line_rect,frame_id,reader)
    return OwnScoreMarginDigits(frame_id,rect,margin,excluded,digits) if digits is not None else None


def read_own_score(screen: MatLike,frame_id: str) -> OwnScoreObservation:
    if screen.shape!=(1280,720,3) or not frame_id:
        return confirm_own_score(frame_id,None,())
    box=find_exam_own_score_box(screen)
    if box is None:
        return confirm_own_score(frame_id,None,())
    crop=screen[box.y1:box.y2,box.x1:box.x2]
    variants: tuple[tuple[Literal['original','grayscale'],MatLike],...]=(
        ('original',crop),('grayscale',cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)))
    reader=en()
    reads=tuple(RuleTextRead(frame_id,kind,tuple(RuleTextLine(row.text,row.confidence)
                for row in reader.ocr(pixels))) for kind,pixels in variants)
    clipped=right_edge_digit(neutral_white_glyphs(crop))
    result=confirm_own_score(frame_id,box.xywh,reads,right_edge_clipped=clipped)
    if result.status!='number_unknown':
        return result
    try:
        supported='only_rec' in signature(reader.ocr).parameters
    except (TypeError,ValueError):
        supported=False
    if not supported or clipped:
        return result
    api=import_module('kotonebot.backend.ocr')
    direct=[]
    for kind,pixels in variants:
        try:
            rows=reader.ocr(pixels,only_rec=True)
        except Exception as error:
            missing=getattr(api,'OnlyRecEngineMissingError',None)
            if missing is not None and isinstance(error,missing):
                return result
            raise
        direct.append(RuleTextRead(frame_id,kind,tuple(RuleTextLine(row.text,row.confidence) for row in rows)))
    result=confirm_own_score(frame_id,box.xywh,reads,tuple(direct),clipped)
    if result.status!='number_unknown':
        return result
    digits=_foreground_digits(crop,box.xywh,frame_id,reader)
    result=confirm_own_score(frame_id,box.xywh,reads,tuple(direct),clipped,digits)
    if result.status!='number_unknown':
        return result
    margin=_margin_digits(crop,box.xywh,frame_id,reader,digits)
    return confirm_own_score(frame_id,box.xywh,reads,tuple(direct),clipped,digits,margin)
