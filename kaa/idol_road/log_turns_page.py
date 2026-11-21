"""只读完整日志回合标题＋手札信息标记，不截图、不选择当前回合或操作弹窗。"""
from typing import Literal
from dataclasses import replace
from inspect import signature
from importlib import import_module
import cv2
import numpy as np
from cv2.typing import MatLike
from kotonebot.backend.ocr import en,jp
from kaa.tasks import R
from .rules import RuleTextRead,RuleTextLine
from .log_turns import LogTurnEvidence,LogRemainingDirect,LogAbsoluteDirect,VisibleLogTurn,confirm_visible_log_turn
from .ocr_crop import neutral_white_glyphs,right_edge_digit
from .log_remaining_glyphs import LogRemainingGlyphs,mixed_remaining_proposal


def _read(pixels: MatLike, frame_id: str, *, japanese: bool) -> tuple[RuleTextRead,...]:
    variants: tuple[tuple[Literal['original','grayscale'],MatLike],...]=(
        ('original',pixels),('grayscale',cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY)))
    reader=jp() if japanese else en()
    return tuple(RuleTextRead(frame_id,kind,tuple(RuleTextLine(row.text,row.confidence)
        for row in reader.ocr(image))) for kind,image in variants)


def _recognize_whole_crop(pixels: MatLike,frame_id: str) -> tuple[RuleTextRead,...]:
    reader=en()
    try:
        if 'only_rec' not in signature(reader.ocr).parameters:return ()
    except (TypeError,ValueError):return ()
    api=import_module('kotonebot.backend.ocr')
    variants: tuple[tuple[Literal['original','grayscale'],MatLike],...]=(
        ('original',pixels),('grayscale',cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY)))
    result=[]
    for kind,image in variants:
        try:
            rows=reader.ocr(image,only_rec=True)
        except Exception as error:
            missing=getattr(api,'OnlyRecEngineMissingError',None)
            if missing is not None and isinstance(error,missing):return ()
            raise
        result.append(RuleTextRead(frame_id,kind,tuple(RuleTextLine(row.text,row.confidence) for row in rows)))
    return tuple(result)


def read_visible_log_turns(screen: MatLike, frame_id: str) -> tuple[VisibleLogTurn,...]:
    if screen.shape!=(1280,720,3) or not frame_id:
        return ()
    road=R.IdolRoad
    search=road.BoxLogTurnSuffixSearch
    reference=road.BoxLogTurnSuffixReference
    suffix=road.LogTurnHeaderSuffix.template.pixels
    pixels=screen[search.y1:search.y2,search.x1:search.x2]
    variants=(suffix,road.LogTurnHeaderSuffixAlternate.template.pixels,road.LogTurnHeaderSuffixScrolled.template.pixels)
    names: tuple[Literal['reference','alternate','scrolled'],...]=('reference','alternate','scrolled')
    grids=np.stack([cv2.matchTemplate(pixels,template,cv2.TM_CCOEFF_NORMED) for template in variants])
    selected=grids.argmax(axis=0)
    scores=grids.max(axis=0)
    matches=[]
    for _ in range(9):
        _,score,_,(x,y)=cv2.minMaxLoc(scores)
        if not .97<=score<=1.000001:
            break
        matches.append((min(1.,float(score)),x+search.x1,y+search.y1,int(selected[y,x])))
        h,w=suffix.shape[:2]
        scores[max(0,y-h+1):y+h,max(0,x-w+1):x+w]=-1
    if len(matches)>8:
        return ()
    result=[]
    for score,x,y,variant in sorted(matches,key=lambda item:item[2]):
        dy=y-reference.y1
        hand=road.BoxLogHandInfoSearch
        hand_crop=screen[hand.y1+dy:hand.y2+dy,hand.x1:hand.x2]
        template=road.LogHandInfoLabel.template.pixels
        hand_variants=(template,road.LogHandInfoLabelAlternate.template.pixels,road.LogHandInfoLabelScrolled.template.pixels)
        hand_variant=0
        if hand_crop.shape[0]<template.shape[0] or hand_crop.shape[1]<template.shape[1]:
            hand_score,hx,hy=0.,0,0
        else:
            hand_matches=[cv2.minMaxLoc(cv2.matchTemplate(hand_crop,item,cv2.TM_CCOEFF_NORMED)) for item in hand_variants]
            hand_variant=max(range(len(hand_matches)),key=lambda index:hand_matches[index][1])
            _,hand_score,_,(hx,hy)=hand_matches[hand_variant]
            hand_score=min(1.,hand_score) if hand_score<=1.000001 else 0.
        turn,remaining=road.BoxLogAbsoluteTurnText,road.BoxLogRemainingDigits
        # 数字位数改变会移动后缀；标题左边界固定，不随后缀右移而截掉前导数字。
        turn_rect=(turn.x1,turn.y1+dy,x+suffix.shape[1]+turn.x2-reference.x2-turn.x1,turn.h)
        remaining_rect=(remaining.x1,remaining.y1+dy,remaining.w,remaining.h)
        def crop(rect):
            xx,yy,w,h=rect
            return screen[yy:yy+h,xx:xx+w]
        readable=(min(turn_rect)>=0 and min(remaining_rect)>=0 and turn_rect[2]>0
                  and .97<=hand_score<=1)
        # 后方有标题但同排标签／完整文字不可靠时保留未知行，不能静默丢弃再选前一行。
        evidence=LogTurnEvidence(frame_id,(x,y,suffix.shape[1],suffix.shape[0]),score,
            (hx+hand.x1,hy+hand.y1+dy,template.shape[1],template.shape[0]),float(hand_score),
            turn_rect,_read(crop(turn_rect),frame_id,japanese=True) if readable else (),
            remaining_rect,_read(crop(remaining_rect),frame_id,japanese=False) if readable else (),
            names[variant],names[hand_variant])
        observed=confirm_visible_log_turn(evidence)
        if readable and observed.absolute_turn is None and observed.status!='conflict' and x>turn_rect[0]:
            number_rect=(turn_rect[0],turn_rect[1],x-turn_rect[0],turn_rect[3])
            original=crop(number_rect)
            glyphs=neutral_white_glyphs(original)
            clipped=right_edge_digit(glyphs) or right_edge_digit(np.fliplr(glyphs))
            reader=en()
            try:
                supported='only_rec' in signature(reader.ocr).parameters
            except (TypeError,ValueError):
                supported=False
            if supported and not clipped:
                reads=[]
                api=import_module('kotonebot.backend.ocr')
                absolute_variants: tuple[tuple[Literal['original','grayscale'],MatLike],...]=(
                    ('original',original),('grayscale',cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)))
                for kind,image in absolute_variants:
                    try:
                        rows=reader.ocr(image,only_rec=True)
                    except Exception as error:
                        missing=getattr(api,'OnlyRecEngineMissingError',None)
                        if missing is not None and isinstance(error,missing):
                            reads=[]
                            break
                        raise
                    reads.append(RuleTextRead(frame_id,kind,tuple(RuleTextLine(row.text,row.confidence) for row in rows)))
                if reads:
                    evidence=replace(evidence,absolute_direct=LogAbsoluteDirect(frame_id,number_rect,tuple(reads),clipped))
                    observed=confirm_visible_log_turn(evidence)
        mixed=mixed_remaining_proposal(evidence.remaining_reads,frame_id) is not None
        if readable and observed.remaining_turns is None and (observed.status!='conflict' or mixed):
            enlarged=cv2.resize(crop(remaining_rect),None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
            evidence=replace(evidence,remaining_scaled_reads=_read(enlarged,frame_id,japanese=False))
            observed=confirm_visible_log_turn(evidence)
        if readable and observed.remaining_turns is None and (observed.status!='conflict' or mixed):
            original=crop(remaining_rect)
            if mixed:
                masks=(neutral_white_glyphs(original),
                    (cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)>245).astype('uint8')*255)
                components=[]
                for mask in masks:
                    _,_,stats,_=cv2.connectedComponentsWithStats(mask)
                    components.append(tuple((int(row[0]),int(row[1]),int(row[2]),int(row[3]),int(row[4]))
                                            for row in stats[1:]))
                evidence=replace(evidence,remaining_glyphs=LogRemainingGlyphs(frame_id,remaining_rect,
                    components[0],components[1],cv2.countNonZero(masks[0]),cv2.countNonZero(masks[1])))
            reader=en()
            try:
                supported='only_rec' in signature(reader.ocr).parameters
            except (TypeError,ValueError):
                supported=False
            clipped=right_edge_digit(neutral_white_glyphs(original))
            if supported and not clipped:
                reads=[]
                api=import_module('kotonebot.backend.ocr')
                direct_variants: tuple[tuple[Literal['original','grayscale'],MatLike],...]=(
                    ('original',original),('grayscale',cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)))
                for kind,image in direct_variants:
                    try:
                        rows=reader.ocr(image,only_rec=True)
                    except Exception as error:
                        missing=getattr(api,'OnlyRecEngineMissingError',None)
                        if missing is not None and isinstance(error,missing):
                            reads=[]
                            break
                        raise
                    reads.append(RuleTextRead(frame_id,kind,tuple(RuleTextLine(row.text,row.confidence) for row in rows)))
                if reads:
                    evidence=replace(evidence,remaining_direct=LogRemainingDirect(frame_id,remaining_rect,tuple(reads),clipped))
                    observed=confirm_visible_log_turn(evidence)
        if readable and observed.remaining_turns is None:
            # Retain detector output; recognize the entire unchanged crop
            # at both sizes and verify both complete white-glyph masks.
            original=crop(remaining_rect)
            masks=(neutral_white_glyphs(original),
                (cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)>245).astype('uint8')*255)
            clipped=right_edge_digit(masks[0])
            if not clipped:
                direct=evidence.remaining_direct
                if direct is None:
                    direct=LogRemainingDirect(frame_id,remaining_rect,_recognize_whole_crop(original,frame_id),False)
                enlarged=cv2.resize(original,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
                scaled=LogRemainingDirect(frame_id,remaining_rect,_recognize_whole_crop(enlarged,frame_id),False,scale=2)
                components=[]
                for mask in masks:
                    _,_,stats,_=cv2.connectedComponentsWithStats(mask)
                    components.append(tuple((int(row[0]),int(row[1]),int(row[2]),int(row[3]),int(row[4])) for row in stats[1:]))
                glyphs=LogRemainingGlyphs(frame_id,remaining_rect,components[0],components[1],
                    cv2.countNonZero(masks[0]),cv2.countNonZero(masks[1]))
                evidence=replace(evidence,remaining_direct=direct,remaining_direct_scaled=scaled,remaining_glyphs=glyphs)
                observed=confirm_visible_log_turn(evidence)
        result.append(observed)
    return tuple(result)
