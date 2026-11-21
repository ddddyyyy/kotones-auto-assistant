"""自己分数的同帧证据；不依赖对手分数或预期卡牌收益。"""
from dataclasses import dataclass,replace
import re
from typing import Literal
from .rules import RuleTextRead


@dataclass(frozen=True)
class OwnScoreMaskDigits:
    frame_id: str
    score_rect: tuple[int,int,int,int]
    content_rect: tuple[int,int,int,int]
    reads: tuple[RuleTextRead,...]
    foreground_pixels: int
    edge_ink: bool = False
    mode: Literal['all_foreground_after_frame_lines'] = 'all_foreground_after_frame_lines'


@dataclass(frozen=True)
class OwnScoreMarginDigits:
    frame_id: str
    score_rect: tuple[int,int,int,int]
    top_margin: int
    excluded_components: tuple[tuple[int,int,int,int,int],...]
    digits: OwnScoreMaskDigits
    boundary_ink: bool = False
    mode: Literal['bounded_top_speckles_full_width'] = 'bounded_top_speckles_full_width'


@dataclass(frozen=True)
class OwnScoreObservation:
    frame_id: str
    rect: tuple[int,int,int,int] | None
    reads: tuple[RuleTextRead,...]
    direct_reads: tuple[RuleTextRead,...] = ()
    right_edge_clipped: bool = False
    status: Literal['owner_unknown','number_unknown','conflict','resolved'] = 'number_unknown'
    value: int | None = None
    mask_digits: OwnScoreMaskDigits | None = None
    margin_digits: OwnScoreMarginDigits | None = None


def score_mask_proposal(digits: OwnScoreMaskDigits | None) -> int | None:
    if digits is None or len(digits.reads)!=2 or {r.representation for r in digits.reads}!={'original','grayscale'}:
        return None
    texts=[]
    for read in digits.reads:
        if (read.frame_id!=digits.frame_id or len(read.lines)!=1 or not 0<=read.lines[0].confidence<=1
            or re.fullmatch(r'0|[1-9][0-9]{0,8}',read.lines[0].text) is None):
            return None
        texts.append(read.lines[0].text)
    return int(texts[0]) if texts[0]==texts[1] else None


def _valid_mask(frame_id: str,rect: tuple[int,int,int,int],digits: OwnScoreMaskDigits) -> bool:
    x,y,w,h=rect
    cx,cy,cw,ch=digits.content_rect
    return bool(digits.frame_id==frame_id and digits.score_rect==rect
        and digits.mode=='all_foreground_after_frame_lines' and digits.edge_ink is False
        and min(cw,ch)>0 and x<=cx and y<=cy and cx+cw<=x+w and cy+ch<=y+h
        and ch>=h*.5 and cw>=5 and type(digits.foreground_pixels) is int
        and 40<=digits.foreground_pixels<=cw*ch)


def _pair(reads: tuple[RuleTextRead,...],frame_id: str) -> tuple[int | None,bool]:
    if (len(reads)!=2 or {read.representation for read in reads}!={'original','grayscale'}
        or any(read.frame_id!=frame_id for read in reads)):
        return None,False
    values=[]
    for read in reads:
        if len(read.lines)!=1:
            return None,len(read.lines)>1
        line=read.lines[0]
        if not .98<=line.confidence<=1:
            continue
        if re.fullmatch(r'0|[1-9][0-9]{0,8}',line.text) is None:
            return None,True
        values.append(int(line.text))
    if len(set(values))>1:
        return None,True
    return (values[0] if len(values)==2 else None),False


def confirm_own_score(frame_id: str,rect: tuple[int,int,int,int] | None,reads: tuple[RuleTextRead,...],
                      direct_reads: tuple[RuleTextRead,...]=(),right_edge_clipped: bool=False,
                      mask_digits: OwnScoreMaskDigits | None=None,
                      margin_digits: OwnScoreMarginDigits | None=None) -> OwnScoreObservation:
    result=OwnScoreObservation(frame_id,rect,reads,direct_reads,right_edge_clipped,
        mask_digits=mask_digits,margin_digits=margin_digits)
    if (not frame_id or rect is None or min(rect)<0 or min(rect[2:])<=0
        or rect[0]+rect[2]>720 or rect[1]+rect[3]>1280):
        return replace(result,status='owner_unknown')
    raw,conflict=_pair(reads,frame_id)
    direct,direct_conflict=_pair(direct_reads,frame_id)
    if conflict or direct_conflict or (raw is not None and direct is not None and raw!=direct):
        return replace(result,status='conflict')
    masked=None
    if mask_digits is not None:
        if (len(reads)!=2 or {read.representation for read in reads}!={'original','grayscale'}
            or any(read.frame_id!=frame_id for read in reads)
            or not _valid_mask(frame_id,rect,mask_digits)):
            return replace(result,status='number_unknown')
        masked,mask_conflict=_pair(mask_digits.reads,frame_id)
        if mask_conflict:
            return replace(result,status='conflict')
    margin_value=None
    if margin_digits is not None:
        x,y,w,h=rect
        margin=margin_digits.top_margin
        excluded=margin_digits.excluded_components
        if (mask_digits is None or margin_digits.frame_id!=frame_id or margin_digits.score_rect!=rect
            or margin_digits.mode!='bounded_top_speckles_full_width' or margin_digits.boundary_ink is not False
            or type(margin) is not int or not 1<=margin<=h//5 or not excluded
            or any(not isinstance(component,tuple) or len(component)!=5 or any(type(v) is not int for v in component)
                for component in excluded)
            or any(min(cx,cy)<0 or not 1<=cw<=2 or not 1<=ch<=2 or not 1<=area<=min(4,cw*ch)
                or cx+cw>w or cy+ch>margin for cx,cy,cw,ch,area in excluded)
            or not 1<=sum(c[4] for c in excluded)<=margin
            or len(set(excluded))!=len(excluded)
            or not _valid_mask(frame_id,(x,y+margin,w,h-margin),margin_digits.digits)
            or margin_digits.digits.foreground_pixels!=mask_digits.foreground_pixels-sum(c[4] for c in excluded)):
            return replace(result,status='number_unknown')
        margin_value,margin_conflict=_pair(margin_digits.digits.reads,frame_id)
        if margin_conflict:
            return replace(result,status='conflict')
        proposal=score_mask_proposal(mask_digits)
        if proposal is None or margin_value!=proposal:
            margin_value=None
    # Every strong reading participates, including an isolated strong reading
    # from a pair whose other side is weak. Supplements never overrule it.
    strong=set()
    for read in (*reads,*direct_reads,*(mask_digits.reads if mask_digits else ()),
                 *(margin_digits.digits.reads if margin_digits else ())):
        if read.frame_id==frame_id and len(read.lines)==1 and .98<=read.lines[0].confidence<=1:
            text=read.lines[0].text
            if re.fullmatch(r'0|[1-9][0-9]{0,8}',text) is None:
                return replace(result,status='conflict')
            strong.add(int(text))
    if len(strong)>1:
        return replace(result,status='conflict')
    # A single strong detector reading must not contradict the strong direct pair.
    if direct is not None:
        for read in reads:
            if (read.frame_id==frame_id and len(read.lines)==1 and .98<=read.lines[0].confidence<=1
                and re.fullmatch(r'0|[1-9][0-9]{0,8}',read.lines[0].text)
                and int(read.lines[0].text)!=direct):
                return replace(result,status='conflict')
    complete=(len(reads)==2 and {read.representation for read in reads}=={'original','grayscale'}
              and all(read.frame_id==frame_id and len(read.lines)==1 for read in reads))
    value=raw if raw is not None else direct if complete and direct is not None else masked
    if value is None:
        value=margin_value
    if right_edge_clipped is not False:
        value=None
    return replace(result,status='resolved' if value is not None else 'number_unknown',value=value)
