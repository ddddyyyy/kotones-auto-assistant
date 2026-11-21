"""可见日志行的两种回合数字；不把滚动历史行当作当前回合。"""
from dataclasses import dataclass
import re
from typing import Literal
import unicodedata
from .rules import RuleTextRead
from .log_remaining_glyphs import (LogRemainingGlyphs,mixed_remaining_proposal,matches_single_glyph_row,
                                 matches_bounded_digit_row,detector_agrees_or_is_weak)


@dataclass(frozen=True)
class LogRemainingDirect:
    frame_id: str
    rect: tuple[int,int,int,int]
    reads: tuple[RuleTextRead,...]
    right_edge_clipped: bool
    mode: Literal['whole_crop_only_rec'] = 'whole_crop_only_rec'
    scale: Literal[1,2] = 1


@dataclass(frozen=True)
class LogAbsoluteDirect:
    frame_id: str
    rect: tuple[int,int,int,int]
    reads: tuple[RuleTextRead,...]
    edge_clipped: bool
    mode: Literal['full_prefix_only_rec'] = 'full_prefix_only_rec'


@dataclass(frozen=True)
class LogTurnEvidence:
    frame_id: str
    suffix_rect: tuple[int,int,int,int]
    suffix_score: float
    hand_info_rect: tuple[int,int,int,int]
    hand_info_score: float
    turn_rect: tuple[int,int,int,int]
    turn_reads: tuple[RuleTextRead,...]
    remaining_rect: tuple[int,int,int,int]
    remaining_reads: tuple[RuleTextRead,...]
    suffix_variant: Literal['reference','alternate','scrolled'] = 'reference'
    hand_info_variant: Literal['reference','alternate','scrolled'] = 'reference'
    remaining_scaled_reads: tuple[RuleTextRead,...] = ()
    remaining_direct: LogRemainingDirect | None = None
    absolute_direct: LogAbsoluteDirect | None = None
    remaining_glyphs: LogRemainingGlyphs | None = None
    remaining_direct_scaled: LogRemainingDirect | None = None


@dataclass(frozen=True)
class VisibleLogTurn:
    evidence: LogTurnEvidence
    status: Literal['resolved','partial','evidence_unknown','conflict']
    absolute_turn: int | None = None
    remaining_turns: int | None = None
    scope: Literal['visible_log_row_only'] = 'visible_log_row_only'
    currentness_confirmed: Literal[False] = False


def _weak_split(read: RuleTextRead) -> bool:
    return len(read.lines)==2 and all(0<=line.confidence<.98 for line in read.lines)


def _pair(reads: tuple[RuleTextRead,...], frame_id: str, *, suffix: str, maximum: int,
          weak_split_unknown: bool=False) -> tuple[int | None,bool]:
    if (len(reads)!=2 or {read.representation for read in reads}!={'original','grayscale'}
        or any(read.frame_id!=frame_id for read in reads)):
        return None,False
    values=[]
    for read in reads:
        if len(read.lines)!=1:
            if weak_split_unknown and _weak_split(read):
                continue
            return None,len(read.lines)>1
        line=read.lines[0]
        if not .98<=line.confidence<=1:
            continue
        text=re.sub(r'\s+','',unicodedata.normalize('NFKC',line.text))
        match=re.fullmatch(r'([1-9][0-9]*)'+suffix,text)
        if match is None or not 1<=int(match[1])<=maximum:
            return None,True
        values.append(int(match[1]))
    if len(set(values))>1:
        return None,True
    return (values[0],False) if len(values)==2 else (None,False)


def confirm_visible_log_turn(evidence: LogTurnEvidence) -> VisibleLogTurn:
    unknown=lambda:VisibleLogTurn(evidence,'evidence_unknown')
    rects=(evidence.suffix_rect,evidence.hand_info_rect,evidence.turn_rect,evidence.remaining_rect)
    if (not evidence.frame_id or not .97<=evidence.suffix_score<=1 or not .97<=evidence.hand_info_score<=1
        or any(min(rect)<0 or min(rect[2:])<=0 or rect[0]+rect[2]>720 or rect[1]+rect[3]>1280 for rect in rects)):
        return unknown()
    sx,sy,sw,sh=evidence.suffix_rect
    hx,hy,hw,hh=evidence.hand_info_rect
    tx,ty,tw,th=evidence.turn_rect
    rx,ry,rw,rh=evidence.remaining_rect
    if (rx+rw>tx or tx>sx or tx+tw<sx+sw or hx<sx+sw
        or not (hy<sy+sh and sy<hy+hh and ty<=sy and ty+th>=sy+sh
                and ry<sy+sh and sy<ry+rh)):
        return unknown()
    turn,turn_conflict=_pair(evidence.turn_reads,evidence.frame_id,suffix='ターン目',maximum=99)
    direct_turn=evidence.absolute_direct
    if (direct_turn is not None and direct_turn.frame_id==evidence.frame_id
        and direct_turn.rect==(tx,ty,sx-tx,th) and sx>tx
        and direct_turn.mode=='full_prefix_only_rec' and direct_turn.edge_clipped is False):
        number,conflict=_pair(direct_turn.reads,evidence.frame_id,suffix='',maximum=99)
        turn_conflict=turn_conflict or conflict or (number is not None and turn is not None and number!=turn)
        # The full title remains as evidence and must agree literally. No HUD
        # arithmetic, historical row, confidence reduction or string repair.
        raw_agrees=(number is not None and len(evidence.turn_reads)==2
            and {read.representation for read in evidence.turn_reads}=={'original','grayscale'}
            and all(read.frame_id==evidence.frame_id and len(read.lines)==1
                and 0<=read.lines[0].confidence<=1
                and re.sub(r'\s+','',unicodedata.normalize('NFKC',read.lines[0].text))==f'{number}ターン目'
                for read in evidence.turn_reads))
        if raw_agrees and not turn_conflict:
            turn=number
    if turn_conflict:
        turn=None
    remaining,remaining_conflict=_pair(evidence.remaining_reads,evidence.frame_id,suffix='',maximum=30,weak_split_unknown=True)
    if remaining is None and not remaining_conflict and evidence.remaining_scaled_reads:
        raw=evidence.remaining_reads
        # Same full crop only: a high-confidence raw numeric seed must agree
        # with both enlarged representations. Never trim a low-confidence
        # string or use HUD/history to select the desired number.
        valid=(len(raw)==2 and {read.representation for read in raw}=={'original','grayscale'}
               and all(read.frame_id==evidence.frame_id and len(read.lines)==1 for read in raw))
        seeds=[int(re.sub(r'\s+','',unicodedata.normalize('NFKC',read.lines[0].text)))
               for read in raw if valid and .98<=read.lines[0].confidence<=1
               and re.fullmatch(r'[1-9][0-9]*',re.sub(r'\s+','',unicodedata.normalize('NFKC',read.lines[0].text)))]
        scaled,conflict=_pair(evidence.remaining_scaled_reads,evidence.frame_id,suffix='',maximum=30)
        if conflict or (scaled is not None and seeds and set(seeds)!={scaled}):
            remaining_conflict=True
        elif valid and scaled is not None and seeds and set(seeds)=={scaled}:
            remaining=scaled
    direct=evidence.remaining_direct
    if direct is not None:
        valid=(direct.frame_id==evidence.frame_id and direct.rect==evidence.remaining_rect
               and direct.mode=='whole_crop_only_rec' and direct.scale==1 and direct.right_edge_clipped is False)
        if valid:
            number,conflict=_pair(direct.reads,evidence.frame_id,suffix='',maximum=30)
            scaled,scaled_conflict=_pair(evidence.remaining_scaled_reads,evidence.frame_id,suffix='',maximum=30)
            # Direct original-size recognition is independent strong evidence; the full enlarged
            # crop must also agree. Low-confidence detector text is never sliced into a seed.
            remaining_conflict=remaining_conflict or conflict or scaled_conflict
            if number is not None:
                raw_values=[int(re.sub(r'\s+','',unicodedata.normalize('NFKC',read.lines[0].text)))
                            for read in evidence.remaining_reads if read.frame_id==evidence.frame_id and len(read.lines)==1
                            and .98<=read.lines[0].confidence<=1
                            and re.fullmatch(r'[1-9][0-9]*',re.sub(r'\s+','',unicodedata.normalize('NFKC',read.lines[0].text)))]
                if ((remaining is not None and remaining!=number) or (scaled is not None and scaled!=number)
                    or any(value!=number for value in raw_values)):
                    remaining_conflict=True
                if (not conflict and not scaled_conflict and scaled==number
                    and mixed_remaining_proposal(evidence.remaining_reads,evidence.frame_id)==number
                    and matches_single_glyph_row(evidence.remaining_glyphs,evidence.frame_id,evidence.remaining_rect,number)):
                    # All raw numeric proposals, both full-crop recognitions and
                    # both complete masks agree. Preserve the detector's weak
                    # extra text; it is not deleted or used to choose a number.
                    remaining_conflict=False
                    remaining=number
                raw_complete=(len(evidence.remaining_reads)==2
                    and {read.representation for read in evidence.remaining_reads}=={'original','grayscale'}
                    and all(read.frame_id==evidence.frame_id and (len(read.lines)==1 or _weak_split(read))
                            for read in evidence.remaining_reads))
                if remaining is None and not remaining_conflict and raw_complete and scaled==number:
                    remaining=number
    enlarged=evidence.remaining_direct_scaled
    if (direct is not None and enlarged is not None
        and direct.frame_id==enlarged.frame_id==evidence.frame_id
        and direct.rect==enlarged.rect==evidence.remaining_rect
        and direct.mode==enlarged.mode=='whole_crop_only_rec'
        and direct.scale==1 and enlarged.scale==2
        and direct.right_edge_clipped is False and enlarged.right_edge_clipped is False):
        first,first_conflict=_pair(direct.reads,evidence.frame_id,suffix='',maximum=30)
        second,second_conflict=_pair(enlarged.reads,evidence.frame_id,suffix='',maximum=30)
        if (first is not None and first==second and not first_conflict and not second_conflict
            and detector_agrees_or_is_weak(evidence.remaining_reads,evidence.frame_id,first)
            and detector_agrees_or_is_weak(evidence.remaining_scaled_reads,evidence.frame_id,first)
            and matches_bounded_digit_row(evidence.remaining_glyphs,evidence.frame_id,evidence.remaining_rect,first)):
            if remaining is not None and remaining!=first:
                remaining_conflict=True
            else:
                remaining,remaining_conflict=first,False
    if remaining_conflict:
        remaining=None
    status=('conflict' if turn_conflict or remaining_conflict else 'resolved'
            if turn is not None and remaining is not None else 'partial'
            if turn is not None or remaining is not None else 'evidence_unknown')
    return VisibleLogTurn(evidence,status,turn,remaining)
