"""固定已核验样式的集中阈值／回合末收益声明，不执行事件。"""
from dataclasses import dataclass
import re
from typing import Literal
from .rules import RuleTextRead
from .concentration_rule import compact


@dataclass(frozen=True)
class ThresholdTurnEndEvidence:
    frame_id: str
    pattern: Literal['concentration_two', 'score_five']
    template_rect: tuple[int,int,int,int]
    template_score: float
    comparator_score: float
    threshold_rect: tuple[int,int,int,int]
    gain_rect: tuple[int,int,int,int]
    threshold_reads: tuple[RuleTextRead,...]
    gain_reads: tuple[RuleTextRead,...]


@dataclass(frozen=True)
class ThresholdTurnEndDeclaration:
    start_turn: int
    concentration_at_least: int
    reward_kind: Literal['concentration', 'score']
    amount: int
    executable: Literal[False] = False


@dataclass(frozen=True)
class ThresholdTurnEndAnalysis:
    evidence: ThresholdTurnEndEvidence
    declaration: ThresholdTurnEndDeclaration | None
    status: Literal['confirmed_visible_declaration','evidence_unknown','high_confidence_conflict']


def _text(reads: tuple[RuleTextRead,...], frame: str) -> str | None:
    if (len(reads)!=2 or {read.representation for read in reads}!={'original','grayscale'}
        or any(read.frame_id!=frame or len(read.lines)!=1
               or not .98<=read.lines[0].confidence<=1 for read in reads)):
        return None
    values=[compact(read.lines[0].text) for read in reads]
    return values[0] if values[0]==values[1] else None


def interpret_threshold_turn_end(turn: int | None, evidence: ThresholdTurnEndEvidence,
                                 original: tuple[RuleTextRead,...]) -> ThresholdTurnEndAnalysis:
    unknown=ThresholdTurnEndAnalysis(evidence,None,'evidence_unknown')
    patterns: dict[str,tuple[int,Literal['concentration','score'],int]]={
        'concentration_two':(3,'concentration',2),'score_five':(19,'score',5)}
    expected=patterns.get(evidence.pattern)
    if (expected is None or not evidence.frame_id or type(turn) is not int or turn<2
        or not .97<=evidence.template_score<=1 or not .97<=evidence.comparator_score<=1
        or len(original)!=2 or {read.representation for read in original}!={'original','grayscale'}
        or any(read.frame_id!=evidence.frame_id for read in original)):
        return unknown
    x,y,w,h=evidence.template_rect
    for xx,yy,ww,hh in (evidence.threshold_rect,evidence.gain_rect):
        if (min(x,y,xx,yy)<0 or min(w,h,ww,hh)<=0 or xx<x or yy<y
            or xx+ww>x+w or yy+hh>y+h):
            return unknown
    if evidence.gain_rect[1]<evidence.threshold_rect[1]+evidence.threshold_rect[3]:
        return unknown
    threshold,kind,amount=expected
    if (_text(evidence.threshold_reads,evidence.frame_id)!=str(threshold)
        or _text(evidence.gain_reads,evidence.frame_id)!=f'+{amount}'):
        return unknown
    for read in original:
        for line in read.lines:
            if not .98<=line.confidence<=1:
                continue
            text=compact(line.text)
            condition=re.search(r'集中が([0-9]+)(以上|以下|未満|超)',text)
            gain=re.search(r'(集中|スコア)([+-])([0-9]+)',text)
            if ((condition is not None and (int(condition[1])!=threshold or condition[2]!='以上'))
                or 'ターン開始' in text
                or (gain is not None and (gain[1]!=('集中' if kind=='concentration' else 'スコア')
                    or gain[2]!='+' or int(gain[3])!=amount))):
                return ThresholdTurnEndAnalysis(evidence,None,'high_confidence_conflict')
    return ThresholdTurnEndAnalysis(evidence,
        ThresholdTurnEndDeclaration(turn,threshold,kind,amount),'confirmed_visible_declaration')
