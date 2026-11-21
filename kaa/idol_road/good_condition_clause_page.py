"""Same-crop fixed-word verification; never repairs a numeric OCR value."""
import cv2
from typing import Literal
from cv2.typing import MatLike
from kotonebot.backend.ocr import jp
from kaa.tasks import R
from .rules import RuleTextRead,RuleTextLine
from .good_condition_rule import GoodConditionLexicalEvidence,read_good_condition_requirement


def read_good_condition_lexical(body: MatLike,frame_id: str,offset: tuple[int,int],turn: int | None,
                                reads: tuple[RuleTextRead,...]) -> GoodConditionLexicalEvidence | None:
    if not any(read.lines and read.lines[0].text.startswith('好調が') for read in reads):return None
    ink=lambda p:(cv2.cvtColor(p,cv2.COLOR_BGR2GRAY)<110).astype('uint8')*255
    candidates=[]
    for resource in (R.IdolRoad.GoodConditionClauseCompact,R.IdolRoad.GoodConditionClauseRegular,
                     R.IdolRoad.GoodConditionClauseRecovery):
        target=ink(resource.template.pixels);h,w=target.shape
        if body.shape[0]<h or body.shape[1]<w:continue
        scores=cv2.matchTemplate(ink(body),target,cv2.TM_CCOEFF_NORMED)
        _,score,_,(x,y)=cv2.minMaxLoc(scores)
        if not .97<=score<=1.000001 or y>h:continue
        others=scores.copy();others[max(0,y-h+1):y+h,max(0,x-w+1):x+w]=-1
        if cv2.minMaxLoc(others)[1]<.97:candidates.append((score,x,y,w,h))
    if not candidates:return None
    score,x,y,w,h=max(candidates);ox,oy=offset
    cue=GoodConditionLexicalEvidence(frame_id,(ox,oy,body.shape[1],body.shape[0]),
        (ox+x,oy+y,w,h),min(1.,float(score)),True)
    if read_good_condition_requirement(turn,reads,cue).requirement is not None:return cue
    large=cv2.resize(body,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    variants: tuple[tuple[Literal['original','grayscale'],MatLike],...]=(
        ('original',large),('grayscale',cv2.cvtColor(large,cv2.COLOR_BGR2GRAY)))
    scaled=tuple(RuleTextRead(frame_id,kind,tuple(RuleTextLine(row.text,row.confidence)
        for row in jp().ocr(image))) for kind,image in variants)
    return GoodConditionLexicalEvidence(cue.frame_id,cue.body_rect,cue.suffix_rect,cue.suffix_score,True,scaled)
