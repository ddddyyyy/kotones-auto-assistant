"""在当前完整规则正文内匹配整个固定短语；不包含费用数字。"""
import cv2
from typing import Literal
from cv2.typing import MatLike
from kotonebot.backend.ocr import jp
from kaa.tasks import R
from .recurring_hp_cost import RecurringHpTiming
from .rules import RuleTextRead,RuleTextLine


def read_recurring_hp_timing(body: MatLike,frame_id: str,offset: tuple[int,int]) -> RecurringHpTiming | None:
    template=R.IdolRoad.RecurringHpTurnStartPhrase.template.pixels
    h,w=template.shape[:2]
    if not frame_id or body.shape[0]<h or body.shape[1]<w:
        return None
    scores=cv2.matchTemplate(cv2.cvtColor(body,cv2.COLOR_BGR2GRAY),cv2.cvtColor(template,cv2.COLOR_BGR2GRAY),cv2.TM_CCOEFF_NORMED)
    _,score,_,(x,y)=cv2.minMaxLoc(scores)
    if not .97<=score<=1.000001:
        return None
    others=scores.copy()
    others[max(0,y-h+1):y+h,max(0,x-w+1):x+w]=-1
    if cv2.minMaxLoc(others)[1]>=.97:
        return None
    ox,oy=offset
    phrase=body[y:y+h,x:x+w]
    variants: tuple[tuple[Literal['original','grayscale'],MatLike],...]=(
        ('original',phrase),('grayscale',cv2.cvtColor(phrase,cv2.COLOR_BGR2GRAY)))
    reads=tuple(RuleTextRead(frame_id,kind,tuple(RuleTextLine(row.text,row.confidence) for row in jp().ocr(image)))
        for kind,image in variants)
    return RecurringHpTiming(frame_id,(ox,oy,body.shape[1],body.shape[0]),(ox+x,oy+y,w,h),
        min(1.,float(score)),True,(w,h),phrase_reads=reads)
