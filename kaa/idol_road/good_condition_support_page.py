"""仅校准当前卡片的蓝色持续应援标志，不据此推导收益数值。"""
import cv2
from cv2.typing import MatLike
from kaa.tasks import R
from .rule_symbols import RuleSymbolEvidence,RuleSymbolKind


def read_good_condition_support(card: MatLike, frame_id: str,
                                 offset: tuple[int,int]) -> RuleSymbolEvidence | None:
    candidates=[]
    for resource in (R.IdolRoad.GoodConditionPositiveSupport,R.IdolRoad.GoodConditionPositiveSupportCurrent,
                     R.IdolRoad.GoodConditionRecoverySupport):
        kind: RuleSymbolKind = ('positive_debuff_recovery' if resource is R.IdolRoad.GoodConditionRecoverySupport
            else 'positive_turn_end_support')
        pixels=resource.template.pixels
        if card.shape[0]<pixels.shape[0] or card.shape[1]<pixels.shape[1]:continue
        scores=cv2.matchTemplate(card,pixels,cv2.TM_CCOEFF_NORMED)
        _,score,_,(x,y)=cv2.minMaxLoc(scores)
        if not .97<=score<=1.000001:continue
        h,w=pixels.shape[:2];others=scores.copy()
        others[max(0,y-h+1):y+h,max(0,x-w+1):x+w]=-1
        if cv2.minMaxLoc(others)[1]>=.97:continue
        candidates.append(RuleSymbolEvidence(frame_id,kind,
            (x+offset[0],y+offset[1],w,h),min(1.,float(score))))
    return max(candidates,key=lambda candidate:candidate.score) if candidates else None
