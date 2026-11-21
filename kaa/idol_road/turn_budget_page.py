"""校准当前完整可见的时钟＋ターン追加+1片段，仅报告固定预算不兼容。"""
import cv2
from cv2.typing import MatLike
from kaa.tasks import R
from .rule_symbols import RuleSymbolEvidence


def read_turn_addition(body: MatLike, frame_id: str, offset: tuple[int,int]) -> RuleSymbolEvidence | None:
    pixels=R.IdolRoad.RuleTurnAdditionMarker.template.pixels
    if body.shape[0]<pixels.shape[0] or body.shape[1]<pixels.shape[1]:
        return None
    _,score,_,(x,y)=cv2.minMaxLoc(cv2.matchTemplate(body,pixels,cv2.TM_CCOEFF_NORMED))
    if not .97<=score<=1:
        return None
    return RuleSymbolEvidence(frame_id,'turn_addition',
        (x+offset[0],y+offset[1],pixels.shape[1],pixels.shape[0]),float(score))
