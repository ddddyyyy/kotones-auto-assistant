"""完整固定图样＋独立以上符号＋双表示数字；静态字段区域来自资源。"""
from typing import Literal
import cv2
from cv2.typing import MatLike
from kotonebot.backend.ocr import en
from kaa.tasks import R
from .rules import RuleTextRead,RuleTextLine
from .threshold_turn_end import ThresholdTurnEndEvidence,ThresholdTurnEndAnalysis,interpret_threshold_turn_end


def _digits(crop: MatLike, frame: str, *, scale: int) -> tuple[RuleTextRead,...]:
    variants: tuple[tuple[Literal['original','grayscale'],MatLike],...]=(
        ('original',crop),('grayscale',cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)))
    return tuple(RuleTextRead(frame,kind,tuple(RuleTextLine(row.text,row.confidence)
        for row in en().ocr(cv2.resize(image,None,fx=scale,fy=scale)))) for kind,image in variants)


def read_threshold_turn_end(body: MatLike, frame: str, turn: int | None,
                            original: tuple[RuleTextRead,...]) -> ThresholdTurnEndAnalysis | None:
    road=R.IdolRoad
    patterns=(('concentration_two',road.RuleConcentrationTurnEndTwo,road.BoxConcentrationTurnEndReference,
               road.BoxConcentrationTurnEndThreshold,road.BoxConcentrationTurnEndGain,
               road.RuleConcentrationTurnEndAtLeast,road.BoxConcentrationTurnEndAtLeast),
              ('score_five',road.RuleScoreTurnEndFive,road.BoxScoreTurnEndReference,
               road.BoxScoreTurnEndThreshold,road.BoxScoreTurnEndGain,
               road.RuleScoreTurnEndAtLeast,road.BoxScoreTurnEndAtLeast))
    found=[]
    for kind,template,reference,threshold,gain,comparison,comparison_box in patterns:
        pixels=template.template.pixels
        if body.shape[0]<pixels.shape[0] or body.shape[1]<pixels.shape[1]:
            continue
        _,score,_,(x,y)=cv2.minMaxLoc(cv2.matchTemplate(body,pixels,cv2.TM_CCOEFF_NORMED))
        if score<.97:
            continue
        def field(box):
            xx,yy=x+box.x1-reference.x1,y+box.y1-reference.y1
            return (xx,yy,box.w,box.h),body[yy:yy+box.h,xx:xx+box.w]
        threshold_rect,threshold_crop=field(threshold)
        gain_rect,gain_crop=field(gain)
        _,comparison_crop=field(comparison_box)
        comparison_score=float(cv2.matchTemplate(comparison_crop,comparison.template.pixels,cv2.TM_CCOEFF_NORMED)[0,0])
        evidence=ThresholdTurnEndEvidence(frame,'concentration_two' if kind=='concentration_two' else 'score_five',
            (x,y,pixels.shape[1],pixels.shape[0]),float(score),comparison_score,
            threshold_rect,gain_rect,_digits(threshold_crop,frame,scale=2),_digits(gain_crop,frame,scale=1))
        found.append(interpret_threshold_turn_end(turn,evidence,original))
    return found[0] if len(found)==1 else None
