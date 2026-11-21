"""仅在当前体力提示标签明确时读取完整比值，不操作提示开关。"""
from typing import Literal
import cv2
from cv2.typing import MatLike
from kotonebot.backend.image import find
from kotonebot.backend.ocr import en
from kaa.tasks import R
from .hp_ratio import HpRatioRead,HpRatioObservation,confirm_hp_ratio


def read_hp_ratio(screen: MatLike,frame_id: str) -> HpRatioObservation | None:
    road=R.IdolRoad
    found=find(screen,road.BattleHpTooltipLabel.template,rect=road.BoxBattleHpTooltip,threshold=.97)
    if found is None:
        return None
    reference,number=road.BoxBattleHpTooltipReference,road.BoxBattleHpFraction
    x,y=number.x1+found.rect.x1-reference.x1,number.y1+found.rect.y1-reference.y1
    crop=screen[y:y+number.h,x:x+number.w]
    variants: tuple[tuple[Literal['original','grayscale'],MatLike],...]=(
        ('original',crop),('grayscale',cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)))
    reads=[]
    for kind,image in variants:
        rows=en().ocr(image)
        reads.append(HpRatioRead(frame_id,kind,rows[0].text if len(rows)==1 else None,
                                 rows[0].confidence if len(rows)==1 else None))
    label=screen[found.rect.y1:found.rect.y2,found.rect.x1:found.rect.x2]
    score=float(cv2.matchTemplate(label,road.BattleHpTooltipLabel.template.pixels,cv2.TM_CCOEFF_NORMED)[0,0])
    return confirm_hp_ratio(frame_id,found.rect.xywh,score,tuple(reads))
