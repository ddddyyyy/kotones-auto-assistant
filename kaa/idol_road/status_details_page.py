"""已标注的状态详情当前视口；不输入、不推断未显示效果或关闭后的剩余值。"""
from uuid import uuid4
from typing import Literal
import cv2
from cv2.typing import MatLike
from kotonebot.backend.image import find
from kotonebot.backend.ocr import jp
from kaa.tasks import R
from .battle_observation import BattleObservationSession
from .rules import RuleTextRead,RuleTextLine
from .status_details import BattleStatusDetails,StatusRowEvidence,StatusSplitValueEvidence,confirm_status_row


def _read_crop(screen: MatLike,frame_id: str,rect: tuple[int,int,int,int]) -> tuple[RuleTextRead,...]:
    x,y,w,h=rect
    crop=screen[y:y+h,x:x+w]
    variants: tuple[tuple[Literal['original','grayscale'],MatLike],...]=(
        ('original',crop),('grayscale',cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)))
    return tuple(RuleTextRead(frame_id,kind,tuple(RuleTextLine(row.text,row.confidence)
                                                for row in jp().ocr(image))) for kind,image in variants)


def find_status_close(screen: MatLike,*,allow_scrolled: bool=False):
    road=R.IdolRoad
    header=any(find(screen,template,rect=road.BoxBattleStatusHeader,threshold=.97) is not None
               for template in (road.BattleStatusHeader.template,road.BattleStatusHeaderInitial.template))
    if not header and not (allow_scrolled and any(
        find(screen,template,rect=road.BoxBattleStatusTitles,threshold=.97) is not None for template in (
            road.BattleStatusLessonBuff.template,road.BattleStatusLessonBuffInitial.template,road.BattleStatusExtraPlay.template))):
        return None
    return find(screen,road.BattleStatusClose.template,rect=road.BoxBattleStatusClose,threshold=.97)


def read_status_details(screen: MatLike,session: BattleObservationSession,*,allow_scrolled: bool=False) -> BattleStatusDetails | None:
    close=find_status_close(screen,allow_scrolled=allow_scrolled)
    if (not session.battle_id or not session.character_id or type(session.stage) is not int or session.stage<1
        or close is None):
        return None
    road=R.IdolRoad
    frame_id=uuid4().hex
    rows=[]
    for kind,templates in (('lesson_buff',(road.BattleStatusLessonBuff.template,road.BattleStatusLessonBuffInitial.template)),
                           ('extra_play',(road.BattleStatusExtraPlay.template,))):
        template=templates[0]
        matched=None
        for template in templates:
            matched=find(screen,template,rect=road.BoxBattleStatusTitles,threshold=.97)
            if matched is not None:
                break
        if matched is None:
            continue
        rect=matched.rect
        reference=road.BoxBattleStatusReference
        reads=[]
        crops=[]
        for box in (road.BoxBattleStatusTitle,road.BoxBattleStatusValue,road.BoxBattleStatusBody):
            x,y=rect.x1+box.x1-reference.x1,rect.y1+box.y1-reference.y1
            crops.append((x,y,box.w,box.h))
            reads.append(_read_crop(screen,frame_id,(x,y,box.w,box.h)))
        split=None
        if kind=='extra_play':
            reference=road.BoxBattleStatusPlayReference
            cells=[]
            cell_reads=[]
            for box in (road.BoxBattleStatusPlayCount,road.BoxBattleStatusPlayTurns):
                cell=(rect.x1+box.x1-reference.x1,rect.y1+box.y1-reference.y1,box.w,box.h)
                cells.append(cell)
                cell_reads.append(_read_crop(screen,frame_id,cell))
            split=StatusSplitValueEvidence(frame_id,cells[0],cells[1],cell_reads[0],cell_reads[1])
        pixels=screen[rect.y1:rect.y2,rect.x1:rect.x2]
        score=float(cv2.matchTemplate(pixels,template.pixels,cv2.TM_CCOEFF_NORMED)[0,0])
        occluded=any(x<close.rect.x2 and x+w>close.rect.x1 and y<close.rect.y2 and y+h>close.rect.y1
                     for x,y,w,h in crops)
        rows.append(confirm_status_row(StatusRowEvidence(frame_id,rect.xywh,score,reads[0],reads[1],reads[2],split,
                                                        occluded=occluded)))
    return BattleStatusDetails(frame_id,session.battle_id,session.character_id,session.stage,tuple(rows),session.identity_source)
