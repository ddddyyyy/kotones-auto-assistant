"""当前日志控件／滚动条的只读识别，不恢复历史身份或执行手势。"""
from __future__ import annotations
from hashlib import sha256
from uuid import uuid4
from typing import TYPE_CHECKING
import cv2
import numpy as np
from cv2.typing import MatLike
from kotonebot.backend.image import find
from kaa.tasks import R
from .battle_observation import BattleFrameObservation
from .battle_log import LogEntryEvidence,LogScrollEvidence,LogViewport
from .log_turns_page import read_visible_log_turns
from .log_hand import LogHandCard,LogHandObservation,confirm_log_hand
from .short_log import ShortLogEvidence
if TYPE_CHECKING:
    from kaa.tasks.produce.new.play_cards.page import LessonBattleContext


def read_log_hand(screen: MatLike, frame: BattleFrameObservation, *,
                  context: LessonBattleContext | None=None) -> LogHandObservation | None:
    if screen.shape!=(1280,720,3) or frame.values.resource_state!='current':
        return None
    from kaa.tasks.produce.new.play_cards.page import LessonBattleContext
    from kaa.tasks.produce.shared.cards import skill_card_count
    from .turn_end_page import read_empty_hand
    context=context if context is not None else LessonBattleContext(screen,single_line_titles=True)
    if context.fetch_screen() is not screen:
        raise ValueError('手牌采集Context不属于当前截图')
    hands=context.fetch_hands()
    count=skill_card_count(screen)
    if len(hands)!=count or any(hand is None or hand.card is None or hand._screenshot is not screen for hand in hands):
        return None
    cards=tuple(LogHandCard(hand.card._id,hand.card.upgrade_count,hand.available,hand.rect.xywh,hand.has_growth_marker)
                for hand in hands if hand is not None and hand.card is not None)
    return confirm_log_hand(frame.values.frame_id,count,cards,empty_confirmed=count==0 and read_empty_hand(screen))


def read_log_entry(screen: MatLike, frame: BattleFrameObservation, *, hand: LogHandObservation | None=None) -> LogEntryEvidence | None:
    if screen.shape!=(1280,720,3) or frame.values.resource_state!='current':
        return None
    road=R.IdolRoad
    box=road.BoxBattleLogEntrySearch
    crop=screen[box.y1:box.y2,box.x1:box.x2]
    white=lambda image:(image.min(axis=2)>220).astype('uint8')*255
    pixels=road.BattleLogEntry.template.pixels
    template=white(pixels)
    # 完整图样的抗锯齿亮度会跨过白字阈值；灰度仍匹配全部图样，不能用局部线段代替。
    white_scores=cv2.matchTemplate(white(crop),template,cv2.TM_CCOEFF_NORMED)
    gray_scores=cv2.matchTemplate(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),
                                 cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY),cv2.TM_CCOEFF_NORMED)
    scores=np.maximum(white_scores,gray_scores)
    _,score,_,(x,y)=cv2.minMaxLoc(scores)
    if not .97<=score<=1.000001:
        return None
    # 两种表示都只允许同一个完整入口；不能从多个候选中取最高分点击。
    h,w=template.shape
    others=scores.copy()
    others[max(0,y-h+1):y+h,max(0,x-w+1):x+w]=-1
    if cv2.minMaxLoc(others)[1]>=.97:
        return None
    if hand is None:
        hand=read_log_hand(screen,frame)
    if (hand is None or hand.frame_id!=frame.values.frame_id
        or confirm_log_hand(hand.frame_id,hand.count,hand.cards,empty_confirmed=hand.empty_confirmed)!=hand):
        return None
    return LogEntryEvidence(frame.values.frame_id,(x+box.x1,y+box.y1,template.shape[1],template.shape[0]),
                            min(1.,float(score)),hand.fingerprint,hand.cards)


def _scrollbar(screen: MatLike, frame_id: str) -> LogScrollEvidence | None:
    box=R.IdolRoad.BoxBattleLogScrollTrack
    column=np.median(screen[box.y1:box.y2,box.x1:box.x2],axis=1)
    neutral=column.max(axis=1)-column.min(axis=1)<15
    dark=neutral&(column.min(axis=1)>70)&(column.max(axis=1)<180)
    light=neutral&(column.min(axis=1)>=200)&(column.max(axis=1)<=235)
    edges=np.diff(np.r_[False,dark,False].astype('int8'))
    starts,ends=np.flatnonzero(edges==1),np.flatnonzero(edges==-1)
    runs=[(int(start),int(end)) for start,end in zip(starts,ends) if end-start>=12]
    if len(runs)!=1:
        return None
    start,end=runs[0]
    return LogScrollEvidence(frame_id,box.xywh,(box.x1,box.y1+start,box.w,end-start),
                              float((dark|light).mean()),int(light.sum()))


def _short_list(screen: MatLike, frame_id: str) -> ShortLogEvidence | None:
    road=R.IdolRoad
    content,top,track=road.BoxShortLogContent,road.BoxShortLogTopClear,road.BoxBattleLogScrollTrack
    pixels=screen[content.y1:content.y2,content.x1:content.x2]
    occupied=np.flatnonzero((pixels.min(axis=2)<240).any(axis=1))
    if not len(occupied):
        return None
    last_y=content.y1+int(occupied[-1])
    tail_y=last_y+12
    if content.y2-tail_y<160:
        return None
    white=lambda pixels:float((pixels.min(axis=2)>240).mean())
    text=road.BoxShortLogText
    return ShortLogEvidence(frame_id,road.BoxShortLogHeaderBand.xywh,content.xywh,last_y,
        (content.x1,tail_y,content.w,content.y2-tail_y),top.xywh,track.xywh,
        white(screen[top.y1:top.y2,top.x1:top.x2]),white(screen[track.y1:track.y2,track.x1:track.x2]),
        white(screen[tail_y:content.y2,content.x1:content.x2]),text.xywh,
        sha256(screen[text.y1:text.y2,text.x1:text.x2].tobytes()).hexdigest())


def read_log_viewport(screen: MatLike, *, owned_request: bool=False) -> LogViewport | None:
    if screen.shape!=(1280,720,3):
        return None
    road=R.IdolRoad
    close=find(screen,road.BattleLogClose.template,rect=road.BoxBattleLogClose,threshold=.97)
    if close is None:
        return None
    gutter=road.BoxBattleLogGutter
    if (screen[gutter.y1:gutter.y2,gutter.x1:gutter.x2].min(axis=2)>240).mean()<.97:
        return None
    marker=(find(screen,road.LogHandInfoLabel.template,rect=road.BoxLogAnyHandInfo,threshold=.97)
            or find(screen,road.LogHandInfoLabelAlternate.template,rect=road.BoxLogAnyHandInfo,threshold=.97)
            or find(screen,road.LogHandInfoLabelScrolled.template,rect=road.BoxLogAnyHandInfo,threshold=.97))
    if marker is None and not owned_request:
        return None
    frame_id=uuid4().hex
    body=road.BoxBattleLogBody
    padding=tuple(float((screen[box.y1:box.y2,box.x1:box.x2].min(axis=2)>240).mean())
                  for box in (road.BoxLogTailClearLeft,road.BoxLogTailClearRight))
    scrollbar=_scrollbar(screen,frame_id)
    return LogViewport(frame_id,sha256(screen[body.y1:body.y2,body.x1:body.x2].tobytes()).hexdigest(),close.rect.xywh,
        read_visible_log_turns(screen,frame_id) if marker is not None else (),scrollbar,padding,
        marker is not None,_short_list(screen,frame_id) if marker is not None and scrollbar is None else None)
