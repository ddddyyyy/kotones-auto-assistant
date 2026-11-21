"""已采集的当前帧战斗 HUD 适配；不截图、不执行动作或导入 Planner。"""
import re
from typing import Callable
from uuid import uuid4
import cv2
from cv2.typing import MatLike
from kotonebot.backend.image import find
from kotonebot.backend.ocr import en
from kaa.tasks import R
from .battle_observation import BattleFrameObservation, BattleFrameValues, BattleObservationSession, bind_battle_frame
from .concentration_page import read_concentration
from .hp_ratio_page import read_hp_ratio
from .good_condition_page import read_good_condition


def read_formation_total_turns(screen: MatLike) -> int | None:
    box = R.IdolRoad.BoxFormationTotalTurns
    crop = screen[box.y1:box.y2, box.x1:box.x2]
    values = []
    for image in (crop, cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)):
        rows = en().ocr(image)
        if len(rows) != 1 or rows[0].confidence < .98 or re.fullmatch(r'[1-9][0-9]*', rows[0].text) is None:
            return None
        values.append(int(rows[0].text))
    return values[0] if values[0] == values[1] and 1 <= values[0] <= 30 else None


def read_battle_frame(screen: MatLike, session: BattleObservationSession, *,read_details: bool=True,
                      details_when: Callable[[BattleFrameObservation],bool] | None=None,
                      feedback_only: bool=False,include_queued_turns: bool=False) -> BattleFrameObservation | None:
    if feedback_only and (read_details or details_when is not None):
        raise ValueError('反馈专用观察不能同时请求完整状态')
    if screen.shape[:2] != (1280, 720) or len(screen.shape) != 3 or screen.shape[2] != 3:
        return None
    road = R.IdolRoad
    box = road.BoxBattleRemainingLabel
    title = screen[box.y1:box.y2, box.x1:box.x2]
    template = road.BattleRemainingLabel.template.pixels
    # 动态背景不参与标题确认；匹配同一白字掩膜，阈值仍为 .97。
    title_mask = (title.min(axis=2) > 220).astype('uint8') * 255
    template_mask = (template.min(axis=2) > 220).astype('uint8') * 255
    score = cv2.minMaxLoc(cv2.matchTemplate(title_mask, template_mask, cv2.TM_CCOEFF_NORMED))[1]
    if score<.97:
        return None
    if find(screen, road.BattleSkipLabel.template, rect=road.BoxBattleSkipLabel, threshold=.97) is None:
        from .turn_end_page import read_enabled_skip
        # The native SKIP background pulses even with cards remaining. Its
        # enabled white control and two exact high-confidence text readings
        # confirm the same HUD anchor without depending on that animation.
        if not read_enabled_skip(screen):
            return None
    from kaa.tasks.produce.new.play_cards.page import LessonBattleContext
    frame_id=uuid4().hex
    detail_mode='feedback_only' if feedback_only else 'complete' if read_details else 'core_only'
    selection_box=road.BoxBattleSelectionLabel
    selected=screen[selection_box.y1:selection_box.y2,selection_box.x1:selection_box.x2]
    selected_template=road.BattleSelectedLabel.template.pixels
    selected_mask=(selected.min(axis=2)>220).astype('uint8')*255
    selected_template_mask=(selected_template.min(axis=2)>220).astype('uint8')*255
    selected_score=cv2.minMaxLoc(cv2.matchTemplate(selected_mask,selected_template_mask,cv2.TM_CCOEFF_NORMED))[1]
    if selected_score>=.97:
        return bind_battle_frame(session,BattleFrameValues(frame_id,True,None,None,resource_state='selection_preview',detail_mode=detail_mode))
    panel=road.BoxBattleCardInfoPanel
    panel_pixels=screen[panel.y1:panel.y2,panel.x1:panel.x2]
    if (panel_pixels.min(axis=2)>240).mean()>.75:
        return bind_battle_frame(session,BattleFrameValues(frame_id,True,None,None,resource_state='unconfirmed_overlay',detail_mode=detail_mode))
    context = LessonBattleContext(screen)
    remaining,hp=context.fetch_exam_remaining_turns(),context.fetch_exam_hp()
    queued=None
    if include_queued_turns:
        from .queued_turns_page import read_queued_turns
        queued=read_queued_turns(screen,frame_id,remaining)
        remaining=queued.remaining if queued is not None else None
    if hp is None:
        from .hp_page import read_masked_hp
        hp=read_masked_hp(screen)
    core=bind_battle_frame(session,BattleFrameValues(frame_id,True,remaining,hp,
        detail_mode='feedback_only' if feedback_only else 'core_only',queued_turns=queued))
    if not read_details or core is None or (details_when is not None and not details_when(core)):
        return core
    values = BattleFrameValues(frame_id,True,remaining,hp,
                                concentration=read_concentration(screen,frame_id),hp_ratio=read_hp_ratio(screen,frame_id),
                                good_condition=read_good_condition(screen,frame_id),queued_turns=queued)
    return bind_battle_frame(session, values)
