"""Complete optional planning fields from the supplied capture, never another screenshot."""
from dataclasses import replace
from cv2.typing import MatLike
from .battle_observation import BattleFrameObservation,bind_battle_frame
from .concentration_page import read_concentration
from .good_condition_page import read_good_condition
from .hp_ratio_page import read_hp_ratio


def read_planning_details(screen: MatLike,frame: BattleFrameObservation) -> BattleFrameObservation | None:
    name=frame.values.frame_id
    values=replace(frame.values,detail_mode='complete',concentration=read_concentration(screen,name),
        hp_ratio=read_hp_ratio(screen,name),good_condition=read_good_condition(screen,name))
    return bind_battle_frame(frame.session,values)
