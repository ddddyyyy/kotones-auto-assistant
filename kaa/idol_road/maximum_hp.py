"""用户确认的本场恒定最大体力参数；当前HP和回合仍须当前帧观察。"""
from __future__ import annotations
from dataclasses import dataclass,replace
from typing import Literal,TYPE_CHECKING
from .hp_ratio import HpRatioObservation,confirm_hp_ratio
if TYPE_CHECKING:
    from .battle_observation import BattleObservationSession,BattleFrameObservation
    from .battle_inputs import BattleInputState


@dataclass(frozen=True)
class MaximumHpCalibration:
    session: BattleObservationSession
    request_id: int
    input_sequence: int
    source_frame_id: str
    ratio: HpRatioObservation
    max_hp: int
    source: Literal['user_confirmed_idol_road_battle_constant'] = 'user_confirmed_idol_road_battle_constant'


@dataclass(frozen=True)
class BoundMaximumHp:
    calibration: MaximumHpCalibration
    current_frame_id: str
    input_sequence: int


def confirmed_maximum_hp(frame: BattleFrameObservation) -> int | None:
    bound=frame.maximum_hp_parameter
    if bound is None:
        return None
    cal=bound.calibration
    values,state=frame.values,frame.observation
    checked=confirm_hp_ratio(cal.ratio.frame_id,cal.ratio.label_rect,cal.ratio.label_score,cal.ratio.reads)
    if (cal.session!=frame.session or frame.session.identity_source!='confirmed_formation_start_request'
        or cal.source!='user_confirmed_idol_road_battle_constant' or type(cal.request_id) is not int or cal.request_id<1
        or type(cal.input_sequence) is not int or cal.input_sequence<0 or type(bound.input_sequence) is not int
        or bound.input_sequence<cal.input_sequence or not cal.source_frame_id or cal.ratio.frame_id!=cal.source_frame_id
        or checked!=cal.ratio or checked.status!='resolved' or checked.max_hp!=cal.max_hp
        or type(cal.max_hp) is not int or cal.max_hp<=0
        or bound.current_frame_id!=values.frame_id or not values.frame_id or state.frame_id!=values.frame_id
        or values.detail_mode!='complete' or values.resource_state!='current' or not values.hud_confirmed
        or state.source!='confirmed_current_battle_observation' or {'identity','hp','max_hp'}&set(state.conflicted_fields)
        or (state.battle_id,state.character_id,state.stage)!=(frame.session.battle_id,frame.session.character_id,frame.session.stage)
        or type(state.hp) is not int or not 0<=state.hp<=cal.max_hp
        or (state.max_hp is not None and state.max_hp!=cal.max_hp)):
        return None
    return cal.max_hp


class MaximumHpTracker:
    def __init__(self,session: BattleObservationSession):
        if session.identity_source!='confirmed_formation_start_request' or not session.battle_id:
            raise ValueError('最大体力参数缺少本程序开战来源')
        self.session=session
        self.calibration: MaximumHpCalibration | None=None
        self.request_sequence: int | None=None
        self.request_id=0
        self.opening_frame: str | None=None
        self.last_sequence=0
        self.invalid_reason: str | None=None

    def cancel_request(self) -> None:
        self.request_sequence=None
        self.opening_frame=None

    def invalidate(self,reason: str) -> None:
        self.invalid_reason=reason
        self.calibration=None
        self.cancel_request()

    def request(self,frame: BattleFrameObservation,inputs: BattleInputState) -> None:
        if (self.invalid_reason is not None or frame.session!=self.session or inputs.battle_id!=self.session.battle_id
            or inputs.pending is not None or inputs.expired or type(inputs.sequence) is not int
            or inputs.sequence<self.last_sequence or frame.values.detail_mode!='complete'
            or frame.values.resource_state!='current' or frame.observation.source!='confirmed_current_battle_observation'
            or not frame.values.hud_confirmed or not frame.values.frame_id or frame.observation.frame_id!=frame.values.frame_id
            or (frame.observation.battle_id,frame.observation.character_id,frame.observation.stage)
               !=(self.session.battle_id,self.session.character_id,self.session.stage)
            or {'identity','hp','max_hp'}&set(frame.observation.conflicted_fields)):
            raise ValueError('当前会话／输入未确认，不能请求最大体力采集')
        self.request_id+=1
        self.request_sequence=inputs.sequence
        self.opening_frame=frame.values.frame_id

    def observe(self,frame: BattleFrameObservation,inputs: BattleInputState) -> None:
        if self.invalid_reason is not None:
            return
        if (frame.session!=self.session or inputs.battle_id!=self.session.battle_id
            or type(inputs.sequence) is not int or inputs.sequence<self.last_sequence):
            self.invalidate('session_or_input_sequence_changed')
            return
        self.last_sequence=inputs.sequence
        if self.request_sequence is not None and (inputs.sequence!=self.request_sequence or inputs.pending is not None or inputs.expired):
            self.cancel_request()
        values,state=frame.values,frame.observation
        ratio=values.hp_ratio
        if {'hp','max_hp'}&set(state.conflicted_fields):
            self.invalidate('current_hp_conflict')
            return
        if self.calibration is not None and type(state.hp) is int and state.hp>self.calibration.max_hp:
            self.invalidate('current_hp_exceeds_calibrated_maximum')
            return
        if ratio is None:
            return
        checked=confirm_hp_ratio(ratio.frame_id,ratio.label_rect,ratio.label_score,ratio.reads)
        if checked!=ratio or ratio.frame_id!=values.frame_id or checked.status=='conflict':
            self.invalidate('ratio_evidence_conflict')
            return
        if checked.status!='resolved':
            return
        if self.calibration is not None and checked.max_hp!=self.calibration.max_hp:
            self.invalidate('maximum_changed_against_mode_contract')
            return
        if (self.request_sequence is None or values.frame_id==self.opening_frame
            or values.detail_mode!='complete' or values.resource_state!='current'
            or state.source!='confirmed_current_battle_observation' or state.frame_id!=values.frame_id
            or (state.battle_id,state.character_id,state.stage)!=(self.session.battle_id,self.session.character_id,self.session.stage)
            or checked.hp!=state.hp or checked.max_hp!=state.max_hp):
            return
        assert checked.max_hp is not None
        self.calibration=MaximumHpCalibration(self.session,self.request_id,inputs.sequence,values.frame_id,checked,checked.max_hp)
        self.cancel_request()

    def bind(self,frame: BattleFrameObservation,inputs: BattleInputState) -> BattleFrameObservation:
        raw=replace(frame,maximum_hp_parameter=None)
        if (self.invalid_reason is not None or self.calibration is None or frame.session!=self.session
            or inputs.battle_id!=self.session.battle_id or type(inputs.sequence) is not int
            or inputs.sequence<self.last_sequence or inputs.sequence<self.calibration.input_sequence or inputs.expired):
            return raw
        result=replace(raw,maximum_hp_parameter=BoundMaximumHp(self.calibration,frame.values.frame_id,inputs.sequence))
        return result if confirmed_maximum_hp(result) is not None else raw
