"""720×1280 偶像之路页面适配；只在当前截图确认的控件上执行动作。"""

from dataclasses import dataclass, field,replace
from hashlib import sha256
import re
import unicodedata
from typing import Literal,TYPE_CHECKING
if TYPE_CHECKING:
    from .extra_play import ExtraPlayObservation
    from .hud_initialization import InitialHudRound
    from .hud_round import HudRoundResolution

import cv2
import numpy as np
from cv2.typing import MatLike
from kotonebot.backend.image import find
from kotonebot.backend.ocr import en, jp
from kotonebot.primitives import Rect

from kaa.db.tower import tower_characters
from kaa.tasks import R

from .models import Action, Decision, Observation, Outcome, Scene, Stage
from .stars import read_stars, resize_star
from .objectives import StarRequirement, read_rank_requirement
from .objective_page import read_requirement_text
from .dialog_binding import FormationDialogBinding
from .item_page import read_item_viewport
from .memory_picker_page import read_sub_memory_selector
from .memory_replacement import MemoryCommand, selected_candidate
from .memory_picker import sub_alternatives
from .rule_page import read_rule_viewport, read_support_stage
from .battle_observation import BattleFrameObservation, BattleObservationSession, create_battle_session, status_inspection_target,current_absolute_turn
from .battle_page import read_battle_frame, read_formation_total_turns
from .status_details_page import find_status_close,read_status_details
from .status_scan import status_viewport_key
from .formation_page import read_formation_slots,read_recommended_effect
from .rules import StageRules
from .battle_log import BattleLogBinding, LogEntryEvidence, LogViewport
from .battle_log_page import read_log_entry, read_log_viewport,read_log_hand
from .log_hand import LogHandObservation
from .battle_inputs import BattleInputs
from .round_anchor import InitialRoundTracker
from .maximum_hp import MaximumHpTracker,confirmed_maximum_hp
from .skip_probe import PendingSkipPermit,pending_skip_permit
from .card_probe import PendingCardPermit,pending_card_permit,assess_card_log
from .hand_context import CapturedHandContext
from .battle_profile import SenseBattleProfile,sense_profile_from_start


def _compact(text: str) -> str:
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))


def _match(screen: MatLike, template, *, rect: Rect | None = None):
    return find(screen, template.template, rect=rect, threshold=0.95)


def _ranking_ending_crest(screen: MatLike) -> bool:
    road=R.IdolRoad;box=road.BoxRankingEndingCrestSearch
    white=lambda pixels:(pixels.min(axis=2)>230).astype('uint8')*255
    template=white(road.RankingEndingCrest.template.pixels);h,w=template.shape
    scores=cv2.matchTemplate(white(screen[box.y1:box.y2,box.x1:box.x2]),template,cv2.TM_CCOEFF_NORMED)
    _,score,_,(x,y)=cv2.minMaxLoc(scores)
    if .97<=score<=1.000001:
        scores[max(0,y-h+1):y+h,max(0,x-w+1):x+w]=-1
        if cv2.minMaxLoc(scores)[1]<.97:return True
    # Ranks have different digits and decorative star counts. Require both
    # complete side laurels at their native relative offset instead of
    # treating one rank's stars as every ending's template.
    search=road.BoxEndingLaurelSearch
    region=white(screen[search.y1:search.y2,search.x1:search.x2])
    def part(resource):
        target=white(resource.template.pixels);hh,ww=target.shape
        mask=cv2.dilate((target>0).astype('uint8'),np.ones((3,3),dtype='uint8'))
        matches=cv2.matchTemplate(region,target,cv2.TM_CCORR_NORMED,mask=mask)
        matches=np.nan_to_num(matches,nan=-1.,posinf=-1.,neginf=-1.)
        _,confidence,_,(xx,yy)=cv2.minMaxLoc(matches)
        if not .97<=confidence<=1.000001:return None
        observed=region[yy:yy+hh,xx:xx+ww]
        if np.count_nonzero((observed>0)&(target>0))<.95*np.count_nonzero(target):return None
        matches[max(0,yy-hh+1):yy+hh,max(0,xx-ww+1):xx+ww]=-1
        return (xx,yy) if cv2.minMaxLoc(matches)[1]<.97 else None
    left=part(road.EndingLaurelLeft);right=part(road.EndingLaurelRight)
    offset=road.PointEndingLaurelOffset
    return bool(left is not None and right is not None
        and abs(right[0]-left[0]-offset.x)<=2 and abs(right[1]-left[1]-offset.y)<=2)


def _number(screen: MatLike, rect: Rect) -> int | None:
    """完整原色与完整白字掩膜须高置信一致，不截取子串或补历史值。

    此规则只读页面关卡／编成战力，不提供战斗支付或确定收益证据。
    """
    crop = screen[rect.y1:rect.y2, rect.x1:rect.x2]
    mask = cv2.inRange(crop, np.array([210, 210, 210]), np.array([255, 255, 255]))
    values = []
    for image in (crop, mask):
        results = en().ocr(image)
        if len(results) != 1 or results[0].confidence < 0.98:
            return None
        text = _compact(results[0].text)
        if re.fullmatch(r'[1-9][0-9]*', text) is None:
            return None
        values.append(int(text))
    return values[0] if values[0] == values[1] else None


@dataclass
class PageSnapshot:
    observation: Observation
    targets: dict[Action, Rect] = field(default_factory=dict)
    battle_frame: BattleFrameObservation | None = None
    log_entry: LogEntryEvidence | None = None
    log_hand: LogHandObservation | None = None
    hand_context: CapturedHandContext | None = field(default=None,repr=False,compare=False)
    capture_digest: str | None = None
    log_view: LogViewport | None = None


class IdolRoadPage:
    _feedback_mode: Literal['log','hud']='log'
    _hud_capture_frame: BattleFrameObservation | None=None
    _hud_capture_hand: LogHandObservation | None=None
    _hud_authorization: tuple[BattleFrameObservation,int] | None=None

    def __init__(self,*,feedback_only_returns: bool=False) -> None:
        if type(feedback_only_returns) is not bool:
            raise ValueError('反馈专用观察开关必须是布尔值')
        self._feedback_only_returns=feedback_only_returns
        self._feedback_mode='log'
        self._hud_capture_frame=None
        self._hud_capture_hand=None
        self._hud_authorization=None
        self._names = {item.id: item.name for item in tower_characters()}
        self._battle_identity: tuple[str, int] | None = None
        self._battle_session: BattleObservationSession | None = None
        self._battle_sense_profile: SenseBattleProfile | None = None
        self._log_binding=BattleLogBinding()
        self._log_span_capture: tuple[str,str,MatLike] | None=None
        self._battle_inputs: BattleInputs | None=None
        self._round_tracker: InitialRoundTracker | None=None
        self._maximum_hp: MaximumHpTracker | None=None
        self._input_probe_requests: set[tuple[str,int]]=set()
        self._status_request_session: BattleObservationSession | None = None
        self._status_has_scrolled = False
        self._scroll_attempts: dict[str, int] = {}
        self._conditions_identity: tuple[str, int] | None = None
        self._conditions_closing = False
        self._items_dialog = FormationDialogBinding()
        self._support_dialog = FormationDialogBinding()
        self._auto_dialog = FormationDialogBinding()
        self._auto_result_expected=False
        self._memory_command: MemoryCommand | None=None
        self._memory_identity: tuple[str,int] | None=None
        self._memory_receipt: str | None=None

    @property
    def battle_session(self) -> BattleObservationSession | None:
        return self._battle_session

    @property
    def feedback_mode(self) -> Literal['log','hud']:
        return self._feedback_mode

    @property
    def battle_sense_profile(self) -> SenseBattleProfile | None:
        profile=getattr(self,'_battle_sense_profile',None)
        return profile if profile is not None and profile.session==self._battle_session else None

    @property
    def battle_inputs(self) -> BattleInputs | None:
        return self._battle_inputs if self._battle_inputs is not None and self._battle_inputs.session==self._battle_session else None

    @property
    def battle_rounds(self) -> InitialRoundTracker | None:
        tracker=getattr(self,'_round_tracker',None)
        return tracker if tracker is not None and tracker.session==self._battle_session else None

    @property
    def battle_maximum_hp(self) -> MaximumHpTracker | None:
        tracker=getattr(self,'_maximum_hp',None)
        return tracker if tracker is not None and tracker.session==self._battle_session else None

    def _sync_log_inputs(self) -> PendingSkipPermit | PendingCardPermit | None:
        inputs=self.battle_inputs
        if inputs is None:
            return None
        state=inputs.snapshot()
        skip=pending_skip_permit(inputs.session,state,now=inputs.clock())
        card=pending_card_permit(inputs.session,state,now=inputs.clock())
        self._log_binding.update_inputs(inputs.sequence,inputs.pending is not None,skip_probe=skip,card_probe=card)
        return skip or card

    def begin_hud_feedback(self,session: BattleObservationSession) -> None:
        """Choose HUD feedback only for this fresh owned START, before inputs."""
        inputs=self.battle_inputs
        if (session!=self._battle_session or inputs is None or inputs.session!=session
            or inputs.sequence!=0 or inputs.pending is not None or self._log_binding.pending):
            raise ValueError('HUD反馈只能在新的自有战斗初始化时启用')
        self._feedback_mode='hud'
        self._hud_capture_frame=None
        self._hud_capture_hand=None
        self._hud_authorization=None

    def accept_initial_hud(self,producer: 'InitialHudRound') -> BattleFrameObservation:
        """Register only the producer's exact last captured initial frame."""
        from .hud_initialization import InitialHudRound
        inputs=self.battle_inputs
        frame=producer.bound if isinstance(producer,InitialHudRound) else None
        maximum=self.battle_maximum_hp
        if (self._feedback_mode!='hud' or inputs is None or frame is None or maximum is None
            or frame.session!=self._battle_session or producer.session!=frame.session
            or inputs.sequence!=0 or inputs.pending is not None or inputs.clock()>=producer.deadline
            or frame.observation.turn_source!='owned_initial_hud' or current_absolute_turn(frame) is None
            or frame.values.hp_ratio is not None or frame.maximum_hp_parameter is None
            or frame.maximum_hp_parameter.calibration!=maximum.calibration):
            raise ValueError('HUD首回合缺少当前自有初始化来源')
        raw=replace(frame,absolute_turn=None,observation=replace(frame.observation,turn=None,turn_source=None))
        if raw!=self._hud_capture_frame or producer.bound_hand is None or producer.bound_hand!=self._hud_capture_hand:
            raise ValueError('HUD首回合不是最近采集的完整画面')
        self._hud_authorization=(frame,0)
        return frame

    def begin_battle_input(self, frame: BattleFrameObservation, kind: Literal['card','skip'], card_id: str | None=None,*,
                           extra_play: 'ExtraPlayObservation | None'=None) -> None:
        inputs=self.battle_inputs
        if inputs is None or self._log_binding.pending:
            raise ValueError('缺少本场输入记录或日志尚未关闭，不能输入')
        rounds=self.battle_rounds
        if frame.session.turn_budget_mode=='variable_budget':
            if self._feedback_mode=='hud':
                if (self._hud_authorization!=(frame,inputs.sequence) or self._hud_capture_frame is None
                    or self._hud_capture_frame.values.frame_id!=frame.values.frame_id or current_absolute_turn(frame) is None
                    or self._hud_capture_hand is None):
                    raise ValueError('当前HUD输入没有同帧回合授权')
            elif rounds is None or rounds.state.anchor is None or current_absolute_turn(rounds.bind(frame,inputs.snapshot())) is None:
                raise ValueError('当前返回帧锚点已失效，不能使用旧帧输入')
        inputs.begin(kind,frame,card_id,extra_play=extra_play,
            origin_hand=self._hud_capture_hand if self._feedback_mode=='hud' else None)
        self._hud_authorization=None
        maximum=self.battle_maximum_hp
        if maximum is not None:
            maximum.cancel_request()
        if rounds is not None:
            rounds.invalidate('battle_input_requested')
        # 每个请求在实际设备操作之前撤销稳定入口和旧日志候选。
        self._log_binding.reset()
        self._log_binding.update_inputs(inputs.sequence,True)

    def accept_hud_return(self,source: PageSnapshot,frame: BattleFrameObservation,
                           resolution: 'HudRoundResolution') -> None:
        """Grant one next input only after same-capture completion and settlement."""
        inputs=self.battle_inputs
        if (self._feedback_mode!='hud' or inputs is None or self._log_binding.pending
            or source.battle_frame!=self._hud_capture_frame or source.log_hand!=self._hud_capture_hand
            or source.battle_frame!=resolution.evidence.frame or source.log_hand!=resolution.evidence.hand
            or frame.session!=self._battle_session or frame.values.frame_id!=resolution.evidence.frame.values.frame_id
            or not inputs.complete_hud_round(frame,resolution)):
            raise ValueError('HUD返回缺少当前捕获或已结算回合来源')
        self._hud_capture_frame=frame
        self._hud_authorization=(frame,inputs.sequence)

    def _requirements(self, screen: MatLike) -> tuple[StarRequirement, ...]:
        road = R.IdolRoad
        filled = (road.ConditionsFilledStar1.template.pixels,
                  road.ConditionsFilledStar2.template.pixels,
                  road.ConditionsFilledStar3.template.pixels)
        empty = road.ConditionsEmptyStar.template.pixels
        # Native panel rasterization phases are not tied to a particular row.
        # Reuse only observed full bright-star glyphs, with the same threshold.
        alternates=(road.ConditionsFilledStarRow2.template.pixels,
                    road.ConditionsFilledStarRow3.template.pixels,
                    road.ConditionsFilledAlternateRow1.template.pixels,
                    road.ConditionsFilledAlternateRow2.template.pixels,
                    road.ConditionsFilledAlternateRow3.template.pixels)
        slots = (road.BoxConditionsStar1, road.BoxConditionsStar2, road.BoxConditionsStar3)
        requirements = []
        for row_box in (road.BoxConditionsRow1, road.BoxConditionsRow2, road.BoxConditionsRow3):
            dy = row_box.y1 - road.BoxConditionsRow1.y1
            shifted = tuple(Rect(slot.x1, slot.y1 + dy, slot.w, slot.h) for slot in slots)
            stars = read_stars(screen, (shifted[0], shifted[1], shifted[2]), filled, empty,
                               filled_alternates=alternates)
            crop = screen[row_box.y1:row_box.y2, row_box.x1:row_box.x2]
            text=read_requirement_text(crop)
            requirements.append(read_rank_requirement(stars if stars else None, text))
        return tuple(requirements)

    def _character(self, screen: MatLike, results, rect: Rect) -> str | None:
        results = [*results, *jp().ocr(screen, rect=rect)]
        found = {
            char_id for result in results if result.confidence >= 0.95
            and rect.x1 <= result.original_rect.center.x <= rect.x2
            and rect.y1 <= result.original_rect.center.y <= rect.y2
            for char_id, name in self._names.items()
            if _compact(result.text) == _compact(name)
        }
        if found:
            return next(iter(found)) if len(found) == 1 else None
        # 小号地图姓名可在完整同帧区域放大复核；只复核原图已有的精确候选。
        proposed = {
            char_id for result in results
            if rect.x1 <= result.original_rect.center.x <= rect.x2
            and rect.y1 <= result.original_rect.center.y <= rect.y2
            for char_id, name in self._names.items() if _compact(result.text) == _compact(name)
        }
        if len(proposed) != 1:
            return None
        char_id = next(iter(proposed))
        expected = _compact(self._names[char_id])
        crop = screen[rect.y1:rect.y2, rect.x1:rect.x2]
        for variant in (crop, cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)):
            rows = jp().ocr(cv2.resize(variant, None, fx=2, fy=2))
            if len(rows) != 1 or rows[0].confidence < .98 or _compact(rows[0].text) != expected:
                return None
        return char_id

    def _stars(self, screen: MatLike, group: Literal['Formation', 'Map', 'Result']) -> int | None:
        road = R.IdolRoad
        slots = (
            getattr(road, f'Box{group}Star1'), getattr(road, f'Box{group}Star2'),
            getattr(road, f'Box{group}Star3'),
        )
        if group == 'Formation':
            formation_filled = road.FormationFilledStar.template.pixels
            filled = (formation_filled, formation_filled, formation_filled)
            empty = road.FormationEmptyStar.template.pixels
        elif group == 'Map':
            filled = (road.MapFilledStar.template.pixels, road.MapFilledStar2.template.pixels,
                      road.MapFilledStar3.template.pixels)
            empty = road.MapEmptyStar.template.pixels
        else:
            filled = (road.ResultFilledStar.template.pixels, road.ResultFilledStar2.template.pixels,
                      road.ResultFilledStar3.template.pixels)
            # 灰星来自实测地图图案，只作候选模板；不匹配时保持未知。
            map_empty = road.MapEmptyStar.template.pixels
            map_filled = road.MapFilledStar.template.pixels
            result_filled = road.ResultFilledStar.template.pixels
            size = (
                round(map_empty.shape[1] * result_filled.shape[1] / map_filled.shape[1]),
                round(map_empty.shape[0] * result_filled.shape[0] / map_filled.shape[0]),
            )
            empty = resize_star(map_empty, size)
        return read_stars(screen, slots, filled, empty)

    def _next_node(self, screen: MatLike, number: int) -> Rect | None:
        box = R.IdolRoad.BoxMapNextNode
        candidates = []
        for variant in (screen, cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)):
            matches = [
                row for row in en().ocr(variant, rect=box)
                if row.confidence >= 0.99 and _compact(row.text) == str(number)
            ]
            if len(matches) != 1:
                return None
            candidates.append(matches[0].original_rect)
        first, second = candidates
        if abs(first.center.x - second.center.x) > 5 or abs(first.center.y - second.center.y) > 5:
            return None
        return first

    def _character_candidate_confirmed(self, screen: MatLike, result, name: str) -> bool:
        """整帧姓名只作候选；低置信时用同帧完整姓名区域复核，不降阈值。"""
        expected = _compact(name)
        if _compact(result.text) != expected:
            return False
        if result.confidence >= 0.95:
            return True
        box = result.original_rect
        padding = max(1, round(box.h * 0.4))
        crop = screen[max(0, box.y1 - padding):min(screen.shape[0], box.y2 + padding),
                      max(0, box.x1 - padding):min(screen.shape[1], box.x2 + padding)]
        for variant in (crop, cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)):
            rows = jp().ocr(variant)
            if len(rows) != 1 or rows[0].confidence < 0.95 or _compact(rows[0].text) != expected:
                return False
        return True

    def observe(self, screen: MatLike, character_id: str | None) -> PageSnapshot:
        if self._feedback_mode=='hud':
            self._hud_capture_frame=None
            self._hud_capture_hand=None
        if screen.shape[:2] != (1280, 720):
            raise ValueError('偶像之路页面要求归一化后的 720×1280 截图')
        road = R.IdolRoad
        targets: dict[Action, Rect] = {}
        rounds=self.battle_rounds
        if rounds is not None and rounds.state.anchor is not None:
            rounds.invalidate('new_page_observation')
        inputs=self.battle_inputs
        if inputs is not None:
            self._sync_log_inputs()
        if self._battle_session is None and self._log_binding.request is not None:
            self._log_binding.reset()
        memory_selector=read_sub_memory_selector(screen)
        if memory_selector is not None:
            identity=getattr(self,'_memory_identity',None)
            command=getattr(self,'_memory_command',None)
            return PageSnapshot(Observation(Scene.SUB_MEMORY_SELECTOR,memory_selector=memory_selector,
                character_id=identity[0] if identity else None,stage=identity[1] if identity else None,
                memory_replacement_token=command.token if command else None))

        if all(find(screen,template.template,rect=road.BoxResumePrompt,threshold=.97) is not None for template in (
            road.ResumePromptTitle,road.ResumePromptPurpose,road.ResumePromptWarning)):
            # The game can retain a stage across a daily reset or process restart.
            # A resume prompt proves neither that stage's identity nor our ownership.
            self._battle_session=None
            self._log_span_capture=None
            self._battle_identity=None
            self._battle_inputs=None
            self._round_tracker=None
            self._maximum_hp=None
            self._status_request_session=None
            self._status_has_scrolled=False
            self._conditions_identity=None
            self._conditions_closing=False
            self._items_dialog.clear()
            self._support_dialog.clear()
            self._auto_dialog.clear()
            self._auto_result_expected=False
            self._log_binding.reset()
            return PageSnapshot(Observation(Scene.RESUME_PROMPT,
                battle_input_state=inputs.snapshot() if inputs is not None else None))

        if _match(screen,road.AutoFormationResultTitle,rect=road.BoxAutoFormationResultHeader) is not None:
            close=_match(screen,road.SupportClose,rect=road.BoxSupportClose)
            box=road.BoxAutoFormationResultPurpose
            crop=screen[box.y1:box.y2,box.x1:box.x2]
            purpose=True
            for variant in (crop,cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)):
                rows=jp().ocr(variant)
                if len(rows)!=1 or rows[0].confidence<.98 or rows[0].text!='おまかせでメモリーが編成されました':
                    purpose=False
                    break
            if close is None or not purpose:
                return PageSnapshot(Observation(Scene.UNKNOWN))
            identity=self._auto_dialog.identity or (None,None)
            return PageSnapshot(Observation(Scene.AUTO_FORMATION_RESULT,character_id=identity[0],stage=identity[1]),
                                {Action.CLOSE_AUTO_FORMATION_RESULT:close.rect})

        if _match(screen,road.AutoFormationTitle,rect=road.BoxAutoFormationHeader) is not None:
            cancel=_match(screen,road.AutoFormationCancel,rect=road.BoxAutoFormationCancel)
            confirm=_match(screen,road.AutoFormationConfirm,rect=road.BoxAutoFormationConfirm)
            box=road.BoxAutoFormationPurpose
            crop=screen[box.y1:box.y2,box.x1:box.x2]
            purpose=True
            for variant in (crop,cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)):
                rows=jp().ocr(variant)
                if len(rows)!=1 or rows[0].confidence<.98 or rows[0].text!='自動でメモリーを編成します':
                    purpose=False
                    break
            box=road.BoxAutoFormationConfirm
            hsv=cv2.cvtColor(screen[box.y1:box.y2,box.x1:box.x2],cv2.COLOR_BGR2HSV)
            orange=float((cv2.inRange(hsv,np.array([5,150,180]),np.array([30,255,255]))>0).mean())>.5
            identity=self._auto_dialog.identity or (None,None)
            if cancel is not None:
                targets[Action.CANCEL_AUTO_FORMATION]=cancel.rect
            if purpose and confirm is not None and orange:
                targets[Action.APPLY_AUTO_FORMATION]=confirm.rect
            return PageSnapshot(Observation(Scene.AUTO_FORMATION,character_id=identity[0],stage=identity[1],
                auto_formation_confirm_available=Action.APPLY_AUTO_FORMATION in targets),targets)

        owned_log=(self._battle_session is not None and self._log_binding.request is not None
                   and self._log_binding.request.session==self._battle_session)
        log_view=read_log_viewport(screen,owned_request=owned_log)
        if log_view is not None:
            overlap=None
            previous=self._log_binding.pending_header_view
            capture=getattr(self,'_log_span_capture',None)
            request=self._log_binding.request
            if (previous is not None and request is not None and capture is not None
                and (capture[0],capture[1])==(request.request_id,previous.frame_id)
                and not self._log_binding.expired and log_view.headers):
                from .log_span_page import read_log_overlap
                overlap=read_log_overlap(screen,capture[2],log_view,previous)
            inspection=self._log_binding.inspect(log_view,self._battle_session,overlap=overlap)
            if inspection.header_scroll_needed and inspection.request is not None:
                self._log_span_capture=(inspection.request.request_id,log_view.frame_id,screen.copy())
            request=inspection.request
            return PageSnapshot(Observation(Scene.BATTLE_LOG,
                character_id=request.session.character_id if request else None,
                stage=request.session.stage if request else None,battle_log_inspection=inspection),
                {Action.CLOSE_BATTLE_LOG:Rect(*log_view.close_rect)})

        owned_status=self._battle_session is not None and self._status_request_session==self._battle_session
        allow_scrolled=owned_status and self._status_has_scrolled
        status_close=find_status_close(screen,allow_scrolled=allow_scrolled)
        if status_close is not None:
            session=self._battle_session
            details=read_status_details(screen,session,allow_scrolled=allow_scrolled) if session is not None else None
            return PageSnapshot(Observation(Scene.BATTLE_STATUS_DETAILS,
                character_id=session.character_id if session else None,stage=session.stage if session else None,
                battle_status_details=details,
                battle_status_scroll_available=owned_status and details is not None and any(r.evidence.occluded for r in details.rows)),
                {Action.CLOSE_BATTLE_STATUS:status_close.rect})

        if _match(screen, road.TextSupportDetails, rect=road.BoxSupportHeader) is not None:
            close = _match(screen, road.SupportClose, rect=road.BoxSupportClose)
            if close is None:
                return PageSnapshot(Observation(Scene.UNKNOWN))
            identity = self._support_dialog.identity
            targets[Action.CLOSE_SUPPORT] = close.rect
            # 标题关卡独立核对；来源身份仍只来自本适配器打开的编成。
            stage = read_support_stage(screen) if identity is not None else None
            viewport = read_rule_viewport(screen) if identity is not None and stage == identity[1] else None
            return PageSnapshot(Observation(
                Scene.SUPPORT_DETAILS, character_id=identity[0] if identity else None,
                stage=stage, rule_viewport=viewport,
            ), targets)

        if _match(screen, road.TextItemDetails, rect=road.BoxItemDetailsHeader) is not None:
            close = _match(screen, road.ItemDetailsClose, rect=road.BoxItemDetailsClose)
            if close is None:
                return PageSnapshot(Observation(Scene.UNKNOWN))
            identity = self._items_dialog.identity or (None, None)
            targets[Action.CLOSE_ITEM_DETAILS] = close.rect
            viewport = read_item_viewport(screen)
            return PageSnapshot(Observation(
                Scene.ITEM_DETAILS, character_id=identity[0], stage=identity[1],
                visible_items=viewport.visible_items, item_viewport=viewport,
            ), targets)

        if _match(screen, road.TextConditions, rect=road.BoxConditionsHeader) is not None:
            close = _match(screen, road.TextConditionsClose, rect=road.BoxConditionsClose)
            if close is None:
                return PageSnapshot(Observation(Scene.UNKNOWN))
            identity = self._conditions_identity or (None, None)
            targets[Action.CLOSE_CONDITIONS] = close.rect
            return PageSnapshot(Observation(
                Scene.CONDITIONS, character_id=identity[0], stage=identity[1],
                requirements=self._requirements(screen),
            ), targets)

        # 专用奖励说明先于通用弹窗拦截；不是获得奖励弹窗，不设置结果身份。
        info_title = _match(screen, road.TextRewardInfo, rect=road.BoxRewardInfoHeader)
        info_body = _match(screen, road.TextRewardInfoDescription, rect=road.BoxRewardInfoHeader)
        if info_title is not None and info_body is not None:
            close = _match(screen, road.TextRewardInfoClose, rect=road.BoxRewardInfoClose)
            if close is not None:
                targets[Action.CLOSE_INFO] = close.rect
                return PageSnapshot(Observation(Scene.REWARD_INFO), targets)
            return PageSnapshot(Observation(Scene.UNKNOWN))

        # 尚未适配的确认／关闭弹窗不能透传为底层导航页面，也不自动确认。
        if any(_match(screen, control) is not None for control in (
            R.Common.IconButtonCheck, R.Common.IconButtonCross, R.Common.ButtonClose,
        )):
            return PageSnapshot(Observation(Scene.UNKNOWN))

        # 结算页先于底层画面；身份仅来自本次由本适配器启动的挑战。
        back = _match(screen, road.TextReturnToList)
        retry = _match(screen, road.TextRetry)
        if back is not None and retry is not None:
            self._battle_session = None
            self._log_binding.reset()
            identity = self._battle_identity or (None, None)
            outcome = Outcome.UNKNOWN
            stage = identity[1]
            stars = None
            if _match(screen, road.TextFailure) is not None:
                outcome = Outcome.FAILURE
            else:
                title = en().ocr(screen, rect=road.BoxClearTitle)
                clear = _match(screen, road.TextPerfect) is not None or (
                    len(title) == 1 and title[0].confidence >= 0.99
                    and _compact(title[0].text).upper() in ('CLEAR', 'PERFECT')
                )
                if clear:
                    stage = _number(screen, road.BoxClearStage)
                    stars = self._stars(screen, 'Result')
                    if stage is not None and stage == identity[1]:
                        outcome = Outcome.SUCCESS
            targets[Action.DISMISS_RESULT] = back.rect
            return PageSnapshot(Observation(
                Scene.RESULT, character_id=identity[0], stage=stage, outcome=outcome, stars=stars,
            ), targets)

        next_button = _match(screen, road.TextResultNext)
        if next_button is not None and _match(screen, road.TextFirstRank) is not None:
            identity = self._battle_identity or (None, None)
            targets[Action.ADVANCE_RESULT] = next_button.rect
            return PageSnapshot(Observation(
                Scene.JUDGING, character_id=identity[0], stage=identity[1],
                result_next_available=True,
            ), targets)

        inputs=self._battle_inputs
        if (self._battle_session is not None and self._battle_identity is not None
            and inputs is not None and inputs.session==self._battle_session
            and inputs.pending is not None and inputs.pending.frame.values.remaining_turns==1
            and _ranking_ending_crest(screen)):
            # The native ending crest retires battle input, but proves no
            # stars or clear. There is no click target during this animation.
            return PageSnapshot(Observation(Scene.JUDGING,character_id=self._battle_identity[0],
                stage=self._battle_identity[1],result_next_available=False))

        # 已绑定战斗只读取当前HUD；弹窗和结算仍先处理。
        # 不让每次动画采样都先为导航做一次整帧日文OCR。
        if (self._battle_session is not None and self._log_binding.request is not None
            and self._log_binding.request.session==self._battle_session and not self._log_binding.closing):
            # 开窗请求尚未看见面板时，只等待；底层HUD不能结算输入或成为新选牌依据。
            return PageSnapshot(Observation(Scene.UNKNOWN,character_id=self._battle_session.character_id,
                stage=self._battle_session.stage,battle_log_pending=True,
                battle_input_state=inputs.snapshot() if inputs is not None else None),targets)
        if self._battle_session is not None:
            core_only=inputs is not None and inputs.pending is not None and not self._log_binding.closing
            request=self._log_binding.request
            staged_return=(inputs is not None and inputs.pending is not None and self._log_binding.closing
                           and request is not None and request.pending_probe is not None)
            return_context=None
            return_hand=None
            if staged_return and getattr(self,'_feedback_only_returns',False):
                from kaa.tasks.produce.new.play_cards.page import LessonBattleContext
                frame=read_battle_frame(screen,self._battle_session,read_details=False,feedback_only=True,
                    include_queued_turns=self._feedback_mode=='hud')
                if frame is not None:
                    return_context=LessonBattleContext(screen,single_line_titles=True)
                    return_hand=read_log_hand(screen,frame,context=return_context)
            elif staged_return:
                from kaa.tasks.produce.new.play_cards.page import LessonBattleContext
                def details_when(core: BattleFrameObservation) -> bool:
                    nonlocal return_context,return_hand
                    return_context=LessonBattleContext(screen,single_line_titles=True)
                    return_hand=read_log_hand(screen,core,context=return_context)
                    return request is not None and return_hand is not None and return_hand.fingerprint==request.hand_fingerprint
                frame=read_battle_frame(screen,self._battle_session,details_when=details_when,
                    include_queued_turns=self._feedback_mode=='hud')
            else:
                frame=read_battle_frame(screen,self._battle_session,read_details=not core_only,
                    feedback_only=core_only and self._feedback_mode=='hud',include_queued_turns=self._feedback_mode=='hud')
            if frame is not None:
                inputs=self.battle_inputs
                permit=None
                if inputs is not None:
                    inputs.observe_hud(frame)
                    maximum=self.battle_maximum_hp
                    if maximum is not None:
                        maximum.observe(frame,inputs.snapshot())
                        frame=maximum.bind(frame,inputs.snapshot())
                    permit=self._sync_log_inputs()
                wants_log=(self._battle_session.turn_budget_mode=='variable_budget' or self._log_binding.pending
                           or isinstance(permit,PendingCardPermit))
                hand_context=return_context
                if wants_log and inputs is not None and (inputs.pending is None or permit is not None):
                    if hand_context is None:
                        from .trouble_hand import TroubleHandContext
                        hand_context=TroubleHandContext(screen,frame)
                hand=(return_hand if staged_return else read_log_hand(screen,frame,context=hand_context)
                      if hand_context is not None else None)
                entry=read_log_entry(screen,frame,hand=hand) if hand is not None and self._feedback_mode=='log' else None
                log_view=self._log_binding.last_view
                returned=self._log_binding.observe_hud(frame,entry,hand=hand)
                rounds=self.battle_rounds
                skip_resolution=None
                card_resolution=None
                if inputs is not None and returned is not None and log_view is not None and returned.request.card_probe is not None:
                    card_resolution=assess_card_log(returned.request.card_probe,returned,log_view,frame,inputs.snapshot(),now=inputs.clock())
                if rounds is not None and inputs is not None:
                    if returned is not None and log_view is not None:
                        if card_resolution is not None:
                            rounds.accept_card(frame,returned,log_view,card_resolution,inputs.snapshot(),now=inputs.clock())
                        elif returned.request.skip_probe is not None:
                            skip_resolution=inputs.finish_skip_log(returned,log_view,frame)
                            if skip_resolution is not None:
                                rounds.accept_skip(frame,returned,skip_resolution,inputs.snapshot())
                        elif returned.request.card_probe is None:
                            rounds.accept(frame,returned,log_view,inputs.snapshot())
                    frame=rounds.bind(frame,inputs.snapshot(),entry=entry,hand=hand)
                elif inputs is not None and returned is not None and log_view is not None and returned.request.skip_probe is not None:
                    skip_resolution=inputs.finish_skip_log(returned,log_view,frame)
                available=self._feedback_mode=='log' and self._log_binding.can_open(frame,entry)
                if permit is not None and (permit.pending.frame.session.battle_id,permit.sequence) in self._input_probe_requests:
                    available=False
                if available and entry is not None:
                    targets[Action.INSPECT_BATTLE_LOG]=Rect(*entry.rect)
                can_inspect=(frame.values.detail_mode=='complete' and frame.values.resource_state=='current'
                             and frame.values.hp_ratio is None)
                if can_inspect:
                    targets[Action.INSPECT_BATTLE_HP]=road.BoxBattleHpInspect
                if (frame.values.hp_ratio is not None and frame.values.hp_ratio.status=='resolved'
                    and confirmed_maximum_hp(frame) is not None and inputs is not None and inputs.pending is None):
                    targets[Action.CLOSE_BATTLE_HP]=road.BoxBattleHpInspect
                status_target=status_inspection_target(frame)
                if status_target is not None:
                    targets[Action.INSPECT_BATTLE_STATUS]=Rect(*status_target)
                if self._feedback_mode=='hud':
                    self._hud_capture_frame=frame
                    self._hud_capture_hand=hand
                return PageSnapshot(Observation(Scene.BATTLE, character_id=self._battle_session.character_id,
                                                 stage=self._battle_session.stage,battle_state=frame.observation,
                                                 battle_hp_info_available=can_inspect,
                                                 battle_status_info_available=status_target is not None,
                                                 battle_log_available=available,battle_log_pending=self._log_binding.pending,
                                                 battle_input_state=inputs.snapshot() if inputs is not None else None,
                                                 battle_round_anchor=rounds.state if rounds is not None else None,
                                                 battle_skip_resolution=skip_resolution,
                                                 battle_card_resolution=card_resolution,
                                                 battle_log_hand=hand,
                                                 battle_log_return=returned),targets,battle_frame=frame,log_entry=entry,log_hand=hand,
                                    capture_digest=(sha256(screen.tobytes()).hexdigest() if frame.values.detail_mode=='feedback_only' else None),
                                    log_view=log_view if returned is not None else None,
                                    hand_context=(CapturedHandContext(frame.session.battle_id,frame.values.frame_id,hand_context)
                                                  if hand is not None and hand_context is not None else None))

            if inputs is not None and inputs.pending is not None:
                # 已确认会话的待结动画无需扫描导航页；保持真实输入时限，不能恢复旧状态。
                return PageSnapshot(Observation(Scene.UNKNOWN,character_id=self._battle_session.character_id,
                    stage=self._battle_session.stage,battle_input_state=inputs.snapshot(),
                    battle_log_pending=self._log_binding.pending),targets)

        # 姓名用整帧检测结果再按区域筛选，避免紧裁剪改变日文检测。
        texts = jp().ocr(screen)

        if _match(screen, road.TextTitle) is not None and _match(screen, road.TextTotalRewards) is not None:
            self._battle_session = None
            self._log_binding.reset()
            self._support_dialog.clear()
            self._items_dialog.clear()
            self._conditions_identity = None
            self._conditions_closing = False
            if character_id not in self._names:
                raise ValueError(f'游戏塔数据中没有配置角色：{character_id}')
            box = road.BoxCharacters
            for result in texts:
                if (box.x1 <= result.original_rect.center.x <= box.x2
                    and box.y1 <= result.original_rect.center.y <= box.y2
                    and self._character_candidate_confirmed(screen, result, self._names[character_id])):
                    if Action.SELECT_CHARACTER in targets:
                        raise ValueError('角色列表出现多个同名候选，停止选择')
                    targets[Action.SELECT_CHARACTER] = result.original_rect
            return PageSnapshot(Observation(
                Scene.CHARACTERS, character_selectable=Action.SELECT_CHARACTER in targets,
            ), targets)

        main = _match(screen, road.TextMain)
        if main is not None and _match(screen, road.TextSupportTrouble) is not None:
            char_id = self._character(screen, texts, road.BoxFormationCharacter)
            stage = _number(screen, road.BoxFormationStage)
            if char_id is None or stage is None:
                return PageSnapshot(Observation(Scene.UNKNOWN))
            self._log_binding.reset()
            self._items_dialog.on_formation(char_id, stage)
            if self._battle_session is not None and (self._battle_session.character_id, self._battle_session.stage) != (char_id, stage):
                self._battle_session = None
            self._support_dialog.on_formation(char_id, stage)
            self._auto_dialog.on_formation(char_id,stage)
            if self._conditions_closing or self._conditions_identity != (char_id, stage):
                self._conditions_identity = None
                self._conditions_closing = False
            start = _match(screen, road.TextStart, rect=road.BoxStartButton)
            power_box = road.BoxMainPower
            dx, dy = main.rect.x1 - road.PointMainLabel.x, main.rect.y1 - road.PointMainLabel.y
            main_power = _number(screen, Rect(power_box.x1 + dx, power_box.y1 + dy, power_box.w, power_box.h))
            box = road.BoxStartButton
            hsv = cv2.cvtColor(screen[box.y1:box.y2, box.x1:box.x2], cv2.COLOR_BGR2HSV)
            orange = np.count_nonzero(cv2.inRange(hsv, np.array([5, 150, 180]), np.array([30, 255, 255]))) / hsv.shape[0] / hsv.shape[1]
            ready = start is not None and main_power is not None and orange > 0.5
            if start is not None:
                targets[Action.START_BATTLE] = start.rect
            info = _match(screen, road.ConditionsInfoIcon, rect=road.BoxConditionsInfo)
            if info is not None:
                targets[Action.OPEN_CONDITIONS] = info.rect
            items = _match(screen, road.TextOpenItemDetails, rect=road.BoxOpenItemDetails)
            if items is not None:
                targets[Action.OPEN_ITEM_DETAILS] = items.rect
            targets[Action.OPEN_SUPPORT] = road.BoxOpenSupport
            slots=read_formation_slots(screen,main.rect)
            auto=_match(screen,road.FormationAutoLabel,rect=road.BoxFormationAuto)
            if auto is not None:
                targets[Action.OPEN_AUTO_FORMATION]=auto.rect
            return PageSnapshot(Observation(
                Scene.CONFIRM, character_id=char_id, stage=stage, formation_ready=ready,
                stars=self._stars(screen, 'Formation'),
                conditions_available=info is not None,
                items_available=items is not None,
                support_available=True,
                total_turns=read_formation_total_turns(screen),
                formation_slots=slots,auto_formation_available=auto is not None,
                auto_formation_result_pending=self._auto_result_expected,
                formation_recommended_effect=read_recommended_effect(screen,main.rect),
                memory_replacement_receipt=(getattr(self,'_memory_receipt',None)
                    if getattr(self,'_memory_identity',None)==(char_id,stage) else None),
            ), targets)

        challenge = _match(screen, road.TextMapChallenge, rect=road.BoxMapChallenge)
        map_visible = _match(screen, road.TextAchievement) is not None
        if map_visible:
            self._battle_session = None
            self._log_binding.reset()
            self._support_dialog.clear()
            self._items_dialog.clear()
            self._conditions_identity = None
            self._conditions_closing = False
        recenter = _match(screen, road.TextCurrentStage) if map_visible else None
        if map_visible and challenge is not None:
            char_id = self._character(screen, texts, road.BoxMapCharacter)
            if char_id is None:
                return PageSnapshot(Observation(Scene.UNKNOWN))
            # 该版本仅支持实测默认聚焦布局，滚动改变位置后保持未知。
            stage = _number(screen, road.BoxMapStage)
            stars = self._stars(screen, 'Map')
            focused = Stage(stage, True, stars) if stage is not None else None
            targets[Action.SELECT_STAGE] = challenge.rect
            if stage is not None and stars is not None and stars >= 1:
                next_node = self._next_node(screen, stage + 1)
                if next_node is not None:
                    targets[Action.NEXT_STAGE] = next_node
            if recenter is not None:
                targets[Action.SCAN_STAGES] = recenter.rect
                # 浏览过的关卡不能冒充游戏聚焦的挑战中关卡；先复位再重读。
                if stars is None or stars == 0:
                    focused = None
            return PageSnapshot(Observation(
                Scene.STAGES, character_id=char_id, focused_stage=focused,
            ), targets)

        if map_visible and recenter is not None:
            targets[Action.SCAN_STAGES] = recenter.rect
            return PageSnapshot(Observation(
                Scene.STAGES, character_id=self._character(screen, texts, road.BoxMapCharacter),
            ), targets)

        if map_visible and _match(screen, road.TextStageRanking, rect=road.BoxMapSidebar) is not None:
            char_id = self._character(screen, texts, road.BoxMapCharacter)
            if char_id is not None:
                # 只确认地图身份；不从灰色、缺按钮或数据库推断锁定／通关。
                return PageSnapshot(Observation(Scene.MAP_OVERVIEW, character_id=char_id))

        home = _match(screen, R.Daily.ButtonHomeCurrent)
        contest = _match(screen, R.Common.ButtonContest)
        if home is not None and contest is not None:
            self._battle_session = None
            self._log_binding.reset()
            self._support_dialog.clear()
            self._items_dialog.clear()
            self._conditions_identity = None
            self._conditions_closing = False
            targets[Action.OPEN_CONTEST] = contest.rect
            return PageSnapshot(Observation(Scene.HOME), targets)
        road_entry = _match(screen, R.Daily.TextRoadToIdol)
        if road_entry is not None:
            self._battle_session = None
            self._log_binding.reset()
            self._support_dialog.clear()
            self._items_dialog.clear()
            self._conditions_identity = None
            self._conditions_closing = False
            targets[Action.OPEN_ROAD] = road_entry.rect
            return PageSnapshot(Observation(Scene.CONTEST_HUB), targets)
        return PageSnapshot(Observation(Scene.UNKNOWN))

    def execute(self, snapshot: PageSnapshot, decision: Decision, *, stage_rules: StageRules | None = None,
                observed_turns: bool=False) -> None:
        from kotonebot import device

        if decision.action in (Action.WAIT, Action.READY, Action.STOP):
            return
        maximum=self.battle_maximum_hp
        if maximum is not None and decision.action!=Action.INSPECT_BATTLE_HP:
            maximum.cancel_request()
        rounds=self.battle_rounds
        if rounds is not None:
            rounds.invalidate('page_action_requested')
        if decision.action==Action.INSPECT_BATTLE_LOG:
            frame,entry=snapshot.battle_frame,snapshot.log_entry
            inputs=self.battle_inputs
            permit=self._sync_log_inputs()
            if (snapshot.observation.scene!=Scene.BATTLE or frame is None or entry is None
                or frame.session!=self._battle_session or inputs is None or (inputs.pending is not None and permit is None)
                or (decision.character_id,decision.stage)!=(frame.session.character_id,frame.session.stage)
                or snapshot.targets.get(decision.action)!=Rect(*entry.rect)):
                raise ValueError('没有本次战斗的日志入口证据')
            if permit is not None and (frame.session.battle_id,permit.sequence) in self._input_probe_requests:
                raise ValueError('该输入的只读日志窗口已使用，不能重复打开')
            self._log_binding.open(frame,entry,input_sequence=inputs.sequence,
                skip_probe=permit if isinstance(permit,PendingSkipPermit) else None,
                card_probe=permit if isinstance(permit,PendingCardPermit) else None)
            self._log_span_capture=None
            if permit is not None:
                self._input_probe_requests.add((frame.session.battle_id,permit.sequence))
        if decision.action in (Action.SCROLL_BATTLE_LOG,Action.SCROLL_BATTLE_LOG_HEADER,Action.CLOSE_BATTLE_LOG):
            inputs=self.battle_inputs
            if inputs is not None:
                self._sync_log_inputs()
            inspection=snapshot.observation.battle_log_inspection
            if (snapshot.observation.scene!=Scene.BATTLE_LOG or inspection is None
                or inspection.viewport!=self._log_binding.last_view
                or snapshot.targets.get(Action.CLOSE_BATTLE_LOG)!=Rect(*inspection.viewport.close_rect)):
                raise ValueError('没有当前日志面板证据')
            if decision.action in (Action.SCROLL_BATTLE_LOG,Action.SCROLL_BATTLE_LOG_HEADER):
                request=inspection.request
                if (request is None or request.session!=self._battle_session
                    or (decision.character_id,decision.stage)!=(request.session.character_id,request.session.stage)):
                    raise ValueError('没有当前会话日志的滚动授权')
                if decision.action==Action.SCROLL_BATTLE_LOG_HEADER:
                    self._log_binding.scroll_to_header(inspection)
                    start,end=R.IdolRoad.PointLogScrollTo,R.IdolRoad.PointLogScrollFrom
                else:
                    self._log_binding.scroll(inspection)
                    start,end=R.IdolRoad.PointLogScrollFrom,R.IdolRoad.PointLogScrollTo
                device.swipe(start.x,start.y,end.x,end.y,duration=.5)
                return
            self._log_binding.close(inspection)
        if decision.action==Action.CLOSE_BATTLE_STATUS and snapshot.observation.scene!=Scene.BATTLE_STATUS_DETAILS:
            raise ValueError('没有确认的战斗状态详情，不能关闭')
        if decision.action==Action.SCROLL_BATTLE_STATUS:
            details=snapshot.observation.battle_status_details
            session=self._battle_session
            if (snapshot.observation.scene!=Scene.BATTLE_STATUS_DETAILS or session is None or details is None
                or self._status_request_session!=session or self._status_has_scrolled
                or session.identity_source!='confirmed_formation_start_request'
                or details.identity_source!=session.identity_source or status_viewport_key(details) is None
                or not any(row.evidence.occluded for row in details.rows)
                or (details.character_id,details.stage,details.battle_id)!=(session.character_id,session.stage,session.battle_id)
                or (decision.character_id,decision.stage)!=(session.character_id,session.stage)
                or not snapshot.observation.battle_status_scroll_available):
                raise ValueError('没有当前请求绑定的裁切状态详情，不能滚动')
            start,end=R.IdolRoad.PointBattleStatusScrollStart,R.IdolRoad.PointBattleStatusScrollEnd
            self._status_has_scrolled=True
            device.swipe(start.x,start.y,end.x,end.y,duration=.5)
            return
        if decision.action==Action.INSPECT_BATTLE_STATUS:
            frame=snapshot.battle_frame
            target=status_inspection_target(frame) if frame is not None else None
            if (snapshot.observation.scene!=Scene.BATTLE or frame is None or target is None
                or self._battle_session!=frame.session
                or (decision.character_id,decision.stage)!=(frame.session.character_id,frame.session.stage)
                or snapshot.targets.get(decision.action)!=Rect(*target)):
                raise ValueError('没有当前会话的可靠状态图标，不能查看状态详情')
        if decision.action == Action.INSPECT_BATTLE_HP:
            state=snapshot.battle_frame
            if (snapshot.observation.scene!=Scene.BATTLE or state is None or self._battle_session is None
                or state.values.resource_state!='current' or state.session!=self._battle_session
                or state.observation.source!='confirmed_current_battle_observation'
                or (decision.character_id,decision.stage)!=(state.session.character_id,state.session.stage)):
                raise ValueError('没有当前已确认战斗帧，不能查看体力提示')
            inputs=self.battle_inputs
            if inputs is None or maximum is None or maximum.session!=state.session:
                raise ValueError('缺少本场最大体力采集协议')
            maximum.request(state,inputs.snapshot())
        if decision.action == Action.CLOSE_BATTLE_HP:
            frame=snapshot.battle_frame
            inputs=self.battle_inputs
            if (snapshot.observation.scene!=Scene.BATTLE or frame is None or frame.session!=self._battle_session
                or inputs is None or inputs.pending is not None or inputs.snapshot().expired
                or (decision.character_id,decision.stage)!=(frame.session.character_id,frame.session.stage)
                or frame.values.hp_ratio is None or frame.values.hp_ratio.status!='resolved'
                or confirmed_maximum_hp(frame) is None
                or snapshot.targets.get(decision.action)!=R.IdolRoad.BoxBattleHpInspect):
                raise ValueError('没有当前自有体力提示，不能切换关闭')
        if decision.action == Action.SCROLL_ITEMS:
            observation = snapshot.observation
            if (observation.scene != Scene.ITEM_DETAILS or observation.character_id != decision.character_id
                or observation.stage != decision.stage or observation.character_id is None
                or observation.item_viewport is None or observation.item_viewport.key is None):
                raise ValueError('没有已绑定的可靠道具视口，不能滚动')
            start, end = R.IdolRoad.PointItemsScrollStart, R.IdolRoad.PointItemsScrollEnd
            device.swipe(start.x, start.y, end.x, end.y, duration=0.5)
            return
        if decision.action == Action.SELECT_CHARACTER and decision.action not in snapshot.targets:
            key = decision.character_id or ''
            attempt = self._scroll_attempts.get(key, 0)
            if attempt >= 12:
                raise ValueError(f'有限滚动后仍未识别角色：{key}（可能未解锁或被遮挡）')
            start, end = R.IdolRoad.PointScrollStart, R.IdolRoad.PointScrollEnd
            if attempt < 4:
                start, end = end, start
            device.swipe(start.x, start.y, end.x, end.y, duration=0.5)
            self._scroll_attempts[key] = attempt + 1
            return
        if decision.action == Action.SCROLL_SUPPORT:
            if (snapshot.observation.scene != Scene.SUPPORT_DETAILS
                or snapshot.observation.rule_viewport is None
                or snapshot.observation.rule_viewport.key is None
                or snapshot.observation.character_id is None
                or snapshot.observation.character_id != decision.character_id
                or snapshot.observation.stage != decision.stage):
                raise ValueError('没有当前已绑定应援／妨碍弹窗的滚动证据')
            start, end = R.IdolRoad.PointSupportScrollStart, R.IdolRoad.PointSupportScrollEnd
            device.swipe(start.x, start.y, end.x, end.y, duration=0.5)
            return
        if decision.action == Action.BACK_TO_CHARACTERS:
            if snapshot.observation.scene not in (Scene.STAGES, Scene.MAP_OVERVIEW, Scene.RESULT):
                raise ValueError('没有经过验证的返回角色页路径')
            # 失败先回地图，再由下一帧处理返回，避免连续盲点。
            if snapshot.observation.scene == Scene.RESULT:
                device.click(snapshot.targets[Action.DISMISS_RESULT])
            else:
                device.click(R.IdolRoad.PointBack)
            return
        if decision.action==Action.REPLACE_SUB_MEMORY:
            observation=snapshot.observation
            command=decision.memory_command
            identity=(decision.character_id,decision.stage)
            if (command is None or decision.character_id is None or decision.stage is None
                or (observation.character_id,observation.stage)!=identity):
                raise ValueError('换组请求缺少当前页面身份')
            if command.phase=='open':
                slots=observation.formation_slots
                if (observation.scene!=Scene.CONFIRM or observation.stars!=0 or not observation.formation_ready
                    or slots is None or slots!=command.before or slots.frame_id!=command.frame_id
                    or not slots.full or not 0<=command.slot_index<len(slots.slots)
                    or slots.slots[command.slot_index].role!='sub'
                    or slots.slots[command.slot_index].state!='occupied'):
                    raise ValueError('没有当前完整Sub槽位证据')
                self._memory_command=command
                self._memory_identity=(decision.character_id,decision.stage)
                self._memory_receipt=None
                device.click(Rect(*slots.slots[command.slot_index].rect))
                return
            previous=getattr(self,'_memory_command',None)
            view=observation.memory_selector
            if (previous is None or command.token!=previous.token
                or command.before!=previous.before or command.slot_index!=previous.slot_index
                or getattr(self,'_memory_identity',None)!=identity
                or observation.scene!=Scene.SUB_MEMORY_SELECTOR or view is None
                or view.sub_number!=command.before.sub_number(command.slot_index)
                or view.frame_id!=command.frame_id or command.candidate is None):
                raise ValueError('Sub选择请求不属于当前原生页面')
            if command.phase=='pick':
                if previous.phase!='open' or command.candidate not in view.candidates:
                    raise ValueError('Sub选择不是当前候选')
                chosen=command.candidate
                if chosen not in sub_alternatives(view,command.before,command.slot_index):
                    raise ValueError('Sub候选不符合本次替换条件')
                self._memory_command=command
                device.click(Rect(*chosen.rect))
                return
            if (previous.phase!='pick' or previous.candidate!=command.candidate
                or not selected_candidate(view,command.candidate) or view.confirm_rect is None):
                raise ValueError('新Sub原生高亮未确认')
            self._memory_command=command
            self._memory_receipt=command.token
            device.click(Rect(*view.confirm_rect))
            return
        if decision.action == Action.CANCEL_MEMORY_SELECTOR:
            if snapshot.observation.scene!=Scene.SUB_MEMORY_SELECTOR or snapshot.observation.memory_selector is None:
                raise ValueError('没有已确认的Sub记忆选择界面')
            device.click(R.IdolRoad.PointBack)
            self._memory_command=None
            self._memory_identity=None
            self._memory_receipt=None
            return
        if decision.action == Action.BACK_TO_STAGES:
            if snapshot.observation.scene != Scene.CONFIRM:
                raise ValueError('当前不在已确认的编成页，不能返回地图')
            device.click(R.IdolRoad.PointBack)
            return
        if decision.action == Action.NEXT_STAGE:
            focused = snapshot.observation.focused_stage
            if focused is None or focused.stars is None or focused.stars < 1 or decision.stage != focused.number + 1:
                raise ValueError('没有当前关至少一星及相邻关卡身份的证据')
        target = snapshot.targets.get(decision.action)
        if target is None:
            if decision.action == Action.NEXT_STAGE:
                raise ValueError('当前关已有星级，但下一关无法确认，已停止；不会重打当前关')
            raise ValueError(f'当前截图没有动作所需控件：{decision.action.value}')
        if decision.action == Action.START_BATTLE:
            slots=snapshot.observation.formation_slots
            if slots is not None and not slots.full:
                raise ValueError('当前层编组槽位未确认填满，不能开战')
            if decision.character_id is None or decision.stage is None:
                raise ValueError('开战身份缺失')
            if (snapshot.observation.character_id, snapshot.observation.stage) != (decision.character_id, decision.stage):
                raise ValueError('开战编成与请求身份不一致')
            self._battle_session = create_battle_session(snapshot.observation,stage_rules=stage_rules,observed_turns=observed_turns)
            self._feedback_mode='log'
            self._hud_capture_frame=None
            self._hud_capture_hand=None
            self._hud_authorization=None
            self._battle_sense_profile=sense_profile_from_start(self._battle_session,snapshot.observation)
            self._battle_inputs=BattleInputs(self._battle_session)
            self._maximum_hp=MaximumHpTracker(self._battle_session)
            self._input_probe_requests.clear()
            self._round_tracker=(InitialRoundTracker(self._battle_session)
                                 if self._battle_session.turn_budget_mode=='variable_budget' else None)
            self._log_binding.reset()
            self._battle_identity = (decision.character_id, decision.stage)
        if decision.action==Action.OPEN_AUTO_FORMATION:
            observation=snapshot.observation
            slots=observation.formation_slots
            if (observation.scene!=Scene.CONFIRM or observation.stars!=0 or slots is None
                or not slots.known or slots.empty_count<1
                or (observation.character_id,observation.stage)!=(decision.character_id,decision.stage)
                or decision.character_id is None or decision.stage is None):
                raise ValueError('没有当前层明确空槽证据，不能自动编排')
            self._auto_dialog.open(decision.character_id,decision.stage)
        if decision.action==Action.APPLY_AUTO_FORMATION:
            if (snapshot.observation.scene!=Scene.AUTO_FORMATION
                or not snapshot.observation.auto_formation_confirm_available
                or self._auto_dialog.identity is None
                or self._auto_dialog.identity!=(decision.character_id,decision.stage)):
                raise ValueError('自动编排确认没有本次空槽请求的来源')
            self._auto_result_expected=True
            self._items_dialog.clear()
            self._support_dialog.clear()
            self._conditions_identity=None
            self._battle_session=None
            self._battle_identity=None
        if decision.action==Action.CANCEL_AUTO_FORMATION:
            self._auto_dialog.close()
        if decision.action==Action.CLOSE_AUTO_FORMATION_RESULT:
            self._auto_result_expected=False
            self._auto_dialog.close()
        if decision.action == Action.OPEN_CONDITIONS:
            observation = snapshot.observation
            if (observation.scene != Scene.CONFIRM or observation.stars != 0
                or not observation.formation_ready or decision.character_id is None or decision.stage is None
                or observation.character_id != decision.character_id or observation.stage != decision.stage):
                raise ValueError('没有当前零星编成身份的证据，不能绑定关卡条件')
            self._conditions_identity = (decision.character_id, decision.stage)
            self._conditions_closing = False
        if decision.action == Action.CLOSE_CONDITIONS:
            self._conditions_closing = True
        if decision.action == Action.OPEN_ITEM_DETAILS:
            observation = snapshot.observation
            if (observation.scene != Scene.CONFIRM or observation.stars != 0
                or not observation.formation_ready or decision.character_id is None or decision.stage is None
                or observation.character_id != decision.character_id or observation.stage != decision.stage):
                raise ValueError('没有当前零星编成身份的证据，不能绑定 P 道具详情')
            self._items_dialog.open(decision.character_id, decision.stage)
        if decision.action == Action.CLOSE_ITEM_DETAILS:
            self._items_dialog.close()
        if decision.action == Action.OPEN_SUPPORT:
            observation = snapshot.observation
            if (observation.scene != Scene.CONFIRM or observation.stars != 0
                or not observation.formation_ready or decision.character_id is None or decision.stage is None
                or observation.character_id != decision.character_id or observation.stage != decision.stage):
                raise ValueError('没有当前零星编成身份的证据，不能绑定应援／妨碍详情')
            self._support_dialog.open(decision.character_id, decision.stage)
        if decision.action == Action.CLOSE_SUPPORT:
            self._support_dialog.close()
        if decision.action==Action.INSPECT_BATTLE_STATUS:
            self._status_request_session=self._battle_session
            self._status_has_scrolled=False
        if decision.action==Action.CLOSE_BATTLE_STATUS:
            self._status_request_session=None
            self._status_has_scrolled=False
        device.click(target)
