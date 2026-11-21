"""将爬塔当前帧的集中证据传给共享Planner，不修改其评分算法。"""
from cv2.typing import MatLike
from typing import Callable,TYPE_CHECKING
if TYPE_CHECKING:
    from .extra_play import ExtraPlayObservation
    from kaa.tasks.produce.new.play_cards.ui import CardGameObject
    from .skip_recovery import SkipRecoveryObservation
from kaa.tasks.produce.new.play_cards.page import LessonBattleContext
from kaa.tasks.produce.new.play_cards.planner.hud_effects import EffectIconObservation
from .battle_observation import BattleFrameObservation,current_absolute_turn,bind_battle_frame
from .concentration import confirm_concentration
from .good_condition import confirm_good_condition
from kaa.tasks.produce.new.play_cards.planner.turn_observation import AbsoluteTurnObservation
from .hand_context import CapturedHandContext
from .own_score import OwnScoreObservation,confirm_own_score
from .hp_ratio import confirm_hp_ratio
from .maximum_hp import confirmed_maximum_hp
from .score_percent import ScorePercentObservation, confirm_score_percent


def _require_current(frame: BattleFrameObservation) -> None:
    state, values = frame.observation, frame.values
    if values.detail_mode!='complete':
        raise ValueError('当前帧仅有日志探针必要信息，不能用于Planner状态或出牌')
    if frame.session.turn_budget_mode not in ('fixed_budget','variable_budget'):
        raise ValueError('Planner回合预算类型未知')
    if (frame.session.identity_source != 'confirmed_formation_start_request'
        or values.resource_state != 'current' or state.source != 'confirmed_current_battle_observation'
        or (state.character_id, state.stage, state.battle_id) != (
            frame.session.character_id, frame.session.stage, frame.session.battle_id)
        or not values.frame_id or state.frame_id != values.frame_id):
        raise ValueError('Planner缺少当前爬塔战斗帧，不能读取状态')
    if frame.session.turn_budget_mode=='variable_budget':
        if current_absolute_turn(frame) is None:
            raise ValueError('追加回合战斗尚无可靠当前回合，不能进入选牌')


def bind_concentration(observations: list[EffectIconObservation], frame: BattleFrameObservation) -> list[EffectIconObservation]:
    _require_current(frame)
    state, values = frame.observation, frame.values
    # 通用数字投票没有数值置信度，不能成为爬塔集中未知时的后备。
    # 图标也须独立核验，不能保留未确认的通用存在标记。
    result = [item for item in observations if item.effect != 'lesson_buff']
    evidence = values.concentration
    if evidence is None or evidence.frame_id != values.frame_id:
        return result
    checked = confirm_concentration(evidence.frame_id, evidence.icon_rect, evidence.icon_score, evidence.reads,evidence.scaled_reads,evidence.neutral_reads,evidence.expanded)
    if (checked != evidence or checked.status not in ('resolved','number_unknown')
        or checked.value != state.concentration or 'concentration' in state.conflicted_fields):
        return result
    assert checked.icon_rect is not None and checked.icon_score is not None
    x, y, width, height = checked.icon_rect
    result = [item for item in result if item.effect != 'lesson_buff']
    result.append(EffectIconObservation('lesson_buff', checked.icon_score, x, y, width, height, checked.value))
    return result


def bind_good_condition(observations: list[EffectIconObservation], frame: BattleFrameObservation) -> list[EffectIconObservation]:
    _require_current(frame)
    # 未核验图标不能作为存在证据，通用投票不能补未知持续时间。
    result = [item for item in observations if item.effect != 'parameter_buff']
    good = frame.values.good_condition
    if good is None or good.frame_id != frame.values.frame_id or 'good_condition' in frame.observation.conflicted_fields:
        return result
    checked = confirm_good_condition(good.frame_id, good.icon_rect, good.icon_score, good.active_blue_fraction,
                                     good.number_rect, good.reads, good.scaled_reads,good.digits)
    if checked != good or checked.status not in ('resolved', 'number_unknown'):
        return result
    if checked.value != frame.observation.good_condition_turns or checked.icon_rect is None or checked.icon_score is None:
        return result
    x, y, w, h = checked.icon_rect
    result.append(EffectIconObservation('parameter_buff', checked.icon_score, x, y, w, h, checked.value))
    return result


class IdolRoadBattleContext(LessonBattleContext):
    def __init__(self, screen: MatLike, frame: BattleFrameObservation, *,
                 before_commit: Callable[[BattleFrameObservation,str],None] | None=None,
                 hand_context: CapturedHandContext | None=None):
        super().__init__(screen,single_line_titles=True)
        if hand_context is not None:
            hand_context.validate(screen,frame)
        self._hand_context=hand_context
        self._road_frame = frame
        self._road_effects: list[EffectIconObservation] | None = None
        self._before_commit=before_commit
        self._road_score: OwnScoreObservation | None=None
        self._road_percent: ScorePercentObservation | None=None
        self._road_skip_recovery: SkipRecoveryObservation | None=None
        self._road_extra_play: ExtraPlayObservation | None=None

    def fetch_extra_play_observation(self) -> 'ExtraPlayObservation | None':
        _require_current(self._road_frame)
        if self._road_extra_play is None:
            from .extra_play_page import read_extra_play
            self._road_extra_play=read_extra_play(self.fetch_screen(),self._road_frame.values.frame_id)
        return self._road_extra_play

    def fetch_skip_recovery_observation(self) -> 'SkipRecoveryObservation | None':
        _require_current(self._road_frame)
        if self._road_skip_recovery is None:
            from .skip_recovery_page import read_skip_recovery
            self._road_skip_recovery=read_skip_recovery(self.fetch_screen(),self._road_frame.values.frame_id)
        from .skip_recovery import confirm_skip_recovery
        observed=self._road_skip_recovery
        if (observed.evidence.frame_id!=self._road_frame.values.frame_id
            or confirm_skip_recovery(observed.evidence)!=observed):
            return None
        return observed

    def fetch_exam_score_percent(self) -> int | None:
        _require_current(self._road_frame)
        if self._road_percent is None:
            from .score_percent_page import read_score_percent
            self._road_percent=read_score_percent(self.fetch_screen(),self._road_frame.values.frame_id)
        percent=self._road_percent
        checked=confirm_score_percent(percent.frame_id,percent.rect,percent.reads,percent.edge_clipped)
        if (checked!=percent or percent.frame_id!=self._road_frame.values.frame_id or percent.status!='resolved'):
            return None
        return percent.value

    def fetch_exam_current_score(self) -> int | None:
        _require_current(self._road_frame)
        if self._road_score is None:
            from .own_score_page import read_own_score
            self._road_score=read_own_score(self.fetch_screen(),self._road_frame.values.frame_id)
        score=self._road_score
        checked=confirm_own_score(score.frame_id,score.rect,score.reads,score.direct_reads,score.right_edge_clipped,
            score.mask_digits,score.margin_digits)
        if (checked!=score or score.frame_id!=self._road_frame.values.frame_id or score.status!='resolved'):
            return None
        return score.value

    def fetch_exam_standings(self):
        score=self.fetch_exam_current_score()
        if score is None:
            return None
        standings=super().fetch_exam_standings()
        return standings if standings is not None and standings.own_score==score else None

    def fetch_exam_max_hp(self) -> int | None:
        _require_current(self._road_frame)
        state=self._road_frame.observation
        ratio=self._road_frame.values.hp_ratio
        if {'hp','max_hp'}&set(state.conflicted_fields):
            return None
        if ratio is None:
            return confirmed_maximum_hp(self._road_frame)
        checked=confirm_hp_ratio(ratio.frame_id,ratio.label_rect,ratio.label_score,ratio.reads)
        if (checked!=ratio or checked.status!='resolved' or ratio.frame_id!=self._road_frame.values.frame_id
            or checked.hp!=state.hp or checked.max_hp!=state.max_hp):
            return None
        return checked.max_hp

    def fetch_hands(self):
        if self._hand_context is None:
            return super().fetch_hands()
        return self._hand_context.read(self.fetch_screen(),self._road_frame)

    def fetch_absolute_turn_observation(self) -> AbsoluteTurnObservation | None:
        _require_current(self._road_frame)
        return current_absolute_turn(self._road_frame)

    def fetch_exam_remaining_turns(self) -> int | None:
        _require_current(self._road_frame)
        return self._road_frame.values.remaining_turns

    def fetch_exam_hp(self) -> int | None:
        _require_current(self._road_frame)
        state,values=self._road_frame.observation,self._road_frame.values
        checked=bind_battle_frame(self._road_frame.session,values)
        if (checked is None or 'hp' in state.conflicted_fields or type(state.hp) is not int
            or state.hp<0 or checked.observation.hp!=state.hp):
            return None
        return state.hp

    def commit(self, card: 'CardGameObject'):
        _require_current(self._road_frame)
        if self._before_commit is not None:
            if card.card is None or card._screenshot is not self.fetch_screen():
                raise ValueError('输入卡牌不属于本次爬塔截图')
            self._before_commit(self._road_frame,card.card._id)
        super().commit(card)

    def fetch_effect_observations(self) -> list[EffectIconObservation]:
        if self._road_effects is None:
            self._road_effects = bind_good_condition(
                bind_concentration(super().fetch_effect_observations(), self._road_frame), self._road_frame)
        return list(self._road_effects)

    def fetch_unknown_effects(self) -> frozenset[str]:
        observations = self.fetch_effect_observations()
        return frozenset(effect for effect in ('lesson_buff','parameter_buff') if not any(
            item.effect==effect and item.value is not None for item in observations))
