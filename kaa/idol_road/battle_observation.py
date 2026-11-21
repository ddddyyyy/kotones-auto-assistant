"""单场战斗的身份与同帧数值绑定；不补历史值或执行规则。"""
from dataclasses import dataclass
from typing import Literal,TYPE_CHECKING
if TYPE_CHECKING:
    from .maximum_hp import BoundMaximumHp
from uuid import uuid4
from .models import Observation, Scene
from .rule_conditions import RuleConditionObservation
from .concentration import ConcentrationObservation, confirm_concentration
from .hp_ratio import HpRatioObservation,confirm_hp_ratio
from .good_condition import GoodConditionObservation, confirm_good_condition
from .rules import StageRules
from .turn_budget import fixed_turn_budget_conflicts
from .queued_turns import QueuedTurnObservation,queued_turns_valid
from kaa.battle_turn import AbsoluteTurnObservation


@dataclass(frozen=True)
class BattleObservationSession:
    battle_id: str
    character_id: str
    stage: int
    total_turns: int | None
    """编成页初始显示回合数，不表示追加回合后的总数。"""
    identity_source: str
    turn_budget_mode: Literal['fixed_budget', 'variable_budget'] = 'fixed_budget'

    @property
    def initial_total_turns(self) -> int | None:
        return self.total_turns


@dataclass(frozen=True)
class BattleFrameValues:
    frame_id: str
    hud_confirmed: bool
    remaining_turns: int | None
    hp: int | None
    reader_source: Literal['ranked_hud_same_frame'] = 'ranked_hud_same_frame'
    resource_state: Literal['current', 'selection_preview', 'unconfirmed_overlay'] = 'current'
    concentration: ConcentrationObservation | None = None
    hp_ratio: HpRatioObservation | None = None
    good_condition: GoodConditionObservation | None = None
    detail_mode: Literal['complete','core_only','feedback_only'] = 'complete'
    """core_only用于日志探针，feedback_only用于自有输入反馈；未读取的状态不是零，均不能用于Planner。"""
    queued_turns: QueuedTurnObservation | None = None


@dataclass(frozen=True)
class BattleFrameObservation:
    session: BattleObservationSession
    values: BattleFrameValues
    observation: RuleConditionObservation
    absolute_turn: AbsoluteTurnObservation | None = None
    maximum_hp_parameter: 'BoundMaximumHp | None' = None


def current_absolute_turn(frame: BattleFrameObservation) -> AbsoluteTurnObservation | None:
    return _current_turn(frame,feedback=False)


def current_feedback_turn(frame: BattleFrameObservation) -> AbsoluteTurnObservation | None:
    """An input-return feedback frame cannot become a planning observation."""
    return _current_turn(frame,feedback=True)


def _current_turn(frame: BattleFrameObservation,*,feedback: bool) -> AbsoluteTurnObservation | None:
    """只消费已由模式适配器绑定的首次返回帧，不从剩余数补回合。"""
    absolute,state,values,session=frame.absolute_turn,frame.observation,frame.values,frame.session
    if (absolute is None or session.identity_source!='confirmed_formation_start_request'
        or session.turn_budget_mode!='variable_budget' or values.resource_state!='current' or not values.hud_confirmed
        or (values.detail_mode!='complete' and not (feedback and values.detail_mode=='feedback_only'
            and state.turn_source in ('owned_card_log_return','owned_skip_log_return')))
        or state.source!='confirmed_current_battle_observation'
        or state.turn_source not in ('owned_initial_log_return','owned_skip_log_return','owned_card_log_return','owned_initial_hud','owned_card_hud_return','owned_skip_hud_return')
        or type(state.turn) is not int or not 1<=state.turn<=99
        or type(session.initial_total_turns) is not int or not 1<=session.initial_total_turns<=30
        or (state.turn_source in ('owned_initial_log_return','owned_initial_hud') and (state.turn!=1 or values.remaining_turns!=session.initial_total_turns))
        or (state.turn_source in ('owned_skip_log_return','owned_skip_hud_return') and state.turn<2)
        or state.frame_id!=values.frame_id or {'identity','turn'}&set(state.conflicted_fields)
        or (state.battle_id,state.character_id,state.stage)!=(session.battle_id,session.character_id,session.stage)
        or (absolute.battle_id,absolute.frame_id,absolute.turn,absolute.remaining_turns)
        !=(session.battle_id,values.frame_id,state.turn,values.remaining_turns)):
        return None
    try:
        absolute.validate()
    except ValueError:
        return None
    return absolute


def status_inspection_target(frame: BattleFrameObservation) -> tuple[int,int,int,int] | None:
    """只点击本帧核验的当前集中图标，不用预告／卡面／旧帧位置。"""
    state,values=frame.observation,frame.values
    evidence=values.concentration
    if (state.source!='confirmed_current_battle_observation' or values.resource_state!='current'
        or frame.session.identity_source!='confirmed_formation_start_request'
        or (state.character_id,state.stage,state.battle_id)!=(frame.session.character_id,frame.session.stage,frame.session.battle_id)
        or state.frame_id!=values.frame_id or not values.frame_id or state.conflicted_fields
        or evidence is None or evidence.frame_id!=values.frame_id):
        return None
    checked=confirm_concentration(evidence.frame_id,evidence.icon_rect,evidence.icon_score,evidence.reads,evidence.scaled_reads,evidence.neutral_reads,evidence.expanded)
    if (checked!=evidence or checked.status not in ('resolved','number_unknown')
        or checked.value!=state.concentration or len(evidence.reads)!=3
        or {read.representation for read in evidence.reads}!={'original','color_white_mask','gray_white_mask'}
        or any(read.frame_id!=values.frame_id for read in evidence.reads)
        or len({read.rect for read in evidence.reads})!=1):
        return None
    return checked.icon_rect


def create_battle_session(formation: Observation, *, stage_rules: StageRules | None = None,
                          observed_turns: bool=False) -> BattleObservationSession:
    if type(observed_turns) is not bool:raise ValueError('回合观察模式未知')
    if formation.formation_slots is not None and not formation.formation_slots.full:
        raise ValueError('本层编组槽位未确认填满，不能建立开战会话')
    if (formation.scene != Scene.CONFIRM or formation.stars != 0 or not formation.formation_ready
        or not formation.character_id or type(formation.stage) is not int or formation.stage < 1):
        raise ValueError('没有确认的零星可用编成，不能建立战斗观察会话')
    total = formation.total_turns
    if type(total) is not int or not 1 <= total <= 30:
        total = None
    if stage_rules is not None and (
        (stage_rules.character_id,stage_rules.stage)!=(formation.character_id,formation.stage)
        or stage_rules.source!='visible_support_details'
        or stage_rules.identity_source!='opened_from_confirmed_formation'):
        raise ValueError('战斗回合预算缺少当前编成的规则来源')
    if observed_turns and (stage_rules is None or total is None):
        raise ValueError('观察回合需要当前编成规则与初始回合数')
    # The owned clock also handles floors without a declared turn addition.
    # variable_budget means observing transitions rather than deriving time.
    mode='variable_budget' if observed_turns or fixed_turn_budget_conflicts(formation.character_id,formation.stage,stage_rules) else 'fixed_budget'
    return BattleObservationSession(uuid4().hex, formation.character_id, formation.stage, total,
                                     'confirmed_formation_start_request',mode)


def bind_battle_frame(session: BattleObservationSession, values: BattleFrameValues) -> BattleFrameObservation | None:
    if (values.hud_confirmed is not True or values.reader_source != 'ranked_hud_same_frame'
        or values.detail_mode not in ('complete','core_only','feedback_only')
        or (values.detail_mode!='complete' and any(value is not None for value in (values.concentration,values.hp_ratio,values.good_condition)))
        or not values.frame_id or not session.battle_id or not session.character_id
        or type(session.stage) is not int or session.stage < 1):
        return None
    remaining, total = values.remaining_turns, session.initial_total_turns
    if values.queued_turns is not None and not queued_turns_valid(values.queued_turns,values.frame_id,remaining):
        return None
    turn = None
    turn_source = None
    conflicts: list[Literal['turn', 'hp', 'max_hp', 'concentration', 'good_condition']] = []
    if session.turn_budget_mode not in ('fixed_budget','variable_budget'):
        conflicts.append('turn')
    if (session.turn_budget_mode=='fixed_budget' and type(remaining) is int and 1 <= remaining <= 30
        and type(total) is int and 1 <= total <= 30):
        if remaining <= total:
            turn = total - remaining + 1
            turn_source = 'fixed_budget_assumption'
        else:
            conflicts.append('turn')
    hp = values.hp
    if hp is not None and (type(hp) is not int or hp < 0):
        hp = None
        conflicts.append('hp')
    source = ('confirmed_current_battle_observation'
              if session.identity_source == 'confirmed_formation_start_request'
              else 'historical_or_unconfirmed_battle_observation')
    max_hp=None
    ratio=values.hp_ratio
    if ratio is not None:
        checked_ratio=confirm_hp_ratio(ratio.frame_id,ratio.label_rect,ratio.label_score,ratio.reads)
        if (ratio.frame_id!=values.frame_id or checked_ratio!=ratio or checked_ratio.status=='conflict'
            or (checked_ratio.status=='resolved' and hp is not None and hp!=checked_ratio.hp)):
            hp=None
            conflicts.extend(('hp','max_hp'))
        elif checked_ratio.status=='resolved':
            hp,max_hp=checked_ratio.hp,checked_ratio.max_hp
    concentration=None
    evidence=values.concentration
    if evidence is not None:
        checked=confirm_concentration(evidence.frame_id,evidence.icon_rect,evidence.icon_score,evidence.reads,evidence.scaled_reads,evidence.neutral_reads,evidence.expanded)
        if evidence.frame_id!=values.frame_id or checked!=evidence or checked.status=='conflict':
            conflicts.append('concentration')
        elif checked.status=='resolved':
            concentration=checked.value
    good_condition_turns = None
    good = values.good_condition
    if good is not None:
        checked_good = confirm_good_condition(good.frame_id, good.icon_rect, good.icon_score,
            good.active_blue_fraction, good.number_rect, good.reads, good.scaled_reads,good.digits)
        if good.frame_id != values.frame_id or checked_good != good or checked_good.status == 'conflict':
            conflicts.append('good_condition')
        elif checked_good.status == 'resolved':
            good_condition_turns = checked_good.value
    if values.resource_state!='current':
        turn=hp=max_hp=concentration=good_condition_turns=None
        turn_source=None
        source='unconfirmed_resource_preview'
    observation = RuleConditionObservation(session.character_id, session.stage, session.battle_id,
                                            values.frame_id, turn, concentration=concentration,hp=hp,max_hp=max_hp,source=source,
                                            conflicted_fields=tuple(conflicts), good_condition_turns=good_condition_turns,
                                            turn_source=turn_source)
    return BattleFrameObservation(session, values, observation)
