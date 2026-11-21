"""在调用方确认的单帧状态上检查声明条件；不判断事件已触发或禁用仍有效。"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from .conditional_rule import ConditionalPlayDeclaration
from .health_rule import HealthMentalBlockDeclaration
from .concentration_rule import ConcentrationDoubleDeclaration
from .threshold_turn_end import ThresholdTurnEndDeclaration
from .rule_catalog import RuleDeclarationCatalog, RuleDeclarationSighting, UnknownRuleSighting


class ConditionVerdict(str, Enum):
    SATISFIED = 'satisfied'
    NOT_SATISFIED = 'not_satisfied'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class RuleConditionObservation:
    character_id: str
    stage: int
    battle_id: str
    frame_id: str
    turn: int | None
    concentration: int | None = None
    hp: int | None = None
    max_hp: int | None = None
    source: str = 'unconfirmed'
    """仅调用方核验的当前战斗单帧可用 confirmed_current_battle_observation。"""
    conflicted_fields: tuple[Literal['identity', 'turn', 'concentration', 'hp', 'max_hp', 'good_condition'], ...] = ()
    good_condition_turns: int | None = None
    turn_source: Literal['fixed_budget_assumption','owned_initial_log_return','owned_skip_log_return','owned_card_log_return','owned_initial_hud','owned_card_hud_return','owned_skip_hud_return'] | None = None


ConditionReason = Literal[
    'condition_met', 'condition_not_met', 'context_unconfirmed', 'identity_conflict',
    'battle_not_confirmed', 'frame_not_confirmed', 'observation_conflict', 'turn_unknown',
    'outside_declared_turn', 'concentration_unknown', 'hp_ratio_unknown', 'unsupported_declaration',
]


@dataclass(frozen=True)
class RuleConditionResult:
    sighting: RuleDeclarationSighting
    verdict: ConditionVerdict
    reason: ConditionReason
    timing: Literal['declared_turn', 'other_turn', 'unknown'] = 'unknown'


@dataclass(frozen=True)
class RuleConditionReport:
    observation: RuleConditionObservation
    expected_battle_id: str
    checks: tuple[RuleConditionResult, ...]
    unknown_rule_sightings: tuple[UnknownRuleSighting, ...]
    context_issue: ConditionReason | None = None
    scope: Literal['declaration_condition_only'] = 'declaration_condition_only'
    trigger_phase_confirmed: Literal[False] = False
    all_rules_known: Literal[False] = False
    executable: Literal[False] = False


def _nonnegative(value: int | None) -> bool:
    return type(value) is int and value >= 0


def _context_issue(catalog: RuleDeclarationCatalog, observation: RuleConditionObservation,
                   battle_id: str) -> ConditionReason | None:
    if catalog.status != 'partial' or observation.source != 'confirmed_current_battle_observation':
        return 'context_unconfirmed'
    if ('identity' in observation.conflicted_fields
        or set(observation.conflicted_fields) - {'identity', 'turn', 'concentration', 'hp', 'max_hp', 'good_condition'}):
        return 'observation_conflict'
    if (not catalog.character_id or type(catalog.stage) is not int or catalog.stage < 1
        or type(observation.stage) is not int
        or (catalog.character_id, catalog.stage) != (observation.character_id, observation.stage)):
        return 'identity_conflict'
    if not battle_id or not observation.battle_id or observation.battle_id != battle_id:
        return 'battle_not_confirmed'
    if not observation.frame_id:
        return 'frame_not_confirmed'
    return None


def _check(sighting: RuleDeclarationSighting, observation: RuleConditionObservation,
            context_issue: ConditionReason | None) -> RuleConditionResult:
    def unknown(reason: ConditionReason, timing: Literal['declared_turn', 'other_turn', 'unknown'] = 'unknown'):
        return RuleConditionResult(sighting, ConditionVerdict.UNKNOWN, reason, timing)

    if context_issue is not None:
        return unknown(context_issue)
    declaration = sighting.declaration
    if type(declaration.start_turn) is not int or declaration.start_turn < 1:
        return unknown('unsupported_declaration')
    if 'turn' in observation.conflicted_fields:
        return unknown('observation_conflict')
    if type(observation.turn) is not int or observation.turn < 1:
        return unknown('turn_unknown')
    if observation.turn != declaration.start_turn:
        # 下一回合的当前资源不能倒推上一回合是否触发，也不判断两回合禁用的到期。
        return unknown('outside_declared_turn', 'other_turn')
    if ((sighting.kind == 'concentration_extra_play' and isinstance(declaration, ConditionalPlayDeclaration))
        or (sighting.kind == 'concentration_double' and isinstance(declaration, ConcentrationDoubleDeclaration))
        or (sighting.kind == 'concentration_turn_end' and isinstance(declaration,ThresholdTurnEndDeclaration))):
        if (type(declaration.concentration_at_least) is not int or declaration.concentration_at_least < 1
            or (isinstance(declaration, ConditionalPlayDeclaration)
                and (type(declaration.extra_plays) is not int or declaration.extra_plays != 1))
            or (isinstance(declaration, ConcentrationDoubleDeclaration) and declaration.multiplier != 2)
            or (isinstance(declaration,ThresholdTurnEndDeclaration)
                and (declaration.reward_kind,declaration.amount) not in (('concentration',2),('score',5)))):
            return unknown('unsupported_declaration', 'declared_turn')
        if 'concentration' in observation.conflicted_fields:
            return unknown('observation_conflict', 'declared_turn')
        concentration = observation.concentration
        if not _nonnegative(concentration) or concentration is None:
            return unknown('concentration_unknown', 'declared_turn')
        met = concentration >= declaration.concentration_at_least
    elif sighting.kind == 'hp_mental_block' and isinstance(declaration, HealthMentalBlockDeclaration):
        if (type(declaration.hp_at_most_percent) is not int or not 1 <= declaration.hp_at_most_percent <= 100
            or type(declaration.duration_turns) is not int or declaration.duration_turns != 2
            or declaration.blocked_category != 'mental'):
            return unknown('unsupported_declaration', 'declared_turn')
        if {'hp', 'max_hp'} & set(observation.conflicted_fields):
            return unknown('observation_conflict', 'declared_turn')
        hp, max_hp = observation.hp, observation.max_hp
        if (hp is None or max_hp is None or not _nonnegative(hp) or not _nonnegative(max_hp)
            or max_hp == 0 or hp > max_hp):
            return unknown('hp_ratio_unknown', 'declared_turn')
        # 整数交叉相乘，避免百分比取整把边界外的 HP 错判为满足。
        met = hp * 100 <= declaration.hp_at_most_percent * max_hp
    else:
        return unknown('unsupported_declaration', 'declared_turn')
    return RuleConditionResult(sighting, ConditionVerdict.SATISFIED if met else ConditionVerdict.NOT_SATISFIED,
                               'condition_met' if met else 'condition_not_met', 'declared_turn')


def evaluate_rule_conditions(catalog: RuleDeclarationCatalog, observation: RuleConditionObservation,
                              *, battle_id: str) -> RuleConditionReport:
    """battle_id 必须由当前战斗调用方提供，不能仅复用输入里的历史战斗标识。

    满足只表示该声明所需数值条件成立，不证明触发时机、已生效、牌合法或收益确定。
    每条分屏观察独立输出，禁止求和为额外出牌次数或合并为禁用状态。
    """
    issue = _context_issue(catalog, observation, battle_id)
    return RuleConditionReport(observation, battle_id,
                               tuple(_check(row, observation, issue) for row in catalog.sightings),
                               catalog.unknown_sightings, context_issue=issue)
