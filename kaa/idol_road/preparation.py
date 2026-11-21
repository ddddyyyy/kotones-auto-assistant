"""战斗输入的只读预检查；不是开战授权，不调用设备、DB 或 Planner。"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from .items import ItemInventory, is_confirmed_item_reference
from .models import Observation, Scene
from .objectives import StageObjectives
from .rules import StageRules
from .rule_catalog import RuleDeclarationCatalog, build_rule_catalog
from .good_condition_rule import GoodConditionRequirement, good_condition_requirements
from .turn_budget import TurnBudgetConflict, fixed_turn_budget_conflicts


class PreparationIssueCode(str, Enum):
    FORMATION_UNKNOWN = 'formation_scene_unknown'
    IDENTITY_CONFLICT = 'formation_identity_unconfirmed'
    STARS_UNKNOWN = 'actual_stars_unknown'
    ALREADY_CLEARED = 'stage_already_cleared'
    FORMATION_UNAVAILABLE = 'formation_unavailable'
    FORMATION_SLOTS_INCOMPLETE = 'formation_slots_incomplete'
    OBJECTIVES_UNKNOWN = 'objectives_unknown'
    OBJECTIVES_IDENTITY = 'objectives_identity_conflict'
    OBJECTIVES_SOURCE = 'objectives_source_unconfirmed'
    TARGET_UNKNOWN = 'one_star_rank_goal_unknown'
    INVENTORY_UNKNOWN = 'inventory_unknown'
    INVENTORY_IDENTITY = 'inventory_identity_conflict'
    INVENTORY_SOURCE = 'inventory_source_unconfirmed'
    INVENTORY_INCOMPLETE = 'inventory_incomplete'
    ITEM_BINDING_UNKNOWN = 'item_binding_unconfirmed'
    ITEM_SECTION_UNKNOWN = 'item_section_unknown'
    ITEM_INSTANCES_UNKNOWN = 'item_instances_unconfirmed'
    ARCHETYPE_UNKNOWN = 'archetype_unknown'
    LINEUP_UNKNOWN = 'lineup_identity_unknown'
    DECK_UNKNOWN = 'initial_deck_unknown'
    RULES_UNKNOWN = 'stage_rules_unconfirmed'
    RULES_IDENTITY = 'stage_rules_identity_conflict'
    RULES_SOURCE = 'stage_rules_source_unconfirmed'
    RULES_INCOMPLETE = 'stage_rules_incomplete'
    TURN_BUDGET_UNSUPPORTED = 'dynamic_turn_budget_unimplemented'
    EFFECTS_UNSUPPORTED = 'item_effect_execution_unimplemented'
    ADAPTER_UNAVAILABLE = 'battle_adapter_unimplemented'


_MESSAGES = {
    PreparationIssueCode.FORMATION_UNKNOWN: '当前不在已确认编成页',
    PreparationIssueCode.IDENTITY_CONFLICT: '当前角色或关卡身份未确认',
    PreparationIssueCode.STARS_UNKNOWN: '实际星数未知',
    PreparationIssueCode.ALREADY_CLEARED: '当前关已有星数，不应重新挑战',
    PreparationIssueCode.FORMATION_UNAVAILABLE: '当前编成不可用',
    PreparationIssueCode.FORMATION_SLOTS_INCOMPLETE: '本层编组槽位未知或有空缺',
    PreparationIssueCode.OBJECTIVES_UNKNOWN: '未取得星级条件',
    PreparationIssueCode.OBJECTIVES_IDENTITY: '星级条件属于其他角色或关卡',
    PreparationIssueCode.OBJECTIVES_SOURCE: '星级条件来源未确认',
    PreparationIssueCode.TARGET_UNKNOWN: '一星名次目标未知或尚不支持',
    PreparationIssueCode.INVENTORY_UNKNOWN: '未取得装备观察',
    PreparationIssueCode.INVENTORY_IDENTITY: '装备观察属于其他角色或关卡',
    PreparationIssueCode.INVENTORY_SOURCE: '装备观察来源未确认',
    PreparationIssueCode.INVENTORY_INCOMPLETE: '装备完整性尚未证明',
    PreparationIssueCode.ITEM_BINDING_UNKNOWN: '存在未绑定或有歧义的道具',
    PreparationIssueCode.ITEM_SECTION_UNKNOWN: '存在归属未知的道具',
    PreparationIssueCode.ITEM_INSTANCES_UNKNOWN: '跨屏装备实体尚未确认',
    PreparationIssueCode.ARCHETYPE_UNKNOWN: '流派尚未确认',
    PreparationIssueCode.LINEUP_UNKNOWN: '稳定编组身份尚未确认',
    PreparationIssueCode.DECK_UNKNOWN: '初始牌组尚未初始化',
    PreparationIssueCode.RULES_UNKNOWN: '当前关卡规则尚未确认',
    PreparationIssueCode.RULES_IDENTITY: '应援／妨碍观察属于其他角色或关卡',
    PreparationIssueCode.RULES_SOURCE: '应援／妨碍观察来源未确认',
    PreparationIssueCode.RULES_INCOMPLETE: '应援／妨碍列表完整性尚未证明',
    PreparationIssueCode.TURN_BUDGET_UNSUPPORTED: '当前场地可追加回合，固定初始总回合不能作为绝对回合预算',
    PreparationIssueCode.EFFECTS_UNSUPPORTED: '道具事件结算尚未接入',
    PreparationIssueCode.ADAPTER_UNAVAILABLE: '完整自动爬塔尚未就绪（仅提供显式单场实验模式）',
}


@dataclass(frozen=True)
class PreparationIssue:
    code: PreparationIssueCode
    message: str


@dataclass(frozen=True)
class BattleSpec:
    character_id: str
    stage: int
    rank_at_most: int | None
    rank_goal_source: Literal['conditions_dialog'] | None = None
    mode: Literal['idol_road'] = 'idol_road'
    target_stars: Literal[1] = 1
    archetype: None = None
    lineup_identity: None = None
    """未提供稳定编组身份，不能据此复用固定牌序或行动脚本。"""


@dataclass(frozen=True)
class BattlePreparation:
    formation_confirmed: bool
    spec: BattleSpec | None
    issues: tuple[PreparationIssue, ...]
    automatic_battle_ready: Literal[False] = False
    """单场实验适配器不等于完整自动爬塔；此报告不能授权自动开战。"""
    rule_declarations: RuleDeclarationCatalog | None = None
    good_condition_requirements: tuple[GoodConditionRequirement, ...] = ()
    """只确认条件的独立字段；完整规则和收益仍可能未知。"""
    turn_budget_conflicts: tuple[TurnBudgetConflict, ...] = ()


def prepare_battle(
    character_id: str | None, stage: int | None, observation: Observation | None,
    *, objectives: StageObjectives | None = None, inventory: ItemInventory | None = None,
    stage_rules: StageRules | None = None,
) -> BattlePreparation:
    codes: list[PreparationIssueCode] = []

    def add(code: PreparationIssueCode) -> None:
        if code not in codes:
            codes.append(code)

    if observation is None or observation.scene != Scene.CONFIRM:
        add(PreparationIssueCode.FORMATION_UNKNOWN)
    if (not character_id or type(stage) is not int or stage < 1 or observation is None
        or observation.character_id != character_id or observation.stage != stage):
        add(PreparationIssueCode.IDENTITY_CONFLICT)
    if observation is None or observation.stars is None:
        add(PreparationIssueCode.STARS_UNKNOWN)
    elif observation.stars >= 1:
        add(PreparationIssueCode.ALREADY_CLEARED)
    if observation is None or not observation.formation_ready:
        add(PreparationIssueCode.FORMATION_UNAVAILABLE)
    if observation is not None and observation.formation_slots is not None and not observation.formation_slots.full:
        add(PreparationIssueCode.FORMATION_SLOTS_INCOMPLETE)
    formation_confirmed = not codes
    rank = None
    if objectives is None:
        add(PreparationIssueCode.OBJECTIVES_UNKNOWN)
    elif (objectives.character_id, objectives.stage) != (character_id, stage):
        add(PreparationIssueCode.OBJECTIVES_IDENTITY)
    elif (objectives.source != 'conditions_dialog'
          or objectives.identity_source != 'opened_from_confirmed_formation'):
        add(PreparationIssueCode.OBJECTIVES_SOURCE)
    else:
        rank = objectives.rank_limit(1)
        if type(rank) is not int or rank < 1:
            rank = None
            add(PreparationIssueCode.TARGET_UNKNOWN)
    if inventory is None:
        add(PreparationIssueCode.INVENTORY_UNKNOWN)
    elif (inventory.character_id, inventory.stage) != (character_id, stage):
        add(PreparationIssueCode.INVENTORY_IDENTITY)
    elif (inventory.source != 'visible_item_details'
          or inventory.identity_source != 'opened_from_confirmed_formation'):
        add(PreparationIssueCode.INVENTORY_SOURCE)
    else:
        # 当前模型没有完整性证明机制，不能通过手动翻转字段绕过。
        add(PreparationIssueCode.INVENTORY_INCOMPLETE)
        sightings = (inventory.visible_items, *(page.visible_items for page in inventory.viewports))
        if any(not items for items in sightings):
            add(PreparationIssueCode.INVENTORY_UNKNOWN)
        for items in sightings:
            for item in items or ():
                if not is_confirmed_item_reference(item.reference):
                    add(PreparationIssueCode.ITEM_BINDING_UNKNOWN)
                if item.section is None:
                    add(PreparationIssueCode.ITEM_SECTION_UNKNOWN)
        if len(inventory.viewports) > 1:
            add(PreparationIssueCode.ITEM_INSTANCES_UNKNOWN)
    if stage_rules is not None:
        if (stage_rules.character_id, stage_rules.stage) != (character_id, stage):
            add(PreparationIssueCode.RULES_IDENTITY)
        elif (stage_rules.source != 'visible_support_details'
              or stage_rules.identity_source != 'opened_from_confirmed_formation'):
            add(PreparationIssueCode.RULES_SOURCE)
        else:
            add(PreparationIssueCode.RULES_INCOMPLETE)
    # 当前手牌由战斗适配器逐帧读取；完整牌库和全局稳定编组ID不是开战前提。
    # 剩余机制仍须有实际实现，不能用移除牌库门槛宣称完整自动战斗就绪。
    for code in (PreparationIssueCode.ARCHETYPE_UNKNOWN, PreparationIssueCode.RULES_UNKNOWN,
                 PreparationIssueCode.EFFECTS_UNSUPPORTED, PreparationIssueCode.ADAPTER_UNAVAILABLE):
        add(code)
    spec = (BattleSpec(character_id, stage, rank, rank_goal_source='conditions_dialog' if rank is not None else None)
            if formation_confirmed and character_id is not None and stage is not None else None)
    catalog = (build_rule_catalog(character_id, stage, stage_rules)
               if formation_confirmed and character_id is not None and stage is not None and stage_rules is not None else None)
    turn_conflicts=(fixed_turn_budget_conflicts(character_id,stage,stage_rules)
                    if formation_confirmed and character_id is not None and stage is not None else ())
    if turn_conflicts:
        add(PreparationIssueCode.TURN_BUDGET_UNSUPPORTED)
    return BattlePreparation(formation_confirmed, spec, tuple(PreparationIssue(code, _MESSAGES[code]) for code in codes),
                             rule_declarations=catalog,
                             good_condition_requirements=(good_condition_requirements(character_id, stage, stage_rules)
                                 if formation_confirmed and character_id is not None and stage is not None else ()),
                             turn_budget_conflicts=turn_conflicts)
