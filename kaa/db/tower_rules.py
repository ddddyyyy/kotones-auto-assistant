"""爬塔规则／P 道具的只读原始引用，不推断装备、触发或剩余次数。

关卡与效果组的关联必须由调用方另行确认，不能从 ID 名称拼接猜测。
这些数据不是模拟器；fireLimit=0 也不表示无限次数，限制可能位于附魔中。
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal
import json
import logging
import sqlite3

from ._util import register_cache_clear
from .sqlite import select_many


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExamDefinitionFacts:
    id: str
    fields_json: str


@lru_cache(maxsize=256)
def exam_rule_definition(kind: Literal['effect', 'trigger'], definition_id: str) -> ExamDefinitionFacts | None:
    """完整保留显式引用的原始列；解释器必须拒绝未理解的机制字段。"""
    table = {'effect': 'ProduceExamEffect', 'trigger': 'ProduceExamTrigger'}.get(kind)
    if table is None or not definition_id:
        return None
    rows = _query(f'SELECT * FROM {table} WHERE id = ?', definition_id)
    if rows is None or len(rows) != 1 or rows[0]['id'] != definition_id:
        return None
    try:
        return ExamDefinitionFacts(definition_id, json.dumps(dict(rows[0]), ensure_ascii=False))
    except (TypeError, ValueError):
        return None


def _ids(raw: str | None) -> tuple[str, ...] | None:
    """未知／坏数据不同于已声明的空数组；保留顺序和重复项。"""
    if raw is None:
        return None
    try:
        values = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
        return None
    return tuple(values)


def _integer(value: object) -> int | None:
    return value if type(value) is int else None


def _query(sql: str, *keys: str | int):
    try:
        return select_many(sql, *keys)
    except sqlite3.Error:
        logger.warning('爬塔规则数据库查询失败，保持未知：%s', keys, exc_info=True)
        return None


@dataclass(frozen=True)
class ItemNameFacts:
    id: str
    name: str


@lru_cache(maxsize=1)
def produce_item_name_candidates() -> tuple[ItemNameFacts, ...] | None:
    """完整 ID／名称候选快照，不加载描述或效果；规范化匹配由绑定层负责。"""
    rows = _query('SELECT id, name FROM ProduceItem ORDER BY id')
    if rows is None or any(not isinstance(row[key], str) or not row[key].strip()
                           for row in rows for key in ('id', 'name')):
        return None
    return tuple(ItemNameFacts(row['id'], row['name']) for row in rows)


@dataclass(frozen=True)
class ProduceItemFacts:
    id: str
    name: str
    fire_limit: int | None
    fire_interval: int | None
    trigger_id: str | None
    trigger_ids: tuple[str, ...] | None
    item_effect_ids: tuple[str, ...] | None
    effect_group_ids: tuple[str, ...] | None
    description_json: str | None


_ITEM_COLUMNS = '''id, name, fireLimit, fireInterval, produceTriggerId,
    produceTriggerIds, produceItemEffectIds, effectGroupIds, produceDescriptions'''


def _item(row) -> ProduceItemFacts | None:
    if any(not isinstance(row[key], str) or not row[key].strip() for key in ('id', 'name')):
        return None
    return ProduceItemFacts(
        row['id'], row['name'], _integer(row['fireLimit']), _integer(row['fireInterval']),
        row['produceTriggerId'], _ids(row['produceTriggerIds']), _ids(row['produceItemEffectIds']),
        _ids(row['effectGroupIds']), row['produceDescriptions'],
    )


@lru_cache(maxsize=256)
def produce_item_by_id(item_id: str) -> ProduceItemFacts | None:
    rows = _query(f'SELECT {_ITEM_COLUMNS} FROM ProduceItem WHERE id = ?', item_id)
    return _item(rows[0]) if rows is not None and len(rows) == 1 else None


@lru_cache(maxsize=256)
def produce_items_by_name(name: str) -> tuple[ProduceItemFacts, ...] | None:
    """精确名称的所有候选；同名多 ID 不自动选择。None 表示源不可用。"""
    rows = _query(f'SELECT {_ITEM_COLUMNS} FROM ProduceItem WHERE name = ? ORDER BY id', name)
    if rows is None:
        return None
    items = tuple(_item(row) for row in rows)
    return tuple(item for item in items if item is not None) if all(item is not None for item in items) else None


@dataclass(frozen=True)
class ItemEffectFacts:
    id: str
    effect_type: str | None
    effect_turn: int | None
    effect_count: int | None
    produce_effect_id: str | None
    status_enchant_id: str | None


@lru_cache(maxsize=256)
def produce_item_effect(effect_id: str) -> ItemEffectFacts | None:
    rows = _query('''SELECT id, effectType, effectTurn, effectCount,
        produceEffectId, produceExamStatusEnchantId FROM ProduceItemEffect WHERE id = ?''', effect_id)
    if rows is None or len(rows) != 1:
        return None
    row = rows[0]
    return ItemEffectFacts(row['id'], row['effectType'], _integer(row['effectTurn']),
                           _integer(row['effectCount']), row['produceEffectId'], row['produceExamStatusEnchantId'])


@dataclass(frozen=True)
class StatusEnchantFacts:
    id: str
    trigger_id: str | None
    exam_effect_ids: tuple[str, ...] | None
    description_json: str | None


@lru_cache(maxsize=256)
def produce_status_enchant(enchant_id: str) -> StatusEnchantFacts | None:
    rows = _query('''SELECT id, produceExamTriggerId, produceExamEffectIds, produceDescriptions
        FROM ProduceExamStatusEnchant WHERE id = ?''', enchant_id)
    if rows is None or len(rows) != 1:
        return None
    row = rows[0]
    return StatusEnchantFacts(row['id'], row['produceExamTriggerId'],
                              _ids(row['produceExamEffectIds']), row['produceDescriptions'])


@dataclass(frozen=True)
class GimmickFacts:
    group_id: str
    priority: int | None
    start_turn: int | None
    remaining_turn: int | None
    remaining_turn_permil: int | None
    status_type: str | None
    status_value: int | None
    check_type: str | None
    exam_effect_id: str | None
    card_search_id: str | None
    is_positive: bool | None
    description_json: str | None


@dataclass(frozen=True)
class GimmickDescriptionFacts:
    group_id: str
    priority: int
    start_turn: int
    exam_effect_id: str | None
    description_json: str | None


@lru_cache(maxsize=64)
def gimmick_descriptions_at_turn(turn: int) -> tuple[GimmickDescriptionFacts, ...] | None:
    """当前回合的全模式描述候选；不是当前塔关卡关联，也不筛选首个组。"""
    if type(turn) is not int or turn < 1:
        return None
    rows = _query('''SELECT id, priority, startTurn, produceExamEffectId, produceDescriptions
        FROM ProduceExamGimmickEffectGroup WHERE startTurn = ? ORDER BY id, priority''', turn)
    if rows is None or any(not isinstance(row['id'], str) or not row['id'].strip()
                           or type(row['priority']) is not int or type(row['startTurn']) is not int
                           for row in rows):
        return None
    return tuple(GimmickDescriptionFacts(row['id'], row['priority'], row['startTurn'],
                                         row['produceExamEffectId'], row['produceDescriptions']) for row in rows)


@lru_cache(maxsize=256)
def gimmick_group(group_id: str) -> tuple[GimmickFacts, ...] | None:
    """仅查显式组 ID；无记录也保持未知，不据此宣称该关没有规则。"""
    rows = _query('''SELECT id, priority, startTurn, remainingTurn, remainingTurnPermil,
        fieldStatusType, fieldStatusValue, fieldStatusCheckType, produceExamEffectId,
        fieldStatusProduceCardSearchId, isPositive, produceDescriptions
        FROM ProduceExamGimmickEffectGroup WHERE id = ? ORDER BY priority''', group_id)
    if not rows:
        return None
    return tuple(GimmickFacts(
        row['id'], _integer(row['priority']), _integer(row['startTurn']), _integer(row['remainingTurn']),
        _integer(row['remainingTurnPermil']), row['fieldStatusType'], _integer(row['fieldStatusValue']),
        row['fieldStatusCheckType'], row['produceExamEffectId'], row['fieldStatusProduceCardSearchId'],
        bool(row['isPositive']) if type(row['isPositive']) is int and row['isPositive'] in (0, 1) else None,
        row['produceDescriptions'],
    ) for row in rows)


for _cached in (produce_item_by_id, produce_items_by_name, produce_item_effect,
                produce_status_enchant, gimmick_group, produce_item_name_candidates,
                gimmick_descriptions_at_turn, exam_rule_definition):
    register_cache_clear(_cached.cache_clear)
