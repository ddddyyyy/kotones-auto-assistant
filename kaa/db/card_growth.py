"""Conservative growth targets and stamina-cost changes from master data."""
from dataclasses import dataclass
from functools import lru_cache
import json

from ._util import parse_id_list, register_cache_clear
from .skill_card import SkillCard, load_exam_effects
from .sqlite import select, select_many


def _effect_groups(raw: str | None) -> tuple[str, ...] | None:
    """Absent/malformed metadata is not a complete empty group inventory."""
    if raw is None or raw == '':
        return None
    try:
        values = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
        return None
    return tuple(values)


@dataclass(frozen=True)
class CardGrowthRule:
    categories: tuple[str, ...] = ()
    target_known: bool = False
    green_changes: tuple[int, ...] = ()
    red_changes: tuple[int, ...] = ()
    costs_known: bool = False
    output_changed: bool = True
    lesson_changes: tuple[int, ...] = ()
    output_known: bool = False
    effect_groups: tuple[str, ...] | None = ()
    card_ids: tuple[str, ...] | None = ()
    lesson_count_changes: tuple[int, ...] = ()
    block_changes: tuple[int, ...] = ()
    review_changes: tuple[int, ...] = ()
    full_power_changes: tuple[int, ...] = ()

    def matches(self, card: SkillCard) -> bool | None:
        category = getattr(card, 'category', None)
        if self.categories and category and category != 'ProduceCardCategory_Unknown':
            if category not in self.categories:
                return False
        elif self.categories:
            return None
        if self.card_ids is None:
            return None
        if self.card_ids:
            if len(self.card_ids) != 1:
                return None
            card_id = getattr(card, '_id', None)
            if not card_id:
                return None
            if card_id != self.card_ids[0]:
                return False
        if self.effect_groups is None:
            return None
        if self.effect_groups:
            # Multiple search groups need independently verified AND/OR rules.
            if len(self.effect_groups) != 1:
                return None
            card_groups = _effect_groups(getattr(card, '_effect_group_ids', None))
            if card_groups is None:
                return None
            if self.effect_groups[0] not in card_groups:
                return False
        return True if self.target_known else None


@lru_cache(maxsize=1024)
def get_card_growth_rule(effect_id: str) -> CardGrowthRule:
    effect = load_exam_effects({effect_id}).get(effect_id) if effect_id else None
    if effect is None:
        return CardGrowthRule()
    search = select('SELECT * FROM ProduceCardSearch WHERE id = ?', effect._produce_card_search_id or '')
    categories = tuple(parse_id_list(search['cardCategories'])) if search else ()
    groups = _effect_groups(search['effectGroupIds']) if search else None
    card_ids = _effect_groups(search['produceCardIds']) if search else None
    target_known = bool(search) and all((
        search['cardPositionType'] == 'ProduceCardPositionType_DeckAll',
        not parse_id_list(search['cardRarities']), card_ids is not None and len(card_ids) <= 1,
        not parse_id_list(search['upgradeCounts']), groups is not None and len(groups) <= 1,
        search['planType'] in {'', 'ProducePlanType_Unknown'},
        search['cardStatusType'] in {'', 'ProduceCardSearchStatusType_Unknown'},
        search['orderType'] in {'', 'ProduceCardOrderType_Unknown'},
        search['staminaMinMaxType'] in {'', 'ConditionMinMaxType_Unknown'},
        search['examEffectType'] in {'', 'ProduceExamEffectType_Unknown'},
        search['costType'] in {'', 'ExamCostType_Unknown'},
        not any(search[k] for k in ('cardSearchTag', 'produceCardRandomPoolId',
            'produceCardPoolId', 'limitCount', 'staminaMin', 'staminaMax', 'isSelf', 'isCustomized')),
        effect.pick_range_type == 'ProducePickRangeType_All',
        not effect.pick_count_min, not effect.pick_count_max,
    ))
    ids = parse_id_list(effect._produce_card_grow_effect_ids)
    rows = select_many('SELECT * FROM ProduceCardGrowEffect WHERE id IN ('
                       + ','.join('?' for _ in ids) + ')', *ids) if ids else []
    data = {row['id']: dict(row) for row in rows}
    known = bool(ids) and len(data) == len(set(ids))
    green: list[int] = []
    red: list[int] = []
    output_changed = not known
    output_known = known
    lesson: list[int] = []
    lesson_counts: list[int] = []
    blocks: list[int] = []
    reviews: list[int] = []
    full_power: list[int] = []
    for key in ids:
        row = data.get(key)
        if row is None:
            continue
        kind = row['effectType'].removeprefix('ProduceCardGrowEffectType_')
        value = row['value']
        restricted = bool(
            row.get('playProduceExamTriggerId') or row.get('playEffectProduceExamTriggerId')
            or row.get('playProduceExamEffectId') or row.get('produceCardStatusEnchantId')
            or parse_id_list(row.get('targetPlayEffectProduceExamTriggerIds'))
            or parse_id_list(row.get('targetPlayProduceExamEffectIds'))
            or parse_id_list(row.get('effectGroupIds'))
            or row.get('costType') not in {None, '', 'ExamCostType_Unknown'}
            or row.get('playMovePositionType') not in {None, '', 'ProduceCardMovePositionType_Unknown'}
        )
        if restricted:
            output_changed = True
            output_known = False
            if kind.startswith('Cost'):
                known = False
            continue
        if kind in {'CostAdd', 'CostReduce'}:
            green.append(value if kind == 'CostAdd' else -value)
        elif kind in {'CostPenetrateAdd', 'CostPenetrateReduce'}:
            red.append(value if kind == 'CostPenetrateAdd' else -value)
        elif kind in {'ReviewAdd', 'FullPowerPointAdd'} and value >= 0:
            (reviews if kind == 'ReviewAdd' else full_power).append(value)
            output_changed = True
        elif kind == 'BlockAdd' and value >= 0:
            blocks.append(value)
            output_changed = True
        elif kind == 'LessonCountAdd' and value >= 0:
            lesson_counts.append(value)
            output_changed = True
        elif kind in {'LessonAdd', 'LessonReduce'}:
            lesson.append(value if kind == 'LessonAdd' else -value)
            output_changed = True
        else:
            output_changed = True
            output_known = False
            # Special-resource fees and changes to triggers/effects need more
            # than a static stamina delta to establish the next play's cost.
            if kind.startswith('Cost') or kind in {
                'EffectAdd', 'EffectChange', 'PlayTriggerChange', 'PlayEffectTriggerChange',
                'CardStatusEnchantChange',
            }:
                known = False
    return CardGrowthRule(categories, target_known, tuple(green), tuple(red), known,
                          output_changed, tuple(lesson), output_known, groups, card_ids, tuple(lesson_counts), tuple(blocks),
                          tuple(reviews), tuple(full_power))


register_cache_clear(get_card_growth_rule.cache_clear)
