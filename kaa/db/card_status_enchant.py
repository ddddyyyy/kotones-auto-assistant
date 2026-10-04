"""Intrinsic card growth metadata; declared limits are not observed remaining counts."""
from dataclasses import dataclass
from functools import lru_cache
import json
import sqlite3

from ._util import register_cache_clear
from .sqlite import select, select_many


@dataclass(frozen=True)
class CardStatusEnchant:
    enchant_id: str
    trigger_id: str
    trigger_count: int
    growth_ids: tuple[str, ...]
    growth_types: tuple[str, ...]
    complete: bool
    cost_changed: bool
    output_changed: bool


@lru_cache(maxsize=1024)
def get_card_status_enchant(enchant_id: str) -> CardStatusEnchant | None:
    if not enchant_id:
        return None
    try:
        row = select('SELECT id, produceExamTriggerId, triggerCount, produceCardGrowEffectIds '
                     'FROM ProduceCardStatusEnchant WHERE id = ?', enchant_id)
        if row is None:
            return None
        try:
            ids = json.loads(row['produceCardGrowEffectIds'])
        except (ValueError, TypeError):
            ids = None
        valid = isinstance(ids, list) and all(isinstance(v, str) and v for v in ids)
        growth_ids = tuple(ids) if isinstance(ids, list) and valid else ()
        rows = select_many('SELECT id, effectType FROM ProduceCardGrowEffect WHERE id IN ('
                           + ','.join('?' for _ in growth_ids) + ')', *growth_ids) if growth_ids else []
    except sqlite3.OperationalError:
        return None
    types = {r['id']: r['effectType'] for r in rows}
    complete = valid and len(types) == len(set(growth_ids)) and all(types.values())
    ordered = tuple(types[key] for key in growth_ids if key in types)
    # Effect/trigger replacement may alter both payment and output.
    structural = any(kind.removeprefix('ProduceCardGrowEffectType_') in {
        'EffectAdd', 'EffectChange', 'PlayTriggerChange', 'PlayEffectTriggerChange',
        'CardStatusEnchantChange', 'Unknown',
    } for kind in ordered)
    costs = not complete or structural or any(
        kind.startswith('ProduceCardGrowEffectType_Cost') for kind in ordered)
    output = not complete or structural or any(
        not kind.startswith('ProduceCardGrowEffectType_Cost') for kind in ordered)
    return CardStatusEnchant(row['id'], row['produceExamTriggerId'], row['triggerCount'],
                             growth_ids, ordered, complete, costs, output)


def card_enchant_uncertainty(card) -> tuple[bool, bool]:
    """Return cost/output uncertainty until per-instance growth is observed."""
    enchant_id = getattr(card, '_produce_card_status_enchant_id', None)
    if not enchant_id:
        return False, False
    enchant = get_card_status_enchant(enchant_id)
    if enchant is None:
        return True, True
    return enchant.cost_changed, enchant.output_changed


register_cache_clear(get_card_status_enchant.cache_clear)
