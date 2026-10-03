"""Resolve exam status-enchant trigger and effect chains from game master data."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import lru_cache

from ._util import parse_id_list, register_cache_clear
from .skill_card import ProduceExamEffect, load_exam_effects
from .sqlite import select


@dataclass(frozen=True)
class ExamStatusEnchant:
    enchant_id: str
    asset_id: str
    trigger_id: str
    effects: tuple[ProduceExamEffect, ...]
    chained_effects: tuple[ProduceExamEffect, ...] = ()


def _load_chained_effects(
    roots: tuple[ProduceExamEffect, ...],
) -> tuple[ProduceExamEffect, ...]:
    seen = {effect._id for effect in roots}
    current = roots
    chained: list[ProduceExamEffect] = []
    for _ in range(3):
        ids = {
            effect._chain_produce_exam_effect_id
            for effect in current
            if effect._chain_produce_exam_effect_id
            and effect._chain_produce_exam_effect_id not in seen
        }
        if not ids:
            break
        effect_map = load_exam_effects(ids)
        current = tuple(
            effect_map[effect_id]
            for effect_id in sorted(ids)
            if effect_id in effect_map
        )
        chained.extend(current)
        seen.update(ids)
    return tuple(chained)


@lru_cache(maxsize=2048)
def get_exam_status_enchant(enchant_id: str) -> ExamStatusEnchant | None:
    if not enchant_id:
        return None
    try:
        row = select(
            """
            SELECT id, assetId, produceExamTriggerId, produceExamEffectIds
            FROM ProduceExamStatusEnchant
            WHERE id = ?
            LIMIT 1;
            """,
            enchant_id,
        )
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    effect_ids = parse_id_list(
        row['produceExamEffectIds'],
        context=f'ProduceExamStatusEnchant {enchant_id}.produceExamEffectIds',
    )
    effect_map = load_exam_effects(set(effect_ids))
    effects = tuple(
        effect_map[effect_id]
        for effect_id in effect_ids
        if effect_id in effect_map
    )
    return ExamStatusEnchant(
        enchant_id=row['id'],
        asset_id=row['assetId'],
        trigger_id=row['produceExamTriggerId'],
        effects=effects,
        chained_effects=_load_chained_effects(effects),
    )


register_cache_clear(get_exam_status_enchant.cache_clear)
