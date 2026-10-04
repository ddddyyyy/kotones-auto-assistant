import pytest
from typing import Any, cast

from types import SimpleNamespace
from unittest.mock import patch

from kaa.db.exam_status_enchant import get_exam_status_enchant


def setup_function():
    get_exam_status_enchant.cache_clear()


def test_status_enchant_resolves_ordered_effect_chain():
    row = {
        'id': 'enchant-1',
        'assetId': 'asset-1',
        'produceExamTriggerId': 'trigger-1',
        'produceExamEffectIds': '["effect-2", "effect-1"]',
    }
    effects = {
        'effect-1': SimpleNamespace(
            _id='effect-1', _chain_produce_exam_effect_id=''
        ),
        'effect-2': SimpleNamespace(
            _id='effect-2', _chain_produce_exam_effect_id='effect-3'
        ),
        'effect-3': SimpleNamespace(
            _id='effect-3', _chain_produce_exam_effect_id=''
        ),
    }
    with (
        patch('kaa.db.exam_status_enchant.select', return_value=row),
        patch('kaa.db.exam_status_enchant.load_exam_effects', return_value=effects),
    ):
        enchant = get_exam_status_enchant('enchant-1')

    assert enchant is not None
    assert enchant.trigger_id == 'trigger-1'
    assert [effect._id for effect in enchant.effects] == ['effect-2', 'effect-1']
    assert [effect._id for effect in enchant.chained_effects] == ['effect-3']


@pytest.mark.parametrize('single,multiple,expected', [
    ('one', None, ('one',)),
    ('', '["z", "a"]', ('z', 'a')),
    ('one', '["one", "two", "two"]', ('one', 'two')),
    ('', '[]', ()),
])
def test_effect_row_preserves_both_chain_fields_in_declared_order(single, multiple, expected):
    from kaa.db.skill_card import ProduceExamEffect, ProduceExamEffectRow
    row = ProduceExamEffectRow.model_validate(dict(
        id='root', chainProduceExamEffectId=single, chainProduceExamEffectIds=multiple,
    ))
    effect = ProduceExamEffect.from_row(row)
    assert effect.chain_effect_ids == expected
    assert effect._chain_produce_exam_effect_id == single


def test_effect_chain_traversal_preserves_branches_and_handles_cycles_and_missing_rows():
    from kaa.db.exam_status_enchant import _load_chained_effects
    def effect(key, *children):
        return SimpleNamespace(_id=key, _chain_produce_exam_effect_id='',
                               chain_effect_ids=children)
    roots = (effect('root-b', 'z', 'a'), effect('root-a', 'a', 'missing'))
    data = {'z': effect('z', 'root-b', 'tail'), 'a': effect('a', 'tail'),
            'tail': effect('tail', 'z')}
    calls = []
    def load(ids):
        calls.append(ids)
        return {key: data[key] for key in ids if key in data}
    with patch('kaa.db.exam_status_enchant.load_exam_effects', side_effect=load):
        result = _load_chained_effects(cast(Any, roots))
    assert [entry._id for entry in result] == ['z', 'a', 'tail']
    assert calls == [{'z', 'a', 'missing'}, {'tail'}]


def test_effect_chain_traversal_keeps_existing_depth_limit():
    from kaa.db.exam_status_enchant import _load_chained_effects
    data = {str(i): SimpleNamespace(_id=str(i), _chain_produce_exam_effect_id='',
            chain_effect_ids=(str(i + 1),)) for i in range(6)}
    with patch('kaa.db.exam_status_enchant.load_exam_effects', side_effect=lambda ids: {
            key: data[key] for key in ids}):
        result = _load_chained_effects(cast(Any, (data['0'],)))
    assert [entry._id for entry in result] == ['1', '2', '3']
