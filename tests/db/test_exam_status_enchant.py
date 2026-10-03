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
