from types import SimpleNamespace
import pytest
from kaa.db import card_status_enchant as db


@pytest.fixture(autouse=True)
def clear_cache():
    db.get_card_status_enchant.cache_clear()
    yield
    db.get_card_status_enchant.cache_clear()


def _setup(monkeypatch, raw='["b", "a"]', missing=False):
    row = dict(id='enchant', produceExamTriggerId='trigger', triggerCount=2,
               produceCardGrowEffectIds=raw)
    monkeypatch.setattr(db, 'select', lambda *_: row)
    monkeypatch.setattr(db, 'select_many', lambda *_: [
        dict(id='a', effectType='ProduceCardGrowEffectType_CostAdd'),
    ] if missing else [
        dict(id='a', effectType='ProduceCardGrowEffectType_CostAdd'),
        dict(id='b', effectType='ProduceCardGrowEffectType_LessonCountAdd'),
    ])


def test_declared_limit_and_order_are_preserved_without_remaining_count(monkeypatch):
    _setup(monkeypatch)
    result = db.get_card_status_enchant('enchant')
    assert result is not None and result.complete
    assert result.trigger_count == 2 and result.trigger_id == 'trigger'
    assert result.growth_ids == ('b', 'a')
    assert result.growth_types[0].endswith('LessonCountAdd')
    assert result.cost_changed and result.output_changed


@pytest.mark.parametrize('raw,missing', [('invalid', False), ('[1]', False), ('["b", "a"]', True)])
def test_incomplete_definitions_never_prove_cost_or_output(monkeypatch, raw, missing):
    _setup(monkeypatch, raw, missing)
    result = db.get_card_status_enchant('enchant')
    assert result is not None and not result.complete
    assert db.card_enchant_uncertainty(SimpleNamespace(_produce_card_status_enchant_id='enchant')) == (True, True)


def test_missing_definition_is_unknown_but_absent_enchant_is_unaffected(monkeypatch):
    monkeypatch.setattr(db, 'select', lambda *_: None)
    assert db.card_enchant_uncertainty(SimpleNamespace(_produce_card_status_enchant_id='missing')) == (True, True)
    assert db.card_enchant_uncertainty(SimpleNamespace()) == (False, False)


def test_cache_uses_unified_invalidation(monkeypatch):
    from kaa.db import _util
    _setup(monkeypatch)
    assert db.get_card_status_enchant('enchant') is not None
    monkeypatch.setattr(db, 'select', lambda *_: None)
    monkeypatch.setattr(_util, 'invalidate_connections', lambda: None)
    _util.clear_db_caches()
    assert db.get_card_status_enchant('enchant') is None
