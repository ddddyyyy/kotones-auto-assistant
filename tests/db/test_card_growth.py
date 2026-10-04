from types import SimpleNamespace
from typing import Any, cast

import pytest

from kaa.db import card_growth, _util
from kaa.db.skill_card import ProduceExamEffect, ProduceExamEffectRow


def _search():
    row: dict[str, Any] = dict(cardPositionType='ProduceCardPositionType_DeckAll',
        cardCategories='["ProduceCardCategory_ActiveSkill"]',
        planType='ProducePlanType_Unknown', cardStatusType='ProduceCardSearchStatusType_Unknown',
        orderType='ProduceCardOrderType_Unknown', staminaMinMaxType='ConditionMinMaxType_Unknown',
        examEffectType='ProduceExamEffectType_Unknown', costType='ExamCostType_Unknown')
    for key in ('cardRarities', 'produceCardIds', 'upgradeCounts', 'effectGroupIds'):
        row[key] = '[]'
    for key in ('cardSearchTag', 'produceCardRandomPoolId', 'produceCardPoolId'):
        row[key] = ''
    for key in ('limitCount', 'staminaMin', 'staminaMax', 'isSelf', 'isCustomized'):
        row[key] = 0
    return row


def _setup(monkeypatch, kinds):
    effect = ProduceExamEffect.from_row(ProduceExamEffectRow.model_validate(dict(
        id='growth', produceCardSearchId='target', produceCardGrowEffectIds='["a", "b"]',
        pickRangeType='ProducePickRangeType_All', pickCountMin=0, pickCountMax=0)))
    search = _search()
    monkeypatch.setattr(card_growth, 'load_exam_effects', lambda ids: {'growth': effect})
    monkeypatch.setattr(card_growth, 'select', lambda *_: search)
    monkeypatch.setattr(card_growth, 'select_many', lambda *_: [
        dict(id=key, effectType='ProduceCardGrowEffectType_' + kind, value=value)
        for key, kind, value in kinds])
    card_growth.get_card_growth_rule.cache_clear()
    return effect, search


@pytest.mark.parametrize('kinds,green,red,known,output_changed', [
    ([('a', 'LessonAdd', 14), ('b', 'CostAdd', 1)], (1,), (), True, True),
    ([('a', 'CostReduce', 2), ('b', 'CostPenetrateAdd', 1)], (-2,), (1,), True, False),
    ([('a', 'CostReviewAdd', 2), ('b', 'LessonAdd', 4)], (), (), False, True),
    ([('a', 'CostAdd', 1)], (1,), (), False, True),
])
def test_growth_master_data_distinguishes_stamina_special_fees_and_missing_rows(
        monkeypatch, kinds, green, red, known, output_changed):
    _setup(monkeypatch, kinds)
    try:
        rule = card_growth.get_card_growth_rule('growth')
        assert rule.green_changes == green and rule.red_changes == red
        assert rule.costs_known is known and rule.output_changed is output_changed
        assert rule.matches(cast(Any, SimpleNamespace(category='ProduceCardCategory_ActiveSkill'))) is True
        assert rule.matches(cast(Any, SimpleNamespace(category='ProduceCardCategory_MentalSkill'))) is False
        assert rule.matches(cast(Any, SimpleNamespace(category='ProduceCardCategory_Unknown'))) is None
    finally:
        card_growth.get_card_growth_rule.cache_clear()


@pytest.mark.parametrize('key,value', [
    ('effectGroupIds', '["group"]'), ('cardPositionType', 'ProduceCardPositionType_Hold'),
    ('isSelf', 1), ('limitCount', 1),
])
def test_unsupported_target_filter_is_unknown_not_all_cards(monkeypatch, key, value):
    _, search = _setup(monkeypatch, [('a', 'CostAdd', 1), ('b', 'LessonAdd', 3)])
    search[key] = value
    try:
        rule = card_growth.get_card_growth_rule('growth')
        assert rule.matches(cast(Any, SimpleNamespace(category='ProduceCardCategory_ActiveSkill'))) is None
        assert rule.matches(cast(Any, SimpleNamespace(category='ProduceCardCategory_MentalSkill'))) is False
    finally:
        card_growth.get_card_growth_rule.cache_clear()


def test_growth_cache_uses_database_update_invalidation(monkeypatch):
    _, search = _setup(monkeypatch, [('a', 'CostAdd', 1), ('b', 'LessonAdd', 3)])
    monkeypatch.setattr(_util, 'invalidate_connections', lambda: None)
    try:
        assert card_growth.get_card_growth_rule('growth').target_known
        search['isSelf'] = 1
        assert card_growth.get_card_growth_rule('growth').target_known
        _util.clear_db_caches()
        assert not card_growth.get_card_growth_rule('growth').target_known
    finally:
        card_growth.get_card_growth_rule.cache_clear()
