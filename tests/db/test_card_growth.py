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



def test_master_lesson_add_and_cost_add_have_supported_output(monkeypatch):
    _setup(monkeypatch, [('a', 'LessonAdd', 14), ('b', 'CostAdd', 1)])
    try:
        rule = card_growth.get_card_growth_rule('growth')
        assert rule.lesson_changes == (14,) and rule.output_known
        assert rule.green_changes == (1,) and rule.costs_known
    finally:
        card_growth.get_card_growth_rule.cache_clear()


def test_targeted_lesson_growth_keeps_specific_effect_filter_unknown(monkeypatch):
    _setup(monkeypatch, [('a', 'LessonAdd', 14), ('b', 'CostAdd', 1)])
    monkeypatch.setattr(card_growth, 'select_many', lambda *_: [
        dict(id='a', effectType='ProduceCardGrowEffectType_LessonAdd', value=14,
             targetPlayProduceExamEffectIds='["specific-effect"]'),
        dict(id='b', effectType='ProduceCardGrowEffectType_CostAdd', value=1),
    ])
    try:
        rule = card_growth.get_card_growth_rule('growth')
        assert not rule.lesson_changes and not rule.output_known
        assert rule.costs_known and rule.green_changes == (1,)
    finally:
        card_growth.get_card_growth_rule.cache_clear()



@pytest.mark.parametrize('raw,expected', [
    ('["group", "other"]', True), ('["other"]', False), ('[]', False),
    (None, None), ('', None), ('invalid-json', None), ('{}', None), ('[1]', None),
])
def test_single_effect_group_uses_declared_card_inventory(monkeypatch, raw, expected):
    _, search = _setup(monkeypatch, [('a', 'CostAdd', 7), ('b', 'LessonAdd', 37)])
    search['effectGroupIds'] = '["group"]'
    try:
        rule = card_growth.get_card_growth_rule('growth')
        assert rule.target_known and rule.effect_groups == ('group',)
        card = cast(Any, SimpleNamespace(category='ProduceCardCategory_ActiveSkill', _effect_group_ids=raw))
        assert rule.matches(card) is expected
    finally:
        card_growth.get_card_growth_rule.cache_clear()


@pytest.mark.parametrize('raw', ['["a", "b"]', 'invalid-json', '{}'])
def test_multiple_or_malformed_search_groups_remain_unknown(monkeypatch, raw):
    _, search = _setup(monkeypatch, [('a', 'CostAdd', 7), ('b', 'LessonAdd', 37)])
    search['effectGroupIds'] = raw
    try:
        rule = card_growth.get_card_growth_rule('growth')
        assert not rule.target_known
        card = cast(Any, SimpleNamespace(category='ProduceCardCategory_ActiveSkill', _effect_group_ids='["a", "b"]'))
        assert rule.matches(card) is None
    finally:
        card_growth.get_card_growth_rule.cache_clear()



@pytest.mark.parametrize('candidate,expected', [('target', True), ('other', False), ('', None)])
def test_single_card_id_growth_targets_card_type_not_unique_instance(monkeypatch, candidate, expected):
    _, search = _setup(monkeypatch, [('a', 'LessonCountAdd', 1), ('b', 'CostAdd', 4)])
    search['produceCardIds'] = '["target"]'
    try:
        rule = card_growth.get_card_growth_rule('growth')
        assert rule.target_known and rule.output_known
        assert rule.lesson_count_changes == (1,) and rule.green_changes == (4,)
        card = cast(Any, SimpleNamespace(_id=candidate, category='ProduceCardCategory_ActiveSkill'))
        assert rule.matches(card) is expected
    finally:
        card_growth.get_card_growth_rule.cache_clear()


def test_count_reduction_is_parsed_without_assuming_runtime_minimum(monkeypatch):
    _setup(monkeypatch, [('a', 'LessonCountReduce', 1), ('b', 'CostAdd', 1)])
    try:
        rule = card_growth.get_card_growth_rule('growth')
        assert rule.output_known and rule.lesson_count_changes == (-1,)
    finally:
        card_growth.get_card_growth_rule.cache_clear()


@pytest.mark.parametrize('raw', ['["a", "b"]', 'invalid-json'])
def test_multi_or_malformed_card_id_filter_stays_unknown(monkeypatch, raw):
    _, search = _setup(monkeypatch, [('a', 'LessonCountAdd', 1), ('b', 'CostAdd', 1)])
    search['produceCardIds'] = raw
    try:
        rule = card_growth.get_card_growth_rule('growth')
        assert rule.matches(cast(Any, SimpleNamespace(_id='a', category='ProduceCardCategory_ActiveSkill'))) is None
    finally:
        card_growth.get_card_growth_rule.cache_clear()


@pytest.mark.parametrize('value', [0, 5])
def test_block_growth_loads_with_lesson_count_without_changing_cost(monkeypatch, value):
    _setup(monkeypatch, [('a', 'BlockAdd', value), ('b', 'LessonCountAdd', 2)])
    try:
        rule = card_growth.get_card_growth_rule('growth')
        assert rule.block_changes == (value,) and rule.lesson_count_changes == (2,)
        assert rule.output_known and rule.costs_known
    finally:
        card_growth.get_card_growth_rule.cache_clear()


@pytest.mark.parametrize('kind,field', [('ReviewAdd', 'review_changes'), ('FullPowerPointAdd', 'full_power_changes')])
def test_resource_growth_master_data_has_known_output_and_payment(monkeypatch, kind, field):
    _setup(monkeypatch, [('a', kind, 2), ('b', 'CostPenetrateAdd', 2)])
    try:
        rule = card_growth.get_card_growth_rule('growth')
        assert getattr(rule, field) == (2,)
        assert rule.output_known and rule.costs_known and rule.red_changes == (2,)
    finally:
        card_growth.get_card_growth_rule.cache_clear()


@pytest.mark.parametrize('value', [-1, -3])
def test_negative_count_reduction_metadata_stays_unknown(monkeypatch, value):
    _setup(monkeypatch, [('a', 'LessonCountReduce', value), ('b', 'CostAdd', 1)])
    try:
        rule = card_growth.get_card_growth_rule('growth')
        assert not rule.output_known and not rule.lesson_count_changes
    finally:
        card_growth.get_card_growth_rule.cache_clear()
