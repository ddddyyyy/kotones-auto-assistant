import sqlite3

import pytest

from kaa.db import tower_rules as rules
from kaa.db._util import clear_db_caches


@pytest.fixture
def database(monkeypatch):
    connection = sqlite3.connect(':memory:')
    connection.row_factory = sqlite3.Row
    connection.executescript('''
        CREATE TABLE ProduceItem (id TEXT PRIMARY KEY, name TEXT, fireLimit INTEGER,
            fireInterval INTEGER, produceTriggerId TEXT, produceTriggerIds TEXT,
            produceItemEffectIds TEXT, effectGroupIds TEXT, produceDescriptions TEXT);
        CREATE TABLE ProduceItemEffect (id TEXT PRIMARY KEY, effectType TEXT,
            effectTurn INTEGER, effectCount INTEGER, produceEffectId TEXT, produceExamStatusEnchantId TEXT);
        CREATE TABLE ProduceExamStatusEnchant (id TEXT PRIMARY KEY, produceExamTriggerId TEXT,
            produceExamEffectIds TEXT, produceDescriptions TEXT);
        CREATE TABLE ProduceExamGimmickEffectGroup (id TEXT, priority INTEGER, startTurn INTEGER,
            remainingTurn INTEGER, remainingTurnPermil INTEGER, fieldStatusType TEXT,
            fieldStatusValue INTEGER, fieldStatusCheckType TEXT, produceExamEffectId TEXT,
            fieldStatusProduceCardSearchId TEXT, isPositive INTEGER, produceDescriptions TEXT);
        INSERT INTO ProduceItem VALUES ('item-b','同名道具',0,0,'','[]','["effect"]','[]','[]');
        INSERT INTO ProduceItem VALUES ('item-a','同名道具',0,0,'','[]','["effect"]','[]','[]');
        INSERT INTO ProduceItemEffect VALUES ('effect','enchant',-1,1,'','enchant-id');
        INSERT INTO ProduceExamStatusEnchant VALUES ('enchant-id','trigger','["exam-effect"]', '[]');
        INSERT INTO ProduceExamGimmickEffectGroup VALUES ('explicit-group',7,7,0,0,'unknown',0,'unknown','effect-7','',0,'[]');
        INSERT INTO ProduceExamGimmickEffectGroup VALUES ('explicit-group',3,3,0,0,'concentration',3,'at_least','effect-3','',1,'[]');
    ''')
    monkeypatch.setattr(rules, 'select_many', lambda query, *keys: connection.execute(query, keys).fetchall())
    clear_db_caches()
    yield connection
    clear_db_caches()
    connection.close()


def test_name_query_preserves_ambiguity_and_is_exact(database):
    items = rules.produce_items_by_name('同名道具')
    assert items is not None and [item.id for item in items] == ['item-a', 'item-b']
    assert rules.produce_items_by_name('同名') == ()
    assert rules.produce_items_by_name("' OR 1=1 --") == ()


def test_name_snapshot_is_complete_minimal_and_cache_invalidated(database):
    names = rules.produce_item_name_candidates()
    assert names is not None and [(row.id, row.name) for row in names] == [('item-a', '同名道具'), ('item-b', '同名道具')]
    assert not hasattr(names[0], 'description_json')
    database.execute("UPDATE ProduceItem SET name='更新' WHERE id='item-a'")
    assert rules.produce_item_name_candidates() == names
    clear_db_caches()
    assert rules.produce_item_name_candidates()[0].name == '更新'


def test_invalid_entry_does_not_produce_partial_name_snapshot(database):
    database.execute("UPDATE ProduceItem SET id=NULL WHERE id='item-a'")
    assert rules.produce_item_name_candidates() is None


def test_name_snapshot_source_unavailable(database, monkeypatch):
    def unavailable(*args):
        raise sqlite3.OperationalError('no such table')
    monkeypatch.setattr(rules, 'select_many', unavailable)
    assert rules.produce_item_name_candidates() is None


def test_item_effect_enchant_chain_uses_explicit_references_not_id_patterns(database):
    item = rules.produce_item_by_id('item-a')
    assert item is not None and item.fire_limit == 0 and item.item_effect_ids == ('effect',)
    effect = rules.produce_item_effect(item.item_effect_ids[0])
    assert effect is not None and effect.status_enchant_id == 'enchant-id'
    assert effect.effect_turn == -1 and effect.effect_count == 1
    enchant = rules.produce_status_enchant(effect.status_enchant_id)
    assert enchant is not None and enchant.trigger_id == 'trigger'
    assert enchant.exam_effect_ids == ('exam-effect',)
    # Raw 0/-1 are retained; no remaining-uses or unlimited-use inference exists.
    assert not hasattr(item, 'remaining_uses')


@pytest.mark.parametrize('raw,expected', [
    ('[]', ()), ('["x","x","y"]', ('x', 'x', 'y')),
    (None, None), ('', None), ('{bad', None), ('{}', None),
    ('[null]', None), ('[1]', None), ('[true]', None), ('[""]', None),
])
def test_reference_lists_preserve_unknown_empty_order_and_duplicates(database, raw, expected):
    database.execute('UPDATE ProduceItem SET produceItemEffectIds=? WHERE id=?', (raw, 'item-a'))
    item = rules.produce_item_by_id('item-a')
    assert item is not None and item.item_effect_ids == expected
    assert item.trigger_ids == ()  # Other independently valid fields remain intact.


def test_gimmick_group_uses_explicit_id_and_keeps_trigger_fields(database):
    group = rules.gimmick_group('explicit-group')
    assert group is not None and [row.priority for row in group] == [3, 7]
    assert group[0].start_turn == 3 and group[0].status_value == 3
    assert group[0].check_type == 'at_least' and group[0].is_positive is True
    assert group[1].is_positive is False
    assert rules.gimmick_group('tower_001-hski-stage_013') is None


def test_description_candidates_are_all_modes_and_cache_invalidates(database):
    result = rules.gimmick_descriptions_at_turn(3)
    assert result is not None and len(result) == 1 and result[0].group_id == 'explicit-group'
    database.execute("UPDATE ProduceExamGimmickEffectGroup SET produceExamEffectId='new' WHERE startTurn=3")
    assert rules.gimmick_descriptions_at_turn(3) == result
    clear_db_caches()
    assert rules.gimmick_descriptions_at_turn(3)[0].exam_effect_id == 'new'
    assert rules.gimmick_descriptions_at_turn(99) == ()


def test_exam_definition_retains_unknown_columns_and_cache_refreshes(database):
    import json
    database.executescript('''CREATE TABLE ProduceExamEffect (id TEXT, futureMechanism INTEGER);
        INSERT INTO ProduceExamEffect VALUES ('effect',7);''')
    record = rules.exam_rule_definition('effect', 'effect')
    assert record is not None and json.loads(record.fields_json)['futureMechanism'] == 7
    database.execute('UPDATE ProduceExamEffect SET futureMechanism=8')
    assert rules.exam_rule_definition('effect', 'effect') == record
    clear_db_caches()
    assert json.loads(rules.exam_rule_definition('effect', 'effect').fields_json)['futureMechanism'] == 8
    database.execute("INSERT INTO ProduceExamEffect VALUES ('effect',9)")
    clear_db_caches()
    assert rules.exam_rule_definition('effect', 'effect') is None


def test_missing_exam_definition_source_stays_unknown(database):
    assert rules.exam_rule_definition('trigger', 'missing') is None


@pytest.mark.parametrize('turn', [0, -1, True])
def test_invalid_candidate_turn_does_not_query_a_fake_stage(database, turn):
    assert rules.gimmick_descriptions_at_turn(turn) is None


def test_broken_candidate_identity_is_not_silently_removed(database):
    database.execute('UPDATE ProduceExamGimmickEffectGroup SET priority=NULL WHERE startTurn=3')
    assert rules.gimmick_descriptions_at_turn(3) is None


def test_invalid_polarity_and_numeric_data_stay_unknown(database):
    database.execute('UPDATE ProduceExamGimmickEffectGroup SET isPositive=2,startTurn=?', ('invalid',))
    group = rules.gimmick_group('explicit-group')
    assert group is not None and all(row.start_turn is None and row.is_positive is None for row in group)


@pytest.mark.parametrize('name', [None, '', '  '])
def test_item_with_invalid_essential_identity_stays_unknown(database, name):
    database.execute('UPDATE ProduceItem SET name=? WHERE id=?', (name, 'item-a'))
    assert rules.produce_item_by_id('item-a') is None


def test_invalid_same_name_candidate_is_not_silently_filtered(database):
    database.execute("UPDATE ProduceItem SET id=NULL WHERE id='item-a'")
    assert rules.produce_items_by_name('同名道具') is None


@pytest.mark.parametrize('lookup,key', [
    (rules.produce_item_by_id, 'missing'), (rules.produce_item_effect, 'missing'),
    (rules.produce_status_enchant, 'missing'), (rules.gimmick_group, 'missing'),
])
def test_missing_reference_is_not_fake_empty_effect(lookup, key, database):
    assert lookup(key) is None


def test_cache_invalidation_refreshes_all_views(database):
    assert rules.produce_item_by_id('item-a').name == '同名道具'
    rules.produce_items_by_name('同名道具')
    rules.produce_item_effect('effect')
    rules.produce_status_enchant('enchant-id')
    rules.gimmick_group('explicit-group')
    database.execute("UPDATE ProduceItem SET name='更新' WHERE id='item-a'")
    database.execute('UPDATE ProduceItemEffect SET effectCount=2')
    database.execute("UPDATE ProduceExamStatusEnchant SET produceExamTriggerId='new-trigger'")
    database.execute('UPDATE ProduceExamGimmickEffectGroup SET startTurn=9')
    assert rules.produce_item_by_id('item-a').name == '同名道具'
    clear_db_caches()
    assert rules.produce_item_by_id('item-a').name == '更新'
    assert [item.id for item in rules.produce_items_by_name('同名道具')] == ['item-b']
    assert rules.produce_item_effect('effect').effect_count == 2
    assert rules.produce_status_enchant('enchant-id').trigger_id == 'new-trigger'
    assert rules.gimmick_group('explicit-group')[0].start_turn == 9


@pytest.mark.parametrize('lookup,key', [
    (rules.produce_item_by_id, 'item-a'), (rules.produce_items_by_name, '同名道具'),
    (rules.produce_item_effect, 'effect'), (rules.produce_status_enchant, 'enchant-id'),
    (rules.gimmick_group, 'explicit-group'),
])
def test_database_unavailable_stays_unknown(lookup, key, database, monkeypatch):
    def unavailable(*args):
        raise sqlite3.OperationalError('no such table')
    monkeypatch.setattr(rules, 'select_many', unavailable)
    assert lookup(key) is None
