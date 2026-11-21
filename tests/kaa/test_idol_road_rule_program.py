from dataclasses import replace
import json

import pytest

from kaa.idol_road.rule_program import (CandidateDrawProgram, DrawDeclaration, DrawRuleInputs,
                                        interpret_draw_declaration, summarize_rule_programs)
from kaa.idol_road.rule_references import RuleEffectReference


def effect(effect_id, kind, **changes):
    data = {
        'id': effect_id, 'effectType': kind, 'effectValue1': 0, 'effectValue2': 0,
        'effectCount': 0, 'effectTurn': 0, 'targetProduceCardId': '', 'targetUpgradeCount': 0,
        'targetExamEffectType': 'ProduceExamEffectType_Unknown', 'produceCardSearchId': '',
        'movePositionType': 'ProduceCardMovePositionType_Unknown',
        'pickRangeType': 'ProducePickRangeType_Unknown', 'pickCountReferenceProduceCardSearchId': '',
        'pickCountType': 'ProducePickCountType_Unknown', 'pickCountMin': 0, 'pickCountMax': 0,
        'produceCardSearchId2': '', 'pickRangeType2': 'ProducePickRangeType_Unknown',
        'pickCountReferenceProduceCardSearchId2': '', 'pickCountType2': 'ProducePickCountType_Unknown',
        'pickCountMin2': 0, 'pickCountMax2': 0, 'chainProduceExamEffectId': '',
        'chainProduceExamEffectIds': '[]', 'produceExamStatusEnchantId': '',
        'produceCardStatusEnchantId': '', 'produceCardGrowEffectIds': '[]',
    }
    return json.dumps({**data, **changes})


def inputs(count=1):
    trigger = {'id': 'trigger', 'phaseTypes': '["ProduceExamPhaseType_StartPlay"]', 'phaseValues': '[]',
               'fieldStatusCheckTypes': '[]', 'fieldStatusTypes': '[]', 'fieldStatusValues': '[]',
               'fieldStatusProduceCardSearchIds': '[]', 'produceCardSearchId': '',
               'upperSearchCount': 0, 'lowerSearchCount': 0,
               'cardMovePositionType': 'ProduceCardMovePositionType_Unknown', 'effectTypes': '[]',
               'lessonType': 'ProduceStepLessonType_Unknown'}
    return DrawRuleInputs('root', effect('root', 'ProduceExamEffectType_ExamStatusEnchant',
                                        effectTurn=-1, produceExamStatusEnchantId='enchant'),
                          'enchant', 'trigger', json.dumps(trigger), ('draw',),
                          (effect('draw', 'ProduceExamEffectType_ExamCardDraw', effectValue1=count),))


def modify(raw, **changes):
    return json.dumps({**json.loads(raw), **changes})


def test_explicit_chain_explains_draw_without_turn_count_or_stage_execution_inference():
    declaration, reason = interpret_draw_declaration(inputs())
    assert reason == 'supported_draw_declaration'
    assert declaration == DrawDeclaration('ProduceExamPhaseType_StartPlay', 1, -1)
    assert not hasattr(declaration, 'unlimited') and not hasattr(declaration, 'remaining_uses')


@pytest.mark.parametrize('changes', [
    {'root_json': None}, {'trigger_json': None}, {'effect_ids': None}, {'effect_jsons': (None,)},
    {'root_id': 'different'}, {'enchant_id': 'different'}, {'trigger_id': 'different'},
])
def test_missing_or_wrong_explicit_references_never_create_draw_declaration(changes):
    assert interpret_draw_declaration(replace(inputs(), **changes))[0] is None


@pytest.mark.parametrize('field,value', [
    ('effectCount', 1), ('effectValue2', 2), ('produceCardSearchId', 'filter'),
    ('chainProduceExamEffectId', 'draw'), ('chainProduceExamEffectIds', '["root"]'),
    ('chainProduceExamEffectIds', None), ('produceCardGrowEffectIds', '["grow"]'),
    ('futureMechanism', 0), ('effectValue1', True), ('effectValue1', 0), ('effectValue1', -1),
])
def test_filtered_chained_counted_or_unknown_draw_mechanisms_are_not_simplified(field, value):
    data = inputs()
    data = replace(data, effect_jsons=(modify(data.effect_jsons[0], **{field: value}),))
    assert interpret_draw_declaration(data) == (None, 'unsupported_definition')


@pytest.mark.parametrize('field,value', [
    ('phaseTypes', '["ProduceExamPhaseType_ExamStartTurn"]'),
    ('phaseTypes', '["ProduceExamPhaseType_StartPlay","ProduceExamPhaseType_ExamStartTurn"]'),
    ('phaseValues', '[1]'), ('fieldStatusValues', '[3]'), ('fieldStatusCheckTypes', None),
    ('upperSearchCount', 1), ('effectTypes', '{bad'), ('futureTrigger', ''),
])
def test_trigger_conditions_or_other_phases_are_not_discarded(field, value):
    data = inputs()
    data = replace(data, trigger_json=modify(data.trigger_json, **{field: value}))
    assert interpret_draw_declaration(data) == (None, 'unsupported_definition')


def test_empty_duplicate_or_branching_effect_lists_do_not_become_one_draw():
    for ids, raws in [((), ()), (('draw', 'draw'), inputs().effect_jsons*2),
                      (('draw', 'other'), inputs().effect_jsons*2)]:
        assert interpret_draw_declaration(replace(inputs(), effect_ids=ids, effect_jsons=raws))[0] is None


def test_missing_mechanism_column_is_not_filled_with_default():
    data = inputs()
    trigger = json.loads(data.trigger_json)
    del trigger['fieldStatusValues']
    assert interpret_draw_declaration(replace(data, trigger_json=json.dumps(trigger)))[0] is None


def test_consensus_keeps_every_candidate_and_never_authorizes_execution():
    declaration = interpret_draw_declaration(inputs())[0]
    candidates = tuple(CandidateDrawProgram(RuleEffectReference(group, 1, 1, 'root'), declaration,
                                             'supported_draw_declaration') for group in ('a', 'b'))
    report = summarize_rule_programs(candidates)
    assert report.common_draw_declaration == declaration and report.candidates == candidates
    assert not report.stage_binding_confirmed and not report.executable
    assert report.candidate_scope == 'matched_plain_text_candidates_only'
    conflict = replace(candidates[1], declaration=DrawDeclaration('ProduceExamPhaseType_StartPlay', 2, -1))
    missing = replace(candidates[1], declaration=None, reason='missing_definition')
    assert summarize_rule_programs((candidates[0], conflict)).common_draw_declaration is None
    assert summarize_rule_programs((candidates[0], missing)).common_draw_declaration is None
    assert summarize_rule_programs(()).common_draw_declaration is None
