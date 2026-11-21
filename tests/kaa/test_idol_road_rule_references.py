from dataclasses import replace
import json

import pytest

from kaa.idol_road.rules import RuleTextLine, RuleTextRead
from kaa.idol_road.rule_references import RuleDescriptionCandidate, match_rule_description


def reads(text='以降、ターン開始後、スキルカードを引く'):
    return tuple(RuleTextRead('frame', kind, (RuleTextLine(text, .99),)) for kind in ('original', 'grayscale'))


def candidate(group='one', **kwargs):
    raw = json.dumps([{'produceDescriptionType': 'ProduceDescriptionType_PlainText',
                       'text': '以降、ターン開始後、スキルカードを引く'}])
    return replace(RuleDescriptionCandidate(group, 1, 1, 'effect-'+group, raw), **kwargs)


def test_same_description_preserves_all_references_without_stage_association():
    result = match_rule_description(1, reads(), (candidate(), candidate('two')))
    assert result.status == 'plain_text_candidates'
    assert [row.group_id for row in result.candidates] == ['one', 'two']
    assert not result.complete and result.candidate_scope == 'plain_text_descriptions_only'
    assert not hasattr(result, 'effect_id') and not hasattr(result, 'character_id')
    assert not hasattr(result.candidates[0], 'description_json')


def test_single_match_is_still_only_a_partial_candidate():
    result = match_rule_description(1, reads(), (candidate(),))
    assert len(result.candidates) == 1 and result.complete is False


def test_wrong_turn_and_full_text_mismatch_cannot_bind():
    assert match_rule_description(3, reads(), (candidate(),)).status == 'no_plain_text_match'
    assert match_rule_description(1, reads('ターン開始後、スキルカードを引く'), (candidate(),)).candidates == ()


@pytest.mark.parametrize('raw', [None, '', '{bad', '{}', '[]', '[null]',
    '[{"produceDescriptionType":"ProduceDescriptionType_ProduceExamEffectType","text":"以降、ターン開始後、スキルカードを引く"}]',
    '[{"produceDescriptionType":"ProduceDescriptionType_PlainText","text":"<nobr>以降、ターン開始後、スキルカードを引く</nobr>"}]'])
def test_unsupported_or_broken_descriptions_do_not_become_exact_rules(raw):
    result = match_rule_description(1, reads(), (candidate(description_json=raw),))
    assert not result.candidates and result.complete is False


def test_unknown_and_unavailable_are_separate():
    assert match_rule_description(None, reads(), None).status == 'text_unknown'
    assert match_rule_description(1, reads(), None).status == 'source_unavailable'
    assert match_rule_description(1, reads(), ()).status == 'no_plain_text_match'


def test_read_conflicts_or_different_frames_cannot_use_database_to_force_text():
    left, right = reads()
    for rows in ((left, replace(right, frame_id='other')),
                 (left, replace(right, lines=(RuleTextLine('別の条件', .99),))),
                 (left, replace(right, lines=(RuleTextLine(left.lines[0].text, .97),))),
                 (left, replace(right, lines=(left.lines[0], left.lines[0])))):
        assert match_rule_description(1, rows, (candidate(),)).status == 'text_unknown'
