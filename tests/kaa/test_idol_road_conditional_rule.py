from dataclasses import replace

import pytest

from kaa.idol_road.conditional_rule import (ConditionalPlayEvidence, RuleIconEvidence,
                                           interpret_conditional_play)
from kaa.idol_road.rules import RuleTextLine, RuleTextRead


def reads(text='集中が3以上の場合、スキルカード使用数追加+1', confidence=.99):
    return tuple(RuleTextRead('frame', kind, (RuleTextLine(text, confidence),))
                 for kind in ('original', 'grayscale'))


def evidence(**changes):
    icons = (RuleIconEvidence('frame', 'concentration', (12, 0, 35, 35), .99),
             RuleIconEvidence('frame', 'extra_play', (290, 0, 36, 35), .99))
    return replace(ConditionalPlayEvidence('frame', icons, reads()), **changes)


@pytest.mark.parametrize('turn,threshold', [(3, 3), (5, 9), (9, 18)])
def test_same_frame_complete_text_and_both_icons_form_visible_condition(turn, threshold):
    text = reads(f'集中が{threshold}以上の場合、スキルカード使用数追加+1')
    result = interpret_conditional_play(turn, evidence(text_reads=text), text)
    assert result.status == 'confirmed_visible_declaration' and result.declaration is not None
    assert result.declaration.start_turn == turn
    assert result.declaration.concentration_at_least == threshold
    assert result.declaration.extra_plays == 1 and not result.declaration.executable


def test_positive_one_icon_and_add_one_text_are_redundant_when_ocr_omits_plus():
    observed = reads('集中が3以上の場合、スキルカード使用数追加1')
    assert interpret_conditional_play(3, evidence(text_reads=observed), observed).declaration is not None
    assert interpret_conditional_play(3, evidence(icons=(), text_reads=observed), observed).declaration is None


@pytest.mark.parametrize('text', [
    '集中が3以下の場合、スキルカード使用数追加+1',
    '集中が3以上の場合、スキルカード使用数追加-1',
    '集中が3以上の場合、スキルカード使用数追加+2',
    'スコアが3以上の場合、スキルカード使用数追加+1',
    '集中が3以上の場合、スキルカード使用数追加+1（2ターン）',
    '集中が3以上の場合、', 'スキルカード使用数追加+1',
])
def test_partial_or_different_conditions_are_not_simplified(text):
    result = interpret_conditional_play(3, evidence(text_reads=reads(text)), reads(text))
    assert result.status == 'unsupported_text' and result.declaration is None


@pytest.mark.parametrize('turn', [None, True, 0])
def test_unconfirmed_trigger_turn_cannot_form_a_declaration(turn):
    assert interpret_conditional_play(turn, evidence(), reads()).declaration is None


def test_missing_duplicate_conflicting_or_different_frame_icons_are_unknown():
    left, right = evidence().icons
    for icons in ((), (left,), (left, left), (left, right, right),
                  (replace(left, frame_id='other'), right),
                  (replace(left, score=.96), right), (replace(left, score=float('nan')), right),
                  (left, replace(right, rect=(5, 0, 36, 35))),
                  (left, replace(right, rect=(290, 40, 36, 35)))):
        assert interpret_conditional_play(3, evidence(icons=icons), reads()).status == 'icons_unknown'


@pytest.mark.parametrize('confidence', [.97, float('nan'), 1.1])
def test_weak_masked_text_is_not_repaired_with_icons_or_history(confidence):
    assert interpret_conditional_play(3, evidence(text_reads=reads(confidence=confidence)), reads()).declaration is None


def test_masked_text_conflict_and_different_frames_are_not_resolved_by_voting():
    left, right = reads()
    invalid = [(left, replace(right, frame_id='other')),
               (left, reads('集中が9以上の場合、スキルカード使用数追加+1')[1]), (left,)]
    for rows in invalid:
        assert interpret_conditional_play(3, evidence(text_reads=rows), reads()).declaration is None


@pytest.mark.parametrize('raw', ['集中が9以上の場合、', '集中が3以下の場合、', '使用数追加-1', '使用数追加+2'])
def test_high_confidence_original_conflict_cannot_be_overridden_by_masked_text(raw):
    result = interpret_conditional_play(3, evidence(), reads(raw))
    assert result.status == 'high_confidence_conflict' and result.declaration is None


def test_original_and_masked_reads_must_have_same_frame():
    left, right = reads()
    assert interpret_conditional_play(3, evidence(), (left, replace(right, frame_id='old'))).declaration is None
