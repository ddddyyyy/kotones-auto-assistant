from dataclasses import replace

import pytest

from kaa.config.idol_road import IdolRoadConfig
from kaa.idol_road.controller import IdolRoadController
from kaa.idol_road.models import Action, Observation, Scene
from kaa.idol_road.preparation import PreparationIssueCode as Code, prepare_battle
from kaa.idol_road.rules import (RuleTextLine, RuleTextRead, RuleTurnNumberEvidence, RuleViewport,
                                 StageRules, VisibleStageRule, confirmed_rule_turn, confirmed_rule_turn_with_number)


def reads(text='3ターン目', confidence=.99):
    return tuple(RuleTextRead('frame', kind, (RuleTextLine(text, confidence),))
                 for kind in ('original', 'grayscale'))


def viewport(key='a'):
    return RuleViewport(key, (VisibleStageRule((40, 244, 635, 185), 3, reads(), ()),))


def formation(**kwargs):
    return replace(Observation(Scene.CONFIRM, 'ttmr', 16, stars=0, formation_ready=True,
                               support_available=True), **kwargs)


def controller(mode='preview'):
    return IdolRoadController(IdolRoadConfig(characters=['ttmr'], mode=mode))


def test_confirmed_turn_requires_full_same_frame_agreement():
    assert confirmed_rule_turn(reads('３ ターン目')) == 3
    assert confirmed_rule_turn(reads('13ターン目')) == 13


@pytest.mark.parametrize('text', ['0ターン目', '第3ターン目', '3ターン目以降', '3', 'スコア3ターン目'])
def test_partial_or_unsupported_turn_text_is_unknown(text):
    assert confirmed_rule_turn(reads(text)) is None


@pytest.mark.parametrize('confidence', [.97, float('nan'), 1.1])
def test_low_or_invalid_confidence_is_unknown(confidence):
    assert confirmed_rule_turn(reads(confidence=confidence)) is None


def test_conflicting_or_different_frame_and_missing_reads_are_unknown():
    left, right = reads()
    for invalid in ((), (left,), (left, left),
                    (left, replace(right, frame_id='other')),
                    (left, replace(right, lines=(RuleTextLine('5ターン目', .99),))),
                    (left, replace(right, lines=()))):
        assert confirmed_rule_turn(invalid) is None


def number_evidence(text='3', **kwargs):
    return replace(RuleTurnNumberEvidence('frame', reads(text), .99, (31, 20, 76, 25)), **kwargs)


def test_same_frame_number_and_suffix_recover_low_confidence_mixed_size_title():
    assert confirmed_rule_turn_with_number(reads('ミターン目', .81), number_evidence()) == 3
    assert confirmed_rule_turn_with_number(reads('13ターン目', .90), number_evidence('13')) == 13


@pytest.mark.parametrize('text', ['5ターン目', '第3ターン目', 'スコア3'])
def test_high_confidence_title_conflict_is_not_overridden_by_digits(text):
    assert confirmed_rule_turn_with_number(reads(text), number_evidence()) is None


def test_one_high_confidence_conflict_cannot_be_hidden_by_low_confidence_other_read():
    left, right = reads('5ターン目')
    assert confirmed_rule_turn_with_number((left, replace(right, lines=(RuleTextLine('3ターン目', .8),))), number_evidence()) is None


@pytest.mark.parametrize('change', [
    {'suffix_score': .96}, {'suffix_score': float('nan')}, {'frame_id': 'other'},
    {'reads': reads('3', .97)}, {'reads': reads('0')}, {'reads': reads('3x')},
    {'reads': ()}, {'reads': (reads('3')[0], reads('5')[1])},
])
def test_unconfirmed_supplement_does_not_recover_unknown_title(change):
    assert confirmed_rule_turn_with_number(reads('3ターン目', .9), number_evidence(**change)) is None


def test_confirmed_full_title_still_works_without_supplement():
    assert confirmed_rule_turn_with_number(reads(), None) == 3


def test_weak_template_candidate_needs_high_confidence_same_frame_suffix_text():
    evidence = number_evidence(suffix_score=.95, suffix_reads=reads('ターン目'))
    assert confirmed_rule_turn_with_number(reads('ミターン目', .8), evidence) == 3
    assert confirmed_rule_turn_with_number(reads('ミターン目', .8), replace(evidence, suffix_reads=reads('ターン目', .97))) is None
    assert confirmed_rule_turn_with_number(reads('ミターン目', .8), replace(evidence, suffix_reads=reads('ターン後'))) is None


def test_high_confidence_suffix_conflict_blocks_even_good_template_match():
    assert confirmed_rule_turn_with_number(reads(), number_evidence(suffix_reads=reads('ターン後'))) is None


def test_bounded_support_flow_preserves_observations_and_returns_to_preview():
    c = controller()
    assert c.step(formation()).action == Action.OPEN_SUPPORT
    for key in ('a','b','c'):
        assert c.step(Observation(Scene.SUPPORT_DETAILS,'ttmr',16,rule_viewport=viewport(key))).action==Action.SCROLL_SUPPORT
    repeated=Observation(Scene.SUPPORT_DETAILS,'ttmr',16,rule_viewport=viewport('c'))
    assert c.step(repeated).action==Action.WAIT
    assert c.step(repeated).action==Action.CLOSE_SUPPORT
    rules = c.stage_rules
    assert rules is not None and len(rules.viewports) == 3
    assert not rules.complete and not rules.semantics_confirmed
    assert rules.identity_source == 'opened_from_confirmed_formation'
    assert c.step(formation()).action == Action.READY
    assert c.battles_started == 0


def test_stale_support_frame_waits_then_closes_without_completeness_claim():
    c = controller()
    c.step(formation())
    observation = Observation(Scene.SUPPORT_DETAILS, 'ttmr', 16, rule_viewport=viewport())
    assert c.step(observation).action == Action.SCROLL_SUPPORT
    assert c.step(observation).action == Action.WAIT
    assert c.step(observation).reason == 'support_scroll_not_confirmed'
    assert c.stage_rules is not None and not c.stage_rules.complete


def test_support_scan_reaches_later_turns_but_remains_finite_and_partial():
    c=controller()
    c.step(formation())
    for index in range(6):
        decision=c.step(Observation(Scene.SUPPORT_DETAILS,'ttmr',16,rule_viewport=viewport(str(index))))
        assert decision.action==Action.SCROLL_SUPPORT
    last=c.step(Observation(Scene.SUPPORT_DETAILS,'ttmr',16,rule_viewport=viewport('last')))
    assert last.action==Action.CLOSE_SUPPORT and last.reason=='support_scroll_limit'
    assert len(c.stage_rules.viewports)==7 and not c.stage_rules.complete


def test_unknown_support_viewport_closes_and_is_not_an_empty_rule_set():
    c = controller()
    c.step(formation())
    assert c.step(Observation(Scene.SUPPORT_DETAILS, 'ttmr', 16)).reason == 'support_viewport_unknown'
    assert c.stage_rules is not None and c.stage_rules.viewports == ()
    assert not c.stage_rules.complete and not c.stage_rules.semantics_confirmed


@pytest.mark.parametrize('character,stage', [('hski', 16), ('ttmr', 13), ('ttmr', None)])
def test_rule_identity_conflict_stops_without_recording(character, stage):
    c = controller()
    c.step(formation())
    assert c.step(Observation(Scene.SUPPORT_DETAILS, character, stage, rule_viewport=viewport())).action == Action.STOP
    assert c.stage_rules is None


def test_orphan_dialog_only_closes_and_cannot_fill_configured_character():
    c = controller()
    assert c.step(Observation(Scene.SUPPORT_DETAILS)).reason == 'close_unbound_support'
    assert c.stage_rules is None


def test_owned_support_title_can_be_reobserved_without_guessing_stage_or_recording_rules():
    c=controller()
    c.step(formation())
    unknown=Observation(Scene.SUPPORT_DETAILS,'ttmr',None)
    assert c.step(unknown).reason=='await_support_stage_identity'
    assert c.stage_rules is None
    assert c.step(Observation(Scene.SUPPORT_DETAILS,'ttmr',16,rule_viewport=viewport())).action==Action.SCROLL_SUPPORT
    assert c.stage_rules is not None


def test_owned_support_title_wait_has_a_finite_bound():
    c=controller()
    c.step(formation())
    unknown=Observation(Scene.SUPPORT_DETAILS,'ttmr',None)
    assert c.step(unknown).action==Action.WAIT
    assert c.step(unknown).action==Action.WAIT
    assert c.step(unknown).reason=='stage_not_confirmed'
    assert c.stage_rules is None and c.battles_started==0


@pytest.mark.parametrize('stars', [1, 2, 3])
def test_cleared_stage_skips_before_opening_support(stars):
    assert controller().step(formation(stars=stars)).action == Action.BACK_TO_STAGES


def test_manual_mode_still_starts_after_report_only_preparation():
    c = controller('manual')
    c.step(formation())
    c.step(Observation(Scene.SUPPORT_DETAILS, 'ttmr', 16))
    assert c.step(formation()).action == Action.START_BATTLE


@pytest.mark.parametrize('change,expected', [({}, Code.RULES_INCOMPLETE),
    ({'character_id': 'hski'}, Code.RULES_IDENTITY), ({'stage': 13}, Code.RULES_IDENTITY),
    ({'source': 'guess'}, Code.RULES_SOURCE), ({'identity_source': 'history'}, Code.RULES_SOURCE)])
def test_observed_rules_do_not_imply_supported_effects_or_automatic_readiness(change, expected):
    rules = replace(StageRules('ttmr', 16, (viewport(),)), **change)
    report = prepare_battle('ttmr', 16, formation(), stage_rules=rules)
    assert expected in {row.code for row in report.issues}
    assert Code.RULES_UNKNOWN in {row.code for row in report.issues}
    assert report.automatic_battle_ready is False
