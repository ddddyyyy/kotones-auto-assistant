from dataclasses import replace

import pytest

from kaa.idol_road.conditional_rule import ConditionalPlayDeclaration
from kaa.idol_road.health_rule import HealthMentalBlockDeclaration
from kaa.idol_road.rule_catalog import RuleDeclarationCatalog, RuleDeclarationSighting, UnknownRuleSighting
from kaa.idol_road.rule_conditions import ConditionVerdict as Verdict, RuleConditionObservation, evaluate_rule_conditions


def sighting(kind='concentration_extra_play', declaration=None):
    declaration = declaration or ConditionalPlayDeclaration(3, 3, 1)
    return RuleDeclarationSighting(0, 0, kind, declaration)


def catalog(*rows):
    return RuleDeclarationCatalog('ttmr', 16, 'partial', rows or (sighting(),),
                                   (UnknownRuleSighting(0, 1, 'unsupported_visible_rule'),))


def observation(**changes):
    return replace(RuleConditionObservation('ttmr', 16, 'battle-a', 'current-frame', 3,
                     concentration=3, hp=10, max_hp=50, source='confirmed_current_battle_observation'), **changes)


def check(state=None, rules=None):
    return evaluate_rule_conditions(rules or catalog(), state or observation(), battle_id='battle-a')


@pytest.mark.parametrize('value,verdict', [(0, Verdict.NOT_SATISFIED), (2, Verdict.NOT_SATISFIED),
    (3, Verdict.SATISFIED), (9, Verdict.SATISFIED), (None, Verdict.UNKNOWN),
    (-1, Verdict.UNKNOWN), (True, Verdict.UNKNOWN)])
def test_concentration_threshold_is_inclusive_and_unknown_values_are_not_zero(value, verdict):
    assert check(observation(concentration=value)).checks[0].verdict == verdict


@pytest.mark.parametrize('hp,max_hp,verdict', [(10, 50, Verdict.SATISFIED), (11, 50, Verdict.NOT_SATISFIED),
    (0, 50, Verdict.SATISFIED), (10, 49, Verdict.NOT_SATISFIED), (1, 6, Verdict.SATISFIED),
    (None, 50, Verdict.UNKNOWN), (0, None, Verdict.UNKNOWN), (0, 0, Verdict.UNKNOWN),
    (51, 50, Verdict.UNKNOWN), (-1, 50, Verdict.UNKNOWN), (True, 50, Verdict.UNKNOWN)])
def test_hp_ratio_uses_exact_cross_product_instead_of_rounding(hp, max_hp, verdict):
    rules = catalog(sighting('hp_mental_block', HealthMentalBlockDeclaration(7, 20)))
    result = check(observation(turn=7, hp=hp, max_hp=max_hp), rules).checks[0]
    assert result.verdict == verdict


@pytest.mark.parametrize('changes,reason', [
    ({'character_id': 'hski'}, 'identity_conflict'), ({'stage': 13}, 'identity_conflict'),
    ({'stage': True}, 'identity_conflict'), ({'battle_id': 'old-battle'}, 'battle_not_confirmed'),
    ({'frame_id': ''}, 'frame_not_confirmed'), ({'source': 'historical_snapshot'}, 'context_unconfirmed'),
    ({'turn': None}, 'turn_unknown'), ({'turn': True}, 'turn_unknown'),
    ({'conflicted_fields': ('identity',)}, 'observation_conflict'),
    ({'conflicted_fields': ('turn',)}, 'observation_conflict'),
    ({'conflicted_fields': ('concentration',)}, 'observation_conflict'),
])
def test_missing_or_conflicting_context_does_not_inherit_previous_values(changes, reason):
    result = check(observation(**changes)).checks[0]
    assert result.verdict == Verdict.UNKNOWN and result.reason == reason


@pytest.mark.parametrize('turn', [6, 8, 9])
def test_other_turn_never_infers_whether_two_turn_block_is_active_or_expired(turn):
    rules = catalog(sighting('hp_mental_block', HealthMentalBlockDeclaration(7, 20)))
    result = check(observation(turn=turn, hp=0), rules).checks[0]
    assert result.verdict == Verdict.UNKNOWN and result.reason == 'outside_declared_turn'
    assert result.timing == 'other_turn'


def test_conflict_in_hp_does_not_hide_independent_concentration_check():
    result = check(observation(conflicted_fields=('hp',)))
    assert result.checks[0].verdict == Verdict.SATISFIED
    rules = catalog(sighting('hp_mental_block', HealthMentalBlockDeclaration(7, 20)))
    assert check(observation(turn=7, conflicted_fields=('max_hp',)), rules).checks[0].verdict == Verdict.UNKNOWN


def test_duplicate_sightings_are_not_added_up_or_executed_and_input_stays_unchanged():
    first = sighting()
    rules = catalog(first, replace(first, viewport_index=1))
    state = observation()
    result = check(state, rules)
    assert len(result.checks) == 2 and all(row.verdict == Verdict.SATISFIED for row in result.checks)
    assert result.unknown_rule_sightings == rules.unknown_sightings
    assert not result.all_rules_known and not result.executable and not result.trigger_phase_confirmed
    assert result.scope == 'declaration_condition_only'
    assert state == observation() and rules.sightings == (first, replace(first, viewport_index=1))
    assert not hasattr(result, 'extra_plays') and not hasattr(result, 'blocked_categories')


def test_invalid_catalog_and_expected_battle_are_unknown():
    assert check(rules=replace(catalog(), status='source_unknown')).checks[0].reason == 'context_unconfirmed'
    assert evaluate_rule_conditions(catalog(), observation(), battle_id='').checks[0].reason == 'battle_not_confirmed'


def test_empty_catalog_still_reports_global_context_failure():
    result = check(observation(source='historical_snapshot'), replace(catalog(), sightings=()))
    assert not result.checks and result.context_issue == 'context_unconfirmed'
    assert not result.executable and not result.all_rules_known


@pytest.mark.parametrize('row', [
    sighting(declaration=ConditionalPlayDeclaration(True, 3, 1)),
    sighting(declaration=ConditionalPlayDeclaration(3, 0, 1)),
    sighting('hp_mental_block', ConditionalPlayDeclaration(3, 3, 1)),
    sighting('hp_mental_block', HealthMentalBlockDeclaration(3, 101)),
])
def test_unsupported_declaration_is_not_treated_as_a_default_condition(row):
    result = check(rules=catalog(row)).checks[0]
    assert result.verdict == Verdict.UNKNOWN and result.reason == 'unsupported_declaration'
