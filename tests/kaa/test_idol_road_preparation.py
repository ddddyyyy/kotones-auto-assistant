from dataclasses import replace

import pytest

from kaa.idol_road.items import ItemBindingStatus, ItemInventory, ItemNameRead, ItemReference, ItemViewport, VisibleItem
from kaa.idol_road.models import Observation, Scene
from kaa.idol_road.objectives import StageObjectives, StarRequirement, read_rank_requirement
from kaa.idol_road.preparation import PreparationIssueCode as Code, prepare_battle


def formation(**kwargs):
    return replace(Observation(Scene.CONFIRM, character_id='ttmr', stage=16, stars=0, formation_ready=True), **kwargs)


def objectives():
    return StageObjectives('ttmr', 16, tuple(read_rank_requirement(stars, f'{4-stars}位以上でクリア') for stars in (3, 2, 1)))


def item(*, status=ItemBindingStatus.RESOLVED, section='memory'):
    reads = (ItemNameRead('frame', 'original', '道具', .99), ItemNameRead('frame', 'grayscale', '道具', .99))
    return VisibleItem(ItemReference(reads, status, '道具', ('id',), 'id'), (27, 200, 666, 200), section)


def codes(report):
    return {issue.code for issue in report.issues}


def test_known_one_star_rank_is_goal_not_automatic_authorization():
    report = prepare_battle('ttmr', 16, formation(), objectives=objectives(), inventory=ItemInventory('ttmr', 16, (item(),)))
    assert report.formation_confirmed and report.spec is not None
    assert report.spec.mode == 'idol_road' and report.spec.target_stars == 1 and report.spec.rank_at_most == 3
    assert report.spec.rank_goal_source == 'conditions_dialog'
    assert report.spec.archetype is None and report.spec.lineup_identity is None
    assert report.automatic_battle_ready is False
    assert {Code.INVENTORY_INCOMPLETE, Code.ARCHETYPE_UNKNOWN,
            Code.RULES_UNKNOWN, Code.EFFECTS_UNSUPPORTED, Code.ADAPTER_UNAVAILABLE} <= codes(report)
    assert not {Code.LINEUP_UNKNOWN,Code.DECK_UNKNOWN} & codes(report)


@pytest.mark.parametrize('observation,expected', [
    (None, Code.FORMATION_UNKNOWN), (formation(scene=Scene.RESULT), Code.FORMATION_UNKNOWN),
    (formation(character_id='hski'), Code.IDENTITY_CONFLICT), (formation(stage=13), Code.IDENTITY_CONFLICT),
    (formation(stars=None), Code.STARS_UNKNOWN), (formation(stars=1), Code.ALREADY_CLEARED),
    (formation(stars=2), Code.ALREADY_CLEARED), (formation(stars=3), Code.ALREADY_CLEARED),
    (formation(formation_ready=False), Code.FORMATION_UNAVAILABLE),
])
def test_unconfirmed_or_cleared_formation_never_produces_battle_spec(observation, expected):
    report = prepare_battle('ttmr', 16, observation, objectives=objectives())
    assert expected in codes(report) and report.spec is None and not report.formation_confirmed
    assert report.automatic_battle_ready is False


@pytest.mark.parametrize('character,stage', [(None, 16), ('', 16), ('ttmr', None), ('ttmr', 0), ('ttmr', True)])
def test_invalid_expected_identity_cannot_be_filled_from_history(character, stage):
    report = prepare_battle(character, stage, formation(), objectives=objectives())
    assert Code.IDENTITY_CONFLICT in codes(report) and report.spec is None


@pytest.mark.parametrize('change,expected', [
    ({'character_id': 'hski'}, Code.OBJECTIVES_IDENTITY), ({'stage': 13}, Code.OBJECTIVES_IDENTITY),
    ({'source': 'guess'}, Code.OBJECTIVES_SOURCE), ({'identity_source': 'history'}, Code.OBJECTIVES_SOURCE),
    ({'requirements': ()}, Code.TARGET_UNKNOWN),
    ({'requirements': (StarRequirement(3, 'score', None), StarRequirement(2, 'rank', 2), StarRequirement(1, 'score', None))}, Code.TARGET_UNKNOWN),
])
def test_mismatching_or_unsupported_objectives_do_not_supply_rank_goal(change, expected):
    report = prepare_battle('ttmr', 16, formation(), objectives=replace(objectives(), **change))
    assert expected in codes(report) and report.spec is not None
    assert report.spec.rank_at_most is None and report.spec.rank_goal_source is None


def test_missing_goals_and_inventory_are_not_empty_or_default_rank_one():
    report = prepare_battle('ttmr', 16, formation())
    assert {Code.OBJECTIVES_UNKNOWN, Code.INVENTORY_UNKNOWN} <= codes(report)
    assert report.spec is not None and report.spec.rank_at_most is None


@pytest.mark.parametrize('change,expected', [
    ({'character_id': 'hski'}, Code.INVENTORY_IDENTITY), ({'stage': 13}, Code.INVENTORY_IDENTITY),
    ({'source': 'guess'}, Code.INVENTORY_SOURCE), ({'identity_source': 'history'}, Code.INVENTORY_SOURCE),
])
def test_inventory_identity_and_source_are_checked(change, expected):
    report = prepare_battle('ttmr', 16, formation(), inventory=replace(ItemInventory('ttmr', 16, (item(),)), **change))
    assert expected in codes(report) and report.automatic_battle_ready is False


def test_all_viewport_uncertainties_are_reported_without_counting_duplicate_sightings():
    inventory = ItemInventory('ttmr', 16, (item(),), viewports=(
        ItemViewport('a', (item(),)), ItemViewport('b', (item(status=ItemBindingStatus.AMBIGUOUS, section=None),)),
    ))
    report = prepare_battle('ttmr', 16, formation(), inventory=inventory)
    assert {Code.INVENTORY_INCOMPLETE, Code.ITEM_BINDING_UNKNOWN,
            Code.ITEM_SECTION_UNKNOWN, Code.ITEM_INSTANCES_UNKNOWN} <= codes(report)
    assert len(report.issues) == len(codes(report))
    assert inventory.complete is False and len(inventory.viewports) == 2
    assert not hasattr(report, 'unique_item_count')


def test_falsely_resolved_item_without_name_evidence_is_still_unconfirmed():
    bad = item()
    bad = replace(bad, reference=replace(bad.reference, name_reads=()))
    report = prepare_battle('ttmr', 16, formation(), inventory=ItemInventory('ttmr', 16, (bad,)))
    assert Code.ITEM_BINDING_UNKNOWN in codes(report)


@pytest.mark.parametrize('visible', [None, ()])
def test_unknown_or_empty_sightings_do_not_establish_complete_equipment(visible):
    report = prepare_battle('ttmr', 16, formation(), inventory=ItemInventory('ttmr', 16, visible))
    assert {Code.INVENTORY_UNKNOWN, Code.INVENTORY_INCOMPLETE} <= codes(report)
