import json
from dataclasses import replace

import pytest

from kaa.idol_road.items import ItemCandidate, ItemNameRead, bind_item_reference
from kaa.idol_road.item_description import (
    ItemDescriptionCandidate, ItemDescriptionLine, ItemDescriptionRead, bind_item_description,
)


def scenario():
    reference = bind_item_reference(tuple(ItemNameRead('frame', kind, '同名', .999)
        for kind in ('original', 'grayscale')), (ItemCandidate('six', '同名'), ItemCandidate('eight', '同名')))
    candidates = tuple(ItemDescriptionCandidate(identity, '同名', json.dumps([{'text': text}]))
        for identity, text in (('six', '条件、体力回復<nobr>6</nobr>'), ('eight', '条件、体力回復<nobr>8</nobr>')))
    reads = tuple(ItemDescriptionRead('frame', kind, (ItemDescriptionLine('条件、', .999),
        ItemDescriptionLine('体力回復6', .999))) for kind in ('original', 'grayscale'))
    return reference, candidates, reads


def test_full_description_disambiguates_without_claiming_effects_or_changing_name_reference():
    reference, candidates, reads = scenario()
    result = bind_item_description(reference, reads, candidates)
    assert result.status == 'resolved' and result.item_id == 'six'
    assert result.reference.item_id is None and result.reference.candidate_ids == ('six', 'eight')
    assert result.reads == reads and result.scope == 'description_identity_only'


@pytest.mark.parametrize('confidence', [.979, None, True, float('nan'), float('inf')])
def test_low_or_invalid_number_confidence_stays_unknown(confidence):
    reference, candidates, reads = scenario()
    bad = replace(reads[1], lines=(ItemDescriptionLine('条件、体力回復6', confidence),))
    assert bind_item_description(reference, (reads[0], bad), candidates).item_id is None


def test_cross_frame_or_conflicting_number_is_not_repaired_from_candidates():
    reference, candidates, reads = scenario()
    for bad in (replace(reads[1], frame_id='other'),
                replace(reads[1], lines=(ItemDescriptionLine('条件、体力回復8', .999),))):
        assert bind_item_description(reference, (reads[0], bad), candidates).status == 'unknown_description'


def test_missing_condition_or_wrong_digit_never_matches_a_substring():
    reference, candidates, reads = scenario()
    for text in ('体力回復6', '条件、体力回復ć', '条件、体力回復'):
        partial = tuple(replace(read, lines=(ItemDescriptionLine(text, .999),)) for read in reads)
        assert bind_item_description(reference, partial, candidates).status == 'no_exact_match'


def test_missing_or_malformed_alternative_prevents_a_false_unique_match():
    reference, candidates, reads = scenario()
    for bad in (candidates[:1], tuple(reversed(candidates)),
                (candidates[0], replace(candidates[1], description_json=None)),
                (candidates[0], replace(candidates[1], description_json='[{"text":"<unknown>8"}]'))):
        assert bind_item_description(reference, reads, bad).status == 'candidate_conflict'


def test_identical_descriptions_remain_ambiguous():
    reference, candidates, reads = scenario()
    candidates = (candidates[0], replace(candidates[1], description_json=candidates[0].description_json))
    result = bind_item_description(reference, reads, candidates)
    assert result.status == 'ambiguous' and result.candidate_ids == ('six', 'eight')
