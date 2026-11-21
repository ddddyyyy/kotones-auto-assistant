import pytest

from kaa.idol_road.items import (
    ItemBindingStatus, ItemCandidate, ItemNameRead, bind_item_reference, confirmed_item_name,
)


def reads(text='星のリトルプリンス', confidence=0.99, second_text=None):
    return (
        ItemNameRead('same-frame', 'original', text, confidence),
        ItemNameRead('same-frame', 'grayscale', text if second_text is None else second_text, 0.99),
    )


def test_unique_exact_candidate_is_an_identity_reference_not_effect_evidence():
    result = bind_item_reference(reads(), (ItemCandidate('item-id', '星のリトルプリンス'),))
    assert result.status == ItemBindingStatus.RESOLVED and result.item_id == 'item-id'
    assert result.name_reads == reads()
    assert not hasattr(result, 'remaining_uses') and not hasattr(result, 'effects_active')


@pytest.mark.parametrize('confidence', [None, 0.979, -1, 1.01, float('nan'), float('inf'), True])
def test_unknown_or_invalid_confidence_never_binds_even_with_unique_candidate(confidence):
    result = bind_item_reference(reads(confidence=confidence), (ItemCandidate('id', '星のリトルプリンス'),))
    assert result.status == ItemBindingStatus.UNKNOWN_NAME and result.item_id is None


@pytest.mark.parametrize('samples', [
    (), (ItemNameRead('one', 'original', '道具', 1),),
    (ItemNameRead('one', 'original', '道具', 1), ItemNameRead('two', 'grayscale', '道具', 1)),
    (ItemNameRead('', 'original', '道具', 1), ItemNameRead('', 'grayscale', '道具', 1)),
    (ItemNameRead('one', 'original', '道具', 1), ItemNameRead('one', 'original', '道具', 1)),
])
def test_missing_or_cross_frame_representations_remain_unknown(samples):
    assert confirmed_item_name(samples) is None


def test_high_confidence_kana_conflict_is_not_corrected_from_database():
    result = bind_item_reference(reads('夢へのスタートライン', second_text='夢ヘのスタートライン'),
                                 (ItemCandidate('correct-id', '夢へのスタートライン'),))
    assert result.status == ItemBindingStatus.UNKNOWN_NAME and result.item_id is None


def test_consistently_wrong_kana_does_not_match_manual_candidate():
    result = bind_item_reference(reads('夢ヘのスタートライン'),
                                 (ItemCandidate('correct-id', '夢へのスタートライン'),))
    assert result.status == ItemBindingStatus.CANDIDATE_CONFLICT and result.item_id is None


def test_whitespace_and_compatibility_forms_are_normalized_not_fuzzy_matched():
    assert confirmed_item_name(reads(' 道具 Ａ ', second_text='道具A')) == '道具A'
    assert bind_item_reference(reads(' 道具 Ａ ', second_text='道具A'),
                               (ItemCandidate('id', '道具Ａ'),)).item_id == 'id'
    assert bind_item_reference(reads('道具'), (ItemCandidate('id', '道具＋'),)).item_id is None


@pytest.mark.parametrize('text', ['', '  ', None])
def test_empty_name_never_binds(text):
    assert confirmed_item_name(reads(text)) is None


def test_duplicate_names_preserve_all_variants_without_first_candidate_selection():
    result = bind_item_reference(reads('同名'), (ItemCandidate('basic', '同名'), ItemCandidate('upgraded', '同名')))
    assert result.status == ItemBindingStatus.AMBIGUOUS
    assert result.candidate_ids == ('basic', 'upgraded') and result.item_id is None


def test_same_id_repetition_is_not_silently_deduplicated():
    result = bind_item_reference(reads('同名'), (ItemCandidate('same', '同名'), ItemCandidate('same', '同名')))
    assert result.status == ItemBindingStatus.AMBIGUOUS and result.item_id is None


@pytest.mark.parametrize('candidates,status', [
    (None, ItemBindingStatus.DATABASE_UNAVAILABLE),
    ((), ItemBindingStatus.NO_EXACT_MATCH),
    ((ItemCandidate('', '星のリトルプリンス'),), ItemBindingStatus.CANDIDATE_CONFLICT),
])
def test_unavailable_missing_and_invalid_candidates_are_distinct(candidates, status):
    result = bind_item_reference(reads(), candidates)
    assert result.status == status and result.item_id is None
