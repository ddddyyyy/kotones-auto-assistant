from dataclasses import replace

import pytest

from kaa.idol_road.item_overlap import associate_item_viewports
from kaa.idol_road.items import (
    ItemBindingStatus, ItemNameRead, ItemOverlapStatus, ItemReference, ItemViewport, VisibleItem,
)


def item(item_id, y, *, section='memory', status=ItemBindingStatus.RESOLVED):
    reads = (ItemNameRead('frame', 'original', item_id, 0.99), ItemNameRead('frame', 'grayscale', item_id, 0.99))
    reference = ItemReference(reads, status, item_id, (item_id,), item_id)
    return VisibleItem(reference, (27, y, 666, 200), section)


def pair():
    return (ItemViewport('top', (item('a', 200), item('b', 412), item('c', 624))),
            ItemViewport('bottom', (item('b', 200), item('c', 412), item('d', 624))))


def test_two_distinct_unique_anchors_make_only_candidate_links():
    previous, current = pair()
    result = associate_item_viewports(previous, current)
    assert result.status == ItemOverlapStatus.CANDIDATE and result.content_shift == (0, -212)
    assert [(link.previous_index, link.current_index, link.item_id) for link in result.links] == [(1, 0, 'b'), (2, 1, 'c')]
    assert result.same_instances_confirmed is False
    assert not hasattr(result, 'unique_item_count')


def test_one_anchor_never_deduplicates_same_name():
    result = associate_item_viewports(ItemViewport('a', (item('x', 400),)), ItemViewport('b', (item('x', 200),)))
    assert result.status == ItemOverlapStatus.INSUFFICIENT and result.links == ()


def test_duplicate_id_in_a_viewport_is_not_a_unique_anchor():
    previous, current = pair()
    previous = replace(previous, visible_items=(item('b', 200), item('b', 412), item('c', 624)))
    result = associate_item_viewports(previous, current)
    assert result.status == ItemOverlapStatus.INSUFFICIENT and result.links == ()


@pytest.mark.parametrize('section', [None, 'stage'])
def test_missing_or_conflicting_section_does_not_borrow_previous_section(section):
    previous, current = pair()
    current = replace(current, visible_items=(item('b', 200, section=section), item('c', 412), item('d', 624)))
    assert associate_item_viewports(previous, current).status == ItemOverlapStatus.INSUFFICIENT


@pytest.mark.parametrize('status', [ItemBindingStatus.UNKNOWN_NAME, ItemBindingStatus.AMBIGUOUS,
                                   ItemBindingStatus.NO_EXACT_MATCH, ItemBindingStatus.DATABASE_UNAVAILABLE])
def test_unresolved_binding_never_becomes_anchor(status):
    previous, current = pair()
    current = replace(current, visible_items=(item('b', 200, status=status), item('c', 412), item('d', 624)))
    assert associate_item_viewports(previous, current).status == ItemOverlapStatus.INSUFFICIENT


def test_identical_database_id_without_name_evidence_is_not_anchor():
    previous, current = pair()
    bad = item('b', 200)
    bad = replace(bad, reference=replace(bad.reference, name_reads=()))
    current = replace(current, visible_items=(bad, item('c', 412), item('d', 624)))
    assert associate_item_viewports(previous, current).status == ItemOverlapStatus.INSUFFICIENT


def test_disagreeing_geometry_does_not_cherry_pick_anchors():
    previous, current = pair()
    current = replace(current, visible_items=(item('b', 200), item('c', 425), item('d', 624)))
    assert associate_item_viewports(previous, current).status == ItemOverlapStatus.GEOMETRY_CONFLICT


def test_reordered_cards_do_not_confirm_alignment():
    previous, current = pair()
    current = replace(current, visible_items=(item('c', 200), item('b', 412), item('d', 624)))
    assert associate_item_viewports(previous, current).status == ItemOverlapStatus.ORDER_CONFLICT


def test_same_viewport_is_not_list_end_or_new_equipment():
    previous, _ = pair()
    assert associate_item_viewports(previous, previous).status == ItemOverlapStatus.SAME_VIEWPORT


def test_new_pixel_key_with_stationary_cards_does_not_confirm_scroll():
    previous, _ = pair()
    assert associate_item_viewports(previous, replace(previous, key='changed-pixels')).status == ItemOverlapStatus.DIRECTION_UNKNOWN


@pytest.mark.parametrize('bad', [ItemViewport(None, None), ItemViewport('x', ()), ItemViewport('x', None)])
def test_unknown_page_keeps_unknown(bad):
    previous, _ = pair()
    assert associate_item_viewports(previous, bad).status == ItemOverlapStatus.UNKNOWN


def test_two_sections_of_same_id_are_not_two_distinct_anchors():
    previous = ItemViewport('a', (item('same', 400, section='stage'), item('same', 700)))
    current = ItemViewport('b', (item('same', 200, section='stage'), item('same', 500)))
    assert associate_item_viewports(previous, current).status == ItemOverlapStatus.INSUFFICIENT
