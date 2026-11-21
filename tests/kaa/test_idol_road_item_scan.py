import pytest

from kaa.idol_road.item_scan import ItemScan
from kaa.idol_road.items import ItemBindingStatus, ItemReference, ItemViewport, VisibleItem


def page(key, item_id='same-item'):
    ref = ItemReference((), ItemBindingStatus.RESOLVED, '同名', (item_id,), item_id)
    return ItemViewport(key, (VisibleItem(ref, (27, 171, 666, 235)),))


def test_three_distinct_viewports_bound_scrolling_without_flattening_same_items():
    scan = ItemScan()
    assert scan.observe(page('a'))[0] == 'scroll'
    assert scan.observe(page('b'))[0] == 'scroll'
    assert scan.observe(page('c')) == ('close', 'item_scroll_limit')
    assert scan.scrolls == 2 and len(scan.viewports) == 3
    assert [p.visible_items[0].reference.item_id for p in scan.viewports] == ['same-item'] * 3
    assert not hasattr(scan, 'unique_items')


def test_residual_viewport_does_not_request_another_swipe_or_claim_end():
    scan = ItemScan()
    scan.observe(page('a'))
    assert scan.observe(page('a')) == ('wait', 'await_item_scroll')
    assert scan.observe(page('a')) == ('close', 'item_scroll_not_confirmed')
    assert scan.scrolls == 1 and len(scan.viewports) == 1
    assert scan.observe(page('b')) == ('close', 'item_scroll_not_confirmed')


def test_wait_then_new_viewport_preserves_both_observations():
    scan = ItemScan()
    scan.observe(page('a'))
    scan.observe(page('a'))
    assert scan.observe(page('b'))[0] == 'scroll'
    assert [viewport.key for viewport in scan.viewports] == ['a', 'b']


def test_return_to_old_viewport_is_not_new_equipment():
    scan = ItemScan()
    scan.observe(page('a'))
    scan.observe(page('b'))
    assert scan.observe(page('a')) == ('close', 'item_viewport_repeated')
    assert len(scan.viewports) == 2


@pytest.mark.parametrize('viewport', [ItemViewport(None, None), ItemViewport('x', None), ItemViewport('x', ())])
def test_unknown_viewport_is_not_empty_complete_inventory(viewport):
    scan = ItemScan()
    assert scan.observe(viewport) == ('close', 'item_viewport_unknown')
    assert scan.scrolls == 0 and scan.viewports == ()


@pytest.mark.parametrize('kwargs', [{'max_scrolls': -1}, {'max_stale': 0}])
def test_bad_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ItemScan(**kwargs)


def test_zero_scroll_budget_still_records_first_viewport():
    scan = ItemScan(max_scrolls=0)
    assert scan.observe(page('a')) == ('close', 'item_scroll_limit')
    assert len(scan.viewports) == 1
