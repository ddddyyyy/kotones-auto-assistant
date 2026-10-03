"""Keep preparation checkbox matching away from formation-details controls."""

import pytest

from kaa.tasks.produce.prepare import NOTE_BOOST_REGION, PT_BOOST_REGION


@pytest.mark.parametrize(
    'region, checkbox_center',
    [
        (NOTE_BOOST_REGION, (533, 1027)),
        (PT_BOOST_REGION, (623, 1027)),
    ],
)
def test_boost_region_contains_checkbox_but_excludes_details(region, checkbox_center):
    x, y = checkbox_center
    assert region.x1 <= x < region.x2
    assert region.y1 <= y < region.y2
    # Live false match clicked the icon at (677, 1093).
    assert region.y2 < 1060
    assert not (region.x1 <= 677 < region.x2 and region.y1 <= 1093 < region.y2)
