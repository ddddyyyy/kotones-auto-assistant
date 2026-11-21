"""星数领域边界，不模拟页面点击。"""

import pytest

from kaa.idol_road.models import Observation, Scene, Stage
from kaa.idol_road.stars import count_known_stars


@pytest.mark.parametrize('stars', [0, 1, 2, 3, None])
def test_stage_clear_is_derived_from_explicit_star_count(stars):
    stage = Stage(1, True, stars)
    assert stage.stars == stars
    assert stage.cleared == (None if stars is None else stars >= 1)


@pytest.mark.parametrize('stars', [-1, 4, True, False, 1.5, '1'])
def test_star_count_rejects_invalid_values(stars):
    with pytest.raises(ValueError):
        Stage(1, True, stars)
    with pytest.raises(ValueError):
        Observation(Scene.RESULT, stars=stars)


@pytest.mark.parametrize('slots,expected', [
    ((False, False, False), 0), ((True, False, False), 1),
    ((True, True, False), 2), ((True, True, True), 3),
    ((True, False, True), 2), ((False, True, False), 1),
    ((True, None, True), None), ((None, False, False), None),
    ((True, True), None),
    ((1, 1, 1), None), ((True, 'empty', False), None),
])
def test_count_requires_all_three_slots_without_assuming_prefix_order(slots, expected):
    assert count_known_stars(slots) == expected
