import pytest

from kaa.idol_road.objectives import StageObjectives, read_rank_requirement


@pytest.mark.parametrize('text,rank', [
    ('1位でクリア', 1), ('2位以上でクリア', 2), ('3位以上でクリア', 3),
    ('・３位以上でクリア', 3), (' 2 位 以上でクリア ', 2),
    ('2位でクリア', None), ('0位以上でクリア', None), ('10000点以上', None),
    ('前半3位以上でクリア', None), ('3位以上でクリア追加条件', None), (None, None),
])
def test_rank_condition_requires_complete_supported_text(text, rank):
    row = read_rank_requirement(1, text)
    assert row.rank_at_most == rank and row.text == text


@pytest.mark.parametrize('stars', [0, 4, True, 1.0, '1'])
def test_invalid_star_tier_is_not_accepted(stars):
    with pytest.raises(ValueError):
        read_rank_requirement(stars, '1位でクリア')


def test_rank_goal_does_not_infer_missing_star_tiers():
    complete = tuple(read_rank_requirement(stars, text) for stars, text in (
        (3, '1位でクリア'), (2, '2位以上でクリア'), (1, '3位以上でクリア'),
    ))
    objectives = StageObjectives('ttmr', 16, complete)
    assert [objectives.rank_limit(stars) for stars in (1, 2, 3)] == [3, 2, 1]
    for rows in (complete[:2], (*complete[:2], read_rank_requirement(None, '3位以上でクリア')),
                 (*complete[:2], complete[1])):
        assert StageObjectives('ttmr', 16, rows).rank_limit(1) is None


def test_unknown_supported_goal_retains_raw_text():
    rows = tuple(read_rank_requirement(stars, text) for stars, text in (
        (3, 'スコア10000以上でクリア'), (2, '2位以上でクリア'), (1, None),
    ))
    objectives = StageObjectives('ttmr', 16, rows)
    assert objectives.rank_limit(3) is None and objectives.rank_limit(1) is None
    assert objectives.rank_limit(2) == 2
    assert objectives.requirements[0].text == 'スコア10000以上でクリア'
