import pytest

from kaa.tasks.produce.new.play_cards.planner.state import (
    AnomalyState,
    AnomalyStance,
    BattleState,
    ProduceItemState,
)


def test_anomaly_dimensions_are_independent():
    anomaly = AnomalyState(
        stance=AnomalyStance.PRESERVATION,
        full_power_points=8,
        full_power_active=True,
        enthusiasm=5,
    )
    assert anomaly.stance is AnomalyStance.PRESERVATION
    assert anomaly.full_power_active
    assert anomaly.enthusiasm == 5


def test_item_fire_limit_and_cooldown():
    assert ProduceItemState('item', fired_count=1, fire_limit=2).can_fire
    assert not ProduceItemState('item', fired_count=2, fire_limit=2).can_fire
    assert not ProduceItemState('item', cooldown_turns=1).can_fire


def test_battle_state_validates_observation_confidence():
    with pytest.raises(ValueError):
        BattleState(remaining_turns=1, hp=10, genki=0, observation_confidence=1.1)
