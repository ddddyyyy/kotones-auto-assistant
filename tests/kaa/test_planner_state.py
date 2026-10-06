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


@pytest.mark.parametrize('value', [0, 2, 4, 6, 12])
def test_current_aggressive_hud_replaces_inferred_memory(value):
    from kaa.tasks.produce.new.play_cards.planner.memory import BattleMemory
    from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState
    memory = BattleMemory(buffs=ExamBuffState(aggressive=9, review=4))
    memory.reconcile_effect_values({'aggressive': value, 'review': 4})
    assert memory.buffs.aggressive == value
    assert memory.buffs.review == 4
    memory.reconcile_effect_values({'aggressive': None, 'review': 4})
    assert memory.buffs.aggressive == (value or 1)
    memory.reconcile_effect_values({'review': 4})
    assert memory.buffs.aggressive == (value or 1)
    memory.reconcile_effect_values({'review': 4})
    assert memory.buffs.aggressive == 0
    assert memory.effect_misses['aggressive'] == 2
