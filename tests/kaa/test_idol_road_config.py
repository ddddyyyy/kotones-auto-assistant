"""Profile 默认值与输入约束，不涉及设备或游戏截图。"""

import pytest
from pydantic import ValidationError

from kaa.config.idol_road import IdolRoadConfig
from kaa.config.schema import KaaConfig


def test_old_profile_gets_disabled_preview_without_migration():
    profile = KaaConfig.model_validate({'tasks': {'contest': {'enabled': True}}})
    assert not profile.tasks.idol_road.enabled
    assert profile.tasks.idol_road.mode == 'preview'
    assert profile.tasks.idol_road.auto_fill_missing_formation is True
    assert profile.tasks.contest.enabled
    assert KaaConfig.model_validate_json(profile.model_dump_json()) == profile


def test_character_order_and_profile_isolation():
    first = KaaConfig()
    second = KaaConfig()
    first.tasks.idol_road.characters.extend(['hmsz', 'hski'])
    assert second.tasks.idol_road.characters == []
    assert first.tasks.idol_road.characters == ['hmsz', 'hski']


def test_registry_uses_disabled_profile_toggle():
    from kaa.tasks import TASK_REGISTRY

    info = TASK_REGISTRY['idol_road']
    profile = KaaConfig()
    assert not info.get_enabled(profile)
    profile.tasks.idol_road.enabled = True
    assert info.get_enabled(profile)


@pytest.mark.parametrize('values', [
    {'characters': ['hski', 'hski']},
    {'characters': [' HSKI']},
    {'characters': ['花海咲季']},
    {'max_battles': 0},
    {'max_battles': 101},
    {'timeout_seconds': 0},
    {'mode': 'planner'},
])
def test_invalid_or_unimplemented_options_are_rejected(values):
    with pytest.raises(ValidationError):
        IdolRoadConfig.model_validate(values)
