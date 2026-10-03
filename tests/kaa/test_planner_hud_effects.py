import importlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from kaa.tasks.produce.new.play_cards.planner.hud_effects import (
    _icon_alpha,
    _read_effect_value,
    observe_effect_icons,
)


def _screen_with_icon(effect: str, x: int = 28, y: int = 360) -> np.ndarray:
    screen = np.zeros((1280, 720, 3), dtype=np.uint8)
    source = _icon_alpha(effect)
    height = 34
    width = round(source.shape[1] * height / source.shape[0])
    glyph = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
    mask = glyph > 32
    region = screen[y : y + height, x : x + width]
    region[mask] = 255
    return screen


def test_parameter_buff_icon_is_observed_from_hud_column():
    observations = observe_effect_icons(_screen_with_icon('parameter_buff'))

    assert any(item.effect == 'parameter_buff' for item in observations)


def test_full_power_icon_is_observed_from_hud_column():
    observations = observe_effect_icons(_screen_with_icon('full_power'))

    assert any(item.effect == 'full_power' for item in observations)


def test_review_icon_is_observed_in_existing_battle_fixture():
    fixture = Path(__file__).parents[1] / 'images' / 'produce' / 'in_produce_cards_2.png'
    screen = cv2.imread(str(fixture))
    assert screen is not None

    observations = observe_effect_icons(screen)

    assert any(item.effect == 'review' and item.value == 2 for item in observations)
    assert not any(item.effect == 'full_power' for item in observations)


def test_grey_review_glyph_is_not_treated_as_active():
    fixture = Path(__file__).parents[1] / 'images' / 'produce' / 'in_produce_cards_1.png'
    screen = cv2.imread(str(fixture))
    assert screen is not None

    observations = observe_effect_icons(screen)

    assert not any(item.effect == 'review' for item in observations)


def test_matches_outside_left_effect_column_are_rejected():
    observations = observe_effect_icons(
        _screen_with_icon('lesson_buff', x=130),
    )

    assert not any(item.effect == 'lesson_buff' for item in observations)


def test_effect_value_requires_consensus(monkeypatch):
    class FakeOcr:
        results = iter(['1o', '10', '10', 'z1o', 'noise'])

        def classification(self, _image):
            return next(self.results)

    module = importlib.import_module(
        'kaa.tasks.produce.new.play_cards.planner.hud_effects'
    )
    monkeypatch.setattr(module, '_digit_ocr', lambda: FakeOcr())

    screen = np.zeros((1280, 720, 3), dtype=np.uint8)
    assert _read_effect_value(screen, 'lesson_buff', 27, 490) == 10


def test_strong_stance_is_detected_only_in_top_hud_slot():
    assert any(item.effect == 'concentration' for item in observe_effect_icons(
        _screen_with_icon('concentration', y=250),
    ))
    assert not any(item.effect == 'concentration' for item in observe_effect_icons(
        _screen_with_icon('concentration', y=450),
    ))


@pytest.mark.parametrize('name', ['anomaly_strong_exam_1.png', 'anomaly_strong_exam_2.png'])
def test_strong_stance_matches_real_exam_frames(name):
    fixture = Path(__file__).parents[1] / 'images' / 'produce' / name
    screen = cv2.imread(str(fixture))
    assert screen is not None
    matches = [item for item in observe_effect_icons(screen) if item.effect == 'concentration']
    assert len(matches) == 1
    assert 20 <= matches[0].x <= 40 and 235 <= matches[0].y <= 260
    assert matches[0].value is None  # Do not mistake a counter for stance level.


@pytest.mark.parametrize('name', ['in_produce_cards_1.png', 'in_produce_cards_2.png'])
def test_other_plan_frames_do_not_match_strong_stance(name):
    fixture = Path(__file__).parents[1] / 'images' / 'produce' / name
    screen = cv2.imread(str(fixture))
    assert screen is not None
    assert not any(item.effect == 'concentration' for item in observe_effect_icons(screen))


@pytest.mark.parametrize('name,value', [
    ('anomaly_strong_exam_1.png', 5), ('anomaly_strong_exam_2.png', 8),
    ('anomaly_points_6.png', 6),
])
def test_full_power_points_are_read_from_real_grey_badges(name, value):
    screen = cv2.imread(str(Path(__file__).parents[1] / 'images' / 'produce' / name))
    assert screen is not None
    matches = [item for item in observe_effect_icons(screen) if item.effect == 'full_power_point']
    assert len(matches) == 1
    assert matches[0].value == value


@pytest.mark.parametrize('results,expected', [
    (['0'] * 5, 0),
    (['13'] * 5, 13),
    (['13', '13', '13', '8', '8'], None),
    (['noise', '13', '13', 'noise', 'noise'], None),
    (['label13'] * 5, None),
])
def test_point_badge_requires_strict_number_consensus(monkeypatch, results, expected):
    module = importlib.import_module('kaa.tasks.produce.new.play_cards.planner.hud_effects')
    values = iter(results)
    monkeypatch.setattr(module, '_digit_ocr', lambda: type('Ocr', (), {
        'classification': lambda self, image: next(values),
    })())
    assert _read_effect_value(np.zeros((1280, 720, 3), dtype=np.uint8), 'full_power_point', 89, 248) == expected


def test_unreadable_point_badge_is_not_exposed_as_presence(monkeypatch):
    module = importlib.import_module('kaa.tasks.produce.new.play_cards.planner.hud_effects')
    monkeypatch.setattr(module, '_read_effect_value', lambda *args: None)
    observations = observe_effect_icons(_screen_with_icon('full_power_point', x=89, y=250))
    assert not any(item.effect == 'full_power_point' for item in observations)


@pytest.mark.parametrize('name', ['in_produce_cards_1.png', 'in_produce_cards_2.png'])
def test_other_plan_frames_do_not_match_point_badges(name):
    screen = cv2.imread(str(Path(__file__).parents[1] / 'images' / 'produce' / name))
    assert screen is not None
    assert not any(item.effect == 'full_power_point' for item in observe_effect_icons(screen))
