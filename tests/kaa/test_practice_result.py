from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from kaa.tasks.produce.new.play_cards import practice_result
from kaa.tasks.produce.new.strategies import standard
from kaa.tasks.produce.new.controller import ProduceController
from kaa.tasks.produce.new.page import PracticeContext


def test_practice_result_text_requires_an_explicit_grade():
    assert practice_result.classify_practice_result('PERFECT') == 'PERFECT'
    assert practice_result.classify_practice_result('C L E A R') == 'CLEAR'
    assert practice_result.classify_practice_result('NOT CLEAR') == 'FAILED'
    assert practice_result.classify_practice_result('LESSON NEXT') is None


def test_practice_result_reads_only_the_result_banner(monkeypatch):
    screen = np.zeros((1280, 720, 3), dtype=np.uint8)
    observed = []

    def fake_ocr(image, *, rect):
        observed.append((image, rect))
        return SimpleNamespace(squash=lambda: SimpleNamespace(text='PERFECT'))

    monkeypatch.setattr(
        practice_result,
        'en',
        lambda: SimpleNamespace(ocr=fake_ocr),
    )

    assert practice_result.read_practice_result(screen) == 'PERFECT'
    assert observed[0][0] is screen
    assert observed[0][1] == practice_result.RESULT_BANNER


@pytest.mark.parametrize('battle_hud_visible', [False, True])
def test_practice_result_is_sampled_before_card_animation_skip(monkeypatch, battle_hud_visible):
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    observed = []

    class BattleHud:
        @classmethod
        def __class_getitem__(cls, _items):
            return SimpleNamespace(exists=lambda: battle_hud_visible)

    def fake_do_cards(_is_exam, _threshold, _end, **kwargs):
        kwargs['result_observer'](frame)

    monkeypatch.setattr(standard, 'AnyOf', BattleHud)
    monkeypatch.setattr(standard, '_build_battle_strategy', lambda *_args: None)
    monkeypatch.setattr(standard, 'do_cards', fake_do_cards)
    monkeypatch.setattr(standard, 'read_practice_result', lambda image: observed.append(image) or 'PERFECT')

    strategy = standard.StandardStrategy(cast(ProduceController, SimpleNamespace(page=None)))
    strategy.on_practice_entered(cast(PracticeContext, None))

    assert len(observed) == 1
    assert observed[0] is frame
    assert strategy._practice_result == 'PERFECT'


def test_practice_exit_waits_briefly_for_late_result_banner(monkeypatch):
    strategy = standard.StandardStrategy(cast(ProduceController, SimpleNamespace(page=None)))
    probes = []
    delays = []

    def observe():
        probes.append(1)
        if len(probes) == 2:
            strategy._practice_result = 'CLEAR'

    monkeypatch.setattr(strategy, '_observe_practice_result', observe)
    monkeypatch.setattr(standard, 'sleep', delays.append)

    strategy.on_practice_exited()

    assert len(probes) == 2
    assert delays == [0.6]
    assert strategy._practice_result == 'CLEAR'


def test_failed_ocr_still_collects_unconfirmed_evidence(monkeypatch):
    strategy = standard.StandardStrategy(cast(ProduceController, SimpleNamespace(page=None)))
    frames = []
    strategy._practice_evidence = cast(Any, SimpleNamespace(
        observe=lambda frame, result: frames.append((frame, result)),
    ))
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)

    def fail(_image):
        raise RuntimeError('OCR unavailable')

    monkeypatch.setattr(standard, 'read_practice_result', fail)
    strategy._observe_practice_result(frame)
    assert strategy._practice_result is None
    assert len(frames) == 1
    assert frames[0][0] is frame
    assert frames[0][1] is None
