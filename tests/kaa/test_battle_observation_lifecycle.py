"""Observation ownership and data consistency; no game input operations."""
import gc
import weakref
from types import SimpleNamespace

import numpy as np

from kaa.tasks.produce.new.play_cards import page
from kaa.tasks.produce.new.play_cards.planner import hud_effects


def test_single_line_card_title_option_is_context_local_and_preserves_default_parser(monkeypatch):
    screen=np.zeros((1280,720,3),dtype='uint8')
    calls=[]
    def recognize(image,**options):
        assert image is screen
        calls.append(options)
        return []
    monkeypatch.setattr(page,'locate_cards',recognize)
    ordinary=page.LessonBattleContext(screen)
    direct=page.LessonBattleContext(screen,single_line_titles=True)
    for context in (ordinary,direct,ordinary,direct):
        assert context.fetch_hands()==[]
    assert calls==[{}, {'single_line_titles':True}]


def test_snapshot_is_shared_by_all_parsers_and_next_context_refreshes(monkeypatch):
    captures = []
    inputs = []

    def capture():
        frame = np.full((1280, 720, 3), len(captures), dtype=np.uint8)
        captures.append(frame)
        return frame

    def number(box, *, screen, **kwargs):
        inputs.append(screen)
        return int(screen[0, 0, 0])

    def turns(screen):
        inputs.append(screen)
        return int(screen[0, 0, 0]) + 1

    def hands(screen):
        inputs.append(screen)
        return []

    def target(screen, template):
        inputs.append(screen)
        return None

    monkeypatch.setattr(page, 'device', SimpleNamespace(screenshot=capture))
    monkeypatch.setattr(page, '_fetch_int', number)
    # Score gaps have their own parser now; keep this snapshot-ownership test
    # synthetic rather than running real OCR on the generated black frame.
    monkeypatch.setattr(page, '_read_score_gap', lambda screen: number(None, screen=screen))
    monkeypatch.setattr(page, '_fetch_remaining_turns', turns)
    monkeypatch.setattr(page, '_fetch_exam_remaining_turns', turns)
    monkeypatch.setattr(page, 'locate_cards', hands)
    monkeypatch.setattr(page, 'find', target)
    monkeypatch.setattr(hud_effects, 'observe_effect_icons', hands)
    monkeypatch.setattr(page, 'read_exam_standings', lambda screen: inputs.append(screen))
    context = page.LessonBattleContext()
    context.fetch_hands()
    assert context.fetch_all() == page.HudInfo(1, 0, 0, 0)
    context.fetch_exam_remaining_turns()
    context.fetch_exam_hp()
    context.fetch_exam_stamina()
    context.fetch_exam_standings()
    context.fetch_score_target_is_final()
    context.fetch_effect_observations()
    calls = len(inputs)
    context.fetch_all()
    context.fetch_hands()
    context.fetch_effect_observations()
    assert len(inputs) == calls
    assert len(captures) == 1
    assert all(frame is captures[0] for frame in inputs)
    assert page.LessonBattleContext().fetch_hp() == 1
    assert len(captures) == 2


def test_context_and_cached_card_back_reference_do_not_leak_frames(monkeypatch):
    frames = []
    contexts = []

    def capture():
        frame = np.zeros((1280, 720, 3), dtype=np.uint8)
        frames.append(weakref.ref(frame))
        return frame

    monkeypatch.setattr(page, 'device', SimpleNamespace(screenshot=capture))
    for _ in range(100):
        context = page.LessonBattleContext()
        contexts.append(weakref.ref(context))
        # Even a cached result that refers to its owner is collected by GC.
        monkeypatch.setattr(page, 'locate_cards', lambda frame: [SimpleNamespace(owner=context, frame=frame)])
        context.fetch_hands()
    del context
    monkeypatch.setattr(page, 'locate_cards', lambda frame: [])
    gc.collect()
    assert all(reference() is None for reference in contexts)
    assert all(reference() is None for reference in frames)


def test_explicit_snapshot_does_not_capture_another_frame(monkeypatch):
    def unexpected_capture():
        raise AssertionError('A provided observation must not capture again')

    monkeypatch.setattr(page, 'device', SimpleNamespace(screenshot=unexpected_capture))
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    context = page.LessonBattleContext(frame)
    assert context.fetch_screen() is frame
    assert context.fetch_screen() is frame
