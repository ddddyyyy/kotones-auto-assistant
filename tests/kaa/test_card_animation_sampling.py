from types import SimpleNamespace
import numpy as np

from kaa.tasks.produce.shared import cards


def _clock(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(cards, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    def sleep(seconds):
        clock.now += seconds
    monkeypatch.setattr(cards, 'sleep', sleep)
    monkeypatch.setattr(cards, 'device', SimpleNamespace(screenshot=lambda: clock.now))
    return clock


def test_sampling_catches_banner_in_former_blind_window_without_clicks(monkeypatch):
    clock = _clock(monkeypatch)
    samples = []
    results = []
    monkeypatch.setattr(cards, 'skip', lambda: (_ for _ in ()).throw(AssertionError('no skip')))
    def observe(frame):
        samples.append(frame)
        if 2.2 <= frame <= 2.8:
            results.append('CLEAR')
    cards._finish_card_animation(observe)
    assert results == ['CLEAR']
    assert samples == [0.5 * index for index in range(1, 10)]
    assert clock.now == 4.5


def test_sampling_budget_includes_slow_ocr(monkeypatch):
    clock = _clock(monkeypatch)
    samples = []
    def observe(frame):
        samples.append(frame)
        clock.now += 1.0
    cards._finish_card_animation(observe)
    assert samples == [0.5, 2.0, 3.5]
    assert clock.now == 4.5


def test_sampling_is_bounded_even_if_observer_is_very_slow(monkeypatch):
    clock = _clock(monkeypatch)
    samples = []
    def observe(frame):
        samples.append(frame)
        clock.now += 10
    cards._finish_card_animation(observe)
    assert samples == [0.5]


def test_legacy_animation_keeps_original_skip_and_waits(monkeypatch):
    delays = []
    skipped = []
    monkeypatch.setattr(cards, 'sleep', delays.append)
    monkeypatch.setattr(cards, 'skip', lambda: skipped.append(True))
    cards._finish_card_animation(None)
    assert delays == [1, 3.5]
    assert skipped == [True]


def test_animation_gate_waits_then_releases_on_missing_overlay(monkeypatch):
    clock = _clock(monkeypatch)
    visible = [True]
    monkeypatch.setattr(cards, '_has_resolving_card', lambda screen: visible[0])
    gate = cards.CardAnimationGate()
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    assert gate.should_wait(frame)
    clock.now = 4
    assert gate.should_wait(frame)
    visible[0] = False
    assert not gate.should_wait(frame)
    assert gate.started is None
    visible[0] = True
    assert gate.should_wait(frame)
    assert gate.started == 4


def test_animation_gate_has_bounded_wait_and_resets_after_clear(monkeypatch):
    clock = _clock(monkeypatch)
    visible = [True]
    monkeypatch.setattr(cards, '_has_resolving_card', lambda screen: visible[0])
    gate = cards.CardAnimationGate()
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    assert gate.should_wait(frame)
    clock.now = 8
    assert not gate.should_wait(frame)
    clock.now = 20
    assert not gate.should_wait(frame)
    visible[0] = False
    assert not gate.should_wait(frame)
    visible[0] = True
    assert gate.should_wait(frame)


def test_overlay_detector_uses_central_marker_not_hand(monkeypatch):
    observed = []
    def find(screen, template, *, rect, threshold):
        observed.append((rect, threshold))
        return object() if len(observed) == 2 else None
    monkeypatch.setattr(cards, 'find_template', find)
    assert cards._has_resolving_card(np.zeros((1280, 720, 3), dtype=np.uint8))
    assert len(observed) == 2
    assert all(rect == cards.RESOLVING_CARD_LETTER_RECT and threshold == 0.93 for rect, threshold in observed)
    assert not cards._has_resolving_card(np.zeros((200, 200, 3), dtype=np.uint8))


def test_hand_gate_waits_for_moving_marker_positions(monkeypatch):
    clock = _clock(monkeypatch)
    layout = [((1, 143, 1084), (1, 409, 1084), (1, 553, 1084))]
    monkeypatch.setattr(cards, '_hand_layout', lambda _: layout[0])
    gate = cards.HandLayoutGate()
    assert gate.should_wait(None)
    clock.now = 0.4
    layout[0] = ((1, 136, 1084), (1, 352, 1084), (1, 569, 1084))
    assert gate.should_wait(None)
    clock.now = 0.71
    assert not gate.should_wait(None)
    gate.reset()
    assert gate.should_wait(None)


def test_hand_gate_ignores_minor_jitter_but_tracks_count_and_type(monkeypatch):
    clock = _clock(monkeypatch)
    layout = [((0, 100, 1084),)]
    monkeypatch.setattr(cards, '_hand_layout', lambda _: layout[0])
    gate = cards.HandLayoutGate()
    assert gate.should_wait(None)
    clock.now = 0.4
    layout[0] = ((0, 102, 1085),)
    assert not gate.should_wait(None)
    layout[0] = ((1, 102, 1085),)
    assert gate.should_wait(None)
    layout[0] = ((1, 102, 1085), (1, 300, 1085))
    assert gate.should_wait(None)
    layout[0] = ()
    assert not gate.should_wait(None)
    assert gate.started is None


def test_hand_gate_timeout_is_bounded_until_reset(monkeypatch):
    clock = _clock(monkeypatch)
    monkeypatch.setattr(cards, '_hand_layout', lambda _: ((1, int(clock.now * 10), 1084),))
    gate = cards.HandLayoutGate()
    assert gate.should_wait(None)
    clock.now = 3
    assert not gate.should_wait(None)
    clock.now = 4
    assert not gate.should_wait(None)
    assert gate.warned
    gate.reset()
    assert gate.should_wait(None)


def test_hand_marker_detector_is_confined_to_hand_roi(monkeypatch):
    observed = []
    def find(screen, template, *, rect):
        observed.append(rect)
        return [SimpleNamespace(rect=cards.Rect(136, 1084, 16, 14))]
    monkeypatch.setattr(cards, 'find_all_templates', find)
    assert cards._hand_layout(np.zeros((1280, 720, 3), dtype=np.uint8)) == (
        (0, 136, 1084), (1, 136, 1084), (2, 136, 1084),
    )
    assert observed == [cards.R.InProduce.BoxCardLetter] * 3
    assert cards._hand_layout(np.zeros((100, 100, 3), dtype=np.uint8)) == ()
