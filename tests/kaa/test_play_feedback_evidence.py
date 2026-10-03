import json
from types import SimpleNamespace

import cv2
import numpy as np

from kaa.tasks.produce.new.play_cards.planner import play_evidence as module


def test_default_off(monkeypatch):
    monkeypatch.delenv(module.EVIDENCE_ENV, raising=False)
    assert module.PlayFeedbackEvidence.from_environment() is None


def test_unchanged_play_saves_three_distinct_frames(tmp_path, monkeypatch):
    recorder = module.PlayFeedbackEvidence(tmp_path)
    before = np.zeros((8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(module, 'device', SimpleNamespace(
        screenshot=lambda: np.full_like(before, 80),
    ))
    recorder.before_commit('card', before)
    before[:] = 255  # The saved pre-click frame must be copied.
    recorder.after_commit()
    recorder.feedback('paid_play_observably_unchanged', np.full_like(before, 160))
    files = list(tmp_path.glob('*.png'))
    assert len(files) == 3
    for suffix, value in [('before', 0), ('after-click', 80), ('feedback', 160)]:
        path = next(p for p in files if p.name.endswith(f'-{suffix}.png'))
        assert np.all(cv2.imread(str(path)) == value)
    assert json.loads(next(tmp_path.glob('*.json')).read_text())['card_id'] == 'card'
    assert recorder.pending is None


def test_confirmed_play_is_not_saved_and_bundles_are_bounded(tmp_path):
    recorder = module.PlayFeedbackEvidence(tmp_path)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    recorder.before_commit('ok', frame)
    recorder.feedback('card_left_hand', frame)
    assert list(tmp_path.iterdir()) == []
    for _ in range(12):
        recorder.before_commit('unchanged', frame)
        recorder.feedback('paid_play_observably_unchanged', frame)
    assert len(list(tmp_path.glob('*.json'))) == 8
    assert recorder.saved_count == 8
    assert recorder.pending is None


def test_capture_and_write_failures_do_not_interrupt_play(tmp_path, monkeypatch):
    recorder = module.PlayFeedbackEvidence(tmp_path / 'blocked')
    recorder.directory.write_text('not a directory')
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    def fail():
        raise RuntimeError('capture unavailable')
    monkeypatch.setattr(module, 'device', SimpleNamespace(screenshot=fail))
    recorder.before_commit('card', frame)
    recorder.after_commit()
    recorder.feedback('paid_play_observably_unchanged', frame)
    assert recorder.pending is None
