import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

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
    assert len(list(tmp_path.glob('*.json'))) == 6
    for _ in range(4):
        recorder.before_commit('terminal', frame, terminal=True)
        recorder.feedback('insufficient_evidence', frame)
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


@pytest.mark.parametrize('evidence', [
    'insufficient_evidence', 'turn_advanced_without_play_observation',
    'battle_ended_without_observation',
])
def test_unconfirmed_play_preserves_post_click_frame_without_claiming_stability(
    tmp_path, monkeypatch, evidence,
):
    recorder = module.PlayFeedbackEvidence(tmp_path)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(module, 'device', SimpleNamespace(screenshot=lambda: frame + 40))
    recorder.before_commit('card', frame)
    recorder.after_commit()
    recorder.feedback(evidence, None)
    metadata = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert metadata['evidence'] == evidence
    assert metadata['after_click_captured'] is True
    assert metadata['feedback_captured'] is False
    assert metadata['after_click_is_settled_observation'] is False
    assert len(list(tmp_path.glob('*.png'))) == 2
    assert recorder.pending is None


def test_missing_post_click_frame_is_explicit_and_not_fabricated(tmp_path, monkeypatch):
    recorder = module.PlayFeedbackEvidence(tmp_path)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    def fail():
        raise RuntimeError('capture unavailable')
    monkeypatch.setattr(module, 'device', SimpleNamespace(screenshot=fail))
    recorder.before_commit('card', frame)
    recorder.after_commit()
    recorder.feedback('turn_advanced_without_play_observation', frame)
    metadata = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert metadata['after_click_captured'] is False
    assert metadata['feedback_captured'] is True
    assert not list(tmp_path.glob('*-after-click.png'))


def test_terminal_reserve_does_not_capture_exhausted_ordinary_requests(tmp_path, monkeypatch):
    recorder = module.PlayFeedbackEvidence(tmp_path)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    calls = []
    monkeypatch.setattr(module, 'device', SimpleNamespace(
        screenshot=lambda: (calls.append(1), frame)[1]))
    for _ in range(6):
        recorder.before_commit('ordinary', frame)
        recorder.after_commit()
        recorder.feedback('insufficient_evidence', frame)
    recorder.before_commit('exhausted', frame)
    recorder.after_commit()
    recorder.feedback('insufficient_evidence', frame)
    assert len(calls) == 6
    recorder.before_commit('terminal', frame, terminal=True)
    recorder.after_commit()
    recorder.feedback('battle_ended_without_observation', None)
    assert len(calls) == 7
    rows = [json.loads(p.read_text()) for p in tmp_path.glob('*.json')]
    terminal = next(row for row in rows if row['card_id'] == 'terminal')
    assert terminal['observed_terminal_request'] is True
    assert terminal['after_click_is_settled_observation'] is False
    assert terminal['feedback_captured'] is False
    assert recorder.saved_count == 7
