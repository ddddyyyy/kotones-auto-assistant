from pathlib import Path

import cv2
import numpy as np

from kaa.tasks.produce.new.play_cards.practice_evidence import EVIDENCE_ENV, PracticeResultEvidence


def test_evidence_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(EVIDENCE_ENV, raising=False)
    assert PracticeResultEvidence.from_environment() is None


def test_confirmed_result_saves_the_exact_frame_only_once(tmp_path, monkeypatch):
    monkeypatch.setenv(EVIDENCE_ENV, str(tmp_path))
    evidence = PracticeResultEvidence.from_environment()
    assert evidence is not None
    frame = np.full((12, 15, 3), 71, dtype=np.uint8)
    evidence.observe(frame, None)
    evidence.observe(frame, 'CLEAR')
    evidence.observe(frame, 'PERFECT')
    evidence.finish()
    files = list(tmp_path.glob('*.png'))
    assert len(files) == 1
    assert '-CLEAR-' in files[0].name
    assert np.array_equal(cv2.imread(str(files[0])), frame)
    assert not evidence.frames


def test_unconfirmed_capture_is_bounded_and_copies_frames(tmp_path):
    evidence = PracticeResultEvidence(tmp_path)
    frame = np.zeros((12, 15, 3), dtype=np.uint8)
    for value in range(10):
        frame[:] = value
        evidence.observe(frame, None)
    frame[:] = 99
    assert len(evidence.frames) == 6
    evidence.finish()
    files = sorted(tmp_path.glob('*.png'))
    assert len(files) == 6
    assert all('-unconfirmed-' in path.name for path in files)
    assert [int(cv2.imread(str(path))[0, 0, 0]) for path in files] == list(range(4, 10))
    evidence.finish()
    assert len(list(tmp_path.glob('*.png'))) == 6


def test_capture_write_error_does_not_interrupt_game(tmp_path):
    blocked = tmp_path / 'not-a-directory'
    blocked.write_text('blocked', encoding='utf-8')
    evidence = PracticeResultEvidence(Path(blocked))
    evidence.observe(np.zeros((12, 15, 3), dtype=np.uint8), 'CLEAR')
    assert evidence.finished
