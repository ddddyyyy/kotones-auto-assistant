from types import SimpleNamespace

import numpy as np

from importlib import import_module

produce = import_module('kaa.tasks.produce.shared.produce_end')


def test_failed_exam_tap_prompt_is_recognized_only_in_bottom_area(monkeypatch):
    screen = np.zeros((1280, 720, 3), dtype=np.uint8)
    boxes = []

    def fake_ocr(_screen, *, rect):
        boxes.append(rect)
        return [SimpleNamespace(text='TAPTXA', confidence=0.86)]

    monkeypatch.setattr(produce, 'en', lambda: SimpleNamespace(ocr=fake_ocr))

    assert produce._failed_exam_next_visible(screen)
    assert boxes == [produce.BOX_FAILED_EXAM_NEXT]


def test_failed_exam_tap_prompt_rejects_weak_or_unrelated_ocr(monkeypatch):
    screen = np.zeros((1280, 720, 3), dtype=np.uint8)
    results = [SimpleNamespace(text='TAPTXA', confidence=0.6)]
    monkeypatch.setattr(
        produce,
        'en',
        lambda: SimpleNamespace(ocr=lambda *_args, **_kwargs: results),
    )

    assert not produce._failed_exam_next_visible(screen)
    results[:] = [SimpleNamespace(text='FAILED', confidence=0.99)]
    assert not produce._failed_exam_next_visible(screen)


def test_settlement_stops_clicking_once_home_is_visible(monkeypatch, caplog):
    screen = np.zeros((1280, 720, 3), dtype=np.uint8)
    clicks = []
    generate = object()
    button_next = SimpleNamespace(exists=lambda: True)
    monkeypatch.setattr(produce, 'device', SimpleNamespace(
        screenshot=lambda: screen,
        click=clicks.append,
    ))
    monkeypatch.setattr(produce, 'image', SimpleNamespace(raw=lambda: SimpleNamespace(
        find=lambda *_args: False,
    )))
    monkeypatch.setattr(produce, 'ocr', SimpleNamespace(find=lambda *_args: generate))
    monkeypatch.setattr(produce, 'R', SimpleNamespace(
        InProduce=SimpleNamespace(
            ButtonRetry=SimpleNamespace(find=lambda: None),
            ButtonNextNoIcon=button_next,
        ),
        Common=SimpleNamespace(ButtonClose=SimpleNamespace(find=lambda: None)),
    ))
    monkeypatch.setattr(produce, 'at_home', lambda: True)
    monkeypatch.setattr(produce, 'Loop', lambda **_kwargs: range(2))
    monkeypatch.setattr(produce, 'sleep', lambda *_args: None)
    monkeypatch.setattr(produce, 'wait', lambda *_args, **_kwargs: None)
    caplog.set_level('INFO', logger=produce.__name__)

    produce.produce_end(has_live=False)

    assert clicks == [generate]
    assert 'Produce completed.' in caplog.text
