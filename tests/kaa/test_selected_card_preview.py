from types import SimpleNamespace

import numpy as np

from kaa.tasks.produce.shared import cards
from kotonebot.primitives import Rect


def test_selected_card_preview_is_confirmed_only_when_visible(monkeypatch):
    clicked = []
    observed = []
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    right_label = Rect(533, 1122, 86, 21)

    def locate(screen):
        observed.append(screen)
        return right_label

    monkeypatch.setattr(cards, '_selected_card_preview_rect', locate)
    # There is no screenshot method: confirmation must reuse the supplied frame.
    monkeypatch.setattr(cards, 'device', SimpleNamespace(click=clicked.append))

    assert cards._confirm_selected_card_preview(frame)
    assert len(observed) == 1 and observed[0] is frame
    assert clicked == [right_label]


def test_selected_card_preview_does_not_click_without_select_label(monkeypatch):
    clicked = []
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    monkeypatch.setattr(cards, '_selected_card_preview_rect', lambda _screen: None)
    monkeypatch.setattr(cards, 'device', SimpleNamespace(click=clicked.append))

    assert not cards._confirm_selected_card_preview(frame)
    assert clicked == []
