from types import SimpleNamespace

from kaa.tasks.produce.shared import cards


def test_selected_card_preview_is_confirmed_only_when_visible(monkeypatch):
    clicked = []
    monkeypatch.setattr(cards, 'ocr', SimpleNamespace(
        find=lambda _matcher, *, rect: rect,
    ))
    monkeypatch.setattr(cards, 'device', SimpleNamespace(click=clicked.append))

    assert cards._confirm_selected_card_preview()
    assert clicked == [cards.SELECTED_CARD_CONFIRM_RECT]


def test_selected_card_preview_does_not_click_without_select_label(monkeypatch):
    clicked = []
    monkeypatch.setattr(cards, 'ocr', SimpleNamespace(
        find=lambda _matcher, *, rect: None,
    ))
    monkeypatch.setattr(cards, 'device', SimpleNamespace(click=clicked.append))

    assert not cards._confirm_selected_card_preview()
    assert clicked == []
