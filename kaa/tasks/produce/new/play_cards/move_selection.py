"""Recognize the cards offered by an in-battle move-to-hold dialog."""

from dataclasses import dataclass
from collections.abc import Iterable

from cv2.typing import MatLike
from kotonebot import logging
from kotonebot.primitives import Rect

from kaa.tasks import R
from kotonebot.backend.image import find_all
from kaa.db.skill_card import SkillCard
from kaa.game_ui.skill_card_select import match_card_region


# Covers all visible modal rows; the hand behind the dialog is blurred and is
# rejected by the stricter letter-template threshold in the caller.
MOVE_CARD_AREA = R.InProduce.BoxMoveCards
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MoveCardOption:
    letter_rect: Rect
    card: SkillCard | None


def _has_upgrade_marker(screen: MatLike, letter: Rect) -> bool:
    """The small red plus on the modal card distinguishes base from upgraded."""
    x = int(letter.x1)
    y = int(letter.y1)
    marker = screen[max(0, y - 60):max(0, y - 25), x + 40:x + 70]
    if marker.size == 0:
        return False
    red = (
        (marker[:, :, 2] > 200)
        & (marker[:, :, 1] < 180)
        & (marker[:, :, 0] < 200)
    )
    return int(red.sum()) >= 80


def find_move_scroll_thumb(screen: MatLike) -> int | None:
    """Locate the dark thumb on the move dialog's narrow right-hand track."""
    height, width = screen.shape[:2]
    track = R.InProduce.BoxMoveScrollTrack
    if height < track.y2 or width <= track.x1:
        return None
    column = screen[track.y1:track.y2, track.x1]
    blue = column[:, 0].astype('int16')
    green = column[:, 1].astype('int16')
    red = column[:, 2].astype('int16')
    thumb = (
        (blue > 100) & (blue < 140)
        & (abs(blue - green) <= 3)
        & (abs(green - red) <= 7)
    )
    best_start = best_length = 0
    run_start = None
    for index, is_thumb in enumerate(thumb):
        if is_thumb and run_start is None:
            run_start = index
        elif not is_thumb and run_start is not None:
            length = index - run_start
            if length > best_length:
                best_start, best_length = run_start, length
            run_start = None
    if run_start is not None and len(thumb) - run_start > best_length:
        best_start, best_length = run_start, len(thumb) - run_start
    return track.y1 + best_start + best_length // 2 if best_length >= 25 else None


def recognize_move_cards(
    screen: MatLike,
    letters: Iterable[Rect],
) -> list[MoveCardOption]:
    """Match visible modal cards while keeping duplicate copies separate."""
    height, width = screen.shape[:2]
    ordered = sorted(letters, key=lambda rect: (rect.y1, rect.x1))
    distinct: list[Rect] = []
    for letter in ordered:
        center = letter.center
        if not (
            MOVE_CARD_AREA.x1 <= center.x < MOVE_CARD_AREA.x2
            and MOVE_CARD_AREA.y1 <= center.y < MOVE_CARD_AREA.y2
        ):
            continue
        if any(
            abs(center.x - previous.center.x) < 25
            and abs(center.y - previous.center.y) < 25
            for previous in distinct
        ):
            continue
        distinct.append(letter)

    options: list[MoveCardOption] = []
    for letter in distinct:
        center_x = int(letter.center.x)
        x1, x2 = center_x - 60, center_x + 60
        y1, y2 = int(letter.y1) - 105, int(letter.y2)
        card = None
        if 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height:
            try:
                card = match_card_region(screen[y1:y2, x1:x2])
                if card is not None and card._asset_id and _has_upgrade_marker(screen, letter):
                    card = SkillCard.from_asset_id(card._asset_id, 1) or card
            except Exception:
                logger.exception('Could not identify a move-to-hold card.')
        options.append(MoveCardOption(letter, card))
    return options


def observe_move_cards(screen: MatLike) -> list[MoveCardOption]:
    """Locate letters and recognize card faces from exactly the same frame."""
    # Modal illustrations can resemble a letter at 0.8 (a saved T false match
    # scores 0.821); the reference's six real markers all exceed 0.92.
    letters = [
        match.rect
        for template in (R.InProduce.A, R.InProduce.M, R.InProduce.T)
        for match in find_all(screen, template.template, rect=MOVE_CARD_AREA, threshold=0.9)
    ]
    return recognize_move_cards(screen, letters)
