"""逐星位识别；不从标题、名次或缺失匹配推断星数。"""

import cv2
from cv2.typing import MatLike
from kotonebot.backend.image import find
from kotonebot.primitives import Rect


def count_known_stars(slots: tuple[bool | None, ...]) -> int | None:
    """三个位置均有明确证据才返回总数，不假设亮星必须连续。"""
    if len(slots) != 3 or any(type(slot) is not bool for slot in slots):
        return None
    return sum(slot is True for slot in slots)


def read_stars(
    screen: MatLike, slots: tuple[Rect, Rect, Rect],
    filled: tuple[MatLike, MatLike, MatLike], empty: MatLike,
    *, filled_alternates: tuple[MatLike,...] = (),
) -> int | None:
    observations: list[bool | None] = []
    for slot, filled_template in zip(slots, filled):
        if slot.x1 < 0 or slot.y1 < 0 or slot.x2 > screen.shape[1] or slot.y2 > screen.shape[0]:
            return None
        filled_match = any(find(screen, template, rect=slot, colored=True, threshold=0.97) is not None
                           for template in (filled_template,*filled_alternates))
        empty_match = find(screen, empty, rect=slot, colored=True, threshold=0.97)
        if (not filled_match) == (empty_match is None):
            observations.append(None)
        else:
            observations.append(filled_match)
    return count_known_stars(tuple(observations))


def resize_star(reference: MatLike, size: tuple[int, int]) -> MatLike:
    """只缩放已采集图案；缩放后仍须实际模板匹配，不保证适用于其他页面。"""
    return cv2.resize(reference, size, interpolation=cv2.INTER_LINEAR)
