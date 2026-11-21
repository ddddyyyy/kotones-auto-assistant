"""相邻道具视口的保守候选关联；不合并实体、恢复未知 OCR 或生成编组 ID。"""

from collections import Counter

from .items import (
    ItemOverlap, ItemOverlapLink, ItemOverlapStatus, ItemViewport,
    VisibleItem, is_confirmed_item_reference,
)


def _identity(item: VisibleItem) -> tuple[str, str] | None:
    reference = item.reference
    if (not is_confirmed_item_reference(reference) or reference.item_id is None
        or item.section not in ('stage', 'memory')):
        return None
    return reference.item_id, item.section


def associate_item_viewports(previous: ItemViewport, current: ItemViewport) -> ItemOverlap:
    """调用方须限定同一已绑定编成弹窗内的相邻视口；不能跨角色／会话使用。"""
    def result(status: ItemOverlapStatus) -> ItemOverlap:
        return ItemOverlap(previous.key, current.key, status)

    if not previous.key or not current.key or not previous.visible_items or not current.visible_items:
        return result(ItemOverlapStatus.UNKNOWN)
    if previous.key == current.key:
        return result(ItemOverlapStatus.SAME_VIEWPORT)
    old = [_identity(item) for item in previous.visible_items]
    new = [_identity(item) for item in current.visible_items]
    old_counts, new_counts = Counter(old), Counter(new)
    links = []
    for index, identity in enumerate(old):
        if identity is None or old_counts[identity] != 1 or new_counts[identity] != 1:
            continue
        current_index = new.index(identity)
        section = previous.visible_items[index].section
        if section is None:
            continue
        links.append(ItemOverlapLink(index, current_index, identity[0], section))
    # 同一个道具 ID 即使出现在不同分组，也不能算两个不同锚点。
    if len({link.item_id for link in links}) < 2:
        return result(ItemOverlapStatus.INSUFFICIENT)
    indices = [link.current_index for link in links]
    if indices != sorted(indices):
        return result(ItemOverlapStatus.ORDER_CONFLICT)
    shifts = []
    for link in links:
        left = previous.visible_items[link.previous_index]
        right = current.visible_items[link.current_index]
        if left.reference.confirmed_name != right.reference.confirmed_name:
            return result(ItemOverlapStatus.GEOMETRY_CONFLICT)
        x1, y1, w1, h1 = left.card_rect
        x2, y2, w2, h2 = right.card_rect
        if (any(type(value) is not int for value in (*left.card_rect, *right.card_rect))
            or min(x1, y1, x2, y2) < 0 or min(w1, h1, w2, h2) <= 0
            or abs(w1 - w2) > 2 or abs(h1 - h2) > 2):
            return result(ItemOverlapStatus.GEOMETRY_CONFLICT)
        shifts.append((x2 - x1, y2 - y1))
    dx, dy = shifts[0]
    if abs(dx) > 2 or any(abs(x - dx) > 2 or abs(y - dy) > 2 for x, y in shifts):
        return result(ItemOverlapStatus.GEOMETRY_CONFLICT)
    if any(y >= -2 for _, y in shifts):
        return result(ItemOverlapStatus.DIRECTION_UNKNOWN)
    return ItemOverlap(previous.key, current.key, ItemOverlapStatus.CANDIDATE, tuple(links), (dx, dy))
