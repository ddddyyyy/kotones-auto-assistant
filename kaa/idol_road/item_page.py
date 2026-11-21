"""当前帧完整可见卡片及分组识别，不把分屏观察当作完整装备。"""

from uuid import uuid4
import hashlib
from typing import Literal

import cv2
import numpy as np
from cv2.typing import MatLike
from kotonebot.backend.ocr import jp
from kotonebot.backend.image import find

from kaa.db.tower_rules import produce_item_name_candidates
from kaa.tasks import R

from .items import ItemBindingStatus, ItemCandidate, ItemNameRead, ItemViewport, VisibleItem, bind_item_reference, confirmed_item_name, normalize_item_name
from .item_description_page import read_item_description


def read_item_viewport(screen: MatLike) -> ItemViewport:
    road = R.IdolRoad
    box = road.BoxItemCards
    crop = screen[box.y1:box.y2, box.x1:box.x2]
    mask = cv2.inRange(crop, np.array([230, 230, 230]), np.array([255, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reference = road.BoxItemCardReference
    name_box = road.BoxItemName
    dx, dy = name_box.x1 - reference.x1, name_box.y1 - reference.y1
    cards = sorted((cv2.boundingRect(contour) for contour in contours), key=lambda rect: rect[1])
    frame_id = uuid4().hex
    items = []
    previous_bottom: int | None = None
    previous_section: Literal['stage', 'memory'] | None = None
    fingerprint = hashlib.sha256()
    sections: list[tuple[int, Literal['stage', 'memory']]] = []
    for kind, template in (('stage', road.TextItemsStageSection), ('memory', road.TextItemsMemorySection)):
        matched = find(screen, template.template, rect=road.BoxItemSections, threshold=0.97)
        if matched is not None:
            section: Literal['stage', 'memory'] = 'stage' if kind == 'stage' else 'memory'
            sections.append((matched.rect.y2, section))
    for x, y, width, height in cards:
        if width < reference.w * 0.9 or height < dy + name_box.h:
            continue
        if y <= 0 or y + height >= crop.shape[0] or width < dx + name_box.w:
            # 顶／底截断时，卡片剩余正文不能冒充其名称。
            continue
        x, y = x + box.x1, y + box.y1
        title = screen[y + dy:y + dy + name_box.h, x + dx:x + dx + name_box.w]
        fingerprint.update(str((x, y, width, height)).encode('ascii'))
        fingerprint.update(title.tobytes())
        reads = []
        variants: tuple[tuple[Literal['original', 'grayscale'], MatLike], ...] = (
            ('original', title), ('grayscale', cv2.cvtColor(title, cv2.COLOR_BGR2GRAY)),
        )
        for representation, variant in variants:
            rows = jp().ocr(variant)
            reads.append(ItemNameRead(frame_id, representation,
                                      rows[0].text if len(rows) == 1 else None,
                                      rows[0].confidence if len(rows) == 1 else None))
        name_reads = tuple(reads)
        name = confirmed_item_name(name_reads)
        source = produce_item_name_candidates() if name is not None else None
        candidates = tuple(ItemCandidate(item.id, item.name) for item in source
                           if normalize_item_name(item.name) == name) if source is not None else None
        # 只在本帧的连续卡片段中继承分组；未读到的新标题／段间空隙不能继承旧分组。
        preceding = [entry for entry in sections if 0 <= y - entry[0] <= name_box.h]
        section_name: Literal['stage', 'memory'] | None = None
        if preceding:
            section_name = max(preceding, key=lambda entry: entry[0])[1]
        elif previous_bottom is not None and 0 <= y - previous_bottom <= name_box.h // 2:
            section_name = previous_section
        reference_binding = bind_item_reference(name_reads, candidates)
        card_rect = (x, y, width, height)
        if reference_binding.status == ItemBindingStatus.AMBIGUOUS:
            # Same name with a changed condition/number is a different viewport.
            fingerprint.update(screen[y+dy+name_box.h:y+height, x+dx:x+width].tobytes())
        description = read_item_description(screen, reference_binding, card_rect)
        items.append(VisibleItem(reference_binding, card_rect, section_name, description))
        previous_bottom, previous_section = y + height, section_name
    return ItemViewport(fingerprint.hexdigest() if items else None, tuple(items) if items else None)


def read_visible_items(screen: MatLike) -> tuple[VisibleItem, ...] | None:
    return read_item_viewport(screen).visible_items
