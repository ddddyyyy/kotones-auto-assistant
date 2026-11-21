"""Read the audited four-line item layout; other layouts remain unknown."""

import cv2
import numpy as np
from cv2.typing import MatLike
from kotonebot.backend.ocr import en, jp
from kaa.db.tower_rules import produce_item_by_id
from kaa.tasks import R

from .items import ItemBindingStatus, ItemReference
from .item_description import (
    ItemDescriptionBinding, ItemDescriptionCandidate, ItemDescriptionLine,
    ItemDescriptionRead, bind_item_description,
)


def read_item_description(screen: MatLike, reference: ItemReference,
                          card_rect: tuple[int, int, int, int]) -> ItemDescriptionBinding | None:
    road = R.IdolRoad
    anchor = road.BoxItemCardReference
    x, y, width, height = card_rect
    if reference.status != ItemBindingStatus.AMBIGUOUS or (width, height) != (anchor.w, anchor.h):
        return None
    dx, dy = x - anchor.x1, y - anchor.y1
    body = road.BoxItemDescriptionBody
    bx, by = body.x1 + dx, body.y1 + dy
    if bx < 0 or by < 0 or bx + body.w > screen.shape[1] or by + body.h > screen.shape[0]:
        return None
    slots = (road.BoxItemDescriptionActive, road.BoxItemDescriptionMental,
             road.BoxItemDescriptionCondition, road.BoxItemDescriptionRecovery, road.BoxItemDescriptionNumber)
    covered = np.zeros((body.h, body.w), dtype=np.uint8)
    for box in (*slots, road.BoxItemDescriptionCategoryIcons, road.BoxItemDescriptionEffectIcons):
        covered[box.y1-body.y1:box.y2-body.y1, box.x1-body.x1:box.x2-body.x1] = 1
    crop = screen[by:by+body.h, bx:bx+body.w]
    # Every dark/colored text pixel must lie in a read slot or explicit icon slot.
    # This rejects extra suffixes and extra conditions instead of ignoring them.
    if np.any((np.min(crop, axis=2) < 170) & (covered == 0)):
        return None
    frame_id = reference.name_reads[0].frame_id
    reads = []
    for representation in ('original', 'grayscale'):
        lines = []
        for index, box in enumerate(slots):
            part = screen[box.y1+dy:box.y2+dy, box.x1+dx:box.x2+dx]
            if representation == 'grayscale':
                part = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)
            rows = (en() if index == len(slots)-1 else jp()).ocr(part, only_rec=True)
            lines.append(ItemDescriptionLine(rows[0].text if len(rows) == 1 else '',
                                             rows[0].confidence if len(rows) == 1 else None))
        reads.append(ItemDescriptionRead(frame_id, representation, tuple(lines)))
    candidates = []
    for item_id in reference.candidate_ids:
        item = produce_item_by_id(item_id)
        if item is None:
            return bind_item_description(reference, tuple(reads), None)
        candidates.append(ItemDescriptionCandidate(item.id, item.name, item.description_json))
    return bind_item_description(reference, tuple(reads), tuple(candidates))
