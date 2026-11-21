"""可见道具名称与显式数据库候选的保守绑定，不判断装备完整性或效果生效。"""

from dataclasses import dataclass
from enum import Enum
import math
import re
import unicodedata
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .item_description import ItemDescriptionBinding


@dataclass(frozen=True)
class ItemNameRead:
    frame_id: str
    representation: Literal['original', 'grayscale']
    text: str | None
    confidence: float | None


@dataclass(frozen=True)
class ItemCandidate:
    id: str
    name: str


class ItemBindingStatus(str, Enum):
    UNKNOWN_NAME = 'unknown_name'
    DATABASE_UNAVAILABLE = 'database_unavailable'
    NO_EXACT_MATCH = 'no_exact_match'
    CANDIDATE_CONFLICT = 'candidate_conflict'
    AMBIGUOUS = 'ambiguous'
    RESOLVED = 'resolved'


@dataclass(frozen=True)
class ItemReference:
    name_reads: tuple[ItemNameRead, ...]
    status: ItemBindingStatus
    confirmed_name: str | None
    candidate_ids: tuple[str, ...]
    item_id: str | None = None


@dataclass(frozen=True)
class VisibleItem:
    reference: ItemReference
    card_rect: tuple[int, int, int, int]
    """当前帧卡片 x/y/w/h，不是编组槽位或单件实例 ID。"""
    section: Literal['stage', 'memory'] | None = None
    description_binding: 'ItemDescriptionBinding | None' = None


@dataclass(frozen=True)
class ItemViewport:
    key: str | None
    visible_items: tuple[VisibleItem, ...] | None


class ItemOverlapStatus(str, Enum):
    CANDIDATE = 'candidate_alignment'
    UNKNOWN = 'unknown_viewport'
    SAME_VIEWPORT = 'same_viewport'
    INSUFFICIENT = 'insufficient_distinct_anchors'
    GEOMETRY_CONFLICT = 'geometry_conflict'
    ORDER_CONFLICT = 'order_conflict'
    DIRECTION_UNKNOWN = 'scroll_direction_unconfirmed'


@dataclass(frozen=True)
class ItemOverlapLink:
    previous_index: int
    current_index: int
    item_id: str
    section: Literal['stage', 'memory']


@dataclass(frozen=True)
class ItemOverlap:
    previous_key: str | None
    current_key: str | None
    status: ItemOverlapStatus
    links: tuple[ItemOverlapLink, ...] = ()
    content_shift: tuple[int, int] | None = None
    same_instances_confirmed: Literal[False] = False
    """只是同一弹窗相邻视口的候选对齐，不证明装备实体相同或数量。"""


@dataclass(frozen=True)
class ItemInventory:
    character_id: str
    stage: int
    visible_items: tuple[VisibleItem, ...] | None
    complete: Literal[False] = False
    source: str = 'visible_item_details'
    identity_source: str = 'opened_from_confirmed_formation'
    viewports: tuple[ItemViewport, ...] = ()
    """分屏观察，不能直接拼接成唯一装备清单或按名称去重。"""
    overlaps: tuple[ItemOverlap, ...] = ()


def normalize_item_name(text: str) -> str:
    # 不做假名替换、模糊匹配、词典纠错或子串截取。
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))


def confirmed_item_name(reads: tuple[ItemNameRead, ...]) -> str | None:
    if (len(reads) != 2 or {read.representation for read in reads} != {'original', 'grayscale'}
        or not isinstance(reads[0].frame_id, str) or not reads[0].frame_id
        or reads[0].frame_id != reads[1].frame_id):
        return None
    names = []
    for read in reads:
        confidence = read.confidence
        if (confidence is None or type(confidence) not in (int, float) or not math.isfinite(confidence)
            or not 0.98 <= confidence <= 1 or not isinstance(read.text, str)):
            return None
        name = normalize_item_name(read.text)
        if not name:
            return None
        names.append(name)
    return names[0] if names[0] == names[1] else None


def is_confirmed_item_reference(reference: ItemReference) -> bool:
    return (reference.status == ItemBindingStatus.RESOLVED and bool(reference.item_id)
            and reference.candidate_ids == (reference.item_id,) and reference.confirmed_name is not None
            and confirmed_item_name(reference.name_reads) == reference.confirmed_name)


def bind_item_reference(
    reads: tuple[ItemNameRead, ...], candidates: tuple[ItemCandidate, ...] | None,
) -> ItemReference:
    """双表示同帧姓名确认后，仅允许唯一同名候选；不选最高稀有度或首项。"""
    name = confirmed_item_name(reads)
    if name is None:
        return ItemReference(reads, ItemBindingStatus.UNKNOWN_NAME, None, ())
    if candidates is None:
        return ItemReference(reads, ItemBindingStatus.DATABASE_UNAVAILABLE, name, ())
    ids = tuple(candidate.id for candidate in candidates)
    if any(not isinstance(candidate.id, str) or not candidate.id
           or not isinstance(candidate.name, str) or normalize_item_name(candidate.name) != name for candidate in candidates):
        return ItemReference(reads, ItemBindingStatus.CANDIDATE_CONFLICT, name, ids)
    if not candidates:
        return ItemReference(reads, ItemBindingStatus.NO_EXACT_MATCH, name, ())
    if len(candidates) != 1:
        return ItemReference(reads, ItemBindingStatus.AMBIGUOUS, name, ids)
    return ItemReference(reads, ItemBindingStatus.RESOLVED, name, ids, candidates[0].id)
