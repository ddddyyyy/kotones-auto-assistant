"""日志往返的当前可见手牌摘要；不是完整牌库或稳定编组身份。"""
from dataclasses import dataclass,asdict
from hashlib import sha256
import json


@dataclass(frozen=True)
class LogHandCard:
    card_id: str
    upgrade_count: int | None
    available: bool
    rect: tuple[int,int,int,int]
    growth_marker: bool


@dataclass(frozen=True)
class LogHandObservation:
    frame_id: str
    count: int
    cards: tuple[LogHandCard,...]
    fingerprint: str
    empty_confirmed: bool = False


def fingerprint_log_hand(count: int, cards: tuple[LogHandCard,...], *, empty_confirmed: bool=False) -> str | None:
    if type(count) is not int or not 0<=count<=5 or len(cards)!=count or (count==0 and not empty_confirmed):
        return None
    if any(not item.card_id or type(item.available) is not bool or type(item.growth_marker) is not bool
           or (item.upgrade_count is not None and (type(item.upgrade_count) is not int or item.upgrade_count<0))
           or min(item.rect)<0 or min(item.rect[2:])<=0 or item.rect[0]+item.rect[2]>720
           or item.rect[1]+item.rect[3]>1280 for item in cards):
        return None
    ordered=sorted(cards,key=lambda item:item.rect[0])
    if len({item.rect for item in cards})!=count or len({item.rect[0] for item in cards})!=count:
        return None
    payload={'count':count,'cards':[asdict(item) for item in ordered]}
    return sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def confirm_log_hand(frame_id: str,count: int,cards: tuple[LogHandCard,...],*,empty_confirmed: bool=False) -> LogHandObservation | None:
    if not frame_id:
        return None
    fingerprint=fingerprint_log_hand(count,cards,empty_confirmed=empty_confirmed)
    return LogHandObservation(frame_id,count,cards,fingerprint,empty_confirmed) if fingerprint is not None else None
