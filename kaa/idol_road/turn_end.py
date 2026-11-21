"""整手明确禁用时单次结束回合；未识别的牌不等于禁用。"""
from dataclasses import dataclass
from typing import Literal
from collections.abc import Callable
from time import monotonic
from math import isfinite


@dataclass(frozen=True)
class HandCardState:
    card_id: str | None
    available: bool | None
    rect: tuple[int, int, int, int] | None


def hand_is_blocked(expected_count: int | None, cards: tuple[HandCardState, ...]) -> bool:
    if type(expected_count) is not int or not 1 <= expected_count <= 5 or len(cards) != expected_count:
        return False
    if any(not card.card_id or card.available is not False or card.rect is None
           or card.rect[2] <= 0 or card.rect[3] <= 0 for card in cards):
        return False
    return len({card.rect for card in cards}) == expected_count


class UnplayableTurnGate:
    """按实际战斗和回合去重；点击后只等待新回合，不重复SKIP。"""
    def __init__(self, *, max_waits: int = 3):
        if max_waits < 1:
            raise ValueError('max_waits must be positive')
        self.pending: tuple[str, int] | None = None
        self.waits = 0
        self.max_waits = max_waits
        self._last_seen: tuple[str,int] | None=None

    def step(self, battle_id: str, turn: int, blocked: bool, enabled: bool) -> Literal['click', 'wait', 'none', 'stop']:
        if not battle_id or type(turn) is not int or turn < 1:
            return 'stop'
        key = (battle_id, turn)
        if self._last_seen is not None and (self._last_seen[0]!=battle_id or turn<self._last_seen[1]):
            return 'stop'
        self._last_seen=key
        if self.pending is not None:
            if self.pending == key:
                self.waits += 1
                return 'stop' if self.waits >= self.max_waits else 'wait'
            self.pending = None
            self.waits = 0
        if blocked is True and enabled is True:
            self.pending = key
            return 'click'
        return 'none'


class PlannedTurnWait:
    """规划SKIP后暂停所有选牌，直到实际换回合；动画残帧不能重入策略。"""
    def __init__(self, *, clock: Callable[[], float] = monotonic):
        self.clock = clock
        self.pending: tuple[int, float] | None = None
        self.stopped = False

    def requested(self, turn: int, *, started: float | None = None) -> None:
        now = self.clock()
        origin = now if started is None else started
        if (type(turn) is not int or turn<1 or self.pending is not None or self.stopped
            or type(origin) not in (int,float) or not isfinite(origin)
            or not 0 <= now-origin < 20):
            raise ValueError('结束回合请求缺少可靠回合或已有请求未完成')
        self.pending = (turn, origin)

    def check(self, turn: int | None) -> Literal['continue', 'wait', 'stop']:
        if self.stopped:
            return 'stop'
        if self.pending is None:
            return 'continue'
        previous, started = self.pending
        elapsed = self.clock()-started
        if (not 0 <= elapsed < 20 or (turn is not None
            and (type(turn) is not int or turn < previous))):
            self.stopped = True
            return 'stop'
        if type(turn) is int and turn > previous:
            self.pending = None
            return 'continue'
        return 'wait'
