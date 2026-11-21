"""有界分屏采集；画面不变或预算耗尽都不是列表完整性证明。"""

from typing import Callable, Generic, Literal, TypeVar

T = TypeVar('T')


class ViewportScan(Generic[T]):
    def __init__(self, key: Callable[[T], str | None], has_content: Callable[[T], bool],
                 *, prefix: str, max_scrolls: int = 2, max_stale: int = 2) -> None:
        if max_scrolls < 0 or max_stale < 1:
            raise ValueError('invalid viewport scan limits')
        self._key = key
        self._has_content = has_content
        self._prefix = prefix
        self._max_scrolls = max_scrolls
        self._max_stale = max_stale
        self._pages: list[T] = []
        self._pending: str | None = None
        self._stale = 0
        self.scrolls = 0
        self._terminal_reason: str | None = None

    @property
    def viewports(self) -> tuple[T, ...]:
        return tuple(self._pages)

    def observe(self, viewport: T) -> tuple[Literal['scroll', 'wait', 'close'], str]:
        if self._terminal_reason is not None:
            return 'close', self._terminal_reason
        key = self._key(viewport)
        if key is None or not self._has_content(viewport):
            return self._close('viewport_unknown')
        if self._pending == key:
            self._stale += 1
            if self._stale >= self._max_stale:
                return self._close('scroll_not_confirmed')
            return 'wait', f'await_{self._prefix}_scroll'
        if any(self._key(page) == key for page in self._pages):
            return self._close('viewport_repeated')
        self._pages.append(viewport)
        self._pending = None
        self._stale = 0
        if self.scrolls >= self._max_scrolls:
            return self._close('scroll_limit')
        self.scrolls += 1
        self._pending = key
        return 'scroll', f'collect_next_{self._prefix}_viewport'

    def _close(self, reason: str) -> tuple[Literal['close'], str]:
        self._terminal_reason = f'{self._prefix}_{reason}'
        return 'close', self._terminal_reason
