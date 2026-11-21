"""有限滚动状态；保留分屏观察，不由画面不动推断到达列表末尾。"""

from .items import ItemViewport
from .viewport_scan import ViewportScan


class ItemScan(ViewportScan[ItemViewport]):
    def __init__(self, *, max_scrolls: int = 2, max_stale: int = 2) -> None:
        super().__init__(lambda page: page.key, lambda page: bool(page.visible_items),
                         prefix='item', max_scrolls=max_scrolls, max_stale=max_stale)
