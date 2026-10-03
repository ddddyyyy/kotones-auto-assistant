from dataclasses import dataclass
from functools import cached_property
import os
import cv2
import logging
from typing import Callable
from typing_extensions import override

from cv2.typing import MatLike
from kaa.db.constants import CharacterId
from kotonebot import image
from kotonebot.primitives import Rect
from kotonebot.core import GameObject
from kotonebot.backend import color

from kaa.tasks import R
from kaa.util import paths
from kaa.image_db import ImageDatabase
from kaa.image_db import registry
from kaa.db.skill_card import SkillCard

DEBUG = False
CARD_OFFSET = (57, 148) # 从字母位置到卡片左上角的偏移量 x, y
CARD_SCALE = 168 / 256 # 原始卡面图像 到 1280x720 截图中卡面图像的缩放比例

logger = logging.getLogger(__name__)


def _has_upgrade_marker(img: MatLike, letter_rect: Rect) -> bool:
    """Detect the red ``+`` at the lower-right of an unobscured hand card."""
    center_x = int(letter_rect.center.x)
    top_y = int(letter_rect.y1)
    h, w = img.shape[:2]
    x1, x2 = max(0, center_x + 60), min(w, center_x + 110)
    y1, y2 = max(0, top_y - 65), min(h, top_y - 10)
    if x1 >= x2 or y1 >= y2:
        return False
    marker = img[y1:y2, x1:x2]
    # The marker is saturated pink/red. Requiring an area, rather than one
    # pixel, rejects compression noise and the nearby green cost badge.
    red = (
        (marker[:, :, 2] > 210)
        & (marker[:, :, 1] < 170)
        & (marker[:, :, 0] < 210)
    )
    _, _, stats, _ = cv2.connectedComponentsWithStats(red.astype('uint8'))
    return any(
        width >= 12 and height >= 12 and area >= 100
        for _, _, width, height, area in stats[1:]
    )


def _upgrade_marker_visible(letter_rect: Rect, next_letter: Rect | None) -> bool:
    """Whether the next card leaves the lower-right upgrade badge exposed."""
    return (
        next_letter is None
        or next_letter.center.x - letter_rect.center.x >= 140
    )


def _visible_upgrade_count(img: MatLike, letter_rect: Rect) -> int:
    """Count trailing ``+`` glyphs in an unobscured card's name strip.

    The card artwork is shared by all upgrade levels. The title suffix is the
    only visible distinction between +, ++ and +++. A glyph is accepted only
    when both its horizontal and vertical strokes span the component and it
    belongs to the rightmost, contiguous suffix; Japanese name strokes elsewhere
    in the title must not count as upgrades.
    """
    center_x = int(letter_rect.center.x)
    top_y = int(letter_rect.y1)
    height, width = img.shape[:2]
    x1, x2 = max(0, center_x - 100), min(width, center_x + 100)
    y1, y2 = max(0, top_y + 20), min(height, top_y + 46)
    if x1 >= x2 or y1 >= y2:
        return 0
    title = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(title, cv2.COLOR_BGR2GRAY)
    # Selected cards render the final suffix in cyan rather than dark gray.
    cyan = (
        (title[:, :, 0] > 235)
        & (title[:, :, 1] > 130)
        & (title[:, :, 1] < 230)
        & (title[:, :, 2] < 140)
    )
    faded_cyan = (
        (title[:, :, 0] > 220)
        & (title[:, :, 1] > 130)
        & (title[:, :, 0].astype('int16') - title[:, :, 1] > 5)
        & (title[:, :, 0].astype('int16') - title[:, :, 2] > 12)
    )
    return max(
        _count_upgrade_suffix(((gray < 150) | mask).astype('uint8'))
        for mask in (cyan, faded_cyan)
    )


def _count_upgrade_suffix(ink: MatLike) -> int:
    """Count only contiguous cross-shaped glyphs at the end of a title."""
    labels, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    glyphs: list[tuple[int, int, int, bool]] = []
    for component in range(1, labels):
        x, y, w, h, area = (int(v) for v in stats[component])
        if not (5 <= area and 5 <= w <= 22 and 7 <= h <= 20):
            continue
        shape = ink[y:y+h, x:x+w]
        is_plus = (
            8 <= w <= 12
            and 8 <= h <= 12
            and 14 <= area <= 45
            and shape[h // 2, :].mean() >= 0.8
            and shape[:, w // 2].mean() >= 0.75
        )
        glyphs.append((x, y, w, is_plus))
    glyphs.sort(key=lambda glyph: glyph[0])
    if not glyphs or not glyphs[-1][3]:
        return 0
    suffix = 1
    for previous, following in zip(reversed(glyphs[:-1]), reversed(glyphs[1:])):
        if not previous[3] or following[0] - (previous[0] + previous[2]) > 6:
            break
        if abs(previous[1] - following[1]) > 4:
            break
        suffix += 1
    return min(suffix, 3)


def _observed_upgrade_count(
    img: MatLike, letter_rect: Rect, next_letter: Rect | None,
) -> int:
    """Read an exposed title suffix even if the red badge is faded."""
    if not _upgrade_marker_visible(letter_rect, next_letter):
        return 0
    suffix = _visible_upgrade_count(img, letter_rect)
    if _has_upgrade_marker(img, letter_rect):
        return max(1, suffix)
    # The title crop spans 200 px. A readable badge does not imply that its
    # trailing title is exposed in a crowded hand.
    title_visible = (
        next_letter is None
        or next_letter.center.x - letter_rect.center.x >= 200
    )
    return suffix if title_visible else 0

@dataclass
class CardGameObject(GameObject):
    rect: Rect
    res_name: str
    card: SkillCard | None

    _screenshot: MatLike
    _letter_rect: Rect

    @cached_property
    def available(self):
        return color.find(self._screenshot, '#7a7d7d', rect=self._letter_rect) is None
    
    def __repr__(self) -> str:
        return f'CardGameObject(res_name={self.res_name}, card={self.card}, available={self.available}, _screenshot={"<...>" if self._screenshot is not None else "<None>"}, _letter_rect={self._letter_rect})'

class CardImageDatabase(ImageDatabase):
    @override
    def build(self, progress_cb=None):
        """构建时对每张卡片图像进行裁剪预处理。"""
        class PreprocessingSource:
            def __init__(self, source):
                self._source = source
            
            def __iter__(self):
                for key, img in self._source:
                    # 截取从下边缘中点为右下角，
                    # 到 CARD_OFFSET * CARD_SCALE 为左上角的区域
                    h, w, _ = img.shape
                    x2, y2 = w // 2, h  # 右下角
                    x1 = int(x2 - CARD_OFFSET[0] / CARD_SCALE)
                    y1 = int(y2 - CARD_OFFSET[1] / CARD_SCALE)
                    half_img = img[y1:y2, x1:x2]
                    
                    # if DEBUG:
                    #     debug_img = img.copy()
                    #     cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    #     cv2.imshow('original_with_rect', cv2.resize(debug_img, (0,0), fx=0.5, fy=0.5))
                    #     cv2.imshow('half_img', cv2.resize(half_img, (0,0), fx=2, fy=2))
                    #     cv2.waitKey(0)
                    yield key, half_img
        
        self.source = PreprocessingSource(self.source)
        super().build(progress_cb=progress_cb)

def build_db(
    progress_cb: Callable[[int, int], None] | None = None,
    *,
    source_dir: str | None = None,
    cache_dir: str | None = None,
):
    """构建技能卡图像数据库索引（薄封装，委托注册中心）。

    :param progress_cb: 进度回调 (processed, total)
    :param source_dir: 数据源目录；为 None 时使用活跃游戏数据目录
    :param cache_dir: 索引缓存目录；为 None 时使用默认缓存目录
    """
    registry._build_spec(
        registry.get_spec('skill_cards'),
        source_dir,
        cache_dir,
        progress_cb=progress_cb,
    )


def skill_cards_db() -> ImageDatabase:
    return registry.get_db('skill_cards')


def _show_rects(title: str, img: MatLike, results: list[GameObject] | list[Rect]):
    if not DEBUG:
        return
    debug_img = img.copy()
    for res in results:
        if isinstance(res, Rect):
            x, y, w, h = res.xywh
        else:
            x, y, w, h = res.rect.xywh
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow(title, cv2.resize(debug_img, (0, 0), fx=0.5, fy=0.5))

def _locate_letters(img: MatLike) -> list[Rect]:
    # letters = AnyOf[
    #     R.InProduce.A,
    #     R.InProduce.M,
    #     R.InProduce.T,
    # ].find_all()
    x, y, w, h = R.InProduce.BoxCardLetter.xywh
    img = img[y:y+h, x:x+w]
    results = image.raw().find_all(img, R.InProduce.A.template)
    results += image.raw().find_all(img, R.InProduce.M.template)
    results += image.raw().find_all(img, R.InProduce.T.template)
    # 需要还原为全图坐标
    results2 = []
    for res in results:
        results2.append(Rect(res.rect.x1 + x, res.rect.y1 + y, res.rect.w, res.rect.h))

    _show_rects('letters', img, results2)
    return results2

def locate_cards(img: MatLike):
    letters = _locate_letters(img)
    ordered_letters = sorted(letters, key=lambda item: item.center.x)
    # 根据字母的位置，计算出左半边卡面的范围
    # 字母往上 195px，往左 95px
    card_regions = []
    for l in letters:
        x2, y2 = l.center.x, l.top_left.y # x 坐标取中心，y 坐标取上边缘
        x1 = x2 - CARD_OFFSET[0]
        y1 = y2 - CARD_OFFSET[1]
        card_regions.append(Rect.from_xyxy(x1, y1, x2, y2))
    
    _show_rects('cards', img, card_regions)

    # 根据图像查卡片
    results: list[CardGameObject | None] = []
    for region, letter in zip(card_regions, letters):
        x, y, w, h = region.xywh
        card_img = img[y:y+h, x:x+w]
        result_list = skill_cards_db().query(card_img, k=1, threshold=150)
        result = result_list[0] if result_list else None
        available = color.find(img, '#7a7d7d', rect=letter) is None
        if result is not None:
            # 从资源名称中提取 asset_id
            card_id = result.key.removesuffix('.png')
            for c in CharacterId:
                suffix = '-' + c.value
                card_id = card_id.removesuffix(suffix)
            # 查数据库
            # The marker is visible on any sufficiently separated card, not
            # just the rightmost one. With a crowded hand the next card covers
            # it; never interpret that occlusion as evidence of a base card.
            index = ordered_letters.index(letter)
            next_letter = ordered_letters[index + 1] if index + 1 < len(ordered_letters) else None
            upgrade_count = _observed_upgrade_count(img, letter, next_letter)
            db_card = SkillCard.from_asset_id(card_id, upgrade_count)
            card = CardGameObject(
                rect=region,
                res_name=result.key,
                card=db_card,
                _screenshot=img,
                _letter_rect=letter,
            )
            results.append(card)
            if card.card is not None:
                logger.info(f'Matched skill card: {card.res_name}({card.card.name}) | dist={result.distance:.2f} available={available}')
            else:
                logger.warning(f'Found skill card {card.res_name} but no database match found.')
        else:
            logger.warning('No matching skill card found.')
            results.append(None)
    results.sort(key=lambda r: r.rect.x1 if r is not None else 1e9)

    # 绘制debug结果
    # 新建一张与原图一样大小的空白图，把卡片完整图像绘制到识别区域上
    if DEBUG:
        preview = img.copy() * 0
        for res in results:
            if res is None:
                continue
            draw_rect = res.rect.copy()
            orig_card_path = os.path.join(paths.resource('skill_cards'), res.res_name)
            orig_card_img = cv2.imread(orig_card_path)
            assert orig_card_img is not None
            orig_card_img = cv2.resize(orig_card_img, None, fx=CARD_SCALE, fy=CARD_SCALE)
            x1, y1 = draw_rect.top_left
            x2, y2 = x1 + orig_card_img.shape[1], y1 + orig_card_img.shape[0]
            preview[y1:y2, x1:x2] = orig_card_img

            x, y, w, h = res.rect.xywh
        cv2.imshow('matched_cards', cv2.resize(preview, (0,0), fx=0.5, fy=0.5))
        cv2.waitKey(1)
    return results


if __name__ == "__main__":
    # sample = cv2.imread('f:/1.png')
    # db = skill_cards_db()
    # result = db.match(sample)
    # print(result)

    skill_cards_db()
    from pprint import pprint
    from time import time
    from kotonebot import device
    while True:
        img = device.screenshot()
        start_time = time()
        cards = locate_cards(img)
        end_time = time()
        print(f'Locate cards took {end_time - start_time:.2f} seconds')
        # pprint(cards)
        cv2.waitKey(0)
