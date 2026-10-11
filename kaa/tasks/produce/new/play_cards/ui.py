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
from kotonebot.backend.ocr import jp
from kotonebot.backend import color

from kaa.tasks import R
from kaa.util import paths
from kaa.image_db import ImageDatabase
from kaa.image_db import registry
from kaa.db.skill_card import (
    SkillCard, category_from_asset_id, asset_from_card_title, normalize_card_title,
)
from .card_output import CardOutputPreview, read_card_output_preview

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
    # Unavailable cards dim both the pink cross and its white outline.
    # Keep the normal mask unchanged; only supplement locally dark crosses.
    channels = marker.astype('int16')
    dim_red = (
        (channels[:, :, 2] > 80) & (channels[:, :, 2] < 160)
        & (channels[:, :, 2] - channels[:, :, 1] > 35)
        & (channels[:, :, 2] - channels[:, :, 0] > 15)
    )
    for mask, dimmed in ((red, False), (dim_red, True)):
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'))
        for x, y, width, height, area in stats[1:]:
            if not (12 <= width <= 32 and 12 <= height <= 32 and area >= 100):
                continue
            if dimmed and max(cv2.mean(marker[max(0, y-2):y+height+2, max(0, x-2):x+width+2])[:3]) > 180:
                continue
            shape = mask[y:y+height, x:x+width]
            # Card artwork and orange backgrounds can pass the color/area test.
            # A plus has two central strokes with mostly empty outer corners.
            quarter_w, quarter_h = width // 4, height // 4
            corners = (
                shape[:quarter_h, :quarter_w], shape[:quarter_h, -quarter_w:],
                shape[-quarter_h:, :quarter_w], shape[-quarter_h:, -quarter_w:],
            )
            if (shape[height // 2, :].mean() >= 0.8
                    and shape[:, width // 2].mean() >= 0.75
                    and sum(corner.mean() for corner in corners) / 4 < 0.25):
                return True
    return False


def _upgrade_marker_visible(letter_rect: Rect, next_letter: Rect | None) -> bool:
    """Whether the next card leaves the lower-right upgrade badge exposed."""
    return (
        next_letter is None
        or next_letter.center.x - letter_rect.center.x >= 140
    )


def _visible_upgrade_count(
    img: MatLike, letter_rect: Rect, next_letter: Rect | None = None,
) -> int:
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
    # Stop before the next card's name strip. The existing 100 px half-width
    # also leaves a conservative margin before its physical left edge.
    crowded = next_letter is not None and next_letter.center.x - center_x < 200
    if crowded and next_letter is not None:
        x2 = min(x2, int(next_letter.center.x) - 100)
    y1, y2 = max(0, top_y + 20), min(height, top_y + 46)
    if x1 >= x2 or y1 >= y2:
        return 0
    title = img[y1:y2, x1:x2]
    # A cut through the name can make a Japanese stroke resemble a trailing
    # cross. Require blank space after the suffix in the isolated strip.
    if crowded and (title.shape[1] < 4 or
                    not (cv2.cvtColor(title[:, -4:], cv2.COLOR_BGR2GRAY) >= 190).all()):
        return 0
    return _title_upgrade_suffix(title)[0]


def _title_upgrade_suffix(title: MatLike) -> tuple[int, int | None]:
    """Locate the same upgrade glyphs for level reading and title isolation."""
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
    suffix = max(
        (_upgrade_suffix(((gray < 150) | mask).astype('uint8'))
         for mask in (cyan, faded_cyan)),
        key=lambda suffix: suffix[0],
    )
    if suffix[0]:
        return suffix
    # A pale gray title can lose its cross in the normal dark-ink mask.
    # Require the same trailing cross count and position on both masks of
    # this crop; retain the existing shape and contiguous-suffix checks.
    pale_suffixes = [
        _upgrade_suffix((gray < threshold).astype('uint8'))
        for threshold in (170, 190)
    ]
    if pale_suffixes[0][0] > 0 and pale_suffixes[0] == pale_suffixes[1]:
        return pale_suffixes[0]
    return suffix


def _count_upgrade_suffix(ink: MatLike) -> int:
    """Count only contiguous cross-shaped glyphs at the end of a title."""
    return _upgrade_suffix(ink)[0]


def _upgrade_suffix(ink: MatLike) -> tuple[int, int | None]:
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
        return 0, None
    suffix = 1
    for previous, following in zip(reversed(glyphs[:-1]), reversed(glyphs[1:])):
        if not previous[3] or following[0] - (previous[0] + previous[2]) > 6:
            break
        if abs(previous[1] - following[1]) > 4:
            break
        suffix += 1
    return min(suffix, 3), glyphs[-suffix][0]


def _observed_upgrade_count(
    img: MatLike, letter_rect: Rect, next_letter: Rect | None,
) -> int:
    """Read an exposed title suffix even if the red badge is faded."""
    if not _upgrade_marker_visible(letter_rect, next_letter):
        return 0
    suffix = _visible_upgrade_count(img, letter_rect, next_letter)
    if _has_upgrade_marker(img, letter_rect):
        return max(1, suffix)
    return suffix

@dataclass
class CardGameObject(GameObject):
    rect: Rect
    res_name: str
    card: SkillCard | None

    _screenshot: MatLike
    _letter_rect: Rect
    _recognition_distance: float | None = None
    _recognition_title: str | None = None
    _recognition_title_confidence: float | None = None

    def _read_output_preview(self, *, wide_header: bool) -> CardOutputPreview | None:
        if any(
            0 < letter.center.x - self._letter_rect.center.x < 200
            for letter in _locate_letters(self._screenshot)
        ):
            return None
        return read_card_output_preview(
            self._screenshot, self._letter_rect, wide_header=wide_header,
        )

    @cached_property
    def output_preview(self) -> CardOutputPreview | None:
        return self._read_output_preview(wide_header=False)

    @cached_property
    def wide_output_preview(self) -> CardOutputPreview | None:
        return self._read_output_preview(wide_header=True)

    @cached_property
    def has_growth_marker(self) -> bool:
        """Visible growth indicator only; it does not identify its value or instance."""
        x = int(self._letter_rect.center.x)
        y = self._letter_rect.y1
        height, width = self._screenshot.shape[:2]
        x1, y1 = max(0, x - 120), max(0, y - 240)
        x2, y2 = min(width, x - 70), min(height, y - 185)
        if x2 - x1 < 26 or y2 - y1 < 26:
            return False
        return image.raw().find(
            self._screenshot, R.InProduce.IconCardGrowth.template,
            rect=Rect.from_xyxy(x1, y1, x2, y2), threshold=0.9,
        ) is not None

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

def _locate_typed_letters(img: MatLike) -> list[tuple[Rect, str]]:
    x, y, w, h = R.InProduce.BoxCardLetter.xywh
    crop = img[y:y+h, x:x+w]
    results: list[tuple[Rect, str]] = []
    for resource, category in (
        (R.InProduce.A, 'ProduceCardCategory_ActiveSkill'),
        (R.InProduce.M, 'ProduceCardCategory_MentalSkill'),
        (R.InProduce.T, 'ProduceCardCategory_Trouble'),
    ):
        for match in image.raw().find_all(crop, resource.template):
            rect = match.rect
            results.append((Rect(rect.x1 + x, rect.y1 + y, rect.w, rect.h), category))
    _show_rects('letters', crop, [rect for rect, _ in results])
    return results


def _locate_letters(img: MatLike) -> list[Rect]:
    return [rect for rect, _ in _locate_typed_letters(img)]


def _read_card_title(
    img: MatLike, letter: Rect, next_letter: Rect | None,
) -> tuple[str, float] | None:
    """Verify an exposed title on the current frame, independently of artwork."""
    if next_letter is not None and next_letter.center.x - letter.center.x < 200:
        return None
    x, y = int(letter.center.x), letter.y1
    if x < 100 or x + 100 > img.shape[1] or y + 20 < 0 or y + 46 > img.shape[0]:
        return None
    crop = img[y+20:y+46, x-100:x+100]
    _, suffix_start = _title_upgrade_suffix(crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ink_threshold = 150
    # Unavailable titles dim the background as well as the suffix. Fixed
    # bright-title masks merge both into one component. Isolate only a
    # trailing cross found at the same position by two current-crop masks.
    if suffix_start is None and cv2.mean(gray)[0] < 150:
        threshold, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        dim_suffixes = [
            _upgrade_suffix((gray < level).astype('uint8'))
            for level in (max(0, threshold - 10), threshold)
        ]
        if dim_suffixes[0][0] > 0 and dim_suffixes[0] == dim_suffixes[1]:
            suffix_start = dim_suffixes[0][1]
            ink_threshold = int(threshold)
    # A frame at the left edge can be recognized as a leading '1'. Remove
    # only a thin component touching both vertical crop edges; title glyphs
    # leave padding above and below. Do not delete characters from OCR text.
    _, _, components, _ = cv2.connectedComponentsWithStats(
        (gray < ink_threshold).astype('uint8'),
    )
    frame_end = max(
        (int(cx + width) for cx, cy, width, height, _area in components[1:]
         if cx <= 12 and cy == 0 and width <= 8 and height == crop.shape[0]),
        default=0,
    )
    # Upgrade crosses are not part of the base identity. Both limits come
    # from components on this crop, without guessing or changing its text.
    crop = crop[:, frame_end:suffix_start]
    if crop.size == 0:
        return None
    readings: list[tuple[str, float]] = []
    confident_titles: set[str] = set()
    try:
        for scale in (1, 2, 3):
            enlarged = cv2.resize(crop, None, fx=scale, fy=scale)
            results = jp().ocr(enlarged)
            confident_titles.update(
                normalize_card_title(result.text) for result in results
                if result.confidence >= 0.98
            )
            if len(results) == 1 and results[0].confidence >= 0.98:
                title = normalize_card_title(results[0].text)
                if title:
                    readings.append((title, results[0].confidence))
    except Exception:
        logger.debug('Could not verify card title.', exc_info=True)
        return None
    if len(confident_titles) > 1:
        return None
    if len(readings) >= 2:
        return readings[0][0], min(confidence for _, confidence in readings)
    # Dim titles may remain low confidence on every color enlargement.
    # Strip only full-height narrow frame components, then corroborate text
    # on masks of this same crop. Never override a high-confidence conflict.
    _, _, edges, _ = cv2.connectedComponentsWithStats((gray < 210).astype('uint8'))
    left_edge = max(
        (int(cx + width) for cx, cy, width, height, _area in edges[1:]
         if cx <= 12 and cy == 0 and width <= 12 and height == gray.shape[0]),
        default=0,
    )
    right_edge = min(
        (int(cx) for cx, cy, width, height, _area in edges[1:]
         if cx + width >= gray.shape[1] - 1 and cy == 0
         and width <= 12 and height == gray.shape[0]),
        default=gray.shape[1],
    )
    clean = gray[:, max(frame_end, left_edge):min(suffix_start or gray.shape[1], right_edge)]
    if clean.size == 0:
        return None
    mask_readings: list[tuple[str, float]] = []
    try:
        for threshold in (210, 220):
            mask = cv2.threshold(clean, threshold, 255, cv2.THRESH_BINARY)[1]
            for scale in (2, 3):
                results = jp().ocr(cv2.resize(mask, None, fx=scale, fy=scale))
                confident_titles.update(
                    normalize_card_title(result.text) for result in results
                    if result.confidence >= 0.98
                )
                if len(results) == 1 and results[0].confidence >= 0.98:
                    title = normalize_card_title(results[0].text)
                    if title:
                        mask_readings.append((title, results[0].confidence))
    except Exception:
        logger.debug('Could not verify masked card title.', exc_info=True)
        return None
    # A high-confidence color reading and a corroborating mask reading
    # are also two complete readings of this same title. All high-confidence
    # results above must still agree; a single reading never recovers a card.
    corroborated = readings + mask_readings
    if len(corroborated) < 2 or len(confident_titles) != 1:
        return None
    return corroborated[0][0], min(confidence for _, confidence in corroborated)


def locate_cards(img: MatLike):
    typed_letters = _locate_typed_letters(img)
    letters = [letter for letter, _ in typed_letters]
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
    for region, (letter, category) in zip(card_regions, typed_letters):
        x, y, w, h = region.xywh
        card_img = img[y:y+h, x:x+w]
        result_list = skill_cards_db().query(card_img, k=1, threshold=150)
        result = None
        for candidate in result_list:
            asset_id = candidate.key.removesuffix('.png')
            for character in CharacterId:
                asset_id = asset_id.removesuffix('-' + character.value)
            if category_from_asset_id(asset_id) == category:
                result = candidate
                break
        index = ordered_letters.index(letter)
        next_letter = ordered_letters[index + 1] if index + 1 < len(ordered_letters) else None
        title_reading = None
        card_id = None
        if result is not None:
            card_id = result.key.removesuffix('.png')
            for character in CharacterId:
                card_id = card_id.removesuffix('-' + character.value)
        if result_list:
            # A same-category nearest image can still be the wrong card.
            # Exposed, corroborated title text must resolve uniquely in the DB.
            title_reading = _read_card_title(img, letter, next_letter)
            if title_reading is not None:
                title_asset = asset_from_card_title(title_reading[0], category)
                if title_asset is not None and title_asset != card_id:
                    card_id = title_asset
                    result = None

        available = color.find(img, '#7a7d7d', rect=letter) is None
        if card_id is not None:
            # 查数据库
            # The marker is visible on any sufficiently separated card, not
            # just the rightmost one. With a crowded hand the next card covers
            # it; never interpret that occlusion as evidence of a base card.
            upgrade_count = _observed_upgrade_count(img, letter, next_letter)
            db_card = SkillCard.from_asset_id(card_id, upgrade_count)
            card = CardGameObject(
                rect=region,
                res_name=result.key if result is not None else card_id + '.png',
                card=db_card,
                _screenshot=img,
                _letter_rect=letter,
                _recognition_distance=float(result.distance) if result is not None else None,
                _recognition_title=title_reading[0] if title_reading is not None else None,
                _recognition_title_confidence=title_reading[1] if title_reading is not None else None,
            )
            results.append(card)
            if card.card is not None:
                logger.info('Matched skill card: %s(%s) | dist=%s title=%s available=%s',
                            card.res_name, card.card.name, card._recognition_distance,
                            card._recognition_title, available)
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
            if orig_card_img is None:
                # A title identity need not select a character artwork variant.
                continue
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
