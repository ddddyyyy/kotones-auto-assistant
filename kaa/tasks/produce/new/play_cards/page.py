import re
from typing import Any, NamedTuple
from functools import lru_cache
from typing_extensions import deprecated

import cv2
from cv2.typing import MatLike
from kotonebot.backend.image import find
from kotonebot.backend.ocr import en
from kotonebot.primitives import Rect
from kotonebot import logging, device, Loop, Countdown

from kaa.tasks import R
from .ui import CardGameObject, locate_cards
from kaa.tasks.produce.new.page import eval_once

logger = logging.getLogger(__name__)
HUD_HEIGHT = 226
# The center target can be three digits (for example 256). The old 60-pixel
# crop cut off edge digits and read 90 as 9 or 256 as 5.
BOX_SCORE_GAP = Rect(280, 120, 160, 70)
BOX_REMAINING_TURNS_LEFT = Rect(25, 69, 80, 44)
# Ranked exams show the standings above the resource gauges, shifting both
# badges down by roughly 40 px compared with lessons.
BOX_EXAM_HP = Rect(565, 208, 63, 41)
BOX_EXAM_GENKI = Rect(635, 171, 54, 40)
BOX_EXAM_LEADER_SCORE = Rect(214, 125, 83, 40)


class ExamStandings(NamedTuple):
    own_score: int
    leader_score: int
    own_is_leader: bool


def _own_exam_score_box(screen: MatLike) -> Rect | None:
    """Find the uniquely wide white frame around the player's standings tile."""
    if screen.shape[0] < 204 or screen.shape[1] < 715:
        return None
    region = screen[48:204, 205:715]
    bright_columns = (region.min(axis=2) > 210).sum(axis=0)
    marked = [205 + index for index, count in enumerate(bright_columns) if count >= 90]
    runs: list[tuple[int, int]] = []
    for x in marked:
        if runs and x == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], x)
        else:
            runs.append((x, x))
    pairs = [
        (left[0], right[0])
        for left in runs
        for right in runs
        if 116 <= right[0] - left[0] <= 124
    ]
    if len(pairs) != 1:
        return None
    left, right = pairs[0]
    # Exclude the selection frame and the gauge below the score. Five-digit
    # scores can spill outside the narrow opponent tile; the leader crop is
    # slightly wider for that reason.
    return Rect(left + 4, 125, min(screen.shape[1] - left - 4, right - left - 5), 40)


def _exam_score_variants(screen: MatLike, box: Rect) -> list[MatLike]:
    crop = screen[box.y1:box.y2, box.x1:box.x2]
    if crop.size == 0:
        return []
    # Digits have bright interiors. Strip colourful tile backgrounds and
    # long thin frame remnants, but retain the original as another candidate:
    # binarization can itself lose strokes in small opponent numbers.
    mask = ((crop.min(axis=2) if crop.ndim == 3 else crop) > 200).astype('uint8') * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for index in range(1, count):
        _, _, width, height, _ = stats[index]
        if (width <= 6 and height >= 30) or (height <= 3 and width >= 60):
            mask[labels == index] = 0
    return [crop, mask]


def _read_exam_score(
    screen: MatLike, box: Rect, *, allow_zero_fallback: bool = False,
) -> int | None:
    primary: set[int] = set()
    secondary: set[int] = set()
    variants = _exam_score_variants(screen, box)
    original_zero = False
    try:
        for index, crop in enumerate(variants):
            results = en().ocr(crop)
            if len(results) == 1 and results[0].text.isdecimal() and results[0].confidence >= 0.8:
                primary.add(int(results[0].text))
            verified = _digits_from_image(crop, fallback_raw=True)
            if len(verified) == 1:
                secondary.add(verified[0])
            if index == 0:
                original_zero = verified == [0]
    except Exception:
        logger.warning('Could not verify ranked exam score.', exc_info=True)
        return None
    agreed = primary & secondary
    if len(agreed) == 1:
        return agreed.pop()
    # Preserve the existing tiny-zero fallback, but do not let it override
    # contradictory nonzero primary readings or multiple consensus values.
    if not primary and allow_zero_fallback and original_zero:
        return 0
    logger.debug('Ranked exam score disagreement: primary=%s secondary=%s box=%s', primary, secondary, box)
    return None


def read_exam_standings(screen: MatLike) -> ExamStandings | None:
    """Read only a uniquely identified player tile and the first-place score."""
    own_box = _own_exam_score_box(screen)
    if own_box is None:
        return None
    own_score = _read_exam_score(screen, own_box, allow_zero_fallback=True)
    own_is_leader = own_box.x1 < 250
    leader_score = (
        own_score if own_is_leader
        else _read_exam_score(screen, BOX_EXAM_LEADER_SCORE)
    )
    if own_score is None or leader_score is None or own_score > leader_score:
        return None
    return ExamStandings(own_score, leader_score, own_is_leader)


@lru_cache(maxsize=1)
def _dddd():
    import ddddocr
    return ddddocr.DdddOcr(show_ad=False)


def _digits_from_image(img: MatLike, *, fallback_raw: bool = False) -> list[int]:
    if img is None or img.size == 0:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ok, buf = cv2.imencode(".png", bw)
    if not ok:
        return []

    text = _dddd().classification(buf.tobytes())
    nums = re.findall(r"\d+", text if isinstance(text, str) else "")
    if not nums and fallback_raw:
        # The outlined zero in the genki badge can turn into "o" after Otsu
        # thresholding. On the original color crop the same OCR reads "0".
        ok, raw_buf = cv2.imencode(".png", img)
        if ok:
            raw_text = _dddd().classification(raw_buf.tobytes())
            nums = re.findall(
                r"\d+", raw_text if isinstance(raw_text, str) else ""
            )
    if not nums and fallback_raw:
        # The ranked genki shield's outline can produce "bo" even when its
        # numeral is zero. Recover only a high-confidence literal zero from
        # the existing independent OCR engine; other missing values stay unknown.
        try:
            results = en().ocr(img)
            if len(results) == 1 and results[0].text == '0' and results[0].confidence >= 0.98:
                return [0]
        except Exception:
            logger.debug('Could not verify missing genki zero.', exc_info=True)
    return [int(n) for n in nums]


def _verify_hp_reading(crop: MatLike, value: int) -> int | None:
    """Verify HP against the same crop; ambiguous icon prefixes stay unknown."""
    try:
        results = en().ocr(crop)
        if len(results) == 1 and results[0].confidence >= 0.98:
            text = results[0].text
            if text == str(value) or (
                900 <= value <= 999 and text == str(value % 100)
            ):
                verified = int(text)
                if verified != value:
                    logger.debug('Verified HP outline prefix %d as %d.', value, verified)
                return verified
            # An explicit high-confidence conflict is not rescued by another
            # preprocessing attempt: it must remain unknown.
            return None
        if crop.ndim == 3:
            minimum = crop.min(axis=2)
            white = (minimum > 245) & (
                crop.max(axis=2).astype('int16') - minimum < 12
            )
            results = en().ocr(white.astype('uint8') * 255)
            if len(results) == 1 and results[0].confidence >= 0.98:
                text = results[0].text
                if text == str(value) or (
                    900 <= value <= 999 and text == str(value % 100)
                ):
                    return int(text)
    except Exception:
        logger.debug('Could not verify HP reading.', exc_info=True)
    return None


def _verify_genki_reading(crop: MatLike, value: int) -> int | None:
    """Cross-check a suspicious three-digit genki result on its original crop."""
    if not 100 <= value < 1000:
        return value
    try:
        results = en().ocr(crop)
        if len(results) != 1 or results[0].confidence < 0.98:
            return None
        reading = results[0].text
        raw = str(value)
        if reading == raw:
            return value
        if (
            len(reading) == 2 and reading.isascii() and reading.isdecimal()
            and reading[0] != '0' and reading in {raw[:2], raw[1:]}
        ):
            verified = int(reading)
            logger.warning('Verified genki OCR %d as %d from same crop.', value, verified)
            return verified
    except Exception:
        logger.debug('Could not verify genki reading.', exc_info=True)
    return None


def _fetch_int(
    box: Rect, *, fallback_raw: bool = False, verify_hp: bool = False,
    verify_genki: bool = False,
) -> int | None:
    screen = device.screenshot()
    crop = screen[box.y1:box.y2, box.x1:box.x2]
    nums = _digits_from_image(crop, fallback_raw=fallback_raw)
    if not nums:
        return None
    if verify_hp:
        return _verify_hp_reading(crop, nums[0])
    if verify_genki:
        return _verify_genki_reading(crop, nums[0])
    return nums[0]


def _fetch_remaining_turns() -> int | None:
    """Read both sides of the turn badge to avoid losing a leading digit.

    The original box can cut off the tens digit in the newer battle HUD.  The
    supplementary box ends before the adjacent score, which otherwise can be
    mistaken for part of the turn count.
    """
    screen = device.screenshot()
    candidates: list[int] = []
    for box in (R.InProduce.InLesson.BoxRemainingTurns, BOX_REMAINING_TURNS_LEFT):
        crop = screen[box.y1:box.y2, box.x1:box.x2]
        nums = _digits_from_image(crop)
        if nums and 1 <= nums[0] <= 30:
            candidates.append(nums[0])
    return max(candidates, default=None)

def _fetch_exam_remaining_turns() -> int | None:
    """Read the ranked badge without its adjacent parameter percentage."""
    screen = device.screenshot()
    # Ranked numbers are centered near x=62; the lesson badge is farther right.
    # White digits isolate them from the colored ring and neighboring percent.
    crop = screen[65:115, 12:112]
    if crop.size == 0:
        return None
    mask = (crop.min(axis=2) > 200).astype('uint8') * 255
    digits = _digits_from_image(mask)
    if len(digits) != 1 or not 1 <= digits[0] <= 30:
        return None
    try:
        results = en().ocr(mask)
        if (
            len(results) == 1
            and results[0].text.isdecimal()
            and results[0].confidence >= 0.98
            and int(results[0].text) == digits[0]
        ):
            return digits[0]
    except Exception:
        logger.debug('Could not verify ranked remaining turns.', exc_info=True)
    return None


class HudInfo(NamedTuple):
    remaining_turns: int | None
    hp: int | None
    genki: int | None
    score_gap: int | None

def _screenshot_hud() -> MatLike:
    screen = device.screenshot()
    h, w, _ = screen.shape
    hud = screen[h - HUD_HEIGHT : h, 0 : w]
    return hud

class LessonBattleContext:
    """课程打牌页面"""

    @eval_once
    def fetch_remaining_turns(self) -> int | None:
        """获取剩余回合数"""
        return _fetch_remaining_turns()
    
    @eval_once
    def fetch_exam_remaining_turns(self) -> int | None:
        """Read the ranked exam's distinct circular turn badge."""
        return _fetch_exam_remaining_turns()

    @eval_once
    def fetch_hp(self) -> int | None:
        """获取当前体力"""
        return _fetch_int(R.InProduce.InLesson.BoxHp, verify_hp=True)
    
    @eval_once
    def fetch_stamina(self) -> int | None:
        """获取当前元气值"""
        return _fetch_int(R.InProduce.InLesson.BoxGenki, fallback_raw=True, verify_genki=True)

    @eval_once
    def fetch_exam_hp(self) -> int | None:
        """Read HP from the lower resource gauge in ranked exams."""
        return _fetch_int(BOX_EXAM_HP, verify_hp=True)

    @eval_once
    def fetch_exam_stamina(self) -> int | None:
        """Read genki from the lower resource gauge in ranked exams."""
        return _fetch_int(BOX_EXAM_GENKI, fallback_raw=True, verify_genki=True)

    @eval_once
    def fetch_exam_standings(self) -> ExamStandings | None:
        return read_exam_standings(device.screenshot())

    @eval_once
    def fetch_score_gap(self) -> int | None:
        """获取距离课程 CLEAR/PERFECT 目标还差的参数值。

        考试界面没有中央目标圆环，此时 OCR 会自然返回 ``None``。
        """
        return _fetch_int(BOX_SCORE_GAP)

    @eval_once
    def fetch_score_target_is_final(self) -> bool | None:
        """Distinguish the intermediate CLEAR goal from the final PERFECT goal."""
        if R.InProduce.TextPerfectUntil.exists():
            return True
        if R.InProduce.TextClearUntil.exists():
            return False
        return None
    
    @eval_once
    def fetch_all(self):
        remaining_turns = _fetch_remaining_turns()
        hp = _fetch_int(R.InProduce.InLesson.BoxHp, verify_hp=True)
        genki = _fetch_int(R.InProduce.InLesson.BoxGenki, fallback_raw=True, verify_genki=True)
        score_gap = _fetch_int(BOX_SCORE_GAP)
        return HudInfo(remaining_turns, hp, genki, score_gap)
    
    @eval_once
    def fetch_hands(self):
        """获取手牌信息"""
        img = device.screenshot()
        cards = locate_cards(img)
        return cards

    @eval_once
    def fetch_screen(self) -> MatLike:
        """Capture the current battle HUD for planner-only observations."""
        return device.screenshot()

    def commit(self, card: CardGameObject):
        card.double_click()

    @deprecated('考试与冲刺周冲刺阶段里，背景是动态的，因此效果不佳')
    def wait_stablized(self, *, timeout: float = float('inf'), interval: float = 0.5, stable_time: float = 3) -> bool:
        """
        等待 HUD 展示稳定。用于等待下一次出牌开始。

        :param timeout: 最长等待时间，单位秒
        :param interval: 每次检查间隔时间，单位秒
        :param stable_time: HUD 稳定持续时间，单位秒。只有当 HUD 在该时间内保持不变，才认为稳定。
        :return: 如果在超时时间内 HUD 稳定则返回 True，否则返回 False。
        """
        hud = _screenshot_hud()
        stable_cd = Countdown(stable_time)
        for _ in Loop(timeout=timeout, interval=interval):
            new_hud = _screenshot_hud()
            if find(hud, new_hud, threshold=0.9) is not None:
                stable_cd.start()
                if stable_cd.expired():
                    return True
            else:
                stable_cd.reset()
            hud = new_hud
            
        return False



if __name__ == "__main__":
    from pprint import pprint
    from time import time
    print("Fetching all HUD info...")
    while True:
        start = time()
        ctx = LessonBattleContext()
        # ctx.wait_stablized(stable_time=1)
        info = ctx.fetch_all()
        end = time()
        print(f"Fetched in {end - start:.2f} seconds:")
        pprint(info)
        pprint(ctx.fetch_hands())
    # print("Remaining Turns:", ctx.fetch_remaining_turns())
    # print("HP:", ctx.fetch_hp())
    # print("Genki:", ctx.fetch_genki())
    # print("Waiting for stabilized...")
    # if ctx.wait_stablized(timeout=30):
    #     print("Page stabilized.")
    # else:
    #     print("Timeout waiting for page to stabilize.")
