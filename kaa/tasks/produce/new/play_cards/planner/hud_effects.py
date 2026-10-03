"""Observe visible lesson/exam effects from the left-side HUD stack."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
from cv2.typing import MatLike


HUD_EFFECT_REGION = (0, 210, 230, 730)
_ASSET_DIR = (
    Path(__file__).parents[5]
    / 'application'
    / 'ui'
    / 'assets'
    / 'skill_card'
    / 'exam_effects'
)

_EFFECT_ASSETS = {
    'lesson_buff': 'exam_examlessonbuff.webp',
    'parameter_buff': 'exam_examparameterbuff.webp',
    'parameter_buff_multiple': 'exam_examparameterbuffmultipleperturn.webp',
    'review': 'exam_examreview.webp',
    'aggressive': 'exam_examcardplayaggressive.webp',
    'genki': 'exam_examblock.webp',
    'full_power_point': 'exam_examfullpowerpoint.webp',
    'full_power': 'exam_examfullpower.webp',
    'enthusiasm': 'exam_examenthusiasticadditive.webp',
    'preservation': 'exam_exampreservation.webp',
    'concentration': 'exam_examconcentration.webp',
    'lesson_multiplier': 'exam_examlessonvaluemultiple.webp',
    'stamina_consumption_down': 'exam_examstaminaconsumptiondown.webp',
}

_CONFIDENCE_THRESHOLDS = {
    'parameter_buff': 0.78,
    'lesson_buff': 0.75,
    'review': 0.80,
    'full_power': 0.65,
    'concentration': 0.80,
}


@dataclass(frozen=True)
class EffectIconObservation:
    effect: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    value: int | None = None


@lru_cache(maxsize=None)
def _icon_alpha(effect: str) -> MatLike:
    image = cv2.imread(
        str(_ASSET_DIR / _EFFECT_ASSETS[effect]),
        cv2.IMREAD_UNCHANGED,
    )
    if image is None or image.ndim != 3 or image.shape[2] < 4:
        raise FileNotFoundError(f'Cannot load effect icon asset: {effect}')
    alpha = image[:, :, 3]
    x, y, width, height = cv2.boundingRect((alpha > 32).astype('uint8'))
    return alpha[y : y + height, x : x + width]


def observe_effect_icons(screen: MatLike) -> list[EffectIconObservation]:
    """Find calibrated HUD glyphs, with numeric OCR where supported."""
    x1, y1, x2, y2 = HUD_EFFECT_REGION
    gray = (
        cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        if screen.ndim == 3
        else screen
    )
    roi = gray[y1:y2, x1:x2]
    # Most effect glyphs are white; the point badge uses grayscale below.
    _, bright = cv2.threshold(roi, 188, 255, cv2.THRESH_BINARY)
    observations: list[EffectIconObservation] = []
    point_observation = _observe_full_power_points(screen, gray)
    if point_observation is not None:
        observations.append(point_observation)
    # Only effects calibrated against real battle screenshots are enabled.
    # The remaining assets stay listed so they can be enabled independently
    # after collecting positive and negative samples for each glyph.
    for effect, threshold in _CONFIDENCE_THRESHOLDS.items():
        # Stance glyphs belong to the top HUD slot. Searching the entire
        # effect stack would let a card/item glyph impersonate the stance.
        search = bright[:120, :75] if effect == 'concentration' else bright
        source = _icon_alpha(effect)
        best_confidence = -1.0
        best_rect = (0, 0, 0, 0)
        for target_height in range(24, 51, 2):
            target_width = max(
                1,
                round(source.shape[1] * target_height / source.shape[0]),
            )
            template = cv2.resize(
                source,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
            if template.shape[0] > search.shape[0] or template.shape[1] > search.shape[1]:
                continue
            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, location = cv2.minMaxLoc(result)
            if confidence > best_confidence:
                best_confidence = float(confidence)
                best_rect = (
                    location[0] + x1,
                    location[1] + y1,
                    target_width,
                    target_height,
                )
        # Real effect glyphs occupy the left diamond column. Matches farther
        # right are usually digits, Japanese labels, or P-item artwork.
        if best_confidence >= threshold and best_rect[0] <= 75:
            x, y, width, height = best_rect
            if effect == 'review':
                # The same white glyph is also shown on a grey, inactive HUD
                # diamond. Only the blue, saturated diamond is an active buff.
                icon = screen[y:y + height, x:x + width]
                if icon.ndim != 3 or cv2.cvtColor(
                    icon, cv2.COLOR_BGR2HSV,
                )[:, :, 1].mean() < 30:
                    continue
            observations.append(
                EffectIconObservation(
                    effect,
                    best_confidence,
                    x,
                    y,
                    width,
                    height,
                    _read_effect_value(screen, effect, x, y),
                )
            )
    ranked = sorted(observations, key=lambda item: -item.confidence)
    selected: list[EffectIconObservation] = []
    for candidate in ranked:
        if any(_overlaps(candidate, existing) for existing in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: item.y)


def _observe_full_power_points(screen: MatLike, gray: MatLike) -> EffectIconObservation | None:
    """The point badge is grey/translucent, unlike white effect glyphs."""
    if gray.shape[0] < 330 or gray.shape[1] < 125:
        return None
    roi = gray[230:330, 75:125]
    source = _icon_alpha('full_power_point')
    best_confidence = -1.0
    best_rect = (0, 0, 0, 0)
    for height in range(24, 35, 2):
        width = max(1, round(source.shape[1] * height / source.shape[0]))
        template = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
        _, confidence, _, location = cv2.minMaxLoc(cv2.matchTemplate(
            roi, template, cv2.TM_CCOEFF_NORMED,
        ))
        if confidence > best_confidence:
            best_confidence = float(confidence)
            best_rect = (location[0] + 75, location[1] + 230, width, height)
    x, y, width, height = best_rect
    if best_confidence < 0.82 or not 80 <= x <= 105:
        return None
    value = _read_effect_value(screen, 'full_power_point', x, y)
    # Do not expose presence-only point observations: a visible zero badge
    # would otherwise be interpreted as proving a positive point count.
    if value is None:
        return None
    return EffectIconObservation('full_power_point', best_confidence, x, y, width, height, value)


@lru_cache(maxsize=1)
def _digit_ocr():
    import ddddocr

    return ddddocr.DdddOcr(show_ad=False)


def _read_effect_value(
    screen: MatLike,
    effect: str,
    icon_x: int,
    icon_y: int,
) -> int | None:
    """Read the numeric value printed next to a calibrated effect icon.

    The calibrated effects use different baselines in the game HUD. Multiple
    threshold variants vote on the result because the outlined game font often
    makes OCR confuse ``0``/``o`` and ``1``/``l``.
    """
    boxes = {
        'parameter_buff': (icon_x + 31, icon_y + 24, icon_x + 68, icon_y + 52),
        'lesson_buff': (icon_x + 31, icon_y + 15, icon_x + 103, icon_y + 45),
        'review': (icon_x + 31, icon_y + 24, icon_x + 68, icon_y + 52),
        'full_power_point': (icon_x + 35, icon_y - 3, icon_x + 72, icon_y + 32),
    }
    box = boxes.get(effect)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    height, width = screen.shape[:2]
    crop = screen[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    variants = [gray]
    variants.append(cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1])
    variants.extend(
        cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
        for threshold in (150, 190, 220)
    )

    votes: Counter[int] = Counter()
    substitutions = str.maketrans({
        'o': '0', 'O': '0', 'Q': '0',
        'i': '1', 'I': '1', 'l': '1',
    })
    for variant in variants:
        ok, encoded = cv2.imencode('.png', variant)
        if not ok:
            continue
        result = _digit_ocr().classification(encoded.tobytes())
        if not isinstance(result, str):
            continue
        raw = result
        normalized = raw.translate(substitutions)
        if effect == 'full_power_point':
            # Read the entire badge, not the first digit buried in a label.
            if re.fullmatch(r'\d{1,3}', normalized.strip()) is None:
                continue
            votes[int(normalized.strip())] += 1
            continue
        numbers = re.findall(r'\d+', normalized)
        if not numbers:
            continue
        value = int(numbers[0])
        if 0 < value <= 999:
            votes[value] += 1
            if votes[value] >= 2:
                return value
    if not votes:
        return None
    value, count = votes.most_common(1)[0]
    if effect == 'full_power_point':
        competing = sum(v for k, v in votes.items() if k != value)
        return value if count >= 3 and competing < 2 else None
    return value if count >= 2 else None


def _overlaps(left: EffectIconObservation, right: EffectIconObservation) -> bool:
    x_overlap = max(
        0,
        min(left.x + left.width, right.x + right.width) - max(left.x, right.x),
    )
    y_overlap = max(
        0,
        min(left.y + left.height, right.y + right.height) - max(left.y, right.y),
    )
    intersection = x_overlap * y_overlap
    smaller = min(left.width * left.height, right.width * right.height)
    return smaller > 0 and intersection / smaller >= 0.45
