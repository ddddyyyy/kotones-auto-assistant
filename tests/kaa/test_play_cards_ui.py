import base64
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from kotonebot.primitives import Rect

from kaa.tasks.produce.new.play_cards.ui import (
    _has_upgrade_marker,
    _upgrade_marker_visible,
    _visible_upgrade_count,
    _observed_upgrade_count,
)
from kaa.tasks.produce.new.play_cards import page
from kaa.tasks.produce.new.play_cards.page import _digits_from_image


def test_upgrade_marker_detected_in_reference_screenshot():
    image = cv2.imread(
        'kotonebot-resource/sprites/jp/in_produce/screenshot_1_cards.png'
    )

    assert image is not None
    assert _has_upgrade_marker(image, Rect(352, 1083, 17, 15))


def test_normal_card_does_not_look_upgraded():
    image = cv2.imread(
        'kotonebot-resource/sprites/jp/in_produce/screenshot_4_cards.png'
    )

    assert image is not None
    # Rightmost card's A marker; the nearby red stamina cost is below the ROI.
    assert not _has_upgrade_marker(image, Rect(598, 1083, 17, 15))


def test_separated_cards_expose_each_upgrade_marker():
    first = Rect(143, 1083, 17, 15)
    second = Rect(358, 1083, 17, 15)
    assert _upgrade_marker_visible(first, second)
    assert _upgrade_marker_visible(second, None)


def test_crowded_hand_does_not_read_occluded_upgrade_marker():
    first = Rect(143, 1083, 17, 15)
    second = Rect(259, 1083, 17, 15)
    assert not _upgrade_marker_visible(first, second)


def test_title_suffix_distinguishes_one_two_and_three_upgrades():
    letter = Rect(598, 1083, 17, 15)
    image = np.full((1280, 720, 3), 240, dtype=np.uint8)

    def add_plus(x: int) -> None:
        cv2.rectangle(image, (x + 4, 1112), (x + 5, 1121), (30, 30, 30), -1)
        cv2.rectangle(image, (x, 1116), (x + 9, 1117), (30, 30, 30), -1)

    # A cross-shaped character earlier in the title is not part of the suffix.
    add_plus(550)
    cv2.rectangle(image, (600, 1112), (605, 1122), (30, 30, 30), -1)
    assert _visible_upgrade_count(image, letter) == 0
    add_plus(640)
    assert _visible_upgrade_count(image, letter) == 1
    add_plus(653)
    assert _visible_upgrade_count(image, letter) == 2
    add_plus(666)
    assert _visible_upgrade_count(image, letter) == 3


def test_reference_card_title_has_one_upgrade_suffix():
    image = cv2.imread(
        'kotonebot-resource/sprites/jp/in_produce/screenshot_1_cards.png'
    )
    assert image is not None
    assert _visible_upgrade_count(image, Rect(352, 1083, 17, 15)) == 1


def test_lesson_score_gap_is_read_from_reference_screenshot():
    image = cv2.imread(
        'kotonebot-resource/sprites/jp/in_produce/screenshot_lesson_5_cards.png'
    )

    assert image is not None
    box = page.BOX_SCORE_GAP
    assert (box.x1, box.y1, box.x2, box.y2) == (280, 120, 440, 190)
    assert _digits_from_image(image[box.y1:box.y2, box.x1:box.x2]) == [6]


def test_genki_zero_uses_raw_crop_when_binary_ocr_misses_it():
    encoded = Path(__file__).with_name('lesson_genki_zero.png.b64').read_text(
        encoding='ascii'
    )
    crop = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    assert crop is not None
    assert _digits_from_image(crop) == []
    assert _digits_from_image(crop, fallback_raw=True) == [0]


def test_stamina_reading_enables_raw_fallback_without_changing_hp(monkeypatch):
    calls: list[tuple[object, bool, bool]] = []

    def fake_fetch(box, *, fallback_raw=False, verify_hp=False, verify_genki=False):
        calls.append((box, fallback_raw, verify_genki))
        return 0

    monkeypatch.setattr(page, '_fetch_int', fake_fetch)
    context = page.LessonBattleContext()

    assert context.fetch_stamina() == 0
    assert context.fetch_hp() == 0
    assert calls == [
        (page.R.InProduce.InLesson.BoxGenki, True, True),
        (page.R.InProduce.InLesson.BoxHp, False, False),
    ]


@pytest.mark.parametrize(
    ('screenshot', 'hp', 'genki'),
    [
        ('screenshot_1_cards.png', 28, 11),
        ('screenshot_4_cards.png', 27, 29),
    ],
)
def test_ranked_exam_resource_boxes_read_below_standings(
    monkeypatch, screenshot: str, hp: int, genki: int,
):
    image = cv2.imread(f'kotonebot-resource/sprites/jp/in_produce/{screenshot}')
    assert image is not None
    monkeypatch.setattr(page, 'device', SimpleNamespace(screenshot=lambda: image))

    context = page.LessonBattleContext()
    assert context.fetch_exam_hp() == hp
    assert context.fetch_exam_stamina() == genki


@pytest.mark.parametrize('score_gap', [90, 110, 256])
def test_lesson_score_gap_keeps_edge_digits_from_live_battle(score_gap, monkeypatch):
    # Only the target number's 160x70 crop is stored, not a full game screen.
    encoded = Path(__file__).with_name(
        f'lesson_score_gap_{score_gap}.png.b64'
    ).read_text(encoding='ascii')
    crop = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    box = page.BOX_SCORE_GAP
    screen = np.zeros((1280, 720, 3), dtype=np.uint8)
    screen[box.y1:box.y2, box.x1:box.x2] = crop
    monkeypatch.setattr(page, 'device', SimpleNamespace(screenshot=lambda: screen))

    assert page._fetch_int(box) == score_gap


def test_remaining_turns_prefers_uncropped_double_digit_reading(monkeypatch):
    monkeypatch.setattr(
        page,
        'device',
        SimpleNamespace(screenshot=lambda: np.zeros((200, 720, 3), dtype=np.uint8)),
    )
    readings = iter(([2], [12]))
    monkeypatch.setattr(page, '_digits_from_image', lambda _image: next(readings))

    assert page._fetch_remaining_turns() == 12


def test_remaining_turns_rejects_adjacent_score_digits(monkeypatch):
    monkeypatch.setattr(
        page,
        'device',
        SimpleNamespace(screenshot=lambda: np.zeros((200, 720, 3), dtype=np.uint8)),
    )
    readings = iter(([111], [11]))
    monkeypatch.setattr(page, '_digits_from_image', lambda _image: next(readings))

    assert page._fetch_remaining_turns() == 11


@pytest.mark.parametrize('tile_left, own_score, leader_score, is_leader', [
    (212, 1900, 1900, True),
    (288, 1400, 1900, False),
    (363, 7742, 13195, False),
    (590, 0, 1866, False),
])
def test_exam_standings_identify_player_by_wide_frame(
    monkeypatch, tile_left, own_score, leader_score, is_leader,
):
    screen = np.zeros((1280, 720, 3), dtype=np.uint8)
    screen[48:204, tile_left:tile_left + 2] = 255
    screen[48:204, tile_left + 120:tile_left + 122] = 255

    def fake_score(_screen, box, *, allow_zero_fallback=False):
        if box.x1 == tile_left + 4:
            assert allow_zero_fallback
            return own_score
        assert box == page.BOX_EXAM_LEADER_SCORE
        return leader_score

    monkeypatch.setattr(page, '_read_exam_score', fake_score)

    assert page.read_exam_standings(screen) == page.ExamStandings(
        own_score, leader_score, is_leader,
    )


def test_exam_standings_ignore_ambiguous_player_frames(monkeypatch):
    screen = np.zeros((1280, 720, 3), dtype=np.uint8)
    for left in (288, 590):
        screen[48:204, left:left + 2] = 255
        screen[48:204, left + 120:left + 122] = 255
    monkeypatch.setattr(page, '_read_exam_score', lambda *args, **kwargs: 100)

    assert page.read_exam_standings(screen) is None


def test_exam_zero_requires_second_ocr_confirmation(monkeypatch):
    screen = np.zeros((1280, 720, 3), dtype=np.uint8)
    box = Rect(590, 125, 123, 50)
    monkeypatch.setattr(page, 'en', lambda: SimpleNamespace(ocr=lambda *args, **kwargs: []))
    monkeypatch.setattr(page, '_digits_from_image', lambda *args, **kwargs: [0])

    assert page._read_exam_score(screen, box, allow_zero_fallback=True) == 0
    assert page._read_exam_score(screen, box) is None
    monkeypatch.setattr(page, '_digits_from_image', lambda *args, **kwargs: [])
    assert page._read_exam_score(screen, box, allow_zero_fallback=True) is None


@pytest.mark.parametrize('primary, secondary, expected', [
    ('31489', [31489], 31489),
    ('51489', [31489], None),
    ('186', [86], None),
    ('1284', [], None),
    ('1284', [12, 84], None),
])
def test_nonzero_exam_score_requires_independent_agreement(monkeypatch, primary, secondary, expected):
    screen = np.zeros((1280, 720, 3), dtype=np.uint8)
    box = page.BOX_EXAM_LEADER_SCORE
    monkeypatch.setattr(page, 'en', lambda: SimpleNamespace(
        ocr=lambda *args, **kwargs: [SimpleNamespace(text=primary, confidence=0.99)],
    ))
    def verify(crop, *, fallback_raw):
        assert crop.shape in {(40, 83, 3), (40, 83)}
        assert fallback_raw
        return secondary
    monkeypatch.setattr(page, '_digits_from_image', verify)
    assert page._read_exam_score(screen, box) == expected


def test_score_variants_remove_frame_without_erasing_digits():
    screen = np.zeros((40, 83, 3), dtype=np.uint8)
    screen[:, 0:2] = 255
    screen[10:25, 15:23] = 255
    original, cleaned = page._exam_score_variants(screen, Rect(0, 0, 83, 40))
    assert original.shape == screen.shape
    assert np.all(cleaned[:, :2] == 0)
    assert np.all(cleaned[10:25, 15:23] == 255)


@pytest.mark.parametrize('primary_values, secondary_values, expected', [
    (['3081', '308'], [[308], [308]], 308),
    (['15622', '13622'], [[13622], [13622]], 13622),
    (['31489', '51489'], [[31489], [51489]], None),
])
def test_score_variants_require_unique_cross_engine_consensus(monkeypatch, primary_values, secondary_values, expected):
    primary = iter(primary_values)
    secondary = iter(secondary_values)
    monkeypatch.setattr(page, 'en', lambda: SimpleNamespace(
        ocr=lambda *args, **kwargs: [SimpleNamespace(text=next(primary), confidence=0.99)],
    ))
    monkeypatch.setattr(page, '_digits_from_image', lambda *args, **kwargs: next(secondary))
    screen = np.zeros((1280, 720, 3), dtype=np.uint8)
    assert page._read_exam_score(screen, page.BOX_EXAM_LEADER_SCORE) == expected


def test_exam_score_verification_failure_is_unknown(monkeypatch):
    monkeypatch.setattr(page, 'en', lambda: SimpleNamespace(
        ocr=lambda *args, **kwargs: [SimpleNamespace(text='1234', confidence=0.99)],
    ))
    def fail(*args, **kwargs):
        raise RuntimeError('OCR unavailable')
    monkeypatch.setattr(page, '_digits_from_image', fail)
    assert page._read_exam_score(np.zeros((1280, 720, 3), dtype=np.uint8), Rect(214, 125, 75, 50)) is None


def test_remaining_turns_reads_twelve_from_reference_screenshot(monkeypatch):
    image = cv2.imread(
        'kotonebot-resource/sprites/jp/in_produce/screenshot_lesson_no_card.png'
    )
    assert image is not None
    monkeypatch.setattr(page, 'device', SimpleNamespace(screenshot=lambda: image))

    assert page._fetch_remaining_turns() == 12


def test_faded_selected_double_upgrade_is_read_without_red_badge():
    # Exact title pixels from e375793's first course, sequence 3.
    raw = Path(__file__).with_name('card_faded_double_upgrade.png.b64').read_text(encoding='ascii')
    title = cv2.imdecode(np.frombuffer(base64.b64decode(raw), dtype=np.uint8), cv2.IMREAD_COLOR)
    image = np.full((1280, 720, 3), 255, dtype=np.uint8)
    image[1103:1129, 260:460] = title
    letter = Rect(352, 1083, 17, 15)
    assert not _has_upgrade_marker(image, letter)
    assert _visible_upgrade_count(image, letter) == 2
    assert _observed_upgrade_count(image, letter, Rect(598, 1083, 17, 15)) == 2
    assert _observed_upgrade_count(image, letter, Rect(515, 1083, 17, 15)) == 0
    # A covered title must not be trusted as an exposed suffix.
    assert _observed_upgrade_count(image, letter, Rect(459, 1083, 17, 15)) == 0


def test_tinted_title_background_without_cross_glyph_is_not_an_upgrade():
    image = np.full((1280, 720, 3), 255, dtype=np.uint8)
    image[1110:1120, 420:450] = (255, 238, 220)
    letter = Rect(352, 1083, 17, 15)
    assert _visible_upgrade_count(image, letter) == 0
    assert _observed_upgrade_count(image, letter, None) == 0


def test_ranked_genki_zero_recovers_exact_missing_ocr_input():
    raw = Path(__file__).with_name('exam_genki_zero.png.b64').read_text(encoding='ascii')
    crop = cv2.imdecode(np.frombuffer(base64.b64decode(raw), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert _digits_from_image(crop) == []
    assert _digits_from_image(crop, fallback_raw=True) == [0]


@pytest.mark.parametrize('text,confidence', [('0', 0.97), ('O', 1.0), ('8', 1.0), ('', 1.0)])
def test_zero_fallback_does_not_invent_other_missing_values(monkeypatch, text, confidence):
    monkeypatch.setattr(page, '_dddd', lambda: SimpleNamespace(classification=lambda data: 'bo'))
    monkeypatch.setattr(page, 'en', lambda: SimpleNamespace(ocr=lambda img: [SimpleNamespace(text=text, confidence=confidence)]))
    assert _digits_from_image(np.full((40, 54, 3), 255, dtype=np.uint8), fallback_raw=True) == []


def test_zero_fallback_ocr_failure_remains_unknown(monkeypatch):
    monkeypatch.setattr(page, '_dddd', lambda: SimpleNamespace(classification=lambda data: 'bo'))
    def fail(img):
        raise RuntimeError('OCR unavailable')
    monkeypatch.setattr(page, 'en', lambda: SimpleNamespace(ocr=fail))
    assert _digits_from_image(np.full((40, 54, 3), 255, dtype=np.uint8), fallback_raw=True) == []


@pytest.mark.parametrize('value', [1, 12])
def test_ranked_turn_badge_ignores_percentage_and_keeps_both_digits(monkeypatch, value):
    raw = Path(__file__).with_name(f'exam_turn_{value}.png.b64').read_text(encoding='ascii')
    crop = cv2.imdecode(np.frombuffer(base64.b64decode(raw), dtype=np.uint8), cv2.IMREAD_COLOR)
    screen = np.zeros((1280, 720, 3), dtype=np.uint8)
    screen[65:115, 12:112] = crop
    monkeypatch.setattr(page, 'device', SimpleNamespace(screenshot=lambda: screen))
    assert page.LessonBattleContext().fetch_exam_remaining_turns() == value


@pytest.mark.parametrize('primary,confidence', [('21', 1.0), ('12', 0.97), ('O', 1.0)])
def test_ranked_turn_ocr_disagreement_stays_unknown(monkeypatch, primary, confidence):
    monkeypatch.setattr(page, 'device', SimpleNamespace(screenshot=lambda: np.zeros((1280, 720, 3), dtype=np.uint8)))
    monkeypatch.setattr(page, '_digits_from_image', lambda img: [12])
    monkeypatch.setattr(page, 'en', lambda: SimpleNamespace(ocr=lambda img: [SimpleNamespace(text=primary, confidence=confidence)]))
    assert page._fetch_exam_remaining_turns() is None


@pytest.mark.parametrize('value,primary,confidence,expected', [
    (935, '35', 0.999, 35), (917, '17', 0.999, 17),
    (935, '935', 0.999, 935), (900, '0', 0.999, 0),
    (935, '35', 0.97, None), (935, '45', 1.0, None),
    (935, 'O35', 1.0, None), (935, '', 1.0, None),
])
def test_hp_heart_prefix_requires_same_crop_corroboration(monkeypatch, value, primary, confidence, expected):
    monkeypatch.setattr(page, 'en', lambda: SimpleNamespace(ocr=lambda img: [SimpleNamespace(text=primary, confidence=confidence)]))
    assert page._verify_hp_reading(np.zeros((41, 63, 3), dtype=np.uint8), value) == expected


@pytest.mark.parametrize('value', [0, 35, 100, 735, 1000])
def test_matching_hp_values_are_preserved_without_a_global_cap(monkeypatch, value):
    monkeypatch.setattr(page, 'en', lambda: SimpleNamespace(ocr=lambda img: [SimpleNamespace(text=str(value), confidence=0.999)]))
    assert page._verify_hp_reading(np.zeros((41, 63, 3), dtype=np.uint8), value) == value


def test_hp_prefix_ocr_failure_remains_unknown(monkeypatch):
    def fail(img):
        raise RuntimeError('OCR unavailable')
    monkeypatch.setattr(page, 'en', lambda: SimpleNamespace(ocr=fail))
    assert page._verify_hp_reading(np.zeros((41, 63, 3), dtype=np.uint8), 935) is None


def test_first_exam_hp_recovers_real_heart_outline_without_prior_state(monkeypatch):
    raw = Path(__file__).with_name('exam_hp_prefix_35.png.b64').read_text(encoding='ascii')
    crop = cv2.imdecode(np.frombuffer(base64.b64decode(raw), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert _digits_from_image(crop) == [935]
    screen = np.zeros((1280, 720, 3), dtype=np.uint8)
    box = page.BOX_EXAM_HP
    screen[box.y1:box.y2, box.x1:box.x2] = crop
    monkeypatch.setattr(page, 'device', SimpleNamespace(screenshot=lambda: screen))
    assert page.LessonBattleContext().fetch_exam_hp() == 35


@pytest.mark.parametrize('value,primary,confidence', [
    (28, '8', 0.999), (57, '7', 0.81), (79, '9', 0.999),
    (10, '10', 0.97), (35, '35', 0.97),
])
def test_ambiguous_hp_values_remain_unknown(monkeypatch, value, primary, confidence):
    monkeypatch.setattr(page, 'en', lambda: SimpleNamespace(ocr=lambda img: [SimpleNamespace(text=primary, confidence=confidence)]))
    assert page._verify_hp_reading(np.zeros((41, 63, 3), dtype=np.uint8), value) is None


@pytest.mark.parametrize('raw_value', [28, 57])
def test_real_two_digit_heart_prefixes_are_not_trusted_as_hp(monkeypatch, raw_value):
    raw = Path(__file__).with_name(f'exam_hp_prefix_{raw_value}.png.b64').read_text(encoding='ascii')
    crop = cv2.imdecode(np.frombuffer(base64.b64decode(raw), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert _digits_from_image(crop) == [raw_value]
    screen = np.zeros((1280, 720, 3), dtype=np.uint8)
    box = page.BOX_EXAM_HP
    screen[box.y1:box.y2, box.x1:box.x2] = crop
    monkeypatch.setattr(page, 'device', SimpleNamespace(screenshot=lambda: screen))
    assert page.LessonBattleContext().fetch_exam_hp() is None


@pytest.mark.parametrize('value', [0, 4, 15])
def test_low_confidence_hp_recovers_from_same_crop_white_digits(monkeypatch, value):
    raw = Path(__file__).with_name(f'hp_white_digits_{value}.png.b64').read_text(encoding='ascii')
    crop = cv2.imdecode(np.frombuffer(base64.b64decode(raw), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert _digits_from_image(crop) == [value]
    assert page._verify_hp_reading(crop, value) == value


def test_high_confidence_hp_conflict_cannot_be_overridden_by_mask(monkeypatch):
    calls = []
    def read(img):
        calls.append(img)
        text = '9' if len(calls) == 1 else '29'
        return [SimpleNamespace(text=text, confidence=1.0)]
    monkeypatch.setattr(page, 'en', lambda: SimpleNamespace(ocr=read))
    assert page._verify_hp_reading(np.zeros((41, 63, 3), dtype=np.uint8), 29) is None
    assert len(calls) == 1


@pytest.mark.parametrize('text,confidence', [('4', 0.97), ('O', 1.0), ('74', 1.0)])
def test_white_digit_fallback_requires_literal_high_confidence_agreement(monkeypatch, text, confidence):
    results = iter([
        [SimpleNamespace(text='74', confidence=0.6)],
        [SimpleNamespace(text=text, confidence=confidence)],
    ])
    monkeypatch.setattr(page, 'en', lambda: SimpleNamespace(ocr=lambda img: next(results)))
    assert page._verify_hp_reading(np.zeros((41, 63, 3), dtype=np.uint8), 4) is None


def test_live_genki_prefix_117_recovers_visible_11():
    raw = Path(__file__).with_name('genki_prefix_117_as_11.png.b64').read_text(encoding='ascii')
    crop = cv2.imdecode(np.frombuffer(base64.b64decode(raw), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert _digits_from_image(crop, fallback_raw=True) == [117]
    assert page._verify_genki_reading(crop, 117) == 11


@pytest.mark.parametrize('independent,confidence,expected', [
    ('13', 0.999, 13), ('117', 0.999, 117), ('77', 0.999, None),
    ('17', 0.97, None), ('O', 0.999, None),
])
def test_genki_prefix_requires_independent_same_crop_confirmation(
    monkeypatch, independent, confidence, expected,
):
    monkeypatch.setattr(page, 'en', lambda: SimpleNamespace(
        ocr=lambda image: [SimpleNamespace(text=independent, confidence=confidence)]))
    raw = 713 if independent == '13' else 117
    assert page._verify_genki_reading(np.zeros((38, 54, 3), dtype=np.uint8), raw) == expected
