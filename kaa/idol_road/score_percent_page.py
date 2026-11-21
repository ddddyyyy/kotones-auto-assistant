"""Read the complete Idol Road attribute percentage from the current screenshot."""
import cv2
from typing import Literal
from cv2.typing import MatLike
from kotonebot.backend.ocr import en
from kaa.tasks import R
from .ocr_crop import neutral_white_glyphs
from .rules import RuleTextRead, RuleTextLine
from .score_percent import ScorePercentObservation, confirm_score_percent


def read_score_percent(screen: MatLike, frame_id: str) -> ScorePercentObservation:
    if screen.shape != (1280, 720, 3) or not frame_id:
        return confirm_score_percent(frame_id, None, ())
    box = R.IdolRoad.BoxBattleScorePercent
    crop = screen[box.y1:box.y2, box.x1:box.x2]
    mask = neutral_white_glyphs(crop)
    height, width = mask.shape
    _, _, components, _ = cv2.connectedComponentsWithStats(mask)
    clipped = any(area >= 4 and (x == 0 or y == 0 or x+w == width or y+h == height)
                  for x, y, w, h, area in components[1:])
    variants: tuple[tuple[Literal['original', 'grayscale'], MatLike], ...] = (
        ('original', crop), ('grayscale', cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
    reads = tuple(RuleTextRead(frame_id, kind, tuple(RuleTextLine(row.text, row.confidence)
                  for row in en().ocr(pixels))) for kind, pixels in variants)
    return confirm_score_percent(frame_id, box.xywh, reads, bool(clipped))
