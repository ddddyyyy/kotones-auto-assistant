"""Current attribute multiplier evidence, separate from score and predicted gains."""
from dataclasses import dataclass, replace
import re
from typing import Literal
from .rules import RuleTextRead


@dataclass(frozen=True)
class ScorePercentObservation:
    frame_id: str
    rect: tuple[int, int, int, int] | None
    reads: tuple[RuleTextRead, ...]
    edge_clipped: bool = False
    status: Literal['region_unknown', 'number_unknown', 'conflict', 'resolved'] = 'number_unknown'
    value: int | None = None


def confirm_score_percent(frame_id: str, rect: tuple[int, int, int, int] | None,
                          reads: tuple[RuleTextRead, ...], edge_clipped: bool = False) -> ScorePercentObservation:
    result = ScorePercentObservation(frame_id, rect, reads, edge_clipped)
    if (not frame_id or rect is None or any(type(v) is not int for v in rect)
        or min(rect) < 0 or min(rect[2:]) <= 0 or rect[0]+rect[2] > 720 or rect[1]+rect[3] > 1280):
        return replace(result, status='region_unknown')
    if (edge_clipped is not False or len(reads) != 2
        or {read.representation for read in reads} != {'original', 'grayscale'}
        or any(read.frame_id != frame_id or len(read.lines) != 1 for read in reads)):
        return result
    values = []
    for read in reads:
        line = read.lines[0]
        if not .98 <= line.confidence <= 1:
            continue
        # Preserve the shared reader's spacing tolerance, but require the full
        # percent suffix and canonical decimal digits; no numeric coercion.
        match = re.fullmatch(r'(0|[1-9][0-9]{0,3})%', line.text.replace(' ', ''))
        if match is None:
            return replace(result, status='conflict')
        values.append(int(match[1]))
    if len(set(values)) > 1:
        return replace(result, status='conflict')
    return replace(result, status='resolved', value=values[0]) if len(values) == 2 else result
