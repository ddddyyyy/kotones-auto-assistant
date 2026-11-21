"""Visible main turn count plus separately queued additions, not absolute time."""
from dataclasses import dataclass


@dataclass(frozen=True)
class QueuedTurnObservation:
    frame_id: str
    base_remaining: int | None
    one_score: float
    two_score: float
    pill_coverage: float
    ink_coverage: float
    queued: int | None
    remaining: int | None


def confirm_queued_turns(frame_id: str,base_remaining: int | None,one_score: float,two_score: float,
                         pill_coverage: float,ink_coverage: float) -> QueuedTurnObservation:
    queued=remaining=None
    if (frame_id and all(type(v) in (int,float) and 0<=v<=1
        for v in (one_score,two_score,pill_coverage,ink_coverage))):
        if one_score>=.97 and two_score<.97 and pill_coverage>=.8 and ink_coverage>=.95:queued=1
        elif two_score>=.97 and one_score<.97 and pill_coverage>=.8 and ink_coverage>=.95:queued=2
        elif max(one_score,two_score)<.97 and max(pill_coverage,ink_coverage)<=.05:queued=0
        if type(base_remaining) is int and type(queued) is int and 1<=base_remaining<=30 and base_remaining+queued<=30:
            remaining=base_remaining+queued
    return QueuedTurnObservation(frame_id,base_remaining,one_score,two_score,pill_coverage,ink_coverage,queued,remaining)


def queued_turns_valid(evidence: QueuedTurnObservation,frame_id: str,remaining: int | None) -> bool:
    return (evidence.frame_id==frame_id and evidence.remaining==remaining
        and confirm_queued_turns(evidence.frame_id,evidence.base_remaining,evidence.one_score,
            evidence.two_score,evidence.pill_coverage,evidence.ink_coverage)==evidence)
