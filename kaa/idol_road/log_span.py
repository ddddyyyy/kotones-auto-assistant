"""One owned upward scroll joins a visible header to a previously observed tail."""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING,Literal
import re
if TYPE_CHECKING:
    from .battle_log import LogRequest,LogViewport
    from .log_turns import VisibleLogTurn


@dataclass(frozen=True)
class LogOverlap:
    top_frame_id: str
    bottom_frame_id: str
    top_body_digest: str
    bottom_body_digest: str
    crop_rect: tuple[int,int,int,int]
    offset: int
    score: float
    alternative_score: float
    signal_pixels: int
    source: Literal['same_panel_blurred_gray_registration'] = 'same_panel_blurred_gray_registration'


@dataclass(frozen=True)
class OwnedLogSpan:
    request_id: str
    battle_id: str
    input_sequence: int
    top: LogViewport
    bottom: LogViewport
    overlap: LogOverlap
    source: Literal['owned_header_to_tail_overlap'] = 'owned_header_to_tail_overlap'


def span_header(span: OwnedLogSpan,request: LogRequest,view: LogViewport) -> VisibleLogTurn | None:
    from .battle_log import log_at_bottom
    from .log_turns import confirm_visible_log_turn
    top,bottom,proof=span.top,span.bottom,span.overlap
    if (span.source!='owned_header_to_tail_overlap' or request.card_probe is None
        or (span.request_id,span.battle_id,span.input_sequence)!=(request.request_id,request.session.battle_id,request.input_sequence)
        or top!=view or top.frame_id==bottom.frame_id or not top.content_confirmed
        or len(top.headers)!=1 or bottom.headers or log_at_bottom(bottom.scrollbar) is not True
        or log_at_bottom(top.scrollbar) is not False or top.scrollbar is None or bottom.scrollbar is None
        or top.scrollbar.frame_id!=top.frame_id or bottom.scrollbar.frame_id!=bottom.frame_id
        or top.scrollbar.track_rect!=bottom.scrollbar.track_rect
        or abs(top.scrollbar.thumb_rect[3]-bottom.scrollbar.thumb_rect[3])>1
        or top.scrollbar.thumb_rect[1]>=bottom.scrollbar.thumb_rect[1]-3
        or len(bottom.tail_clear_fractions)!=2 or any(not .995<=v<=1 for v in bottom.tail_clear_fractions)):
        return None
    if (proof.source!='same_panel_blurred_gray_registration'
        or (proof.top_frame_id,proof.bottom_frame_id)!=(top.frame_id,bottom.frame_id)
        or (proof.top_body_digest,proof.bottom_body_digest)!=(top.body_fingerprint,bottom.body_fingerprint)
        or any(re.fullmatch('[0-9a-f]{64}',v) is None for v in (proof.top_body_digest,proof.bottom_body_digest))
        or min(proof.crop_rect)<0 or min(proof.crop_rect[2:])<=0
        or proof.crop_rect[0]+proof.crop_rect[2]>720 or proof.crop_rect[1]+proof.crop_rect[3]>1280
        or type(proof.offset) is not int or not 3<=proof.offset<=proof.crop_rect[3]*.4
        or not .98<=proof.score<=1 or not -1<=proof.alternative_score<.9
        or proof.score-proof.alternative_score<.05 or type(proof.signal_pixels) is not int
        or not 500<=proof.signal_pixels<=proof.crop_rect[2]*(proof.crop_rect[3]-proof.offset)):
        return None
    track,upper,lower=top.scrollbar.track_rect,top.scrollbar.thumb_rect,bottom.scrollbar.thumb_rect
    expected=(lower[1]-upper[1])*track[3]/upper[3]
    if abs(proof.offset-expected)>max(8,expected*.05):
        return None
    header=top.headers[0]
    if (header.evidence.frame_id!=top.frame_id or confirm_visible_log_turn(header.evidence)!=header
        or header.status!='resolved' or header.remaining_turns!=request.remaining_turns
        or header.evidence.turn_rect[1]+header.evidence.turn_rect[3]-proof.offset>bottom.scrollbar.track_rect[1]):
        return None
    return header
