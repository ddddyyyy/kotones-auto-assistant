"""无滚动条的短日志可见尾部证据；不认证当前回合或全部规则。"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ShortLogEvidence:
    frame_id: str
    header_band: tuple[int,int,int,int]
    content_rect: tuple[int,int,int,int]
    last_content_y: int
    tail_rect: tuple[int,int,int,int]
    top_clear_rect: tuple[int,int,int,int]
    track_rect: tuple[int,int,int,int]
    top_clear_fraction: float
    track_clear_fraction: float
    tail_clear_fraction: float
    text_rect: tuple[int,int,int,int]
    text_fingerprint: str


def short_log_end_visible(evidence: ShortLogEvidence | None,frame_id: str,
                          first_header_rect: tuple[int,int,int,int]) -> bool:
    if evidence is None or evidence.frame_id!=frame_id or not frame_id:
        return False
    rects=(evidence.header_band,evidence.content_rect,evidence.tail_rect,evidence.top_clear_rect,evidence.track_rect,
           evidence.text_rect,first_header_rect)
    if any(min(rect)<0 or min(rect[2:])<=0 or rect[0]+rect[2]>720 or rect[1]+rect[3]>1280 for rect in rects):
        return False
    hx,hy,hw,hh=evidence.header_band
    x,y,w,h=evidence.content_rect
    tx,ty,tw,th=evidence.tail_rect
    cx,cy,cw,ch=evidence.top_clear_rect
    sx,sy,sw,sh=evidence.track_rect
    fx,fy,fw,fh=first_header_rect
    px,py,pw,ph=evidence.text_rect
    return (hx<=fx and fx+fw<=hx+hw and hy<=fy and fy+fh<=hy+hh
            and cx<=hx and cx+cw>=hx+hw-5 and cy+ch<=hy and cw>=500
            and hy+hh<=y and y+100<=evidence.last_content_y<ty
            and tx==x and tw==w and ty>=evidence.last_content_y+10 and ty+th==y+h and th>=160
            and w>=600 and sx>=x+w and sy<=hy and sh>=950 and sw>=3
            and x<=px<=hx and px+pw==x+w and py==y and ph==h
            and re.fullmatch('[0-9a-f]{64}',evidence.text_fingerprint) is not None
            and all(.999<=value<=1 for value in (evidence.top_clear_fraction,evidence.track_clear_fraction,evidence.tail_clear_fraction)))
