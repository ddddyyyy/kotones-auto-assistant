"""Read the complete fixed extra-play control from this capture only."""
import cv2
import numpy as np
from cv2.typing import MatLike
from kaa.tasks import R
from .extra_play import ExtraPlayObservation,confirm_extra_play


def read_extra_play(screen: MatLike,frame_id: str) -> ExtraPlayObservation | None:
    if screen.shape!=(1280,720,3) or not frame_id:return None
    box=R.IdolRoad.BoxBattleExtraPlaySearch
    white=lambda p:(p.min(axis=2)>220).astype('uint8')*255
    region=screen[box.y1:box.y2,box.x1:box.x2]
    pixels=white(region)
    one=_literal_control(region,pixels,R.IdolRoad.BattleExtraPlayOneControl.template.pixels)
    two=_literal_control(region,pixels,R.IdolRoad.BattleExtraPlayTwoControl.template.pixels)
    if one is None or two is None:return None
    return confirm_extra_play(frame_id,box.xywh,one[0],one[1],one[2],one[3],one[4],
        two_score=two[0],two_stroke_coverage=two[1],unique_two=two[2],two_blue_coverage=two[4])


def _literal_control(region: MatLike,pixels: MatLike,reference: MatLike) -> tuple[float,float,bool,float,float] | None:
    white=lambda p:(p.min(axis=2)>220).astype('uint8')*255
    template=white(reference)
    # Keep every foreground stroke, including the separate literal count.
    # Its one-pixel contour rejects extra ink; the surrounding stage animation
    # is not part of the native control and must not penalize a complete glyph.
    mask=cv2.dilate((template>0).astype('uint8'),np.ones((3,3),dtype='uint8'))
    scores=cv2.matchTemplate(pixels,template,cv2.TM_CCORR_NORMED,mask=mask)
    scores=np.nan_to_num(scores,nan=-1.,posinf=-1.,neginf=-1.)
    _,score,_,(x,y)=cv2.minMaxLoc(scores)
    h,w=template.shape
    others=scores.copy();others[max(0,y-h+1):y+h,max(0,x-w+1):x+w]=-1
    unique=cv2.minMaxLoc(others)[1]<.97
    overlap=cv2.matchTemplate((pixels/255).astype('float32'),(template/255).astype('float32'),cv2.TM_CCORR)
    coverage=float(overlap.max()/np.count_nonzero(template))
    def blue(p):
        b,g,r=cv2.split(p)
        return ((b>180)&(g>130)&(g<235)&(r<100)).astype('float32')
    blue_target=blue(reference);blue_pixels=blue(region)
    total=float(np.count_nonzero(blue_target))
    if total==0:return None
    blue_overlap=cv2.matchTemplate(blue_pixels,blue_target,cv2.TM_CCORR)
    blue_coverage=float(blue_overlap.max()/total)
    one_blue=float(blue_overlap[y,x]/total)
    return (max(0.,min(1.,float(score))),max(0.,min(1.,coverage)),unique,
        max(0.,min(1.,blue_coverage)),max(0.,min(1.,one_blue)))
