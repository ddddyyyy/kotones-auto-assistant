"""Read the complete +1/+2 native badge; failed matching never means zero."""
import cv2
import numpy as np
from cv2.typing import MatLike
from kaa.tasks import R
from .queued_turns import QueuedTurnObservation,confirm_queued_turns


def read_queued_turns(screen: MatLike,frame_id: str,base_remaining: int | None) -> QueuedTurnObservation | None:
    if screen.shape!=(1280,720,3) or not frame_id:return None
    box=R.IdolRoad.BoxQueuedTurns
    pixels=screen[box.y1:box.y2,box.x1:box.x2]
    def ink(p):
        b,g,r=cv2.split(p)
        return ((b>220)&(g>100)&(g<220)&(r<100)).astype('uint8')*255
    white=lambda p:(p.min(axis=2)>230).astype('uint8')*255
    scores=[];pill=[];coverage=[]
    for template in (R.IdolRoad.QueuedTurnOne.template.pixels,R.IdolRoad.QueuedTurnTwo.template.pixels):
        target=ink(template);observed=ink(pixels)
        mask=cv2.dilate((target>0).astype('uint8'),np.ones((3,3),dtype='uint8'))
        score=float(cv2.matchTemplate(observed,target,cv2.TM_CCORR_NORMED,mask=mask)[0,0])
        scores.append(max(0.,min(1.,score)) if np.isfinite(score) else 0.)
        shape=(white(template)>0)|(target>0)
        pill.append(float(np.count_nonzero((white(pixels)>0)&shape)/np.count_nonzero(shape)))
        coverage.append(float(np.count_nonzero((observed>0)&(target>0))/max(1,np.count_nonzero(target))))
    return confirm_queued_turns(frame_id,base_remaining,scores[0],scores[1],max(pill),max(coverage))
