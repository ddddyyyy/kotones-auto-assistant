"""Register the whole existing log-text crop; no digit/title inference or input."""
from hashlib import sha256
import cv2
from cv2.typing import MatLike
from kaa.tasks import R
from .battle_log import LogViewport
from .log_span import LogOverlap


def read_log_overlap(top: MatLike,bottom: MatLike,top_view: LogViewport,bottom_view: LogViewport) -> LogOverlap | None:
    if top.shape!=(1280,720,3) or bottom.shape!=top.shape or top_view.frame_id==bottom_view.frame_id:
        return None
    body=R.IdolRoad.BoxBattleLogBody
    if (sha256(top[body.y1:body.y2,body.x1:body.x2].tobytes()).hexdigest()!=top_view.body_fingerprint
        or sha256(bottom[body.y1:body.y2,body.x1:body.x2].tobytes()).hexdigest()!=bottom_view.body_fingerprint):
        return None
    box=R.IdolRoad.BoxShortLogText
    images=[cv2.GaussianBlur(cv2.cvtColor(im[box.y1:box.y2,box.x1:box.x2],cv2.COLOR_BGR2GRAY),(3,3),0)
            for im in (top,bottom)]
    scores=[]
    for offset in range(3,int(box.h*.4)+1):
        upper,lower=images[0][offset:],images[1][:-offset]
        if cv2.countNonZero((upper<210).astype('uint8'))<500 or cv2.countNonZero((lower<210).astype('uint8'))<500:
            continue
        score=float(cv2.matchTemplate(upper,lower,cv2.TM_CCOEFF_NORMED)[0,0])
        scores.append((score,offset))
    if not scores:
        return None
    score,offset=max(scores)
    alternative=max((value for value,shift in scores if abs(shift-offset)>3),default=1.)
    if not .98<=score<=1 or alternative>=.9 or score-alternative<.05:
        return None
    return LogOverlap(top_view.frame_id,bottom_view.frame_id,top_view.body_fingerprint,bottom_view.body_fingerprint,
                      box.xywh,offset,score,alternative,cv2.countNonZero((images[0][offset:]<210).astype('uint8')))
