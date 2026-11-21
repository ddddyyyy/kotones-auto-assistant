"""Read only the current status-column Slump glyph, never a forecast or old frame."""
import cv2
from cv2.typing import MatLike
from kaa.tasks import R
from .score_suppression import CurrentScoreSuppression


def read_score_suppression(screen: MatLike,frame_id: str) -> CurrentScoreSuppression | None:
    if screen.shape!=(1280,720,3) or not frame_id:return None
    box=R.IdolRoad.BoxBattleConcentrationStack
    crop=screen[box.y1:box.y2,box.x1:box.x2]
    white=lambda p:(p.min(axis=2)>220).astype('uint8')*255
    template=white(R.IdolRoad.BattleCurrentSlumpIcon.template.pixels);h,w=template.shape
    scores=cv2.matchTemplate(white(crop),template,cv2.TM_CCOEFF_NORMED)
    _,score,_,(x,y)=cv2.minMaxLoc(scores)
    if not .97<=score<=1.000001:return None
    others=scores.copy();others[max(0,y-h+1):y+h,max(0,x-w+1):x+w]=-1
    if cv2.minMaxLoc(others)[1]>=.97:return None
    hsv=cv2.cvtColor(crop[y:y+h,x:x+w],cv2.COLOR_BGR2HSV)
    fraction=float(((((hsv[:,:,0]<=15)|(hsv[:,:,0]>=170)))
        &(hsv[:,:,1]>=160)&(hsv[:,:,2]>=150)).mean())
    if fraction<=.15:return None
    return CurrentScoreSuppression(frame_id,(box.x1+x,box.y1+y,w,h),min(1.,float(score)),fraction,True)
