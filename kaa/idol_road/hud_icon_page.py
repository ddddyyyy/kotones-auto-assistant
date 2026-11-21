"""当前状态列的校准图样匹配；不截图、不读历史或解释数字。"""
from dataclasses import dataclass
import cv2
from cv2.typing import MatLike
from kotonebot.primitives import Rect


@dataclass(frozen=True)
class HudIconMatch:
    rect: tuple[int,int,int,int]
    score: float
    active_blue_fraction: float


def find_current_blue_icon(screen: MatLike, search: Rect,
                            templates: tuple[MatLike, ...]) -> HudIconMatch | None:
    if screen.shape != (1280,720,3):
        return None
    white = lambda image: (image.min(axis=2)>220).astype('uint8')*255
    crop = screen[search.y1:search.y2,search.x1:search.x2]
    candidates = []
    for pixels in templates:
        template = white(pixels)
        h,w = template.shape
        if crop.shape[0]<h or crop.shape[1]<w:
            continue
        scores = cv2.matchTemplate(white(crop),template,cv2.TM_CCOEFF_NORMED)
        _,score,_,(x,y) = cv2.minMaxLoc(scores)
        if not .97<=score<=1:
            continue
        others = scores.copy()
        others[max(0,y-h+1):y+h,max(0,x-w+1):x+w] = -1
        if cv2.minMaxLoc(others)[1]>=.97:
            return None
        candidates.append((score,x,y,w,h))
    if not candidates:
        return None
    score,x,y,w,h = max(candidates)
    if any(abs(xx-x)>=min(w,ww)/2 or abs(yy-y)>=min(h,hh)/2 for _,xx,yy,ww,hh in candidates):
        return None
    x,y = x+search.x1,y+search.y1
    hsv = cv2.cvtColor(screen[y:y+h,x:x+w],cv2.COLOR_BGR2HSV)
    blue = float(((hsv[:,:,0]>=85)&(hsv[:,:,0]<=115)
                   &(hsv[:,:,1]>=160)&(hsv[:,:,2]>=150)).mean())
    return HudIconMatch((x,y,w,h),float(score),blue) if blue>.15 else None
