"""Initial native Sub-memory group only; no full roster or card inventory scan."""
from hashlib import sha256
from uuid import uuid4
import math
from typing import Literal
import cv2
import numpy as np
from cv2.typing import MatLike
from kotonebot.backend.image import find
from kotonebot.backend.ocr import jp
from kotonebot.primitives import Rect
from kaa.tasks import R
from .formation_page import read_memory_power
from .memory_picker import MemoryCandidate,SubMemoryViewport
from .rules import RuleTextRead,RuleTextLine


def read_sub_memory_selector(screen: MatLike) -> SubMemoryViewport | None:
    if screen.shape!=(1280,720,3):
        return None
    road=R.IdolRoad
    box=road.BoxMemorySubConfirm
    reference=road.BoxMemorySubConfirmReference;number=road.BoxMemorySubConfirmNumber
    white=lambda p:(p.min(axis=2)>220).astype('uint8')*255
    template=white(road.MemorySubConfirm.template.pixels)
    mask=cv2.dilate((template>0).astype('uint8'),np.ones((3,3),dtype='uint8'))
    mask[number.y1-reference.y1:number.y2-reference.y1,number.x1-reference.x1:number.x2-reference.x1]=0
    scores=cv2.matchTemplate(white(screen[box.y1:box.y2,box.x1:box.x2]),template,cv2.TM_CCORR_NORMED,mask=mask)
    scores=np.nan_to_num(scores,nan=-1.,posinf=-1.,neginf=-1.)
    _,confirm_score,_,(x,y)=cv2.minMaxLoc(scores)
    search=find(screen,road.MemorySelectorSearch.template,rect=road.BoxMemorySelectorSearch,threshold=.97)
    if not .97<=confirm_score<=1.000001 or search is None:
        return None
    name=uuid4().hex
    h,w=template.shape
    confirm=Rect(box.x1+x,box.y1+y,w,h)
    crop=screen[confirm.y1:confirm.y2,confirm.x1:confirm.x2]
    variants: tuple[tuple[Literal['original','grayscale'],MatLike],...]=(
        ('original',crop),('grayscale',cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)))
    reads=tuple(RuleTextRead(name,kind,tuple(RuleTextLine(r.text,r.confidence) for r in jp().ocr(pixels)))
        for kind,pixels in variants)
    editing=find(screen,road.MemoryEditingGroup.template,rect=road.BoxMemoryEditingGroup,threshold=.97)
    if editing is None:
        return SubMemoryViewport(name,(),False,confirm_rect=confirm.xywh,confirm_reads=reads,confirm_control_score=min(1.,confirm_score))
    reference=road.BoxMemoryCandidateReference
    candidates=[]
    for row in range(2):
        for col in range(3):
            dx=col*road.PointMemoryCandidateColumnStride.x
            dy=row*road.PointMemoryCandidateRowStride.y
            def shifted(box):return Rect(box.x1+dx,box.y1+dy,box.w,box.h)
            rect=shifted(reference)
            value=read_memory_power(screen,shifted(road.BoxMemoryCandidatePower))
            badge=shifted(road.BoxMemoryCandidateBadgeSearch)
            crop=screen[badge.y1:badge.y2,badge.x1:badge.x2]
            score=float(cv2.minMaxLoc(cv2.matchTemplate(crop,road.MemoryAssignedBadge.template.pixels,cv2.TM_CCOEFF_NORMED))[1])
            assigned=(True if score>=.97 else False if score<=.8 else None) if math.isfinite(score) else None
            outer=shifted(road.BoxMemoryCandidateHighlightRing)
            pixels=screen[outer.y1:outer.y2,outer.x1:outer.x2]
            hsv=cv2.cvtColor(pixels,cv2.COLOR_BGR2HSV)
            orange=(hsv[:,:,0]>=5)&(hsv[:,:,0]<=40)&(hsv[:,:,1]>130)&(hsv[:,:,2]>170)
            ring=np.ones(orange.shape,dtype='bool')
            ring[rect.y1-outer.y1:rect.y2-outer.y1,rect.x1-outer.x1:rect.x2-outer.x1]=False
            fraction=float(orange[ring].mean())
            highlighted=True if fraction>=.18 else False if fraction<=.10 else None
            art=shifted(road.BoxMemoryCandidatePresentation)
            fingerprint=sha256(screen[art.y1:art.y2,art.x1:art.x2].tobytes()).hexdigest()
            candidates.append(MemoryCandidate(name,rect.xywh,value,fingerprint,assigned,highlighted))
    return SubMemoryViewport(name,tuple(candidates),True,confirm_rect=confirm.xywh,
        confirm_reads=reads,confirm_control_score=min(1.,confirm_score))
