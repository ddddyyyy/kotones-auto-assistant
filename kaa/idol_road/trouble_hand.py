"""Correct gray Trouble-card availability using this capture's complete fee badge."""
from functools import cached_property

import cv2
import numpy as np
from cv2.typing import MatLike

from kaa.tasks import R
from kaa.tasks.produce.new.play_cards.page import LessonBattleContext
from kaa.tasks.produce.new.play_cards.planner.card_payment import CardPaymentObservation, valid_card_payment
from .battle_observation import BattleFrameObservation


def _read_sleep_payment(screen: MatLike, frame_id: str, rect: tuple[int,int,int,int]) -> CardPaymentObservation | None:
    if screen.shape != (1280,720,3) or not frame_id:
        return None
    x,y,w,h=rect
    # Native badge has a fixed offset from the full-size hand title anchor.
    x+=105; y+=124
    if w!=57 or h!=148 or x<3 or y<3 or x+48>720 or y+43>1280:
        return None
    region=screen[y-3:y+43,x-3:x+48]
    white=lambda p:(p.min(axis=2)>220).astype('uint8')*255
    def green(p):
        hsv=cv2.cvtColor(p,cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv,np.array([20,80,120]),np.array([90,255,255]))>0
    matches=[]
    for value,reference in ((0,R.IdolRoad.SleepCost0.template.pixels),
                            (0,R.IdolRoad.SleepCost0Right.template.pixels),
                            (2,R.IdolRoad.SleepCost2.template.pixels),
                            (2,R.IdolRoad.SleepCost2Center.template.pixels)):
        template=white(reference);pixels=white(region)
        mask=cv2.dilate((template>0).astype('uint8'),np.ones((3,3),dtype='uint8'))
        scores=cv2.matchTemplate(pixels,template,cv2.TM_CCORR_NORMED,mask=mask)
        scores=np.nan_to_num(scores,nan=-1.,posinf=-1.,neginf=-1.)
        _,score,_,(dx,dy)=cv2.minMaxLoc(scores)
        score=max(0.,min(1.,score))
        crop=region[dy:dy+40,dx:dx+45];pixels=white(crop)
        ink=template>0;style=green(reference)
        coverage=float(np.count_nonzero(ink & (pixels>0))/max(1,np.count_nonzero(ink)))
        style_coverage=float(np.count_nonzero(style & green(crop))/max(1,np.count_nonzero(style)))
        payment=CardPaymentObservation(frame_id,'p_card-00-acc-0_002',0,(x-3+dx,y-3+dy,45,40),value,
            score,coverage,style_coverage,True)
        if valid_card_payment(payment,frame_id):matches.append(payment)
    return max(matches,key=lambda p:p.score) if len({p.green for p in matches})==1 else None


class TroubleHandContext(LessonBattleContext):
    def __init__(self, screen: MatLike, frame: BattleFrameObservation):
        super().__init__(screen,single_line_titles=True)
        self.frame=frame
        self.payments: tuple[CardPaymentObservation,...]=()
        self.unresolved_sleep_payments: tuple[tuple[int,int,int,int],...]=()

    @cached_property
    def corrected_hands(self):
        hands=super().fetch_hands()
        sleep=[h for h in hands if h is not None and h.card is not None and h.card._id=='p_card-00-acc-0_002']
        self.unresolved_sleep_payments=tuple(h.rect.xywh for h in sleep)
        payments=[_read_sleep_payment(self.fetch_screen(),self.frame.values.frame_id,h.rect.xywh) for h in sleep]
        if (sleep and all(p is not None for p in payments)
            and all(h.card is not None and h.card.upgrade_count==0 for h in sleep)):
            confirmed=tuple(p for p in payments if p is not None)
            if len({p.rect for p in confirmed})==len(sleep) and len({p.green for p in confirmed})==1:
                self.payments=confirmed
                self.unresolved_sleep_payments=()
                cost=confirmed[0].green
                hp=self.frame.observation.hp
                genki=0 if type(hp) is int and hp>=cost else self.fetch_exam_stamina()
                legal=(type(hp) is int and hp>=0 and type(genki) is int and genki>=0 and hp+genki>=cost)
                for hand in sleep:
                    hand.__dict__['available']=legal
        return hands

    def fetch_hands(self):
        return self.corrected_hands
