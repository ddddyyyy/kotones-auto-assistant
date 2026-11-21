"""Same-crop HP corroboration when the shared classifier cannot verify HP."""
import re
import cv2
from cv2.typing import MatLike
from kotonebot.backend.ocr import en
from kaa.tasks import R
from kaa.digit_ocr import digit_classifier


def read_masked_hp(screen: MatLike) -> int | None:
    """Require independent whole-mask digits and strong current-crop OCR.

    A colored heart can become a digit in the shared Otsu classifier. This
    reader never removes a prefix or borrows HP from another frame.
    """
    if screen.shape != (1280,720,3):
        return None
    box=R.InProduce.InExam.BoxHp
    crop=screen[box.y1:box.y2,box.x1:box.x2]
    minimum=crop.min(axis=2)
    white=((minimum>245)&(crop.max(axis=2).astype('int16')-minimum<12)).astype('uint8')*255
    ok,encoded=cv2.imencode('.png',white)
    if not ok:
        return None
    text=digit_classifier().classification(encoded.tobytes())
    if not isinstance(text,str) or re.fullmatch(r'0|[1-9][0-9]*',text) is None:
        return None
    # A strong conflicting reading in any representation blocks recovery.
    # At least two complete readings, including the mask, must agree.
    votes=0
    for pixels in (crop,cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),white):
        rows=en().ocr(pixels)
        strong=[row for row in rows if row.confidence>=.98]
        if any(row.text!=text for row in strong):
            return None
        matched=len(rows)==1 and len(strong)==1
        if pixels is white and not matched:
            return None
        votes+=int(matched)
    return int(text) if votes>=2 else None
