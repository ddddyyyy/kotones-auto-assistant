"""Read a current condition row, rechecking an agreeing rank proposal only."""
import re
import unicodedata
import cv2
from cv2.typing import MatLike
from kotonebot.backend.ocr import jp
from .objectives import read_rank_requirement


def _canonical(text: str) -> str:
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',text)).lstrip('・•')


def read_requirement_text(crop: MatLike) -> str | None:
    variants=(crop,cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY))
    pair=[jp().ocr(image) for image in variants]
    if any(len(rows)!=1 for rows in pair):return None
    texts=[_canonical(rows[0].text) for rows in pair]
    if not texts[0] or texts[0]!=texts[1]:return None
    if all(.98<=rows[0].confidence<=1 for rows in pair):return texts[0]
    if read_rank_requirement(None,texts[0]).rank_at_most is None:return None
    for image in variants:
        rows=jp().ocr(cv2.resize(image,None,fx=2,fy=2))
        if (len(rows)!=1 or not .98<=rows[0].confidence<=1
            or _canonical(rows[0].text)!=texts[0]):
            return None
    return texts[0]
