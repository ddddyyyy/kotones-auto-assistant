"""Full-crop white glyph evidence for a narrowly defined detector text artifact."""
from dataclasses import dataclass
import re
import unicodedata
from .rules import RuleTextRead


@dataclass(frozen=True)
class LogRemainingGlyphs:
    frame_id: str
    rect: tuple[int,int,int,int]
    neutral_components: tuple[tuple[int,int,int,int,int], ...]
    grayscale_components: tuple[tuple[int,int,int,int,int], ...]
    neutral_pixels: int
    grayscale_pixels: int


def mixed_remaining_proposal(reads: tuple[RuleTextRead,...],frame_id: str) -> int | None:
    if (len(reads)!=2 or {r.representation for r in reads}!={'original','grayscale'}
        or any(r.frame_id!=frame_id or len(r.lines) not in (1,2) for r in reads)
        or not any(len(r.lines)==2 for r in reads)):
        return None
    numbers=[]
    for read in reads:
        high=[]
        for line in read.lines:
            confidence=line.confidence
            text=re.sub(r'\s+','',unicodedata.normalize('NFKC',line.text))
            if type(confidence) not in (int,float) or not 0<=confidence<=1:
                return None
            if confidence>=.98:
                if not re.fullmatch(r'[1-9][0-9]*',text) or not 1<=int(text)<=30:
                    return None
                high.append(int(text))
            elif not text or any(c.isdigit() for c in text) or not any(c.isalpha() for c in text):
                return None
        if len(high)!=1:
            return None
        numbers.extend(high)
    return numbers[0] if numbers[0]==numbers[1] else None


def matches_single_glyph_row(evidence: LogRemainingGlyphs | None,frame_id: str,
                              rect: tuple[int,int,int,int],number: int) -> bool:
    if (evidence is None or evidence.frame_id!=frame_id or evidence.rect!=rect
        or evidence.neutral_components!=evidence.grayscale_components
        or len(evidence.neutral_components)!=len(str(number))):
        return False
    components=evidence.neutral_components
    _,_,width,height=rect
    if not components or type(width) is not int or type(height) is not int or min(width,height)<=0:
        return False
    right=0
    for x,y,w,h,area in components:
        if any(type(v) is not int for v in (x,y,w,h,area)):
            return False
        if (not right<x or not 0<y or x+w>=width or y+h>=height or w<3
            or not .3*height<=h<=.8*height or not 30<=area<=w*h
            or abs(y-components[0][1])>1 or abs(h-components[0][3])>1):
            return False
        right=x+w
    return (type(evidence.neutral_pixels) is int and type(evidence.grayscale_pixels) is int
        and evidence.neutral_pixels==evidence.grayscale_pixels==sum(row[4] for row in components))


def matches_bounded_digit_row(evidence: LogRemainingGlyphs | None,frame_id: str,
                              rect: tuple[int,int,int,int],number: int) -> bool:
    """Two complete masks may differ by one antialiased pixel per glyph.

    Connected components retain their original evidence order. Sorting only
    for comparison accommodates different digit heights, without cropping.
    """
    if (evidence is None or evidence.frame_id!=frame_id or evidence.rect!=rect
        or type(number) is not int or not 1<=number<=30):return False
    _,_,width,height=rect
    if any(len(row)!=5 or any(type(v) is not int for v in row)
           for rows in (evidence.neutral_components,evidence.grayscale_components) for row in rows):return False
    neutral=sorted(evidence.neutral_components)
    grayscale=sorted(evidence.grayscale_components)
    if len(neutral)!=len(str(number)) or len(neutral)!=len(grayscale):return False
    right=0
    center=None
    for first,second in zip(neutral,grayscale):
        x,y,w,h,area=first
        if (first[:4]!=second[:4] or any(type(v) is not int for v in (*first,*second))
            or not right<x or y<=0 or x+w>=width or y+h>=height or w<3
            or not .3*height<=h<=.8*height or not 30<=area<=w*h
            or not 30<=second[4]<=w*h or abs(area-second[4])>1):return False
        midpoint=2*y+h
        if center is not None and abs(midpoint-center)>2:return False
        center=midpoint
        right=x+w
    return (evidence.neutral_pixels==sum(c[4] for c in neutral)
        and evidence.grayscale_pixels==sum(c[4] for c in grayscale))


def detector_agrees_or_is_weak(reads: tuple[RuleTextRead,...],frame_id: str,number: int) -> bool:
    """Weak malformed strings are retained, never repaired into a numeric seed."""
    if (len(reads)!=2 or {r.representation for r in reads}!={'original','grayscale'}
        or any(r.frame_id!=frame_id or not 1<=len(r.lines)<=2 for r in reads)):return False
    for read in reads:
        for line in read.lines:
            text=re.sub(r'\s+','',unicodedata.normalize('NFKC',line.text))
            numeric=re.fullmatch(r'[1-9][0-9]*',text)
            if not text or not 0<=line.confidence<=1:return False
            if numeric is not None and int(text)!=number:return False
            if line.confidence>=.98 and numeric is None:return False
    return True
