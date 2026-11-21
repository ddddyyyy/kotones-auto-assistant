"""当前集中值的同帧证据；未确认图标或高置信冲突保持未知。"""
from dataclasses import dataclass
import re
from typing import Literal
from .rules import RuleTextRead


@dataclass(frozen=True)
class ConcentrationExpandedDigits:
    frame_id: str
    parent_rect: tuple[int,int,int,int]
    rect: tuple[int,int,int,int]
    reads: tuple[RuleTextRead,...]
    right_edge_ink: bool
    mode: Literal['full_width_extension'] = 'full_width_extension'


@dataclass(frozen=True)
class ConcentrationNumberRead:
    frame_id: str
    representation: Literal['original', 'color_white_mask', 'gray_white_mask','neutral_white_mask','neutral_gray_white_mask']
    text: str | None
    confidence: float | None
    rect: tuple[int, int, int, int]
    row_count: int = 1
    right_edge_ink: bool = False
    scale: Literal[1,2] = 1


@dataclass(frozen=True)
class ConcentrationObservation:
    frame_id: str
    icon_rect: tuple[int, int, int, int] | None
    icon_score: float | None
    reads: tuple[ConcentrationNumberRead, ...]
    status: Literal['icon_unknown', 'number_unknown', 'conflict', 'resolved']
    value: int | None = None
    scaled_reads: tuple[ConcentrationNumberRead, ...] = ()
    value_source: Literal['whole_crop_masks','same_crop_double_size','same_crop_neutral_white','expanded_whole_crop'] | None = None
    neutral_reads: tuple[ConcentrationNumberRead, ...] = ()
    expanded: ConcentrationExpandedDigits | None = None


def expanded_concentration_proposal(reads: tuple[ConcentrationNumberRead,...]) -> int | None:
    """A strong raw original and complete agreeing proposals only request a wider capture."""
    if (len(reads)!=3 or {read.representation for read in reads}!={'original','color_white_mask','gray_white_mask'}
        or any(read.row_count!=1 or read.text is None or re.fullmatch(r'0|[1-9][0-9]*',read.text) is None for read in reads)
        or len({read.text for read in reads})!=1):
        return None
    original=next(read for read in reads if read.representation=='original')
    return int(original.text) if original.text is not None and original.confidence is not None and .98<=original.confidence<=1 else None


def neutral_concentration_proposal(reads: tuple[ConcentrationNumberRead,...],
                                   scaled_reads: tuple[ConcentrationNumberRead,...] = ()) -> int | None:
    """原图强数字与另一强表示只授权同框中性白字复核，不直接授值。"""
    strong=[read for read in reads if read.row_count==1 and read.text is not None
            and re.fullmatch(r'0|[1-9][0-9]*',read.text) is not None
            and read.confidence is not None and .98<=read.confidence<=1]
    original=next((read for read in strong if read.representation=='original'),None)
    if original is None or original.text is None:
        return None
    if sum(read.text==original.text for read in strong)>=2:
        return int(original.text)
    # A complete unchanged raw proposal plus strong enlarged original and
    # another enlarged representation may request an independent neutral pair.
    # These readings alone still cannot resolve the number.
    if (len(reads)!=3 or any(read.row_count!=1 or read.text!=original.text for read in reads)
        or len(scaled_reads)!=3
        or {read.representation for read in scaled_reads}!={'original','color_white_mask','gray_white_mask'}
        or any(read.frame_id!=original.frame_id or read.rect!=original.rect or read.scale!=2 for read in scaled_reads)):
        return None
    scaled_strong=[read for read in scaled_reads if read.row_count==1 and read.text==original.text
        and read.confidence is not None and .98<=read.confidence<=1]
    if len(scaled_strong)>=2 and any(read.representation=='original' for read in scaled_strong):
        return int(original.text)
    return None


def confirm_concentration(frame_id: str, icon_rect: tuple[int,int,int,int] | None,
                          icon_score: float | None, reads: tuple[ConcentrationNumberRead,...],
                          scaled_reads: tuple[ConcentrationNumberRead,...] = (),
                          neutral_reads: tuple[ConcentrationNumberRead,...] = (),
                          expanded: ConcentrationExpandedDigits | None = None) -> ConcentrationObservation:
    def result(status, value=None, source=None):
        return ConcentrationObservation(frame_id, icon_rect, icon_score, reads, status, value, scaled_reads, source,neutral_reads,expanded)
    if (not frame_id or icon_rect is None or icon_score is None or not .97 <= icon_score <= 1
        or min(icon_rect) < 0 or min(icon_rect[2:]) <= 0):
        return result('icon_unknown')
    values=[]
    for index,(group,scale,representations) in enumerate(((reads,1,{'original','color_white_mask','gray_white_mask'}),
        (scaled_reads,2,{'original','color_white_mask','gray_white_mask'}),
        (neutral_reads,2,{'neutral_white_mask','neutral_gray_white_mask'}))):
        if index>0 and not group:
            continue
        if (len(group)!=len(representations) or {read.representation for read in group}!=representations
            or any(read.frame_id!=frame_id or read.scale!=scale or min(read.rect)<0
                   or min(read.rect[2:])<=0 for read in group)
            or len({read.rect for read in (*reads,*group)})!=1):
            return result('number_unknown')
        if any(read.row_count>1 for read in group):
            return result('conflict')
        for read in group:
            if read.confidence is not None and .98 <= read.confidence <= 1:
                if read.text is None or re.fullmatch(r'0|[1-9][0-9]*',read.text) is None:
                    return result('conflict')
                values.append(int(read.text))
    expanded_value=None
    if expanded is not None:
        parent=reads[0].rect
        x,y,w,h=expanded.rect
        if (expanded.frame_id!=frame_id or expanded.mode!='full_width_extension' or expanded.parent_rect!=parent
            or any(type(value) is not int for value in (*expanded.rect,*parent))
            or (x,y,h)!=(parent[0],parent[1],parent[3]) or not parent[2]+2*h<=w<=parent[2]+4*h
            or min(expanded.rect)<0 or x+w>720 or y+h>1280
            or expanded.right_edge_ink is not False
            or len(expanded.reads)!=2 or {read.representation for read in expanded.reads}!={'original','grayscale'}
            or any(read.frame_id!=frame_id for read in expanded.reads)):
            return result('number_unknown')
        if any(len(read.lines)>1 for read in expanded.reads):
            return result('conflict')
        for read in expanded.reads:
            if len(read.lines)==1 and .98<=read.lines[0].confidence<=1:
                text=read.lines[0].text
                if re.fullmatch(r'0|[1-9][0-9]*',text) is None:
                    return result('conflict')
                values.append(int(text))
        proposed=expanded_concentration_proposal(reads)
        if proposed is not None and all(len(read.lines)==1 and .98<=read.lines[0].confidence<=1
            and read.lines[0].text==str(proposed) for read in expanded.reads):
            expanded_value=proposed
    if len(set(values))>1:
        return result('conflict')
    if expanded_value is not None:
        return result('resolved',expanded_value,'expanded_whole_crop')
    if any(read.right_edge_ink for read in (*reads,*scaled_reads,*neutral_reads) if read.representation!='original'):
        return result('number_unknown')
    for group,source in ((reads,'whole_crop_masks'),(scaled_reads,'same_crop_double_size')):
        if not group:
            continue
        if source=='same_crop_double_size':
            # 只复核原始整裁剪的完整一致提议；不补造缺字或裁掉不利字符。
            if (any(read.row_count!=1 or read.text is None or re.fullmatch(r'0|[1-9][0-9]*',read.text) is None
                    for read in reads) or len({read.text for read in reads})!=1):
                continue
        masks=[read for read in group if read.representation!='original']
        if all(read.confidence is not None and .98 <= read.confidence <= 1 and read.text is not None
               and re.fullmatch(r'0|[1-9][0-9]*',read.text) is not None for read in masks):
            if source=='same_crop_double_size' and any(read.text!=reads[0].text for read in masks):
                return result('number_unknown')
            return result('resolved',values[0],source)
    proposed=neutral_concentration_proposal(reads,scaled_reads)
    if (proposed is not None and len(neutral_reads)==2
        and all(read.row_count==1 and read.text==str(proposed) and read.confidence is not None
                and .98<=read.confidence<=1 for read in neutral_reads)):
        return result('resolved',proposed,'same_crop_neutral_white')
    return result('number_unknown')
