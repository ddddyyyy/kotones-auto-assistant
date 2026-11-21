"""当前好调持续时间的独立证据；图标缺失不证明持续时间为零。"""
from dataclasses import dataclass
import re
from typing import Literal
from .rules import RuleTextRead


@dataclass(frozen=True)
class GoodConditionNeutralDigits:
    frame_id: str
    rect: tuple[int,int,int,int]
    reads: tuple[RuleTextRead,...]
    right_edge_ink: bool
    mode: Literal['whole_prefix_neutral_white_double_size'] = 'whole_prefix_neutral_white_double_size'


@dataclass(frozen=True)
class GoodConditionDigits:
    frame_id: str
    digit_rect: tuple[int,int,int,int]
    suffix_rect: tuple[int,int,int,int]
    suffix_score: float
    reads: tuple[RuleTextRead,...]
    right_edge_ink: bool = False
    mode: Literal['whole_prefix_only_rec','whole_prefix_double_size'] = 'whole_prefix_only_rec'
    neutral: GoodConditionNeutralDigits | None = None


@dataclass(frozen=True)
class GoodConditionObservation:
    frame_id: str
    icon_rect: tuple[int, int, int, int] | None
    icon_score: float | None
    active_blue_fraction: float | None
    number_rect: tuple[int, int, int, int] | None
    reads: tuple[RuleTextRead, ...]
    scaled_reads: tuple[RuleTextRead, ...] = ()
    status: Literal['icon_unknown', 'number_unknown', 'conflict', 'resolved'] = 'icon_unknown'
    value: int | None = None
    value_source: Literal['original_pair', 'same_crop_double_size','matched_suffix_and_digits','matched_suffix_and_neutral_digits'] | None = None
    digits: GoodConditionDigits | None = None


def _value(text: str) -> int | None:
    match = re.fullmatch(r'(0|[1-9][0-9]*)ターン', text)
    return int(match[1]) if match else None


def duration_digit_proposals(reads: tuple[RuleTextRead,...],scaled_reads: tuple[RuleTextRead,...]) -> frozenset[int]:
    """保留旧整句所有完整提议；弱读数可由新强证据复核，高置信冲突仍在主验证器否决。"""
    values={_value(read.lines[0].text) for read in (*reads,*scaled_reads) if len(read.lines)==1}
    values.discard(None)
    return frozenset(value for value in values if value is not None)


def confirm_good_condition(frame_id: str, icon_rect: tuple[int, int, int, int] | None,
                           icon_score: float | None, active_blue_fraction: float | None,
                           number_rect: tuple[int, int, int, int] | None,
                           reads: tuple[RuleTextRead, ...],
                           scaled_reads: tuple[RuleTextRead, ...] = (),
                           digits: GoodConditionDigits | None = None) -> GoodConditionObservation:
    def result(status, value=None, source=None):
        return GoodConditionObservation(frame_id, icon_rect, icon_score, active_blue_fraction,
                                         number_rect, reads, scaled_reads, status, value, source,digits)
    if (not frame_id or icon_rect is None or min(icon_rect) < 0 or min(icon_rect[2:]) <= 0
        or icon_score is None or not .97 <= icon_score <= 1
        or active_blue_fraction is None or not .15 < active_blue_fraction <= 1):
        return result('icon_unknown')
    if (number_rect is None or min(number_rect) < 0 or min(number_rect[2:]) <= 0
        or number_rect[0] < icon_rect[0]+icon_rect[2] or number_rect[1] < icon_rect[1]):
        return result('number_unknown')
    values = []
    for index, pair in enumerate((reads, scaled_reads)):
        if index == 1 and not pair:
            continue
        if (len(pair) != 2 or {read.representation for read in pair} != {'original', 'grayscale'}
            or any(read.frame_id != frame_id for read in pair)):
            return result('number_unknown')
        if any(len(read.lines) > 1 for read in pair):
            return result('conflict')
        for read in pair:
            if len(read.lines) == 1 and .98 <= read.lines[0].confidence <= 1:
                value = _value(read.lines[0].text)
                if value is None:
                    return result('conflict')
                values.append(value)
    if digits is not None:
        nx,ny,nw,nh=number_rect
        sx,sy,sw,sh=digits.suffix_rect
        if (digits.frame_id!=frame_id or digits.mode not in ('whole_prefix_only_rec','whole_prefix_double_size')
            or not .97<=digits.suffix_score<=1 or min(sx,sy)<0 or min(sw,sh)<=0
            or not nx<sx or sx+sw>nx+nw or not ny<=sy or sy+sh>ny+nh
            or digits.digit_rect!=(nx,ny,sx-nx,nh)
            or len(digits.reads)!=2 or {read.representation for read in digits.reads}!={'original','grayscale'}
            or any(read.frame_id!=frame_id for read in digits.reads)):
            return result('number_unknown')
        if any(len(read.lines)>1 for read in digits.reads):
            return result('conflict')
        for read in digits.reads:
            if len(read.lines)==1 and .98<=read.lines[0].confidence<=1:
                text=read.lines[0].text
                if re.fullmatch(r'0|[1-9][0-9]*',text) is None:
                    return result('conflict')
                values.append(int(text))
        neutral=digits.neutral
        if neutral is not None:
            if (neutral.frame_id!=frame_id or neutral.rect!=digits.digit_rect
                or neutral.mode!='whole_prefix_neutral_white_double_size'
                or len(neutral.reads)!=2 or {read.representation for read in neutral.reads}!={'original','grayscale'}
                or any(read.frame_id!=frame_id for read in neutral.reads)):
                return result('number_unknown')
            if any(len(read.lines)>1 for read in neutral.reads):
                return result('conflict')
            for read in neutral.reads:
                if len(read.lines)==1 and .98<=read.lines[0].confidence<=1:
                    text=read.lines[0].text
                    if re.fullmatch(r'0|[1-9][0-9]*',text) is None:
                        return result('conflict')
                    values.append(int(text))
    if len(set(values)) > 1:
        return result('conflict')
    if digits is not None and digits.right_edge_ink:
        return result('number_unknown')
    if digits is not None and digits.neutral is not None and digits.neutral.right_edge_ink is not False:
        return result('number_unknown')
    # 旧路径仍要求原色／灰度完整一致；新路径另有完整后缀和强数字对，不舍弃高置信冲突。
    proposed = [_value(read.lines[0].text) if len(read.lines)==1 else None for read in reads]
    if proposed[0] is not None and proposed[0]==proposed[1]:
        for pair, source in ((reads, 'original_pair'), (scaled_reads, 'same_crop_double_size')):
            if len(pair) == 2 and all(len(read.lines) == 1 and .98 <= read.lines[0].confidence <= 1 for read in pair):
                if all(_value(read.lines[0].text) == proposed[0] for read in pair):
                    return result('resolved', proposed[0], source)
    proposals=duration_digit_proposals(reads,scaled_reads)
    if (digits is not None and proposals and all(len(read.lines)==1
        and .98<=read.lines[0].confidence<=1 for read in digits.reads)):
        candidate=int(digits.reads[0].lines[0].text)  # High-confidence numeric grammar checked above.
        if candidate in proposals and all(read.lines[0].text==str(candidate) for read in digits.reads):
            return result('resolved',candidate,'matched_suffix_and_digits')
    if digits is not None and digits.neutral is not None and proposals:
        raw=digits.reads
        original=next(read for read in raw if read.representation=='original')
        neutral=digits.neutral.reads
        if (len(original.lines)==1 and .98<=original.lines[0].confidence<=1
            and all(len(read.lines)==1 and read.lines[0].text==original.lines[0].text for read in raw)
            and all(len(read.lines)==1 and .98<=read.lines[0].confidence<=1
                    and read.lines[0].text==original.lines[0].text for read in neutral)):
            candidate=int(original.lines[0].text)
            if candidate in proposals:
                return result('resolved',candidate,'matched_suffix_and_neutral_digits')
    return result('number_unknown')
