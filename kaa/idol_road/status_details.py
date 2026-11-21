"""当前状态详情里的明确数值；不推算触发来源、到期时刻或缺失效果。"""
from dataclasses import dataclass
import re
import unicodedata
from typing import Literal
from .rules import RuleTextRead


@dataclass(frozen=True)
class StatusSplitValueEvidence:
    frame_id: str
    play_count_rect: tuple[int,int,int,int]
    turn_count_rect: tuple[int,int,int,int]
    play_count_reads: tuple[RuleTextRead,...]
    turn_count_reads: tuple[RuleTextRead,...]


@dataclass(frozen=True)
class StatusRowEvidence:
    frame_id: str
    title_rect: tuple[int,int,int,int]
    title_score: float
    title_reads: tuple[RuleTextRead,...]
    value_reads: tuple[RuleTextRead,...]
    description_reads: tuple[RuleTextRead,...]
    split_value: StatusSplitValueEvidence | None = None
    occluded: bool = False


@dataclass(frozen=True)
class VisibleBattleStatus:
    evidence: StatusRowEvidence
    status: Literal['unknown','resolved']
    kind: Literal['lesson_buff','extra_play'] | None = None
    remaining_turns: int | None = None
    available_plays: int | None = None
    """该条状态显示的追加次数，不代表回合总出牌次数或触发来源。"""
    score_increase_percent: int | None = None
    value_source: Literal['whole_line','split_fields'] | None = None


@dataclass(frozen=True)
class BattleStatusDetails:
    frame_id: str
    battle_id: str
    character_id: str
    stage: int
    rows: tuple[VisibleBattleStatus,...]
    identity_source: str = 'unconfirmed'
    source: Literal['visible_battle_status_details'] = 'visible_battle_status_details'
    complete: Literal[False] = False
    trigger_attribution_confirmed: Literal[False] = False


def _text(reads: tuple[RuleTextRead,...], frame_id: str) -> str | None:
    if (len(reads)!=2 or {r.representation for r in reads}!={'original','grayscale'}
        or any(r.frame_id!=frame_id or len(r.lines)!=1 or not .98<=r.lines[0].confidence<=1 for r in reads)):
        return None
    values=[re.sub(r'\s+','',unicodedata.normalize('NFKC',r.lines[0].text)) for r in reads]
    return values[0] if values[0]==values[1] else None


def _extra_value(text: str | None) -> tuple[int,int] | None:
    match=re.fullmatch(r'([1-9][0-9]*)回[|│]?([1-9][0-9]*)ターン',text or '')
    return (int(match[1]),int(match[2])) if match is not None else None


def _split_value(evidence: StatusRowEvidence) -> tuple[int,int] | None:
    split=evidence.split_value
    if (split is None or split.frame_id!=evidence.frame_id or len(evidence.value_reads)!=2
        or {read.representation for read in evidence.value_reads}!={'original','grayscale'}):
        return None
    left,right=split.play_count_rect,split.turn_count_rect
    title=evidence.title_rect
    if (min(left)<0 or min(right)<0 or min(left[2:])<=0 or min(right[2:])<=0
        or left[0]+left[2]>right[0] or left[1]!=right[1] or left[3]!=right[3]
        or left[1]<title[1]+title[3] or left[0]>title[0] or left[0]<title[0]-title[3]
        or right[0]+right[2]>title[0]+title[2]):
        return None
    count=re.fullmatch(r'([1-9][0-9]*)回',_text(split.play_count_reads,evidence.frame_id) or '')
    turns=re.fullmatch(r'([1-9][0-9]*)ターン',_text(split.turn_count_reads,evidence.frame_id) or '')
    if count is None or turns is None:
        return None
    pair=int(count[1]),int(turns[1])
    # 不能通过分段补读绕过原整行任何高置信冲突，包括多行或无法解释的字符。
    for read in evidence.value_reads:
        if read.frame_id!=evidence.frame_id:
            return None
        if any(.98<=line.confidence<=1 for line in read.lines):
            if len(read.lines)!=1 or _extra_value(re.sub(r'\s+','',unicodedata.normalize('NFKC',read.lines[0].text)))!=pair:
                return None
    return pair


def confirm_status_row(evidence: StatusRowEvidence) -> VisibleBattleStatus:
    unknown=VisibleBattleStatus(evidence,'unknown')
    if (evidence.occluded or not evidence.frame_id or not .97<=evidence.title_score<=1
        or min(evidence.title_rect)<0 or min(evidence.title_rect[2:])<=0):
        return unknown
    title=_text(evidence.title_reads,evidence.frame_id)
    value=_text(evidence.value_reads,evidence.frame_id)
    body=_text(evidence.description_reads,evidence.frame_id)
    if title=='好調' and value is not None:
        number=re.fullmatch(r'([1-9][0-9]*)ターン',value)
        if number is not None:
            # 剩余回合直接来自状态名下的明确数值；正文不可靠时不补倍率。
            description=re.fullmatch(r'スコア上昇量を50%増加([1-9][0-9]*)ターン',body or '')
            percent=50 if description is not None and number[1]==description[1] else None
            return VisibleBattleStatus(evidence,'resolved','lesson_buff',int(number[1]),score_increase_percent=percent,
                                      value_source='whole_line')
    if title=='スキルカード使用数追加':
        pair=_split_value(evidence) if evidence.split_value is not None else _extra_value(value)
        if pair is not None and body=='スキルカードを追加で使用できる':
            return VisibleBattleStatus(evidence,'resolved','extra_play',pair[1],available_plays=pair[0],
                                      value_source='split_fields' if evidence.split_value is not None else 'whole_line')
    return unknown
