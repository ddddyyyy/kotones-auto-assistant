"""独立读取好调条件；后半句收益未知时不补造效果或结算时点。"""
from dataclasses import dataclass
import re
from typing import Literal
import unicodedata

from .rules import RuleTextRead, StageRules, VisibleStageRule,confirmed_rule_turn_with_number


@dataclass(frozen=True)
class GoodConditionRequirement:
    start_turn: int
    minimum_turns: int
    evidence_frames: tuple[str, ...]
    scope: Literal['condition_only'] = 'condition_only'
    reward_confirmed: Literal[False] = False
    trigger_phase_confirmed: Literal[False] = False
    executable: Literal[False] = False


@dataclass(frozen=True)
class GoodConditionAnalysis:
    reads: tuple[RuleTextRead, ...]
    requirement: GoodConditionRequirement | None
    status: Literal['confirmed_condition', 'evidence_unknown', 'high_confidence_conflict', 'unsupported_condition']


@dataclass(frozen=True)
class GoodConditionLexicalEvidence:
    frame_id: str
    body_rect: tuple[int,int,int,int]
    suffix_rect: tuple[int,int,int,int]
    suffix_score: float
    unique: bool
    scaled_reads: tuple[RuleTextRead,...]=()


def _lexical_valid(reads: tuple[RuleTextRead,...],cue: GoodConditionLexicalEvidence | None) -> bool:
    if cue is None:return False
    x,y,w,h=cue.body_rect;sx,sy,sw,sh=cue.suffix_rect
    return bool(cue.frame_id and all(read.frame_id==cue.frame_id for read in reads)
        and min(x,y,sx,sy)>=0 and min(w,h,sw,sh)>0 and x+w<=720 and y+h<=1280
        and x<=sx and y<=sy and sx+sw<=x+w and sy+sh<=y+h and sy-y<=sh
        and .97<=cue.suffix_score<=1 and cue.unique is True)


def _verified_turn_word(text: str) -> str:
    # Only the fixed kana word is disambiguated by the complete native suffix.
    # Digits, comparators and every other character are left untouched.
    return re.sub(r'^(好調が[1-9][0-9]*)(?:夕一ン|夕ーン|タ一ン)(以上の場合)',r'\1ターン\2',text)


def _compact(text: str) -> str:
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))


def _read_requirement(turn: int | None, reads: tuple[RuleTextRead, ...],verified_word: bool=False) -> GoodConditionAnalysis:
    def result(status, requirement=None):
        return GoodConditionAnalysis(reads, requirement, status)

    if (type(turn) is not int or not 1 <= turn <= 30 or len(reads) != 2
        or not reads[0].frame_id or reads[0].frame_id != reads[1].frame_id
        or {read.representation for read in reads} != {'original', 'grayscale'}):
        return result('evidence_unknown')
    clauses = []
    for read in reads:
        prefix = ''
        for line in read.lines:
            # 后半句属于未确认收益，不参与这个独立的条件字段。
            if not .98 <= line.confidence <= 1:
                return result('evidence_unknown')
            prefix += _compact(line.text)
            if '、' in prefix:
                prefix = prefix.split('、', 1)[0]
                break
        else:
            return result('evidence_unknown')
        clauses.append(_verified_turn_word(prefix) if verified_word else prefix)
    if clauses[0] != clauses[1]:
        return result('high_confidence_conflict')
    if clauses[0] == '好調状態の場合':
        minimum = 1
    else:
        match = re.fullmatch(r'好調が([1-9][0-9]*)ターン以上の場合', clauses[0])
        if match is None:
            return result('unsupported_condition')
        minimum = int(match[1])
    return result('confirmed_condition', GoodConditionRequirement(turn, minimum, (reads[0].frame_id,)))


def read_good_condition_requirement(turn: int | None,reads: tuple[RuleTextRead,...],
                                    lexical: GoodConditionLexicalEvidence | None=None) -> GoodConditionAnalysis:
    verified=_lexical_valid(reads,lexical)
    native=_read_requirement(turn,reads,verified)
    if not verified or lexical is None or not lexical.scaled_reads:return native
    if any(read.frame_id!=lexical.frame_id for read in lexical.scaled_reads):return native
    scaled=_read_requirement(turn,lexical.scaled_reads,True)
    strong=set();unsupported=False
    for read in (*reads,*lexical.scaled_reads):
        prefix=''
        for line in read.lines:
            if not .98<=line.confidence<=1:break
            prefix+=_compact(line.text)
            if '、' not in prefix:continue
            clause=_verified_turn_word(prefix.split('、',1)[0])
            match=re.fullmatch(r'好調が([1-9][0-9]*)ターン以上の場合',clause)
            if match:strong.add(int(match[1]))
            elif clause=='好調状態の場合':strong.add(1)
            else:unsupported=True
            break
    if (native.status=='high_confidence_conflict' or scaled.status=='high_confidence_conflict'
        or len(strong)>1 or (unsupported and scaled.requirement is not None)
        or (native.requirement and scaled.requirement and native.requirement.minimum_turns!=scaled.requirement.minimum_turns)):
        return GoodConditionAnalysis(reads,None,'high_confidence_conflict')
    if native.requirement or native.status=='unsupported_condition':return native
    return GoodConditionAnalysis(reads,scaled.requirement,scaled.status)


def _row_lexical(row: VisibleStageRule) -> GoodConditionLexicalEvidence | None:
    cue=row.good_condition_lexical
    if cue is None:return None
    x,y,w,h=row.card_rect;bx,by,bw,bh=cue.body_rect
    return cue if min(x,y)>=0 and x<=bx and y<=by and bx+bw<=x+w and by+bh<=y+h else None


def _recycle_support(row: VisibleStageRule) -> bool:
    cue=_row_lexical(row)
    if not _lexical_valid(row.description_reads,cue) or cue is None:return False
    for reads in (row.description_reads,cue.scaled_reads):
        if (len(reads)!=2 or {read.representation for read in reads}!={'original','grayscale'}
            or any(read.frame_id!=cue.frame_id or not read.lines
                or any(not .98<=line.confidence<=1 for line in read.lines) for read in reads)):
            continue
        texts=[_verified_turn_word(_compact(''.join(line.text for line in read.lines))) for read in reads]
        # 力/カ is a glyph alias only inside the fixed 消費体力 compound.
        pattern=r'好調が[1-9][0-9]*ターン以上の場合、除外にある消費体[力カ]が[1-9][0-9]*以上のスキルカードを山札のランダムな位置に移動'
        if texts[0]==texts[1] and re.fullmatch(pattern,texts[0]):return True
    return False


def good_condition_requirements(character_id: str, stage: int,
                                rules: StageRules | None) -> tuple[GoodConditionRequirement, ...]:
    if (not character_id or type(stage) is not int or stage < 1 or rules is None
        or (rules.character_id, rules.stage) != (character_id, stage)
        or rules.source != 'visible_support_details'
        or rules.identity_source != 'opened_from_confirmed_formation'):
        return ()
    grouped: dict[int, list[GoodConditionRequirement]] = {}
    blocked: set[int] = set()
    for viewport in rules.viewports:
        for row in viewport.visible_rules or ():
            turn = confirmed_rule_turn_with_number(row.turn_reads, row.turn_number_evidence)
            if turn != row.turn or type(turn) is not int:
                continue
            analysis = read_good_condition_requirement(turn,row.description_reads,_row_lexical(row))
            if analysis.status == 'high_confidence_conflict':
                blocked.add(turn)
            value = analysis.requirement
            if value is None:
                if analysis.status == 'unsupported_condition' and any(
                    '好調' in line.text for read in row.description_reads for line in read.lines):
                    blocked.add(turn)
                continue
            if any(read.frame_id != value.evidence_frames[0] for read in row.turn_reads):
                blocked.add(turn)
                continue
            grouped.setdefault(turn, []).append(value)
    return tuple(GoodConditionRequirement(turn, values[0].minimum_turns,
                 tuple(dict.fromkeys(frame for value in values for frame in value.evidence_frames)))
                 for turn, values in sorted(grouped.items())
                 if turn not in blocked and len({value.minimum_turns for value in values}) == 1)


def good_condition_planning_requirements(character_id: str, stage: int,
                                         rules: StageRules | None) -> tuple[GoodConditionRequirement, ...]:
    """条件本身可能属于妨碍；只有独立确认的应援样式才生成准备目标。"""
    requirements = good_condition_requirements(character_id,stage,rules)
    if rules is None:
        return ()
    supported = set()
    for viewport in rules.viewports:
        for row in viewport.visible_rules or ():
            symbol = row.good_condition_support
            goal = read_good_condition_requirement(row.turn,row.description_reads,_row_lexical(row)).requirement
            if (goal is not None and _recycle_support(row)
                and confirmed_rule_turn_with_number(row.turn_reads,row.turn_number_evidence)==row.turn
                and all(read.frame_id in goal.evidence_frames for read in row.turn_reads)):
                supported.add((goal.start_turn,goal.minimum_turns,goal.evidence_frames[0]))
            if (symbol is None or goal is None or symbol.kind not in ('positive_turn_end_support','positive_debuff_recovery')
                or not .97 <= symbol.score <= 1 or min(symbol.rect) < 0 or min(symbol.rect[2:]) <= 0
                or symbol.frame_id not in goal.evidence_frames
                or confirmed_rule_turn_with_number(row.turn_reads,row.turn_number_evidence) != row.turn
                or any(read.frame_id != symbol.frame_id for read in row.turn_reads)):
                continue
            x,y,w,h = row.card_rect
            sx,sy,sw,sh = symbol.rect
            if min(x,y) >= 0 and min(w,h) > 0 and x <= sx and y <= sy and sx+sw <= x+w and sy+sh <= y+h:
                supported.add((goal.start_turn,goal.minimum_turns,symbol.frame_id))
    return tuple(goal for goal in requirements if any(
        (goal.start_turn,goal.minimum_turns,frame) in supported for frame in goal.evidence_frames))
