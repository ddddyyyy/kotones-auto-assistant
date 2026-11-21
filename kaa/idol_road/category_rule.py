"""完整可见的末手类别条件；只生成规划目标，不发放场地收益。"""
from dataclasses import dataclass, replace
from typing import Literal
import re
import unicodedata

from .rules import RuleTextRead, StageRules, confirmed_rule_turn_with_number
from .rule_symbols import RuleSymbolEvidence


@dataclass(frozen=True)
class CategoryObjective:
    trigger_turn: int
    category: str
    reward_count: int | None
    evidence_frames: tuple[str, ...]
    timing_source: str = 'user_confirmed_previous_turn_last_card'
    reward_kind: Literal['anti_debuff','debuff_recovery','subsequent_start_draw'] = 'anti_debuff'

    @property
    def deadline_turn(self) -> int:
        return self.trigger_turn - 1


@dataclass(frozen=True)
class CategoryConditionEvidence:
    frame_id: str
    body_rect: tuple[int,int,int,int]
    icon: RuleSymbolEvidence
    reads: tuple[RuleTextRead,...]


def _complete_category(turn: int | None, reads: tuple[RuleTextRead, ...]) -> CategoryObjective | None:
    if (type(turn) is not int or turn < 2 or len(reads) != 2 or not reads[0].frame_id
        or reads[0].frame_id != reads[1].frame_id
        or {read.representation for read in reads} != {'original', 'grayscale'}):
        return None
    texts = []
    for read in reads:
        if not read.lines or any(not .98 <= line.confidence <= 1 for line in read.lines):
            return None
        texts.append(re.sub(r'\s+', '', unicodedata.normalize('NFKC', ''.join(line.text for line in read.lines))))
    if texts[0] != texts[1]:
        return None
    return _category_text(turn,texts[0],reads[0].frame_id)


def _category_text(turn: int,text: str,frame_id: str) -> CategoryObjective | None:
    match = re.fullmatch(r'直前に(メンタル|アクティブ)スキルカードを使用していた場合、(.+)',text)
    if match is None:
        return None
    category = 'ProduceCardCategory_MentalSkill' if match[1] == 'メンタル' else 'ProduceCardCategory_ActiveSkill'
    reward=match[2]
    immunity=re.fullmatch(r'低下状態無効([12])回',reward)
    if immunity is not None:
        return CategoryObjective(turn,category,int(immunity[1]),(frame_id,))
    if reward=='低下状態回復1':
        return CategoryObjective(turn,category,1,(frame_id,),reward_kind='debuff_recovery')
    if reward=='以降、ターン開始後、スキルカードを引く':
        # The displayed clause has no count. Neither identities nor future
        # draw count are known; this only plans its previous-card condition.
        return CategoryObjective(turn,category,None,(frame_id,),reward_kind='subsequent_start_draw')
    return None


def read_category_condition(turn: int | None,reads: tuple[RuleTextRead,...],
                            evidence: CategoryConditionEvidence | None=None) -> CategoryObjective | None:
    native=_complete_category(turn,reads)
    if evidence is None:return native
    icon=evidence.icon;x,y,w,h=icon.rect;bx,by,bw,bh=evidence.body_rect
    if (not evidence.frame_id or any(read.frame_id!=evidence.frame_id for read in (*reads,*evidence.reads))
        or icon.frame_id!=evidence.frame_id or icon.kind not in ('mental','active')
        or not .97<=icon.score<=1 or min(x,y)<0 or min(w,h,bw,bh)<=0
        or min(bx,by)<0 or bx+bw>720 or by+bh>1280 or x+w>bw or y+h>bh):
        return native
    candidate=_complete_category(turn,evidence.reads)
    expected='ProduceCardCategory_MentalSkill' if icon.kind=='mental' else 'ProduceCardCategory_ActiveSkill'
    if candidate is None or candidate.category!=expected:return None
    if native is not None and (native.category,native.reward_kind,native.reward_count)!=(
        candidate.category,candidate.reward_kind,candidate.reward_count):return None
    # A complete high-confidence original body is authoritative. An icon
    # may contribute its literal M/A glyph to OCR; remove only that exact
    # prefix, backed by this capture's verified icon and matching category.
    for read in reads:
        if not read.lines or any(not .98<=line.confidence<=1 for line in read.lines):continue
        letter,word=('M','メンタル') if icon.kind=='mental' else ('A','アクティブ')
        normalized=re.sub(r'\s+','',unicodedata.normalize('NFKC',''.join(line.text for line in read.lines)))
        normalized=re.sub(rf'^直前に{letter}(?={word})','直前に',normalized)
        assert type(turn) is int  # candidate came from the bounded complete parser
        original=_category_text(turn,normalized,read.frame_id)
        if original is None or (original.category,original.reward_kind,original.reward_count)!=(
            candidate.category,candidate.reward_kind,candidate.reward_count):return None
    return candidate


def category_objectives(character_id: str, stage: int, rules: StageRules | None) -> tuple[CategoryObjective, ...]:
    if (rules is None or (rules.character_id, rules.stage) != (character_id, stage)
        or rules.source != 'visible_support_details'
        or rules.identity_source != 'opened_from_confirmed_formation'):
        return ()
    by_turn: dict[int, list[CategoryObjective]] = {}
    conflicts: set[int] = set()
    for viewport in rules.viewports:
        for row in viewport.visible_rules or ():
            turn = confirmed_rule_turn_with_number(row.turn_reads, row.turn_number_evidence)
            if turn != row.turn or type(turn) is not int:
                continue
            cue=row.category_condition
            if cue is not None:
                x,y,w,h=row.card_rect;bx,by,bw,bh=cue.body_rect
                if not (0<=x<=bx and 0<=y<=by and bx+bw<=x+w and by+bh<=y+h):
                    cue=None
            value = read_category_condition(turn,row.description_reads,cue)
            if value is not None:
                # Title and body must belong to the same captured viewport.
                if any(read.frame_id != value.evidence_frames[0] for read in row.turn_reads):
                    conflicts.add(turn)
                else:
                    by_turn.setdefault(turn, []).append(value)
            elif (any('直前' in line.text for read in row.description_reads for line in read.lines)
                  and len(row.description_reads) == 2
                  and all(read.lines and all(.98 <= line.confidence <= 1 for line in read.lines)
                          for read in row.description_reads)):
                conflicts.add(turn)
    result = []
    for turn, values in sorted(by_turn.items()):
        if turn in conflicts or len({(value.category,value.reward_kind,value.reward_count) for value in values}) != 1:
            continue
        value = values[0]
        result.append(replace(value,evidence_frames=tuple(dict.fromkeys(
            frame for item in values for frame in item.evidence_frames))))
    return tuple(result)
