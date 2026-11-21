"""Frame-local random Sub alternatives; no stable memory/deck identity inference."""
from collections import Counter
from dataclasses import dataclass
from random import SystemRandom
from typing import Callable,Literal
import math
import re
from .formation import FormationSlots
from .rules import RuleTextRead


@dataclass(frozen=True)
class MemoryCandidate:
    frame_id: str
    rect: tuple[int,int,int,int]
    power: int | None
    presentation_sha256: str
    assigned: bool | None
    highlighted: bool | None


@dataclass(frozen=True)
class SubMemoryViewport:
    frame_id: str
    candidates: tuple[MemoryCandidate,...]
    editing_group_confirmed: bool
    source: Literal['native_current_sub_memory_group'] = 'native_current_sub_memory_group'
    confirm_rect: tuple[int,int,int,int] | None = None
    confirm_reads: tuple[RuleTextRead,...] = ()
    confirm_control_score: float | None = None

    @property
    def sub_number(self) -> int | None:
        score=self.confirm_control_score
        if (not self.frame_id or self.confirm_rect is None or score is None or type(score) not in (int,float)
            or not math.isfinite(score) or not .97<=score<=1 or len(self.confirm_reads)!=2
            or {r.representation for r in self.confirm_reads}!={'original','grayscale'}):
            return None
        values=[]
        for read in self.confirm_reads:
            if read.frame_id!=self.frame_id or len(read.lines)!=1 or not .98<=read.lines[0].confidence<=1:
                return None
            match=re.fullmatch(r'サブ([1-9])決定',read.lines[0].text)
            if match is None:return None
            values.append(int(match[1]))
        return values[0] if values[0]==values[1] else None


def sub_alternatives(view: SubMemoryViewport,before: FormationSlots,slot_index: int) -> tuple[MemoryCandidate,...]:
    """A bounded visible alternative pool is sufficient; no full roster scan."""
    if (not before.full or type(slot_index) is not int or not 0<=slot_index<len(before.slots)
        or before.slots[slot_index].role!='sub' or before.slots[slot_index].state!='occupied'
        or any(type(s.power) is not int or s.power<=0 for s in before.slots if s.state=='occupied')
        or not view.frame_id or view.editing_group_confirmed is not True
        or view.source!='native_current_sub_memory_group' or view.sub_number!=before.sub_number(slot_index)):
        return ()
    old_power=before.slots[slot_index].power
    # The native assigned marker links this visible group to the edited slot.
    if type(old_power) is not int or not any(c.frame_id==view.frame_id and c.assigned is True and c.highlighted is True
                                            and c.power==old_power for c in view.candidates):
        return ()
    powers={slot.power for slot in before.slots if slot.state=='occupied'}
    candidates=[]
    for candidate in view.candidates:
        rect=candidate.rect
        if (candidate.frame_id!=view.frame_id or candidate.assigned is not False or candidate.highlighted is not False
            or type(candidate.power) is not int or candidate.power<=0 or candidate.power in powers
            or len(candidate.presentation_sha256)!=64 or any(c not in '0123456789abcdef' for c in candidate.presentation_sha256)
            or any(type(v) is not int for v in rect) or min(rect)<0 or min(rect[2:])<=0
            or rect[0]+rect[2]>720 or rect[1]+rect[3]>1280):
            continue
        candidates.append(candidate)
    if not candidates or len({c.rect for c in candidates})!=len(candidates):
        return ()
    return tuple(candidates)


def choose_sub_alternative(view: SubMemoryViewport,before: FormationSlots,slot_index: int,*,
                            choose_index: Callable[[int],int] | None=None) -> MemoryCandidate | None:
    candidates=sub_alternatives(view,before,slot_index)
    if not candidates:return None
    index=(choose_index or SystemRandom().randrange)(len(candidates))
    return candidates[index] if type(index) is int and 0<=index<len(candidates) else None


def replacement_matches_observed_powers(before: FormationSlots,after: FormationSlots,
                                         slot_index: int,selected: MemoryCandidate) -> bool:
    """Only a post-selection power/capacity check; caller must own the UI change."""
    if (not before.full or not after.full or before.capacity!=after.capacity
        or not before.frame_id or before.frame_id==after.frame_id or type(slot_index) is not int
        or not 0<=slot_index<len(before.slots) or before.slots[slot_index].role!='sub'
        or before.slots[slot_index].state!='occupied'
        or type(selected.power) is not int or selected.power<=0 or selected.power==before.slots[slot_index].power
        or any(type(s.power) is not int for slots in (before,after) for s in slots.slots if s.state=='occupied')):
        return False
    expected=Counter((s.role,s.power) for s in before.slots if s.state=='occupied')
    expected[('sub',before.slots[slot_index].power)]-=1
    expected[('sub',selected.power)]+=1
    actual=Counter((s.role,s.power) for s in after.slots if s.state=='occupied')
    return +expected==actual
