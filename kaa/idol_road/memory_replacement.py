"""Owned, bounded native Sub replacement, independent of device input."""
from dataclasses import dataclass
from random import SystemRandom
from typing import Literal
from uuid import uuid4

from .formation import FormationSlots
from .memory_picker import MemoryCandidate, SubMemoryViewport, choose_sub_alternative, replacement_matches_observed_powers


@dataclass(frozen=True)
class MemoryCommand:
    token: str
    phase: Literal['open','pick','apply']
    before: FormationSlots
    slot_index: int
    frame_id: str
    candidate: MemoryCandidate | None = None


def selected_candidate(view: SubMemoryViewport, candidate: MemoryCandidate) -> bool:
    """Native current highlight, not the old assigned badge, confirms a pending choice."""
    highlighted=[c for c in view.candidates if c.highlighted is True]
    return bool(view.editing_group_confirmed and view.confirm_rect is not None
                and len(highlighted)==1 and all(c.highlighted is not None for c in view.candidates)
                and highlighted[0].frame_id==view.frame_id
                and (highlighted[0].rect,highlighted[0].power,highlighted[0].presentation_sha256)
                ==(candidate.rect,candidate.power,candidate.presentation_sha256))


class MemoryReplacement:
    def __init__(self,before: FormationSlots,*,slot_index: int | None=None):
        indices=[i for i,s in enumerate(before.slots) if s.role=='sub' and s.state=='occupied']
        if not before.full or not indices:
            raise ValueError('No complete native Sub formation')
        if slot_index is not None and (type(slot_index) is not int or slot_index not in indices):
            raise ValueError('Invalid native Sub slot')
        self.before=before
        self.slot_index=SystemRandom().choice(indices) if slot_index is None else slot_index
        self.token=uuid4().hex
        self.phase: Literal['open','pick','apply','verify','done']='open'
        self.candidate: MemoryCandidate | None=None
        self.waits=0

    def open(self) -> MemoryCommand:
        if self.phase!='open':raise ValueError('Replacement already opened')
        self.phase='pick'
        return MemoryCommand(self.token,'open',self.before,self.slot_index,self.before.frame_id)

    def observe_selector(self,view: SubMemoryViewport,token: str | None) -> MemoryCommand | None:
        if token!=self.token or self.phase not in ('pick','apply','verify'):
            raise ValueError('Unowned native memory selector')
        self.waits+=1
        if self.waits>3:raise ValueError('Native selection not confirmed')
        if view.sub_number is None:return None
        if view.sub_number!=self.before.sub_number(self.slot_index):raise ValueError('Wrong native Sub number')
        if self.phase=='verify':return None
        if self.phase=='pick':
            self.candidate=choose_sub_alternative(view,self.before,self.slot_index)
            if self.candidate is None:raise ValueError('No confirmed visible Sub alternative')
            self.phase='apply'
            self.waits=0
            return MemoryCommand(self.token,'pick',self.before,self.slot_index,view.frame_id,self.candidate)
        candidate=self.candidate
        if candidate is None:raise ValueError('Missing owned selection')
        if view.frame_id==candidate.frame_id or not selected_candidate(view,candidate):return None
        self.phase='verify'
        self.waits=0
        return MemoryCommand(self.token,'apply',self.before,self.slot_index,view.frame_id,candidate)

    def verify(self,after: FormationSlots | None,receipt: str | None) -> bool:
        if self.phase!='verify' or receipt!=self.token or self.candidate is None:
            raise ValueError('No matching native confirmation receipt')
        self.waits+=1
        if after is not None and replacement_matches_observed_powers(self.before,after,self.slot_index,self.candidate):
            self.phase='done'
            return True
        if self.waits>=3:raise ValueError('Changed formation not confirmed')
        return False
