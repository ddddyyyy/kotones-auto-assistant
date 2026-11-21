"""本层当前可见编组槽位，不借用前一层容量或将图像指纹当稳定牌库身份。"""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FormationSlot:
    role: Literal['main','sub']
    rect: tuple[int,int,int,int]
    state: Literal['occupied','empty','absent','unknown']
    power: int | None = None
    image_sha256: str | None = None


@dataclass(frozen=True)
class FormationSlots:
    frame_id: str
    slots: tuple[FormationSlot,...]
    source: Literal['current_visible_formation'] = 'current_visible_formation'
    stable_lineup_identity: Literal[False] = False

    def sub_number(self,slot_index: int) -> int | None:
        if (not self.full or type(slot_index) is not int or not 0<=slot_index<len(self.slots)
            or self.slots[slot_index].role!='sub' or self.slots[slot_index].state!='occupied'):
            return None
        return sum(slot.role=='sub' and slot.state=='occupied' for slot in self.slots[:slot_index+1])

    @property
    def known(self) -> bool:
        return bool(self.frame_id and self.source=='current_visible_formation' and self.slots
                    and self.slots[0].role=='main' and self.slots[0].state!='absent'
                    and sum(slot.role=='main' for slot in self.slots)==1
                    and len({slot.rect for slot in self.slots})==len(self.slots)
                    and all(slot.state in ('occupied','empty','absent') and min(slot.rect)>=0
                            and min(slot.rect[2:])>0 for slot in self.slots))

    @property
    def capacity(self) -> int | None:
        return sum(slot.state in ('empty','occupied') for slot in self.slots) if self.known else None

    @property
    def empty_count(self) -> int:
        return sum(slot.state=='empty' for slot in self.slots)

    @property
    def full(self) -> bool:
        return self.known and self.empty_count==0 and all(slot.state=='occupied' for slot in self.slots if slot.role=='main')
