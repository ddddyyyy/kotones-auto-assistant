"""Mode-neutral adapter contract; importing it never registers tasks."""
from dataclasses import dataclass


@dataclass(frozen=True)
class AbsoluteTurnObservation:
    battle_id: str
    frame_id: str
    turn: int
    remaining_turns: int

    def validate(self) -> None:
        if (not self.battle_id or not self.frame_id or type(self.turn) is not int
            or not 1<=self.turn<=99 or type(self.remaining_turns) is not int
            or not 1<=self.remaining_turns<=30):
            raise ValueError('Absolute turn observation lacks a current battle/frame or valid numbers')
