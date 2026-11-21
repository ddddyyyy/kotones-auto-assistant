"""Optional task-owned constraints; ordinary lessons keep their existing ranking."""
from dataclasses import dataclass
from typing import Protocol

from kaa.db.skill_card import SkillCard
from .feedback import PendingPlay, PlayFeedback
from .state import BattleState


@dataclass(frozen=True)
class SelectionConstraint:
    preferred_indices: tuple[int, ...] = ()
    end_turn: bool = False
    reason: str = ''
    allowed_indices: tuple[int,...] | None = None
    stop: bool = False


class SelectionPolicyStop(ValueError):
    """A task-owned hard stop; callers must not replace it with recommended input."""


class SelectionPolicy(Protocol):
    def select(self, state: BattleState, cards: tuple[SkillCard, ...]) -> SelectionConstraint:
        """Cards are recognized, currently available slots from this snapshot.

        Model-proven unaffordable cards have already been removed. Unknown
        costs remain unknown; this input contract authorizes only the first
        visible play, never a simulated later play or numeric cost.

        Preferences rank inside allowed_indices. None is unrestricted; an
        empty allowed set or stop=True must not become recommendation fallback.
        """
        ...

    def committed(self, state: BattleState, card: SkillCard) -> None: ...

    def feedback(self, pending: PendingPlay, feedback: PlayFeedback, state: BattleState) -> None: ...
