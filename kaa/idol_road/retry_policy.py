"""Bounded same-lineup attempts before an owned lineup replacement."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RetryDecision:
    action: Literal['retry_same','change_lineup','stop']
    lineup_revision: int
    failures_on_lineup: int
    changes_used: int
    reason: str


class StageRetryPolicy:
    """Task-local revision, never a global memory/deck identity or image hash."""
    def __init__(self,attempts_per_lineup: int=3,max_lineup_changes: int=2):
        if (type(attempts_per_lineup) is not int or not 2<=attempts_per_lineup<=10
            or type(max_lineup_changes) is not int or not 0<=max_lineup_changes<=10):
            raise ValueError('Invalid bounded lineup retry limits')
        self.attempts_per_lineup=attempts_per_lineup
        self.max_lineup_changes=max_lineup_changes
        self.lineup_revision=0
        self.failures_on_lineup=0
        self.changes_used=0
        self._handled: dict[int,RetryDecision]={}
        self._last_attempt=0
        self._pending_change: RetryDecision | None=None
        self._failed_routes: list[tuple[str,...]]=[]
        self._terminal: RetryDecision | None=None

    @property
    def attempt_variant(self) -> int:
        return self.failures_on_lineup

    @property
    def failed_request_routes(self) -> tuple[tuple[str,...],...]:
        return tuple(self._failed_routes)

    def failed(self,attempt_id: int,*,request_route: tuple[str,...]=()) -> RetryDecision:
        """Caller supplies only its own confirmed zero-star failed result."""
        if type(attempt_id) is not int or attempt_id<1:
            raise ValueError('Failure needs an owned attempt identifier')
        if attempt_id in self._handled:
            return self._handled[attempt_id]
        if attempt_id<=self._last_attempt or self._pending_change is not None:
            raise ValueError('Old attempt or lineup replacement still pending')
        if self._terminal is not None:
            return self._terminal
        if any(not isinstance(step,str) or not step for step in request_route):
            raise ValueError('Invalid requested route')
        self._last_attempt=attempt_id
        self.failures_on_lineup+=1
        if request_route:
            self._failed_routes.append(request_route)
        if self.failures_on_lineup<self.attempts_per_lineup:
            action,reason='retry_same','try_another_route_on_same_lineup'
        elif self.changes_used<self.max_lineup_changes:
            action,reason='change_lineup','same_lineup_attempts_exhausted'
        else:
            action,reason='stop','stage_retry_budget_exhausted'
        decision=RetryDecision(action,self.lineup_revision,self.failures_on_lineup,self.changes_used,reason)
        self._handled[attempt_id]=decision
        if action=='change_lineup':self._pending_change=decision
        elif action=='stop':self._terminal=decision
        return decision

    def confirm_lineup_changed(self,request: RetryDecision) -> None:
        """Call only after a different native selection and full lineup are verified."""
        if self._pending_change is None or request!=self._pending_change:
            raise ValueError('No matching owned lineup change request')
        self.lineup_revision+=1
        self.changes_used+=1
        self.failures_on_lineup=0
        self._failed_routes.clear()
        self._pending_change=None

    def failed_continuations(self,prefix: tuple[str,...],candidate: str) -> int:
        """A requested-route hint, not evidence of card effects or play legality."""
        following=(*prefix,candidate)
        return sum(route[:len(following)]==following for route in self._failed_routes)
