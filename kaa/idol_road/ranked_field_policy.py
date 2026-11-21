"""Rank the current field-policy candidates with the shared evaluator only."""
from dataclasses import dataclass, replace
import math
from kaa.db.skill_card import SkillCard
from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluation, CardEvaluator, CompositeEvaluator
from kaa.tasks.produce.new.play_cards.planner.feedback import PendingPlay, PlayFeedback
from kaa.tasks.produce.new.play_cards.planner.selection_policy import SelectionPolicy, SelectionConstraint
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from .retry_route import RetryRoutePreference
from .score_suppression import CurrentScoreSuppression,SuppressionAwareEvaluator,score_suppression_active,score_only_card


@dataclass(frozen=True)
class RankedFieldTrace:
    frame_id: str | None
    field_constraint: SelectionConstraint
    evaluations: tuple[tuple[int, CardEvaluation], ...]
    selected_indices: tuple[int, ...]
    reason: str
    score_suppression: CurrentScoreSuppression | None=None


@dataclass(frozen=True)
class CurrentCardPreview:
    frame_id: str
    card_id: str
    upgrade: int | None
    per_hit: int
    count: int | None


class RankedFieldPolicy:
    """Delegate inside InputBoundPolicy's HP-filtered available-card subset.

    Field preferences remain hard boundaries. Evaluation orders candidates;
    its projected output does not update observations or grant future buffs.
    """
    def __init__(self, field: SelectionPolicy, *, evaluator: CardEvaluator | None = None,
                 retry_route: RetryRoutePreference | None=None):
        self.field = field
        self.evaluator = evaluator or CompositeEvaluator()
        self.trace: RankedFieldTrace | None = None
        self.retry_route=retry_route
        self.current_previews: tuple[CurrentCardPreview,...]=()

    def _stop(self, reason: str) -> SelectionConstraint:
        if self.trace is not None:
            self.trace = replace(self.trace, reason=reason)
        return SelectionConstraint(reason=reason, allowed_indices=(), stop=True)

    def select(self, state: BattleState, cards: tuple[SkillCard, ...]) -> SelectionConstraint:
        constraint = self.field.select(state, cards)
        frame_id = state.absolute_turn_observation.frame_id if state.absolute_turn_observation is not None else None
        suppression=self.evaluator.current if isinstance(self.evaluator,SuppressionAwareEvaluator) else None
        active=score_suppression_active(suppression,state)
        self.trace = RankedFieldTrace(frame_id,constraint,(),(),constraint.reason,suppression if active else None)
        if constraint.stop or constraint.end_turn:
            return constraint
        allowed = constraint.allowed_indices if constraint.allowed_indices is not None else tuple(range(len(cards)))
        if (not allowed or any(type(index) is not int or not 0 <= index < len(cards)
            for index in (*allowed, *constraint.preferred_indices))
            or any(index not in allowed for index in constraint.preferred_indices)):
            return self._stop('field_rank:invalid_candidate_mapping')
        if active and not constraint.preferred_indices:
            allowed=tuple(index for index in allowed if not score_only_card(cards[index]))
            if not allowed:
                reason='score_suppression:preserve_pure_scoring_cards,no_future_draw_promised'
                self.trace=replace(self.trace,reason=reason)
                return SelectionConstraint(end_turn=True,allowed_indices=(),reason=reason)
        candidates = tuple(dict.fromkeys(constraint.preferred_indices or allowed))
        evaluations = []
        for index in candidates:
            card=cards[index]
            previews=[p for p in self.current_previews if p.frame_id==frame_id
                      and (p.card_id,p.upgrade)==(card._id,card.upgrade_count)]
            candidate_state=state
            if (frame_id and len(previews)==1 and sum(c._id==card._id for c in cards)==1
                and type(previews[0].per_hit) is int and 0<=previews[0].per_hit<=999999
                and (previews[0].count is None or type(previews[0].count) is int and previews[0].count==1)):
                preview=previews[0]
                candidate_state=replace(state,observed_card_output=(card._id,preview.per_hit,preview.count),
                    observed_card_output_is_lower_bound=True)
            evaluation = self.evaluator.evaluate(card, candidate_state)
            if (evaluation is not None and evaluation.card_id == cards[index]._id
                and all(math.isfinite(value) for value in
                        (evaluation.score, evaluation.projected_output, evaluation.confirmed_output))
                and evaluation.projected_output >= 0 and evaluation.confirmed_output >= 0):
                evaluations.append((index, evaluation))
        if not evaluations:
            return self._stop('field_rank:no_supported_evaluation; ' + constraint.reason)
        ranked=evaluations
        if self.retry_route is not None:
            try:
                counts={index:self.retry_route.failed_count(cards[index],state) for index,_ in evaluations}
            except ValueError:
                return self._stop('field_rank:retry_route_frame_not_current')
            least=min(counts.values())
            ranked=[(index,value) for index,value in evaluations if counts[index]==least]
        best = max(evaluation.score for _, evaluation in ranked)
        selected = tuple(index for index, evaluation in ranked if evaluation.score == best)
        reason = constraint.reason + f'; field_rank:estimated_score={best},observation_unchanged'
        self.trace = RankedFieldTrace(frame_id, constraint, tuple(evaluations), selected, reason,
            suppression if active else None)
        return replace(constraint, preferred_indices=selected,
            allowed_indices=tuple(index for index, _ in evaluations), reason=reason)

    def committed(self, state: BattleState, card: SkillCard) -> None:
        self.field.committed(state, card)

    def feedback(self, pending: PendingPlay, feedback: PlayFeedback, state: BattleState) -> None:
        self.field.feedback(pending, feedback, state)
