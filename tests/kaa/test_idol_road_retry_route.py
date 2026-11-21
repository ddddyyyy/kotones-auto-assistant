from types import SimpleNamespace
from kaa.idol_road.battle_inputs import BattleInputs
from kaa.idol_road.retry_policy import StageRetryPolicy
from kaa.idol_road.retry_route import RetryRoutePreference
from kaa.idol_road.ranked_field_policy import RankedFieldPolicy
from kaa.tasks.produce.new.play_cards.planner.selection_policy import SelectionConstraint
from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluation,EvaluationConfidence
from tests.kaa.test_idol_road_battle_inputs import SESSION,frame,state


def cards():
    return tuple(SimpleNamespace(_id=name) for name in ('old','alternative','forbidden'))


class Scores:
    def evaluate(self,card,state):
        return CardEvaluation(card._id,{'old':100,'alternative':1,'forbidden':1000}[card._id],
            EvaluationConfidence.MEDIUM,'estimate')


def ranked(constraint=SelectionConstraint()):
    retry=StageRetryPolicy()
    retry.failed(1,request_route=('16:card:old',))
    inputs=BattleInputs(SESSION)
    preference=RetryRoutePreference(retry,inputs,frame())
    field=SimpleNamespace(select=lambda *args:constraint)
    return RankedFieldPolicy(field,evaluator=Scores(),retry_route=preference),inputs


def test_second_attempt_varies_failed_continuation_within_allowed_candidates():
    policy,_=ranked(SelectionConstraint(allowed_indices=(0,1)))
    assert policy.select(state(),cards()).preferred_indices==(1,)


def test_retry_does_not_override_required_field_category():
    policy,_=ranked(SelectionConstraint(preferred_indices=(0,),allowed_indices=(0,1)))
    assert policy.select(state(),cards()).preferred_indices==(0,)


def test_no_failure_hint_uses_shared_score_instead_of_randomly_weakening_every_play():
    policy,_=ranked(SelectionConstraint(allowed_indices=(1,2)))
    assert policy.select(state(),cards()).preferred_indices==(2,)


def test_pending_or_different_frame_cannot_reuse_retry_ranking():
    policy,inputs=ranked()
    inputs.begin('skip',frame())
    result=policy.select(state(),cards())
    assert result.stop and result.reason=='field_rank:retry_route_frame_not_current'


def test_unconfirmed_request_is_retained_as_request_only_and_new_battle_has_empty_route():
    inputs=BattleInputs(SESSION)
    inputs.begin('card',frame(),'old')
    assert inputs.requested_route==('16:card:old',)
    assert inputs.last_owned_card_play is None
    assert BattleInputs(SESSION).requested_route==()
