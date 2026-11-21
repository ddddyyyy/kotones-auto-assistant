from dataclasses import replace
import pytest
from kaa.idol_road.retry_policy import StageRetryPolicy


def test_repeated_attempts_keep_lineup_and_routes_before_requesting_any_change():
    policy=StageRetryPolicy()
    first=policy.failed(1,request_route=('M:1','A:2'))
    assert first.action=='retry_same' and policy.lineup_revision==0 and policy.attempt_variant==1
    assert policy.failed(1,request_route=('ignored',))==first
    assert policy.failures_on_lineup==1
    assert policy.failed(2,request_route=('M:1','M:2')).action=='retry_same'
    change=policy.failed(3,request_route=('A:1','A:2'))
    assert change.action=='change_lineup' and policy.lineup_revision==0
    assert policy.failed_continuations((),'M:1')==2
    assert policy.failed_continuations(('M:1',),'M:2')==1
    with pytest.raises(ValueError):policy.failed(4)
    with pytest.raises(ValueError):policy.confirm_lineup_changed(replace(change,lineup_revision=9))
    assert policy.lineup_revision==0
    policy.confirm_lineup_changed(change)
    assert policy.lineup_revision==1 and policy.changes_used==1 and policy.attempt_variant==0
    assert policy.failed_request_routes==()
    with pytest.raises(ValueError):policy.confirm_lineup_changed(change)
    assert policy.failed(4).action=='retry_same'


def test_stage_budget_is_finite_across_lineup_changes():
    policy=StageRetryPolicy(attempts_per_lineup=3,max_lineup_changes=2)
    actions=[]
    for attempt in range(1,10):
        decision=policy.failed(attempt)
        actions.append(decision.action)
        if decision.action=='change_lineup':policy.confirm_lineup_changed(decision)
    assert actions==['retry_same','retry_same','change_lineup']*2+['retry_same','retry_same','stop']
    assert policy.changes_used==2 and policy.failed(10).action=='stop'


def test_disabling_changes_still_allows_multiple_attempts_on_the_existing_lineup():
    policy=StageRetryPolicy(max_lineup_changes=0)
    assert [policy.failed(i).action for i in (1,2,3)]==['retry_same','retry_same','stop']


@pytest.mark.parametrize('attempts,changes',[(1,2),(0,2),(True,2),(11,2),(3,-1),(3,True),(3,11)])
def test_invalid_limits_cannot_change_on_first_failure_or_loop_without_bounds(attempts,changes):
    with pytest.raises(ValueError):StageRetryPolicy(attempts,changes)


def test_stale_or_invalid_attempts_and_routes_do_not_advance_the_counter():
    policy=StageRetryPolicy()
    for identifier in (0,True):
        with pytest.raises(ValueError):policy.failed(identifier)
    with pytest.raises(ValueError):policy.failed(1,request_route=('',))
    assert policy.failures_on_lineup==0
    policy.failed(2)
    with pytest.raises(ValueError):policy.failed(1)
    assert policy.failures_on_lineup==1
