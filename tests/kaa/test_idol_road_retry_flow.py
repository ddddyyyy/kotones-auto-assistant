"""Retry scheduling and native replacement ownership; no device or OCR mocks."""
from dataclasses import replace

import pytest

from kaa.config.idol_road import IdolRoadConfig
from kaa.idol_road.controller import IdolRoadController
from kaa.idol_road.formation import FormationSlot, FormationSlots
from kaa.idol_road.memory_picker import MemoryCandidate, SubMemoryViewport
from kaa.idol_road.models import Action, Observation, Outcome, Scene, Stage
from tests.kaa.test_idol_road_memory_picker import confirm_reads


def formation(frame='before',powers=(10,20)):
    return FormationSlots(frame,tuple(FormationSlot('main' if i==0 else 'sub',
        (20+i*100,700,90,90),'occupied',p) for i,p in enumerate(powers)))


def obs(scene,**kwargs):
    return Observation(scene,character_id='hski',stage=1,**kwargs)


def confirm(**kwargs):
    return obs(Scene.CONFIRM,stars=0,formation_ready=True,formation_slots=formation(),**kwargs)


def new_flow(**kwargs):
    return IdolRoadController(IdolRoadConfig(characters=['hski'],mode='manual',
        retry_failed_stage=True,**kwargs))


def fail(flow):
    assert flow.step(confirm()).action==Action.START_BATTLE
    return flow.step(obs(Scene.RESULT,outcome=Outcome.FAILURE))


def return_map(flow,*,number=1,stars=0):
    return flow.step(obs(Scene.STAGES,focused_stage=Stage(number,True,stars)))


def exhaust_same_lineup(flow):
    for attempt in range(3):
        decision=fail(flow)
        assert decision.action==Action.DISMISS_RESULT
        assert flow.battles_started==attempt+1
        assert flow.retry_policy is not None
        assert flow.retry_policy.failures_on_lineup==attempt+1
        # A repeated rendering is not a new failure or another dismiss input.
        assert flow.step(obs(Scene.RESULT,outcome=Outcome.FAILURE)).action==Action.WAIT
        assert flow.retry_policy.failures_on_lineup==attempt+1
        assert return_map(flow).action==Action.SELECT_STAGE


def test_first_two_failures_keep_lineup_third_opens_owned_sub_replacement():
    flow=new_flow()
    exhaust_same_lineup(flow)
    decision=flow.step(confirm())
    assert decision.action==Action.REPLACE_SUB_MEMORY
    command=decision.memory_command
    assert command is not None and command.phase=='open' and command.slot_index==1
    assert flow.retry_policy is not None and flow.retry_policy.lineup_revision==0


def test_replacement_requires_new_native_highlight_receipt_and_changed_full_formation():
    flow=new_flow()
    exhaust_same_lineup(flow)
    command=flow.step(confirm()).memory_command
    assert command is not None
    old=MemoryCandidate('pick',(20,700,90,90),20,'a'*64,True,True)
    new=MemoryCandidate('pick',(120,700,90,90),40,'b'*64,False,False)
    view=SubMemoryViewport('pick',(old,new),True,confirm_rect=(300,1050,190,70),
        confirm_reads=confirm_reads('pick'),confirm_control_score=.99)
    def selector(view):
        return obs(Scene.SUB_MEMORY_SELECTOR,memory_selector=view,memory_replacement_token=command.token)
    pick=flow.step(selector(view))
    assert pick.memory_command is not None and pick.memory_command.phase=='pick'
    assert flow.step(selector(view)).action==Action.WAIT
    selected=replace(view,frame_id='chosen',confirm_reads=confirm_reads('chosen'),candidates=(replace(old,frame_id='chosen',highlighted=False),
        replace(new,frame_id='chosen',highlighted=True)))
    applied=flow.step(selector(selected))
    assert applied.memory_command is not None and applied.memory_command.phase=='apply'
    after=replace(confirm(),formation_slots=formation('after',(10,40)),memory_replacement_receipt=command.token)
    assert flow.step(after).action==Action.START_BATTLE
    policy=flow.retry_policy
    assert policy is not None and policy.lineup_revision==1 and policy.failures_on_lineup==0
    assert policy.changes_used==1


def test_failure_return_cannot_retarget_another_stage():
    flow=new_flow()
    fail(flow)
    assert return_map(flow,number=2).reason=='retry_stage_not_confirmed'


def test_confirmed_failure_can_cross_loading_overview_before_same_stage_retry():
    flow=new_flow()
    fail(flow)
    for _ in range(2):
        assert flow.step(obs(Scene.MAP_OVERVIEW)).reason=='await_retry_stage_focus'
    assert return_map(flow).action==Action.SELECT_STAGE
    assert flow.retry_policy.failures_on_lineup==1
    assert flow.battles_started==1


def test_retry_loading_overview_is_bounded_and_does_not_count_another_failure():
    flow=new_flow()
    fail(flow)
    for _ in range(2):flow.step(obs(Scene.MAP_OVERVIEW))
    assert flow.step(obs(Scene.MAP_OVERVIEW)).reason=='retry_stage_focus_missing'
    assert flow.retry_policy.failures_on_lineup==1


def test_loading_overview_cannot_bypass_battle_limit_or_retry_stage_identity():
    flow=new_flow(max_battles=1)
    fail(flow)
    flow.step(obs(Scene.MAP_OVERVIEW))
    assert return_map(flow).reason=='battle_limit'
    flow=new_flow()
    fail(flow)
    flow.step(obs(Scene.MAP_OVERVIEW))
    assert return_map(flow,number=2).reason=='retry_stage_not_confirmed'


def test_map_star_after_failed_banner_can_advance_without_changing_lineup():
    flow=new_flow()
    fail(flow)
    assert return_map(flow,stars=1).action==Action.NEXT_STAGE


def test_budget_exhaustion_stops_without_changing_after_first_failure():
    flow=new_flow(attempts_per_lineup=2,max_lineup_changes=0)
    assert fail(flow).action==Action.DISMISS_RESULT
    assert return_map(flow).action==Action.SELECT_STAGE
    assert fail(flow).reason=='stage_retry_budget_exhausted'


def test_global_battle_limit_applies_before_same_lineup_retry():
    flow=new_flow(max_battles=1)
    fail(flow)
    assert return_map(flow).reason=='battle_limit'


@pytest.mark.parametrize('outcome,stars,reason',[(Outcome.UNKNOWN,None,'result_unknown'),
    (Outcome.FAILURE,1,'result_stars_conflict')])
def test_ambiguous_or_conflicting_result_never_counts_failure(outcome,stars,reason):
    flow=new_flow()
    assert flow.step(confirm()).action==Action.START_BATTLE
    assert flow.step(obs(Scene.RESULT,outcome=outcome,stars=stars)).reason==reason
    assert flow.retry_policy is None


def test_unowned_picker_cannot_replace_lineup():
    flow=new_flow()
    assert flow.step(obs(Scene.SUB_MEMORY_SELECTOR)).reason=='unowned_memory_selector'


def test_wrong_receipt_does_not_reset_failed_routes_or_revision():
    flow=new_flow()
    exhaust_same_lineup(flow)
    command=flow.step(confirm()).memory_command
    assert command is not None
    decision=flow.step(replace(confirm(),formation_slots=formation('after',(10,40)),
        memory_replacement_receipt='foreign'))
    # The outstanding open transition must first leave the confirmation page.
    assert decision.action==Action.WAIT
    assert flow.retry_policy is not None and flow.retry_policy.lineup_revision==0


@pytest.mark.parametrize('values',[{'attempts_per_lineup':1},{'attempts_per_lineup':11},
    {'max_lineup_changes':-1},{'max_lineup_changes':11}])
def test_retry_config_rejects_unbounded_or_first_failure_change(values):
    with pytest.raises(ValueError):IdolRoadConfig(**values)
