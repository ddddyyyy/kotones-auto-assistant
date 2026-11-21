from dataclasses import replace
import pytest
from kaa.idol_road.formation import FormationSlot,FormationSlots
from kaa.idol_road.models import Observation,Scene,Action,Outcome,Stage
from kaa.config.idol_road import IdolRoadConfig
from kaa.idol_road.controller import IdolRoadController
from kaa.idol_road.battle_observation import create_battle_session
from kaa.idol_road.planner_trial import prepare_planner_trial

def slots(capacity=4,missing=True):
    return FormationSlots('current',(FormationSlot('main',(0,0,10,10),'occupied',100,'main'),)+tuple(
        FormationSlot('sub',(20*i,20,10,10),'empty' if missing and i==capacity-1 else 'occupied',
                      None if missing and i==capacity-1 else 90) for i in range(1,capacity)))

def formation(value=None,**changes):
    return replace(Observation(Scene.CONFIRM,'ttmr',19,stars=0,formation_ready=True,total_turns=20,
                               formation_slots=value or slots(),auto_formation_available=True),**changes)

def flow(**changes):
    return IdolRoadController(IdolRoadConfig(characters=['ttmr'],mode='manual',**changes))

def dialog():
    return Observation(Scene.AUTO_FORMATION,'ttmr',19,auto_formation_confirm_available=True)

def test_dynamic_capacity_and_absent_cells_are_separate_from_empty_slots():
    assert slots(3,False).capacity==3 and slots(3,False).full
    assert slots(4).capacity==4 and slots(4).empty_count==1 and not slots(4).full
    extended=replace(slots(),slots=slots().slots+(FormationSlot('sub',(90,90,10,10),'absent'),))
    assert extended.capacity==4 and extended.empty_count==1
    assert not replace(slots(),source='history').known
    assert not replace(slots(),slots=(slots().slots[0],)*2).known

def test_missing_slot_auto_arranges_once_then_rechecks_current_capacity():
    current=flow()
    assert current.step(formation()).action==Action.OPEN_AUTO_FORMATION
    assert current.step(dialog()).action==Action.APPLY_AUTO_FORMATION
    assert current.step(dialog()).action==Action.WAIT
    assert current.step(formation(slots(4,False))).action==Action.START_BATTLE
    assert current.battles_started==0

def test_preview_reports_missing_slots_without_mutating_formation():
    current=IdolRoadController(IdolRoadConfig(characters=['ttmr'],mode='preview'))
    decision=current.step(formation())
    assert decision.action==Action.READY and decision.reason=='preview_missing_formation_slots'

def test_full_formation_and_different_stage_capacity_do_not_trigger_auto():
    for count in (3,4,5):
        current=flow()
        assert current.step(formation(slots(count,False))).action==Action.START_BATTLE

def test_missing_controls_stop_without_auto_or_challenge():
    assert flow().step(formation(auto_formation_available=False)).action==Action.STOP


def test_unknown_slots_wait_without_auto_or_challenge_then_stop_if_persistent():
    current=flow()
    unknown=formation(formation_slots=FormationSlots('frame',()))
    for _ in range(3):
        assert current.step(unknown).action==Action.WAIT
    assert current.step(unknown).reason=='formation_slots_unknown'


def test_transient_slot_loading_can_resolve_to_full_or_explicit_empty():
    for missing,action in ((False,Action.START_BATTLE),(True,Action.OPEN_AUTO_FORMATION)):
        current=flow()
        assert current.step(formation(formation_slots=FormationSlots('frame',()))).action==Action.WAIT
        assert current.step(formation(slots(4,missing))).action==action

def test_disabled_auto_stops_on_missing_slot():
    assert flow(auto_fill_missing_formation=False).step(formation()).reason=='formation_missing_slots'

def test_orphan_auto_dialog_is_cancelled_not_accepted():
    assert flow().step(dialog()).action==Action.CANCEL_AUTO_FORMATION

def test_same_capacity_must_be_filled_after_auto_no_repeat_loop():
    current=flow()
    current.step(formation())
    current.step(dialog())
    assert current.step(formation(slots(3,False))).action==Action.WAIT
    assert current.step(formation(slots(3,False))).action==Action.WAIT
    assert current.step(formation(slots(3,False))).action==Action.STOP

def test_auto_invalidates_old_preparation_even_when_role_and_stage_unchanged():
    current=flow()
    current.step(formation())
    key=('ttmr',19)
    for values in (current._conditions_requested,current._items_requested,current._support_requested):
        values.add(key)
    current._objectives[key]=None
    current._inventories[key]=None
    current._rules[key]=None
    current.step(dialog())
    assert key not in current._objectives and key not in current._inventories and key not in current._rules
    assert not current._conditions_requested and not current._items_requested and not current._support_requested

def test_missing_slots_are_rejected_by_battle_entry_points():
    with pytest.raises(ValueError):
        create_battle_session(formation())
    with pytest.raises(ValueError):
        prepare_planner_trial(formation(),None)

def test_cleared_stages_never_auto_arrange():
    for stars in (1,2,3):
        assert flow().step(formation(stars=stars)).action==Action.BACK_TO_STAGES


def test_auto_result_transient_formation_cannot_start_before_dialog_is_closed():
    current=flow()
    current.step(formation())
    current.step(dialog())
    assert current.step(formation(slots(4,False),auto_formation_result_pending=True)).action==Action.WAIT
    result=Observation(Scene.AUTO_FORMATION_RESULT,'ttmr',19)
    assert current.step(result).action==Action.CLOSE_AUTO_FORMATION_RESULT
    assert current.step(formation(slots(4,False))).action==Action.START_BATTLE


def test_unsupported_recommendation_does_not_silently_reuse_old_concentration_trial():
    with pytest.raises(ValueError,match='推荐效果'):
        prepare_planner_trial(formation(slots(4,False),formation_recommended_effect='review'),None)


def test_next_stage_with_different_capacity_gets_its_own_auto_attempt():
    current=flow()
    assert current.step(formation()).action==Action.OPEN_AUTO_FORMATION
    assert current.step(dialog()).action==Action.APPLY_AUTO_FORMATION
    assert current.step(formation(slots(4,False))).action==Action.START_BATTLE
    assert current.step(Observation(Scene.BATTLE,'ttmr',19)).action==Action.WAIT
    assert current.step(Observation(Scene.RESULT,'ttmr',19,outcome=Outcome.SUCCESS,stars=1)).action==Action.DISMISS_RESULT
    assert current.step(Observation(Scene.STAGES,'ttmr',focused_stage=Stage(19,True,1))).action==Action.NEXT_STAGE
    assert current.step(Observation(Scene.STAGES,'ttmr',focused_stage=Stage(20,True,0))).action==Action.SELECT_STAGE
    assert current.step(formation(slots(3),stage=20)).action==Action.OPEN_AUTO_FORMATION
    assert current.step(replace(dialog(),stage=20)).action==Action.APPLY_AUTO_FORMATION
