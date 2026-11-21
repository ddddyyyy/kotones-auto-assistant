from dataclasses import replace
import pytest
from tests.kaa.test_idol_road_maximum_hp import calibrated,frame
from kaa.idol_road.current_skip import assess_current_skip
from kaa.idol_road.hp_preparation import HpPreparationBudget,RecurringHpCost
from kaa.idol_road.skip_recovery import SkipRecoveryEvidence,confirm_skip_recovery
from kaa.idol_road.rules import RuleTextRead,RuleTextLine
from kaa.tasks.produce.new.play_cards.planner.state import BattleState


def scenario(hp=50,remaining=20,recovery=2):
    owner,inputs,tracker=calibrated()
    raw=frame(owner,'current',hp=hp)
    raw=replace(raw,values=replace(raw.values,remaining_turns=remaining),
                observation=replace(raw.observation,turn=21-remaining))
    current=tracker.bind(raw,inputs.snapshot())
    state=BattleState(remaining,hp,0,max_hp=53)
    reads=tuple(RuleTextRead('current',kind,(RuleTextLine(str(recovery),.999),)) for kind in ('original','grayscale'))
    observed=confirm_skip_recovery(SkipRecoveryEvidence('current',True,(644,750,44,38),
        (652,754,26,30),(654,756,23,26),.99,False,reads))
    budget=HpPreparationBudget((),RecurringHpCost(1,3,('rule',)),True)
    return inputs,current,state,observed,budget


def test_known_operations_match_observed_uncapped_50_to_52_to_49_without_guaranteeing_events():
    result=assess_current_skip(*scenario())
    assert result.status=='known_operations' and result.recovery_root_hp==52
    assert result.hp_if_recovery_first==result.hp_if_charge_first==49
    assert not result.survival_guaranteed and 'recovery_restrictions' in result.unresolved_events


def test_hp_cap_makes_unknown_event_order_material_instead_of_assuming_net_plus_two_minus_three():
    result=assess_current_skip(*scenario(hp=53))
    assert result.recovery_root_hp==53
    assert (result.known_operations_min_hp,result.known_operations_max_hp)==(50,52)


def test_zero_hp_continues_and_later_recovery_still_applies_after_an_earlier_charge():
    result=assess_current_skip(*scenario(hp=2))
    assert (result.hp_if_recovery_first,result.hp_if_charge_first)==(1,2)
    assert result.known_operations_min_hp==1 and not result.survival_guaranteed
    zero=assess_current_skip(*scenario(hp=0))
    assert zero.status=='known_operations'
    assert (zero.hp_if_recovery_first,zero.hp_if_charge_first)==(0,2)


def test_explicit_zero_recovery_is_distinct_from_unknown():
    args=list(scenario(recovery=0))
    result=assess_current_skip(*args)
    assert result.recovery==0 and result.recovery_root_hp==50 and result.known_operations_min_hp==47
    args[3]=replace(args[3],status='unknown',value=None)
    result=assess_current_skip(*args)
    assert result.status=='unknown' and result.recovery is None


@pytest.mark.parametrize('kind',['pending','other_frame','missing_max','stale_max','different_hp','projected_restriction'])
def test_unbound_input_resource_or_frame_evidence_never_projects_recovery(kind):
    inputs,current,state,observed,budget=scenario()
    if kind=='pending':inputs.begin('skip',current)
    elif kind=='other_frame':observed=replace(observed,evidence=replace(observed.evidence,frame_id='after-input'))
    elif kind=='missing_max':current=replace(current,maximum_hp_parameter=None)
    elif kind=='stale_max':current=replace(current,maximum_hp_parameter=replace(current.maximum_hp_parameter,input_sequence=1))
    elif kind=='different_hp':state=replace(state,hp=49)
    else:state=replace(state,projected_resource_restrictions=frozenset({'hp_recovery'}))
    result=assess_current_skip(inputs,current,state,observed,budget)
    assert result.status=='unknown' and result.recovery_root_hp is None


def test_unknown_declared_charge_does_not_become_zero_and_last_scheduled_turn_does_not_invent_next_round():
    inputs,current,state,observed,budget=scenario()
    result=assess_current_skip(inputs,current,state,observed,replace(budget,recurring_known=False))
    assert result.status=='recovery_only' and result.declared_next_charge is None
    result=assess_current_skip(*scenario(remaining=1))
    assert result.reason=='next_round_unconfirmed' and result.declared_next_charge is None


def test_declared_later_start_is_zero_for_this_next_round_only():
    inputs,current,state,observed,budget=scenario()
    result=assess_current_skip(inputs,current,state,observed,replace(budget,recurring=RecurringHpCost(5,3,('rule',))))
    assert result.status=='known_operations' and result.declared_next_charge==0
    assert result.known_operations_min_hp==result.known_operations_max_hp==52
