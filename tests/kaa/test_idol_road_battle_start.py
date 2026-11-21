from dataclasses import replace
import pytest
from kaa.idol_road.battle_start import prepare_variable_battle
from kaa.idol_road.controller import IdolRoadController
from kaa.config.idol_road import IdolRoadConfig
from kaa.idol_road.models import Action,Observation,Outcome,Scene
from tests.kaa.test_idol_road_turn_budget import formation,goals,row
from tests.kaa.test_idol_road_hp_preparation import rule,rules
from tests.kaa.test_idol_road_retry_flow import formation as slots


def ready():
    return replace(formation(),formation_slots=slots())


def current_rules():
    return rules(rule(1,'以降、ターン開始時、','体力消費3'),row())


def test_variable_preflight_needs_current_rank_rules_and_formation_but_not_full_deck():
    spec=prepare_variable_battle(ready(),goals(),current_rules())
    assert (spec.character_id,spec.stage,spec.rank_at_most,spec.total_turns)==('ttmr',20,3,20)


@pytest.mark.parametrize('changes',[{'formation_slots':None},{'formation_recommended_effect':None},
    {'formation_recommended_effect':'enthusiasm'},{'stars':1},{'total_turns':None}])
def test_unsupported_or_incomplete_formation_fails_before_start(changes):
    with pytest.raises(ValueError,match='automatic_variable_formation_unconfirmed'):
        prepare_variable_battle(replace(ready(),**changes),goals(),current_rules())


def test_unknown_future_charge_is_not_filled_from_old_battle_or_used_to_block_current_start():
    from kaa.idol_road.hp_preparation import hp_preparation_budget
    current=rules(row(),rule(18,'体力が50%以上の場合、'))
    budget=hp_preparation_budget('ttmr',20,current)
    assert budget.goals and not budget.recurring_known and budget.recurring is None
    assert prepare_variable_battle(ready(),goals(),current).stage==20
    with pytest.raises(ValueError,match='automatic_one_star_objective_unconfirmed'):
        prepare_variable_battle(ready(),replace(goals(),identity_source='history'),current_rules())


def test_explicit_auto_mode_preflights_before_requesting_a_battle():
    flow=IdolRoadController(IdolRoadConfig(mode='auto',characters=['ttmr']))
    flow._objectives[('ttmr',20)]=goals()
    flow._rules[('ttmr',20)]=current_rules()
    assert flow.step(ready()).action==Action.START_BATTLE
    assert flow.retry_policy is not None and flow.retry_policy.failures_on_lineup==0
    unpriced=IdolRoadController(IdolRoadConfig(mode='auto',characters=['ttmr']))
    unpriced._objectives[('ttmr',20)]=goals()
    unpriced._rules[('ttmr',20)]=rules(row(),rule(18,'体力が50%以上の場合、'))
    assert unpriced.step(ready()).action==Action.START_BATTLE
    assert unpriced.awaiting_battle_start and unpriced.battles_started==0


def test_sense_floor_without_turn_additions_or_hp_goals_uses_owned_hud_observation():
    from kaa.idol_road.battle_observation import create_battle_session
    current=rules(rule(3,'好調状態の場合、','体力回復5'))
    spec=prepare_variable_battle(ready(),goals(),current)
    assert spec.total_turns==20
    assert create_battle_session(ready(),stage_rules=current).turn_budget_mode=='fixed_budget'
    owner=create_battle_session(ready(),stage_rules=current,observed_turns=True)
    assert owner.turn_budget_mode=='variable_budget'
    flow=IdolRoadController(IdolRoadConfig(mode='auto',characters=['ttmr']))
    flow._objectives[('ttmr',20)]=goals();flow._rules[('ttmr',20)]=current
    assert flow.step(ready()).action==Action.START_BATTLE


def test_auto_retries_confirmed_failure_even_when_legacy_stop_flag_is_true():
    flow=IdolRoadController(IdolRoadConfig(mode='auto',characters=['ttmr'],stop_on_failure=True))
    flow._objectives[('ttmr',20)]=goals()
    flow._rules[('ttmr',20)]=current_rules()
    assert flow.step(ready()).action==Action.START_BATTLE
    decision=flow.step(Observation(Scene.RESULT,'ttmr',20,outcome=Outcome.FAILURE))
    assert decision.action==Action.DISMISS_RESULT
    assert flow.retry_policy is not None and flow.retry_policy.failures_on_lineup==1


def test_auto_reobserves_unknown_conditions_without_merging_rows_across_frames():
    flow=IdolRoadController(IdolRoadConfig(mode='auto',characters=['ttmr']))
    assert flow.step(replace(ready(),conditions_available=True)).action==Action.OPEN_CONDITIONS
    unknown=Observation(Scene.CONDITIONS,'ttmr',20)
    assert flow.step(unknown).reason=='await_current_one_star_conditions'
    assert flow.objectives.rank_limit(1) is None
    current=Observation(Scene.CONDITIONS,'ttmr',20,requirements=goals().requirements)
    assert flow.step(current).action==Action.CLOSE_CONDITIONS
    assert flow.objectives.rank_limit(1)==3


def test_unknown_auto_condition_wait_is_bounded_and_does_not_start_a_battle():
    flow=IdolRoadController(IdolRoadConfig(mode='auto',characters=['ttmr']))
    flow.step(replace(ready(),conditions_available=True))
    unknown=Observation(Scene.CONDITIONS,'ttmr',20)
    assert flow.step(unknown).action==Action.WAIT
    assert flow.step(unknown).action==Action.WAIT
    assert flow.step(unknown).action==Action.CLOSE_CONDITIONS
    assert flow.step(ready()).reason=='automatic_one_star_rank_unknown'
    assert flow.battles_started==0
