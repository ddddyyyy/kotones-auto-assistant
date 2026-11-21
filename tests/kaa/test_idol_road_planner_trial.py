from dataclasses import replace
import pytest
from kaa.config.idol_road import IdolRoadConfig
from kaa.idol_road.models import Observation,Scene,Action,Outcome
from kaa.idol_road.objectives import StageObjectives,read_rank_requirement
from kaa.idol_road.planner_trial import prepare_planner_trial
from kaa.idol_road.controller import IdolRoadController
from kaa.idol_road.page import PageSnapshot
from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame
from kaa.idol_road.planner_adapter import TrialFrameGuard


def formation():
    return Observation(Scene.CONFIRM,'ttmr',16,stars=0,formation_ready=True,total_turns=10)


def goals():
    return StageObjectives('ttmr',16,tuple(read_rank_requirement(i,f'{4-i}位以上でクリア') for i in (1,2,3)))


def test_trial_scope_has_explicit_goal_and_leaves_full_automatic_readiness_separate():
    spec=prepare_planner_trial(formation(),goals())
    assert spec.rank_at_most==3 and spec.total_turns==10 and spec.scope=='single_concentration_trial'


@pytest.mark.parametrize('changes',[{'character_id':'hski'},{'stars':1},{'stars':3},
    {'formation_ready':False},{'total_turns':None},{'total_turns':True},{'stage':None}])
def test_trial_rejects_unconfirmed_or_cleared_input(changes):
    with pytest.raises(ValueError):
        prepare_planner_trial(replace(formation(),**changes),goals())


def test_trial_requires_current_goal_provenance():
    for goal in (None,replace(goals(),stage=15),replace(goals(),identity_source='history')):
        with pytest.raises(ValueError):
            prepare_planner_trial(formation(),goal)


def test_missing_assets_fail_preflight_before_battle(monkeypatch):
    from kaa.idol_road.planner_adapter import prepare_planner_assets
    from kaa.tasks.produce.new.play_cards import ui
    def missing():
        raise FileNotFoundError('skill_cards')
    monkeypatch.setattr(ui,'skill_cards_db',missing)
    with pytest.raises(ValueError,match='未发出挑战'):
        prepare_planner_assets()


def test_single_trial_never_starts_second_challenge_or_advances_after_success():
    flow=IdolRoadController(IdolRoadConfig(mode='planner_trial',characters=['ttmr']))
    flow._stage=16
    flow._objectives[('ttmr',16)]=goals()
    assert flow.step(formation()).action==Action.START_BATTLE
    assert flow.step(Observation(Scene.BATTLE,'ttmr',16)).action==Action.WAIT
    end=flow.step(Observation(Scene.RESULT,'ttmr',16,outcome=Outcome.SUCCESS,stars=1))
    assert end.action==Action.STOP and end.reason=='planner_trial_succeeded' and flow.battles_started==1


def test_trial_failure_stops_even_if_legacy_skip_character_option_is_set():
    flow=IdolRoadController(IdolRoadConfig(mode='planner_trial',characters=['ttmr'],stop_on_failure=False))
    flow._stage=16
    flow._objectives[('ttmr',16)]=goals()
    flow.step(formation())
    assert flow.step(Observation(Scene.RESULT,'ttmr',16,outcome=Outcome.FAILURE)).reason=='planner_trial_failed'


def test_trial_multiple_characters_stops_before_navigation():
    flow=IdolRoadController(IdolRoadConfig(mode='planner_trial',characters=['ttmr','hski']))
    assert flow.step(Observation(Scene.HOME)).reason=='planner_trial_requires_one_character'


def test_frame_guard_waits_unknown_and_stops_on_result_without_claiming_success():
    now=[0.]
    session=BattleObservationSession('battle','ttmr',16,10,'confirmed_formation_start_request')
    guard=TrialFrameGuard(session,100,clock=lambda:now[0])
    assert guard.check(PageSnapshot(Observation(Scene.UNKNOWN)))=='wait'
    now[0]=21
    with pytest.raises(ValueError):
        guard.check(PageSnapshot(Observation(Scene.UNKNOWN)))
    assert guard.check(PageSnapshot(Observation(Scene.JUDGING,'ttmr',16)))=='stop'


def test_trial_move_dialog_does_not_fall_through_to_legacy_option_order():
    session=BattleObservationSession('battle','ttmr',16,10,'confirmed_formation_start_request')
    guard=TrialFrameGuard(session,100,clock=lambda:0)
    with pytest.raises(ValueError,match='移牌选择'):
        guard.check(PageSnapshot(Observation(Scene.UNKNOWN)),move_dialog=True)


def test_only_owned_last_turn_allows_longer_input_free_result_wait():
    now=[0.]
    session=BattleObservationSession('battle','ttmr',19,20,'confirmed_formation_start_request')
    guard=TrialFrameGuard(session,200,clock=lambda:now[0])
    frame=bind_battle_frame(session,BattleFrameValues('last',True,1,53))
    assert frame is not None
    assert guard.check(PageSnapshot(Observation(Scene.BATTLE,'ttmr',19),battle_frame=frame))=='continue'
    unknown=PageSnapshot(Observation(Scene.UNKNOWN))
    assert guard.check(unknown)=='wait'
    now[0]=21
    assert guard.check(unknown)=='wait'
    with pytest.raises(ValueError,match='移牌选择'):
        guard.check(unknown,move_dialog=True)
    now[0]=59
    assert guard.check(unknown)=='wait'
    now[0]=60
    with pytest.raises(ValueError,match='持续未知'):
        guard.check(unknown)


def test_last_turn_does_not_extend_unconfirmed_overlay_or_total_timeout():
    now=[0.]
    session=BattleObservationSession('battle','ttmr',19,20,'confirmed_formation_start_request')
    guard=TrialFrameGuard(session,200,clock=lambda:now[0])
    frame=bind_battle_frame(session,BattleFrameValues('last',True,1,53))
    guard.check(PageSnapshot(Observation(Scene.BATTLE,'ttmr',19),battle_frame=frame))
    overlay=bind_battle_frame(session,BattleFrameValues('overlay',True,None,None,resource_state='unconfirmed_overlay'))
    snapshot=PageSnapshot(Observation(Scene.BATTLE,'ttmr',19),battle_frame=overlay)
    assert guard.check(snapshot)=='wait'
    now[0]=21
    with pytest.raises(ValueError,match='持续未知'):
        guard.check(snapshot)
    now[0]=200
    with pytest.raises(ValueError,match='总时限'):
        guard.check(PageSnapshot(Observation(Scene.UNKNOWN)))


def test_frame_guard_rejects_foreign_battle_and_unsupported_dialog_or_timeout():
    session=BattleObservationSession('battle','ttmr',16,10,'confirmed_formation_start_request')
    guard=TrialFrameGuard(session,100,clock=lambda:0)
    frame=bind_battle_frame(session,BattleFrameValues('frame',True,10,48))
    snapshot=PageSnapshot(Observation(Scene.BATTLE,'ttmr',16),battle_frame=frame)
    assert guard.check(snapshot)=='continue'
    overlay=bind_battle_frame(session,BattleFrameValues('overlay',True,None,None,resource_state='unconfirmed_overlay'))
    assert guard.check(replace(snapshot,battle_frame=overlay))=='wait'
    with pytest.raises(ValueError):
        guard.check(replace(snapshot,battle_frame=replace(frame,session=replace(session,battle_id='foreign'))))
    with pytest.raises(ValueError):
        guard.check(PageSnapshot(Observation(Scene.BATTLE_STATUS_DETAILS,'ttmr',16)))
    with pytest.raises(ValueError):
        TrialFrameGuard(session,0,clock=lambda:0).check(snapshot)
