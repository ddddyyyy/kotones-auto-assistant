"""流程规则验证；只输入领域观察，不模拟设备点击或识别算法。"""

import pytest

from kaa.config.idol_road import IdolRoadConfig
from kaa.idol_road.controller import IdolRoadController
from kaa.idol_road.models import Action, Observation, Outcome, Scene, Stage
from kaa.idol_road.objectives import read_rank_requirement
from kaa.idol_road.items import ItemBindingStatus, ItemNameRead, ItemOverlapStatus, ItemReference, ItemViewport, VisibleItem


def observe(scene, *, character='hski', stage=1, **kwargs):
    if scene==Scene.JUDGING:kwargs.setdefault('result_next_available',True)
    if scene == Scene.RESULT and kwargs.get('outcome') == Outcome.SUCCESS:
        kwargs.setdefault('stars', 3)
    if scene == Scene.CONFIRM:
        kwargs.setdefault('formation_ready', True)
        kwargs.setdefault('stars', 0)
    return Observation(scene, character_id=character, stage=stage, **kwargs)


def controller(**kwargs):
    return IdolRoadController(IdolRoadConfig(characters=['hski'], **kwargs))


@pytest.mark.parametrize('mode',['preview','manual','planner_trial'])
def test_unowned_resume_prompt_stops_without_resuming_retiring_or_counting_new_battle(mode):
    flow=controller(mode=mode)
    decision=flow.step(Observation(Scene.RESUME_PROMPT))
    assert decision.action==Action.STOP and decision.reason=='unfinished_stage_requires_native_recovery'
    assert flow.battles_started==0
    assert flow.step(Observation(Scene.HOME))==decision


def test_explicit_cancellation_has_priority_over_resume_recovery():
    decision=controller().step(Observation(Scene.RESUME_PROMPT),cancelled=True)
    assert decision.action==Action.STOP and decision.reason=='cancelled'


def select_stage(flow):
    return flow.step(observe(
        Scene.STAGES, stages_complete=True,
        stages=(Stage(1, True, 0), Stage(2, True, 0)),
    ))


def start_manual(flow):
    select_stage(flow)
    assert flow.step(observe(Scene.CONFIRM)).action == Action.START_BATTLE


def test_delegated_initial_hud_counts_one_real_battle_and_ready_is_not_an_extra_start():
    from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame
    from dataclasses import replace
    flow=controller(mode='manual')
    start_manual(flow)
    owner=BattleObservationSession('new','hski',1,20,'confirmed_formation_start_request','variable_budget')
    frame=bind_battle_frame(owner,BattleFrameValues('hud',True,20,50))
    initial=observe(Scene.BATTLE,battle_state=frame.observation)
    flow.observe_initial_battle(initial)
    flow.observe_initial_battle(initial)
    assert flow.battles_started==1
    with pytest.raises(ValueError):flow.observe_initial_battle(initial,ready=True)
    ready=replace(initial,battle_state=replace(frame.observation,turn=1,turn_source='owned_initial_log_return'))
    flow.observe_initial_battle(ready,ready=True)
    assert flow.battles_started==1


def test_delegated_initial_hud_requires_a_matching_requested_battle():
    from kaa.idol_road.rule_conditions import RuleConditionObservation
    flow=controller(mode='manual')
    state=RuleConditionObservation('hski',1,'new','hud',None,source='confirmed_current_battle_observation')
    with pytest.raises(ValueError):flow.observe_initial_battle(observe(Scene.BATTLE,battle_state=state))
    assert flow.battles_started==0


def test_preview_never_starts_battle_and_remains_terminal():
    flow = controller()
    assert select_stage(flow).stage == 1
    ready = flow.step(observe(Scene.CONFIRM))
    assert ready.action == Action.READY
    assert flow.step(observe(Scene.BATTLE)) == ready
    assert flow.battles_started == 0


def test_complete_list_required_before_selecting_next_stage():
    flow = controller()
    decision = flow.step(observe(Scene.STAGES, stages=(Stage(5, True, 0),)))
    assert decision.action == Action.SCAN_STAGES


def test_unknown_progress_does_not_become_clear_or_uncleared():
    flow = controller()
    decision = flow.step(observe(
        Scene.STAGES, stages_complete=True,
        stages=(Stage(1, True, None), Stage(2, True, 0)),
    ))
    assert decision.reason == 'stage_stars_unknown'
    assert decision.action == Action.STOP


def test_success_rewards_and_stale_list_advance_only_once():
    flow = controller(mode='manual')
    start_manual(flow)
    assert flow.step(observe(Scene.CONFIRM)).action == Action.WAIT
    for _ in range(3):
        assert flow.step(observe(Scene.BATTLE)).action == Action.WAIT
    assert flow.step(observe(Scene.RESULT, outcome=Outcome.SUCCESS)).action == Action.DISMISS_RESULT
    for _ in range(2):
        assert flow.step(observe(Scene.RESULT, outcome=Outcome.SUCCESS)).action == Action.WAIT
    assert flow.battles_started == 1
    assert flow.step(observe(Scene.REWARDS)).action == Action.DISMISS_REWARDS
    decision = select_stage(flow)
    assert decision.action == Action.SELECT_STAGE
    assert decision.stage == 2


def test_battle_limit_allows_result_and_rewards_before_stopping():
    flow = controller(mode='manual', max_battles=1)
    start_manual(flow)
    assert flow.step(observe(Scene.RESULT, outcome=Outcome.SUCCESS)).action == Action.DISMISS_RESULT
    assert flow.step(observe(Scene.REWARDS)).action == Action.DISMISS_REWARDS
    assert select_stage(flow).reason == 'battle_limit'


def test_native_ending_animation_waits_for_controls_without_claiming_clear():
    flow=controller(mode='manual')
    start_manual(flow)
    animation=observe(Scene.JUDGING,result_next_available=False)
    assert flow.step(animation).reason=='await_result_controls'
    assert flow.battles_started==1
    assert flow.step(observe(Scene.JUDGING)).action==Action.ADVANCE_RESULT
    assert flow.step(observe(Scene.STAGES)).reason=='battle_result_missing'


def test_ending_animation_has_separate_bounded_result_wait():
    now=[0.]
    flow=IdolRoadController(IdolRoadConfig(characters=['hski'],mode='manual'),clock=lambda:now[0])
    start_manual(flow)
    animation=observe(Scene.JUDGING,result_next_available=False)
    assert flow.step(animation).action==Action.WAIT
    now[0]=29.
    assert flow.step(animation).action==Action.WAIT
    now[0]=30.
    assert flow.step(animation).reason=='result_controls_not_confirmed'


@pytest.mark.parametrize('scene,outcome,reason', [
    (Scene.RESULT, Outcome.UNKNOWN, 'result_unknown'),
    (Scene.RESULT, Outcome.FAILURE, 'battle_failed'),
    (Scene.REWARDS, Outcome.UNKNOWN, 'reward_without_confirmed_result'),
    (Scene.STAGES, Outcome.UNKNOWN, 'battle_result_missing'),
])
def test_missing_or_failed_result_never_advances(scene, outcome, reason):
    flow = controller(mode='manual')
    start_manual(flow)
    decision = flow.step(observe(scene, outcome=outcome))
    assert decision.action == Action.STOP
    assert decision.reason == reason


def test_failure_skips_character_without_retry_when_configured():
    flow = IdolRoadController(IdolRoadConfig(
        characters=['hski', 'hmsz'], mode='manual', stop_on_failure=False,
    ))
    start_manual(flow)
    decision = flow.step(observe(Scene.RESULT, outcome=Outcome.FAILURE))
    assert decision.action == Action.BACK_TO_CHARACTERS
    assert decision.character_id == 'hmsz'
    assert flow.battles_started == 1


@pytest.mark.parametrize('observation,reason', [
    (observe(Scene.RESULT, character='hmsz', outcome=Outcome.SUCCESS), 'character_not_confirmed'),
    (observe(Scene.RESULT, stage=2, outcome=Outcome.SUCCESS), 'stage_not_confirmed'),
])
def test_result_must_match_requested_character_and_stage(observation, reason):
    flow = controller(mode='manual')
    start_manual(flow)
    assert flow.step(observation).reason == reason


def test_unknown_scenes_have_total_timeout():
    now = [10.0]
    flow = IdolRoadController(
        IdolRoadConfig(characters=['hski'], timeout_seconds=5), clock=lambda: now[0],
    )
    assert flow.step(Observation(Scene.UNKNOWN)).action == Action.WAIT
    now[0] = 15.0
    assert flow.step(Observation(Scene.UNKNOWN)).reason == 'timeout'


def test_cancel_manual_wait_is_terminal():
    flow = controller(mode='manual')
    start_manual(flow)
    decision = flow.step(observe(Scene.BATTLE), cancelled=True)
    assert decision.reason == 'cancelled'
    assert flow.step(observe(Scene.RESULT, outcome=Outcome.SUCCESS)) == decision


def test_unprepared_formation_does_not_autofill():
    flow = controller()
    select_stage(flow)
    assert flow.step(observe(Scene.FORMATION)).reason == 'formation_not_ready'


def test_empty_config_and_finished_locked_list_exit():
    assert IdolRoadController(IdolRoadConfig()).step(Observation(Scene.HOME)).reason == 'no_characters'
    flow = controller()
    decision = flow.step(observe(
        Scene.STAGES, stages_complete=True,
        stages=(Stage(1, True, 3), Stage(2, False, 0)),
    ))
    assert decision.reason == 'characters_finished'


def test_config_is_snapshotted_for_a_run():
    config = IdolRoadConfig(characters=['hski'])
    flow = IdolRoadController(config)
    config.characters.clear()
    assert flow.character_id == 'hski'


@pytest.mark.parametrize('scene', [Scene.HOME, Scene.CHARACTERS, Scene.FORMATION])
def test_lost_battle_does_not_start_navigation_again(scene):
    flow = controller(mode='manual')
    start_manual(flow)
    assert flow.step(observe(scene)).reason == 'unexpected_battle_exit'


def test_result_cannot_be_followed_by_unrequested_retry():
    flow = controller(mode='manual')
    start_manual(flow)
    flow.step(observe(Scene.RESULT, outcome=Outcome.SUCCESS))
    assert flow.step(observe(Scene.BATTLE)).reason == 'battle_after_confirmed_result'


def test_map_focus_does_not_claim_complete_stage_scan():
    flow = controller()
    decision = flow.step(observe(Scene.STAGES, focused_stage=Stage(13, True, 0)))
    assert decision.action == Action.SELECT_STAGE
    assert decision.stage == 13
    assert decision.reason == 'focused_uncleared_stage'


@pytest.mark.parametrize('stars', [1, 2, 3])
def test_map_advances_past_cleared_stage_without_battle(stars):
    flow = controller()
    decision = flow.step(observe(Scene.STAGES, focused_stage=Stage(13, True, stars)))
    assert decision.action == Action.NEXT_STAGE
    assert decision.stage == 14
    assert decision.reason == ('three_star_skip' if stars == 3 else 'stage_already_cleared')
    assert flow.battles_started == 0


def test_unknown_focused_star_count_does_not_advance_or_challenge():
    flow = controller()
    unknown = observe(Scene.STAGES, focused_stage=Stage(13, True, None))
    for _ in range(2):
        waiting=flow.step(unknown)
        assert waiting.action==Action.WAIT and waiting.reason=='await_map_stars'
    decision = flow.step(unknown)
    assert decision.action == Action.STOP
    assert decision.reason == 'stage_stars_unknown'


def test_map_transition_can_refresh_to_current_zero_stars_without_inventing_them():
    flow=controller()
    assert flow.step(observe(Scene.STAGES,focused_stage=Stage(19,True,None))).action==Action.WAIT
    assert flow.step(observe(Scene.UNKNOWN)).action==Action.WAIT
    assert flow.step(observe(Scene.STAGES,focused_stage=Stage(20,True,None))).action==Action.WAIT
    current=flow.step(observe(Scene.STAGES,focused_stage=Stage(20,True,0)))
    assert current.action==Action.SELECT_STAGE and current.stage==20
    assert flow.battles_started==0


def test_unknown_stage_changes_do_not_reset_map_star_refresh_budget():
    flow=controller()
    for number in (19,20):
        assert flow.step(observe(Scene.STAGES,focused_stage=Stage(number,True,None))).action==Action.WAIT
    assert flow.step(observe(Scene.UNKNOWN)).action==Action.WAIT
    stopped=flow.step(observe(Scene.STAGES,focused_stage=Stage(21,True,None)))
    assert stopped.action==Action.STOP and stopped.reason=='stage_stars_unknown'
    assert flow.battles_started==0


def test_confirm_requires_ready_formation_even_in_preview():
    flow = controller()
    select_stage(flow)
    assert flow.step(observe(Scene.CONFIRM, formation_ready=False)).reason == 'formation_not_ready'


def test_requested_battle_waits_when_formation_button_becomes_disabled():
    flow = controller(mode='manual')
    start_manual(flow)
    decision = flow.step(observe(Scene.CONFIRM, formation_ready=False))
    assert decision.action == Action.WAIT
    assert decision.reason == 'await_battle_start'
    assert flow.battles_started == 0


def test_failure_character_switch_waits_for_actual_character_list():
    flow = IdolRoadController(IdolRoadConfig(
        characters=['hski', 'hmsz'], mode='manual', stop_on_failure=False,
    ))
    start_manual(flow)
    flow.step(observe(Scene.RESULT, outcome=Outcome.FAILURE))
    assert flow.step(observe(Scene.STAGES)).action == Action.BACK_TO_CHARACTERS
    assert flow.step(Observation(Scene.UNKNOWN)).action == Action.WAIT
    assert flow.step(Observation(Scene.CHARACTERS)).character_id == 'hmsz'


def test_ranking_advances_without_claiming_success_or_double_counting():
    flow = controller(mode='manual')
    start_manual(flow)
    assert flow.step(observe(Scene.JUDGING)).action == Action.ADVANCE_RESULT
    assert flow.step(observe(Scene.JUDGING)).action == Action.WAIT
    assert flow.battles_started == 1
    # 没有最终成功页，回到关卡列表也不能冒充通关。
    assert flow.step(observe(Scene.STAGES)).reason == 'battle_result_missing'


@pytest.mark.parametrize('stars,reason', [(None, 'result_stars_unknown'), (0, 'result_has_no_star')])
def test_success_title_without_at_least_one_star_cannot_advance(stars, reason):
    flow = controller(mode='manual')
    start_manual(flow)
    decision = flow.step(observe(Scene.RESULT, outcome=Outcome.SUCCESS, stars=stars))
    assert decision.action == Action.STOP
    assert decision.reason == reason


@pytest.mark.parametrize('stars', [1, 2, 3])
def test_confirmed_result_stars_allow_next_stage_and_deduplicate(stars):
    flow = controller(mode='manual')
    start_manual(flow)
    assert flow.step(observe(Scene.RESULT, outcome=Outcome.SUCCESS, stars=stars)).action == Action.DISMISS_RESULT
    assert flow.step(observe(Scene.RESULT, outcome=Outcome.SUCCESS, stars=stars)).action == Action.WAIT
    assert flow.battles_started == 1
    assert select_stage(flow).stage == 2


@pytest.mark.parametrize('stars', [1, 2, 3])
def test_failure_with_positive_stars_is_not_a_clear(stars):
    flow = controller(mode='manual')
    start_manual(flow)
    assert flow.step(observe(Scene.RESULT, outcome=Outcome.FAILURE, stars=stars)).reason == 'result_stars_conflict'


@pytest.mark.parametrize('stars', [1, 2, 3])
def test_preexisting_stars_in_formation_never_start_a_battle(stars):
    flow = controller(mode='manual')
    select_stage(flow)
    decision = flow.step(observe(Scene.CONFIRM, stars=stars))
    assert decision.action == Action.BACK_TO_STAGES
    assert flow.battles_started == 0


def test_success_waits_for_stale_zero_star_map_to_refresh():
    flow = controller(mode='manual')
    start_manual(flow)
    flow.step(observe(Scene.RESULT, outcome=Outcome.SUCCESS, stars=1))
    decision = flow.step(observe(Scene.STAGES, focused_stage=Stage(1, True, 0)))
    assert decision.action == Action.WAIT
    decision = flow.step(observe(Scene.STAGES, focused_stage=Stage(1, True, 1)))
    assert decision.action == Action.NEXT_STAGE
    assert flow.battles_started == 1


def test_complete_list_skips_all_positive_star_counts_without_spending_budget():
    flow = controller()
    decision = flow.step(observe(Scene.STAGES, stages_complete=True, stages=(
        Stage(1, True, 1), Stage(2, True, 2), Stage(3, True, 3), Stage(4, True, 0),
    )))
    assert decision.action == Action.SELECT_STAGE and decision.stage == 4
    assert flow.battles_started == 0


def test_positive_star_count_does_not_replace_result_scene_confirmation():
    flow = controller(mode='manual')
    start_manual(flow)
    assert flow.step(observe(Scene.RESULT, stars=1, outcome=Outcome.UNKNOWN)).reason == 'result_unknown'


def test_starred_stage_navigation_requests_once_then_waits_and_stops():
    flow = controller()
    snapshot = observe(Scene.STAGES, focused_stage=Stage(13, True, 3))
    assert flow.step(snapshot).action == Action.NEXT_STAGE
    assert flow.step(snapshot).action == Action.WAIT
    assert flow.step(snapshot).action == Action.WAIT
    assert flow.step(snapshot).reason == 'next_stage_not_confirmed'
    assert flow.battles_started == 0


def test_starred_navigation_requires_the_requested_adjacent_stage():
    flow = controller()
    flow.step(observe(Scene.STAGES, focused_stage=Stage(13, True, 1)))
    decision = flow.step(observe(Scene.STAGES, focused_stage=Stage(15, True, 0)))
    assert decision.reason == 'unexpected_next_stage'
    assert flow.battles_started == 0


def test_starred_stage_chain_reaches_zero_star_without_challenging_completed_stages():
    flow = controller()
    for number, stars in ((13, 3), (14, 2), (15, 1)):
        decision = flow.step(observe(Scene.STAGES, focused_stage=Stage(number, True, stars)))
        assert decision.action == Action.NEXT_STAGE and decision.stage == number + 1
    decision = flow.step(observe(Scene.STAGES, focused_stage=Stage(16, True, 0)))
    assert decision.action == Action.SELECT_STAGE and decision.stage == 16
    assert flow.battles_started == 0


@pytest.mark.parametrize('mode', ['preview', 'manual'])
def test_unknown_formation_stars_never_start_or_complete_preview(mode):
    flow = controller(mode=mode)
    select_stage(flow)
    decision = flow.step(observe(Scene.CONFIRM, stars=None))
    assert decision.action == Action.STOP and decision.reason == 'stage_stars_unknown'
    assert flow.battles_started == 0


@pytest.mark.parametrize('mode,action', [('preview', Action.READY), ('manual', Action.START_BATTLE)])
def test_resume_from_confirmed_zero_star_formation(mode, action):
    flow = controller(mode=mode)
    decision = flow.step(observe(Scene.CONFIRM, stage=16))
    assert decision.action == action and decision.stage == 16
    assert flow.battles_started == 0


@pytest.mark.parametrize('stars', [1, 2, 3])
def test_resume_starred_formation_only_returns_to_map(stars):
    flow = controller(mode='manual')
    decision = flow.step(observe(Scene.CONFIRM, stage=13, stars=stars))
    assert decision.action == Action.BACK_TO_STAGES
    assert flow.battles_started == 0


@pytest.mark.parametrize('character,stage,reason', [
    (None, 16, 'character_not_confirmed'),
    ('hski', None, 'stage_not_confirmed'),
    ('hski', 0, 'stage_not_confirmed'),
    ('hski', 16, 'stage_stars_unknown'),
])
def test_resume_requires_same_frame_identity_and_stars(character, stage, reason):
    flow = controller(mode='manual')
    assert flow.step(observe(Scene.CONFIRM, character=character, stage=stage, stars=None)).reason == reason


def test_resume_never_adopts_an_in_progress_battle_or_old_result():
    for scene in (Scene.BATTLE, Scene.RESULT, Scene.JUDGING):
        flow = controller(mode='manual')
        assert flow.step(observe(scene)).action == Action.STOP
        assert flow.battles_started == 0


def test_reward_information_does_not_confirm_result_or_spend_budget():
    flow = controller(mode='manual')
    assert flow.step(Observation(Scene.REWARD_INFO)).action == Action.CLOSE_INFO
    start_manual(flow)
    assert flow.step(Observation(Scene.REWARD_INFO)).action == Action.CLOSE_INFO
    assert flow.battles_started == 0
    assert flow.step(observe(Scene.STAGES)).reason == 'battle_result_missing'


def test_resumed_formation_does_not_replace_existing_stage_request():
    flow = controller(mode='manual')
    select_stage(flow)
    assert flow.step(observe(Scene.CONFIRM, stage=16)).reason == 'stage_not_confirmed'


def test_stage_selection_is_not_repeated_on_residual_map():
    flow = controller(mode='manual')
    assert select_stage(flow).action == Action.SELECT_STAGE
    assert select_stage(flow).reason == 'await_navigation'
    assert select_stage(flow).reason == 'await_navigation'
    assert select_stage(flow).reason == 'navigation_not_confirmed'
    assert flow.battles_started == 0


def test_failure_returns_once_per_page_without_skipping_next_character():
    flow = IdolRoadController(IdolRoadConfig(
        characters=['hski', 'hmsz', 'ttmr'], mode='manual', stop_on_failure=False,
    ))
    start_manual(flow)
    failed = observe(Scene.RESULT, outcome=Outcome.FAILURE)
    assert flow.step(failed).action == Action.BACK_TO_CHARACTERS
    assert flow.step(failed).action == Action.WAIT
    assert flow.character_id == 'hmsz'
    assert flow.step(observe(Scene.STAGES)).action == Action.BACK_TO_CHARACTERS
    assert flow.step(observe(Scene.STAGES)).action == Action.WAIT
    assert flow.step(Observation(Scene.CHARACTERS, character_selectable=True)).action == Action.SELECT_CHARACTER
    assert flow.step(Observation(Scene.CHARACTERS, character_selectable=True)).action == Action.WAIT
    assert flow.step(observe(Scene.STAGES, character='hmsz', focused_stage=Stage(3, True, 0))).stage == 3
    assert flow.battles_started == 1


def test_unknown_frame_does_not_allow_repeated_return_click():
    flow = controller()
    formation = observe(Scene.CONFIRM, stars=3)
    assert flow.step(formation).action == Action.BACK_TO_STAGES
    assert flow.step(Observation(Scene.UNKNOWN)).action == Action.WAIT
    assert flow.step(formation).reason == 'await_navigation'


def test_navigation_wait_does_not_override_cancel_or_timeout():
    now = [0.0]
    flow = IdolRoadController(IdolRoadConfig(characters=['hski'], timeout_seconds=5), clock=lambda: now[0])
    select_stage(flow)
    now[0] = 5.0
    assert select_stage(flow).reason == 'timeout'
    flow = controller()
    select_stage(flow)
    assert flow.step(observe(Scene.STAGES), cancelled=True).reason == 'cancelled'


def test_missing_character_still_allows_bounded_adapter_scrolling():
    flow = controller()
    for _ in range(12):
        assert flow.step(Observation(Scene.CHARACTERS)).action == Action.SELECT_CHARACTER


def test_manual_battle_wait_is_not_a_navigation_stall():
    flow = controller(mode='manual')
    start_manual(flow)
    for _ in range(30):
        assert flow.step(observe(Scene.BATTLE)).reason == 'manual_battle_in_progress'
    assert flow.battles_started == 1


def test_initial_other_character_formation_returns_page_by_page_without_challenge():
    flow = controller(mode='manual')
    other = observe(Scene.CONFIRM, character='ttmr', stage=16, stars=None)
    assert flow.step(other).action == Action.BACK_TO_STAGES
    assert flow.step(other).reason == 'await_navigation'
    assert flow.step(observe(Scene.STAGES, character='ttmr')).action == Action.BACK_TO_CHARACTERS
    assert flow.step(Observation(Scene.CHARACTERS, character_selectable=True)).action == Action.SELECT_CHARACTER
    assert select_stage(flow).action == Action.SELECT_STAGE
    assert flow.step(observe(Scene.CONFIRM)).action == Action.START_BATTLE
    assert flow.character_id == 'hski' and flow.battles_started == 0


@pytest.mark.parametrize('scene', [Scene.STAGES, Scene.MAP_OVERVIEW])
def test_initial_other_character_map_can_align_without_changing_character_order(scene):
    flow = controller()
    decision = flow.step(observe(scene, character='ttmr'))
    assert decision.action == Action.BACK_TO_CHARACTERS
    assert decision.reason == 'align_configured_character'
    assert flow.character_id == 'hski' and flow.battles_started == 0


@pytest.mark.parametrize('scene', [Scene.STAGES, Scene.CONFIRM, Scene.MAP_OVERVIEW])
def test_other_character_after_selection_is_not_reinterpreted_as_task_start(scene):
    flow = controller()
    flow.step(Observation(Scene.CHARACTERS, character_selectable=True))
    assert flow.step(observe(scene, character='ttmr')).reason == 'character_not_confirmed'


@pytest.mark.parametrize('scene', [Scene.RESULT, Scene.BATTLE, Scene.JUDGING])
def test_initial_other_character_battle_pages_never_trigger_alignment(scene):
    flow = controller()
    assert flow.step(observe(scene, character='ttmr')).reason == 'character_not_confirmed'


@pytest.mark.parametrize('scene', [Scene.STAGES, Scene.CONFIRM, Scene.MAP_OVERVIEW])
def test_unknown_character_never_triggers_alignment(scene):
    flow = controller()
    assert flow.step(observe(scene, character=None)).reason == 'character_not_confirmed'


def test_map_without_challenge_reenters_same_character_once_then_stops():
    flow = controller()
    overview = observe(Scene.MAP_OVERVIEW, stage=None)
    decision = flow.step(overview)
    assert decision.action == Action.BACK_TO_CHARACTERS and decision.reason == 'restore_map_focus'
    assert flow.step(overview).action == Action.WAIT
    assert flow.step(Observation(Scene.CHARACTERS, character_selectable=True)).character_id == 'hski'
    assert flow.step(overview).reason == 'map_reentry_exhausted'
    assert flow.character_id == 'hski' and flow.battles_started == 0


def test_reentered_map_requires_real_stage_and_star_observation():
    flow = controller()
    flow.step(observe(Scene.MAP_OVERVIEW, stage=None))
    flow.step(Observation(Scene.CHARACTERS, character_selectable=True))
    decision = flow.step(observe(Scene.STAGES, focused_stage=Stage(13, True, 0)))
    assert decision.action == Action.SELECT_STAGE and decision.stage == 13
    assert flow.battles_started == 0


def test_map_overview_does_not_replace_missing_result_or_requested_stage():
    flow = controller(mode='manual')
    start_manual(flow)
    assert flow.step(observe(Scene.MAP_OVERVIEW)).reason == 'battle_result_missing'
    flow = controller()
    select_stage(flow)
    assert flow.step(observe(Scene.MAP_OVERVIEW)).reason == 'await_navigation'
    assert flow.step(observe(Scene.MAP_OVERVIEW)).reason == 'await_navigation'
    assert flow.step(observe(Scene.MAP_OVERVIEW)).reason == 'navigation_not_confirmed'


def test_map_overview_after_success_does_not_start_unconfirmed_reentry():
    flow = controller(mode='manual')
    start_manual(flow)
    flow.step(observe(Scene.RESULT, outcome=Outcome.SUCCESS))
    assert flow.step(observe(Scene.MAP_OVERVIEW)).reason == 'map_focus_missing'


def test_map_overview_does_not_confirm_pending_adjacent_node():
    flow = controller()
    flow.step(observe(Scene.STAGES, focused_stage=Stage(13, True, 3)))
    assert flow.step(observe(Scene.MAP_OVERVIEW)).reason == 'map_focus_missing'


def test_conditions_are_read_once_before_preview_and_not_counted_as_result():
    flow = controller()
    formation = observe(Scene.CONFIRM, conditions_available=True)
    assert flow.step(formation).action == Action.OPEN_CONDITIONS
    assert flow.step(formation).action == Action.WAIT
    requirements = tuple(read_rank_requirement(stars, f'{4-stars}位以上でクリア') for stars in (3, 2, 1))
    info = observe(Scene.CONDITIONS, requirements=requirements)
    assert flow.step(info).action == Action.CLOSE_CONDITIONS
    assert flow.step(info).action == Action.WAIT
    assert flow.objectives is not None and flow.objectives.rank_limit(1) == 3
    assert flow.step(formation).action == Action.READY
    assert flow.battles_started == 0


def test_conditions_without_binding_are_only_closed_then_reread():
    flow = controller()
    assert flow.step(Observation(Scene.CONDITIONS)).reason == 'close_unbound_conditions'
    assert flow.objectives is None
    assert flow.step(observe(Scene.CONFIRM, conditions_available=True)).action == Action.OPEN_CONDITIONS


@pytest.mark.parametrize('stars', [1, 2, 3])
def test_starred_formation_does_not_open_conditions_or_start_battle(stars):
    flow = controller(mode='manual')
    assert flow.step(observe(Scene.CONFIRM, stars=stars, conditions_available=True)).action == Action.BACK_TO_STAGES
    assert flow.objectives is None and flow.battles_started == 0


def test_wrong_conditions_identity_does_not_provide_goal():
    flow = controller()
    flow.step(observe(Scene.CONFIRM, conditions_available=True))
    assert flow.step(observe(Scene.CONDITIONS, stage=2)).reason == 'stage_not_confirmed'
    assert flow.objectives is None


def test_condition_goals_never_replace_actual_result_stars():
    flow = controller(mode='manual')
    formation = observe(Scene.CONFIRM, conditions_available=True)
    flow.step(formation)
    flow.step(observe(Scene.CONDITIONS, requirements=(read_rank_requirement(1, '3位以上でクリア'),)))
    assert flow.step(formation).action == Action.START_BATTLE
    assert flow.step(observe(Scene.RESULT, outcome=Outcome.SUCCESS, stars=None)).reason == 'result_stars_unknown'


def test_items_read_once_then_return_to_preview_with_incomplete_inventory():
    flow = controller()
    formation = observe(Scene.CONFIRM, items_available=True)
    assert flow.step(formation).action == Action.OPEN_ITEM_DETAILS
    assert flow.step(formation).action == Action.WAIT
    info = observe(Scene.ITEM_DETAILS, visible_items=())
    assert flow.step(info).action == Action.CLOSE_ITEM_DETAILS
    assert flow.step(info).action == Action.WAIT
    inventory = flow.item_inventory
    assert inventory is not None and inventory.visible_items == () and inventory.complete is False
    assert flow.step(formation).action == Action.READY and flow.battles_started == 0


def test_items_not_detected_does_not_become_empty_complete_inventory():
    flow = controller()
    flow.step(observe(Scene.CONFIRM, items_available=True))
    flow.step(observe(Scene.ITEM_DETAILS))
    assert flow.item_inventory is not None
    assert flow.item_inventory.visible_items is None and flow.item_inventory.complete is False


def test_unbound_items_are_closed_without_attaching_to_configured_character():
    flow = controller()
    assert flow.step(Observation(Scene.ITEM_DETAILS)).reason == 'close_unbound_item_details'
    assert flow.item_inventory is None
    assert flow.step(observe(Scene.CONFIRM, items_available=True)).action == Action.OPEN_ITEM_DETAILS


@pytest.mark.parametrize('stars', [1, 2, 3])
def test_positive_stars_skip_item_inspection(stars):
    flow = controller(mode='manual')
    assert flow.step(observe(Scene.CONFIRM, items_available=True, stars=stars)).action == Action.BACK_TO_STAGES


def test_conditions_and_items_have_independent_ordered_requests():
    flow = controller()
    formation = observe(Scene.CONFIRM, conditions_available=True, items_available=True)
    assert flow.step(formation).action == Action.OPEN_CONDITIONS
    flow.step(observe(Scene.CONDITIONS))
    assert flow.step(formation).action == Action.OPEN_ITEM_DETAILS
    flow.step(observe(Scene.ITEM_DETAILS))
    assert flow.step(formation).action == Action.READY


def test_wrong_item_identity_stops_without_inventory():
    flow = controller()
    flow.step(observe(Scene.CONFIRM, items_available=True))
    assert flow.step(observe(Scene.ITEM_DETAILS, stage=2)).reason == 'stage_not_confirmed'
    assert flow.item_inventory is None


def test_bounded_item_scrolling_waits_for_changed_viewport_without_claiming_complete():
    flow = controller()
    formation = observe(Scene.CONFIRM, items_available=True)
    flow.step(formation)
    item = VisibleItem(ItemReference((), ItemBindingStatus.UNKNOWN_NAME, None, ()), (27, 171, 666, 235))
    first = observe(Scene.ITEM_DETAILS, visible_items=(item,), item_viewport=ItemViewport('a', (item,)))
    second = observe(Scene.ITEM_DETAILS, visible_items=(item,), item_viewport=ItemViewport('b', (item,)))
    assert flow.step(first).action == Action.SCROLL_ITEMS
    assert flow.step(first).action == Action.WAIT
    assert flow.step(second).action == Action.SCROLL_ITEMS
    assert flow.step(second).action == Action.WAIT
    assert flow.step(second).action == Action.CLOSE_ITEM_DETAILS
    assert flow.item_inventory is not None and flow.item_inventory.complete is False
    assert len(flow.item_inventory.viewports) == 2
    assert flow.step(formation).action == Action.READY
    assert flow.battles_started == 0


def test_pending_item_scroll_can_be_cancelled_without_more_actions():
    flow = controller()
    flow.step(observe(Scene.CONFIRM, items_available=True))
    item = VisibleItem(ItemReference((), ItemBindingStatus.UNKNOWN_NAME, None, ()), (27, 171, 666, 235))
    info = observe(Scene.ITEM_DETAILS, item_viewport=ItemViewport('a', (item,)))
    assert flow.step(info).action == Action.SCROLL_ITEMS
    assert flow.step(info, cancelled=True).reason == 'cancelled'
    assert flow.step(info).action == Action.STOP


def test_inventory_stores_candidate_overlap_without_flattening_or_merging_entities():
    def anchor(name, y):
        reads = (ItemNameRead('frame', 'original', name, .99), ItemNameRead('frame', 'grayscale', name, .99))
        return VisibleItem(ItemReference(reads, ItemBindingStatus.RESOLVED, name, (name,), name),
                           (27, y, 666, 200), 'memory')
    flow = controller()
    flow.step(observe(Scene.CONFIRM, items_available=True))
    first = ItemViewport('a', (anchor('x', 400), anchor('y', 612)))
    second = ItemViewport('b', (anchor('x', 200), anchor('y', 412)))
    flow.step(observe(Scene.ITEM_DETAILS, visible_items=first.visible_items, item_viewport=first))
    flow.step(observe(Scene.ITEM_DETAILS, visible_items=second.visible_items, item_viewport=second))
    inventory = flow.item_inventory
    assert inventory is not None and len(inventory.viewports) == 2 and len(inventory.visible_items) == 2
    assert inventory.overlaps[0].status == ItemOverlapStatus.CANDIDATE
    assert inventory.overlaps[0].same_instances_confirmed is False and inventory.complete is False
