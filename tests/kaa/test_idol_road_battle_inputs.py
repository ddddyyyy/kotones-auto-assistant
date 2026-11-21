"""输入生命周期纯规则；不调用设备、不伪装成实战回合。"""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from kaa.idol_road.battle_inputs import BattleInputs
from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame
from kaa.idol_road.battle_log import BattleLogBinding,LogEntryEvidence
from kaa.idol_road.input_policy import InputBoundPolicy
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from kaa.tasks.produce.new.play_cards.planner.feedback import PendingPlay,PlayFeedback,PlayFeedbackStatus

SESSION=BattleObservationSession('battle','ttmr',20,20,'confirmed_formation_start_request')


def frame(name='before',remaining=5,hp=48,session=SESSION):
    result=bind_battle_frame(session,BattleFrameValues(name,True,remaining,hp))
    assert result is not None
    return result


def state(remaining=5):
    return BattleState(remaining,48,0)


CONFIRMED=PlayFeedback(PlayFeedbackStatus.CONFIRMED,'card_left_hand')
UNKNOWN=PlayFeedback(PlayFeedbackStatus.UNCONFIRMED,'insufficient_evidence')


def started_card(clock=lambda:0):
    inputs=BattleInputs(SESSION,clock=clock)
    inputs.begin('card',frame(),'card')
    inputs.bind_card_state(state(),'card')
    return inputs


def test_pending_input_blocks_all_further_inputs_without_renewing_sequence():
    inputs=started_card()
    old=inputs.snapshot()
    for kind,card_id in (('card','next'),('skip',None)):
        with pytest.raises(ValueError):
            inputs.begin(kind,frame('new'),card_id)
        assert inputs.snapshot()==old
        assert inputs.requested_route==('16:card:card',)


def test_bound_feedback_releases_exact_request_but_does_not_reuse_input_sequence():
    inputs=started_card()
    assert inputs.finish_card(PendingPlay('card',state(),0,0),CONFIRMED,state(),frame('after'))
    assert inputs.pending is None and inputs.sequence==1
    inputs.begin('skip',frame('after'))
    assert inputs.sequence==2


def test_owned_card_history_retains_exact_request_and_return_without_claiming_game_condition():
    inputs=BattleInputs(SESSION,clock=lambda:0)
    inputs.begin('card',frame(),'mental')
    inputs.bind_card_state(state(),'mental',category='ProduceCardCategory_MentalSkill')
    assert inputs.last_owned_card_play is None
    assert inputs.finish_card(PendingPlay('mental',state(),0,0),CONFIRMED,state(),frame('after'))
    record=inputs.last_owned_card_play
    assert record and (record.battle_id,record.input_sequence,record.card_id,record.category)==(
        SESSION.battle_id,1,'mental','ProduceCardCategory_MentalSkill')
    assert (record.request_frame_id,record.return_frame_id)==('before','after')
    assert record.previous_card_condition_confirmed is False
    inputs.begin('skip',frame('after'))
    assert inputs.last_owned_card_play is None
    inputs.observe_hud(frame('next-round',4))
    assert inputs.last_owned_card_play==record
    # Historical evidence survives SKIP, but is still not a game-state claim.
    assert inputs.last_owned_card_play.previous_card_condition_confirmed is False
    inputs.begin('card',frame('next-round',4),'next')
    assert inputs.last_owned_card_play is None
    assert BattleInputs(SESSION).last_owned_card_play is None


@pytest.mark.parametrize('kind',['unknown_category','resource_feedback','next_round','late','wrong_card'])
def test_weak_or_unrelated_card_feedback_does_not_create_category_history(kind):
    now=[0.]
    inputs=BattleInputs(SESSION,clock=lambda:now[0])
    inputs.begin('card',frame(),'card')
    inputs.bind_card_state(state(),'card',category=None if kind=='unknown_category' else 'ProduceCardCategory_MentalSkill')
    if kind=='late': now[0]=20
    pending=PendingPlay('other' if kind=='wrong_card' else 'card',state(),0,0)
    feedback=(UNKNOWN if kind=='next_round' else PlayFeedback(PlayFeedbackStatus.CONFIRMED,'resources_changed')
              if kind=='resource_feedback' else CONFIRMED)
    remaining=4 if kind=='next_round' else 5
    inputs.finish_card(pending,feedback,state(remaining),frame('after',remaining))
    assert inputs.last_owned_card_play is None


@pytest.mark.parametrize('kind',['wrong_card','wrong_before','old_frame','wrong_battle','unknown','unbound_state','preview'])
def test_unrelated_or_unknown_feedback_does_not_release_request(kind):
    inputs=started_card()
    pending=PendingPlay('other' if kind=='wrong_card' else 'card',state(4) if kind=='wrong_before' else state(),0,0)
    after=frame('before' if kind=='old_frame' else 'after',session=replace(SESSION,battle_id='other') if kind=='wrong_battle' else SESSION)
    observed=replace(state(),remaining_turns_known=False) if kind=='unbound_state' else state()
    if kind=='preview':
        after=replace(after,values=replace(after.values,resource_state='selection_preview'))
    assert not inputs.finish_card(pending,UNKNOWN if kind=='unknown' else CONFIRMED,observed,after)
    assert inputs.pending is not None and inputs.sequence==1


def test_new_round_can_end_old_card_wait_without_turning_unknown_feedback_into_success():
    inputs=started_card()
    assert inputs.finish_card(PendingPlay('card',state(),0,0),UNKNOWN,state(4),frame('after',4))
    assert inputs.pending is None and UNKNOWN.status==PlayFeedbackStatus.UNCONFIRMED


def test_hud_changes_alone_cannot_settle_card_or_skip_without_a_new_round():
    inputs=started_card()
    inputs.observe_hud(frame('changed',4,hp=40))
    assert inputs.pending is not None
    skip=BattleInputs(SESSION)
    skip.begin('skip',frame())
    skip.observe_hud(frame('changed',5,hp=40))
    assert skip.pending is not None
    skip.observe_hud(frame('next',4,hp=40))
    assert skip.pending is None and skip.sequence==1


def test_timeout_never_clears_or_renews_pending_input():
    now=[0.]
    inputs=started_card(clock=lambda:now[0])
    now[0]=20
    assert inputs.snapshot().expired
    assert not inputs.finish_card(PendingPlay('card',state(),0,0),CONFIRMED,state(),frame('late'))
    with pytest.raises(ValueError):
        inputs.begin('card',frame('late'),'card')
    assert inputs.pending.started==0 and inputs.sequence==1


@pytest.mark.parametrize('change',[{'session':replace(SESSION,battle_id='other')},
    {'session':replace(SESSION,identity_source='historical_capture_metadata')},
    {'session':replace(SESSION,turn_budget_mode='variable_budget')}])
def test_other_or_uncalibrated_session_cannot_authorize_inputs(change):
    inputs=BattleInputs(SESSION)
    with pytest.raises(ValueError):
        inputs.begin('skip',frame(**change))
    assert inputs.sequence==0


def test_manually_filled_variable_turn_does_not_bypass_missing_producer():
    session=replace(SESSION,turn_budget_mode='variable_budget')
    current=frame(session=session)
    current=replace(current,observation=replace(current.observation,turn=16))
    with pytest.raises(ValueError):
        BattleInputs(session).begin('skip',current)


def test_state_binding_requires_exact_card_and_current_turn_only_once():
    inputs=BattleInputs(SESSION)
    inputs.begin('card',frame(),'card')
    for observed,card_id in ((state(4),'card'),(state(),'other'),(replace(state(),projected_effect_values=()),'card')):
        with pytest.raises(ValueError):
            inputs.bind_card_state(observed,card_id)
    inputs.bind_card_state(state(),'card')
    with pytest.raises(ValueError):
        inputs.bind_card_state(state(),'card')


def test_input_policy_is_active_without_field_goals_and_unknown_feedback_stops():
    inputs=BattleInputs(SESSION)
    latest=[frame('after')]
    policy=InputBoundPolicy(inputs,lambda:latest[0])
    assert not policy.select(state(),()).preferred_indices
    inputs.begin('card',frame(),'card')
    policy.committed(state(),SimpleNamespace(_id='card'))
    with pytest.raises(ValueError):
        policy.feedback(PendingPlay('card',state(),0,0),UNKNOWN,state())
    assert inputs.pending is not None
    policy.feedback(PendingPlay('card',state(),0,0),CONFIRMED,state())
    assert inputs.pending is None


@pytest.mark.parametrize('kind',['missing_frame','other_battle','probe','projected','wrong_remaining','wrong_hp','invalid_hp_literal','pending','expired'])
def test_selection_preflight_stops_before_delegate_without_mutating_the_ledger(kind):
    current=frame()
    observed=state()
    inputs=BattleInputs(SESSION,clock=lambda:0)
    if kind=='missing_frame':current=None
    elif kind=='other_battle':current=frame(session=replace(SESSION,battle_id='other'))
    elif kind=='probe':current=replace(current,values=replace(current.values,detail_mode='core_only'))
    elif kind=='projected':observed=replace(observed,projected_effect_values=())
    elif kind=='wrong_remaining':observed=state(4)
    elif kind=='wrong_hp':observed=replace(observed,hp=47)
    elif kind=='invalid_hp_literal':
        current=frame(hp=1)
        observed=replace(observed,hp=True)
    elif kind in ('pending','expired'):
        inputs.begin('skip',current)
        if kind=='expired':inputs.clock=lambda:20
    before=inputs.snapshot()
    delegate=SimpleNamespace(select=lambda *args:pytest.fail('preflight must precede evaluation'))
    result=InputBoundPolicy(inputs,lambda:current,delegate).select(observed,())
    assert result.stop and result.allowed_indices==() and not result.end_turn
    assert result.reason=='battle_input:selection_state_unbound_or_pending'
    assert inputs.snapshot()==before


def test_variable_selection_requires_the_exact_current_observation_frame():
    from kaa.battle_turn import AbsoluteTurnObservation
    session=replace(SESSION,turn_budget_mode='variable_budget')
    current=frame(session=session)
    absolute=AbsoluteTurnObservation(session.battle_id,'before',16,5)
    current=replace(current,absolute_turn=absolute,observation=replace(current.observation,
        turn=16,turn_source='owned_card_log_return'))
    observed=replace(state(),absolute_turn_observation=absolute)
    inputs=BattleInputs(session)
    assert inputs.can_select(current,observed)
    stale=replace(observed,absolute_turn_observation=replace(absolute,frame_id='old-same-turn'))
    assert not inputs.can_select(current,stale)
    assert not inputs.can_select(current,replace(observed,absolute_turn_observation=None))
    assert inputs.sequence==0 and inputs.pending is None


def test_log_request_preserves_input_sequence_for_future_anchor_rechecks():
    binding=BattleLogBinding()
    binding.update_inputs(7,False)
    for name in ('one','two'):
        binding.observe_hud(frame(name),LogEntryEvidence(name,(449,1179,30,39),.99,'a'*64))
    binding.open(frame('two'),LogEntryEvidence('two',(449,1179,30,39),.99,'a'*64),input_sequence=7)
    assert binding.request.input_sequence==7


def page_with_ready_log():
    from kaa.idol_road.page import IdolRoadPage
    page=object.__new__(IdolRoadPage)
    page._battle_session=SESSION
    page._battle_inputs=BattleInputs(SESSION)
    page._log_binding=BattleLogBinding()
    for name in ('one','two'):
        page._log_binding.observe_hud(frame(name),LogEntryEvidence(name,(449,1179,30,39),.99,'a'*64))
    return page


def test_page_input_invalidates_ready_log_before_any_device_operation():
    page=page_with_ready_log()
    latest=frame('two')
    evidence=LogEntryEvidence('two',(449,1179,30,39),.99,'a'*64)
    assert page._log_binding.can_open(latest,evidence)
    page.begin_battle_input(latest,'card','card')
    assert page.battle_inputs.sequence==1 and page.battle_inputs.pending is not None
    assert not page._log_binding.can_open(latest,evidence)


def test_page_open_log_prevents_new_input_and_preserves_request():
    page=page_with_ready_log()
    page._log_binding.open(frame('two'),LogEntryEvidence('two',(449,1179,30,39),.99,'a'*64))
    request=page._log_binding.request
    with pytest.raises(ValueError):
        page.begin_battle_input(frame('two'),'skip')
    assert page.battle_inputs.sequence==0 and page._log_binding.request==request
