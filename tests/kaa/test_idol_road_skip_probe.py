"""只读SKIP结算的合成领域测试；不调用设备，不证明实机追加回合。"""
from dataclasses import replace
import pytest
from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame,current_absolute_turn
from kaa.idol_road.battle_inputs import BattleInputs,BattleInputState,PendingBattleInput
from kaa.idol_road.battle_log import BattleLogBinding,LogEntryEvidence,LogRequest,LogReturn,LogViewport,LogScrollEvidence,log_inspection_action
from kaa.idol_road.log_turns import LogTurnEvidence,confirm_visible_log_turn
from kaa.idol_road.rules import RuleTextRead,RuleTextLine
from kaa.idol_road.round_anchor import InitialRoundTracker
from kaa.idol_road.skip_probe import pending_skip_permit,assess_skip_log

SESSION=BattleObservationSession('synthetic-test','ttmr',20,20,'confirmed_formation_start_request','variable_budget')


def frame(name,remaining=20,hp=50):
    value=bind_battle_frame(SESSION,BattleFrameValues(name,True,remaining,hp))
    assert value is not None
    return value


def view(turn,remaining):
    def reads(text):
        return tuple(RuleTextRead('log',kind,(RuleTextLine(text,.99),)) for kind in ('original','grayscale'))
    row=confirm_visible_log_turn(LogTurnEvidence('log',(168,436,98,31),.99,(580,440,88,22),.99,
        (138,426,133,48),reads(f'{turn}ターン目'),(63,419,39,47),reads(str(remaining))))
    return LogViewport('log','b'*64,(342,1140,36,38),(row,),
        LogScrollEvidence('log',(684,128,3,1007),(684,717,3,418),1.,589),(1.,1.),True)


def seeded():
    rounds=InitialRoundTracker(SESSION)
    initial=view(1,20)
    returned=LogReturn(LogRequest('initial',SESSION,'open-initial',20,50,'a'*64),'log','seed',True,True,
                       initial.headers[0],input_sequence_consistent=True)
    inputs=BattleInputs(SESSION,clock=lambda:0)
    assert rounds.accept(frame('seed'),returned,initial,inputs.snapshot())
    return rounds,rounds.bind(frame('seed'),inputs.snapshot()),inputs


def entry(name):
    return LogEntryEvidence(name,(449,1179,30,39),.99,'d'*64)


def opened(remaining=19):
    rounds,seed,inputs=seeded()
    inputs.begin('skip',seed)
    permit=pending_skip_permit(SESSION,inputs.snapshot(),now=0)
    assert permit is not None
    now=[0.]
    binding=BattleLogBinding(clock=lambda:now[0])
    binding.update_inputs(1,True,skip_probe=permit)
    for name in ('a','b'):
        binding.observe_hud(frame(name,remaining,49),entry(name))
    assert binding.can_open(frame('b',remaining,49),entry('b'))
    binding.open(frame('b',remaining,49),entry('b'),input_sequence=1,skip_probe=permit)
    return rounds,inputs,permit,binding,now


def test_owned_skip_freezes_previous_round_and_current_round_play_does_not_overwrite_it():
    from kaa.tasks.produce.new.play_cards.planner.state import BattleState
    from kaa.tasks.produce.new.play_cards.planner.feedback import PendingPlay,PlayFeedback,PlayFeedbackStatus
    rounds,seed,inputs=seeded()
    def state_for(value):
        return BattleState(value.values.remaining_turns,value.observation.hp,0,
                           absolute_turn_observation=current_absolute_turn(value))
    def same_turn(value,name):
        return replace(value,values=replace(value.values,frame_id=name),
            observation=replace(value.observation,frame_id=name),
            absolute_turn=replace(value.absolute_turn,frame_id=name))
    before=state_for(seed)
    inputs.begin('card',seed,'mental')
    inputs.bind_card_state(before,'mental',category='ProduceCardCategory_MentalSkill')
    returned_card=same_turn(seed,'after-mental')
    confirmed=PlayFeedback(PlayFeedbackStatus.CONFIRMED,'card_left_hand')
    assert inputs.finish_card(PendingPlay('mental',before,0,0),confirmed,state_for(returned_card),returned_card)
    inputs.begin('skip',returned_card)
    permit=pending_skip_permit(SESSION,inputs.snapshot(),now=0)
    binding=BattleLogBinding(clock=lambda:0)
    binding.update_inputs(2,True,skip_probe=permit)
    for name in ('a','b'):
        binding.observe_hud(frame(name,19,49),entry(name))
    binding.open(frame('b',19,49),entry('b'),input_sequence=2,skip_probe=permit)
    log=view(2,19)
    binding.close(binding.inspect(log,SESSION))
    returned=binding.observe_hud(frame('back',19,49),entry('back'))
    resolution=inputs.finish_skip_log(returned,log,frame('back',19,49))
    previous=inputs.previous_round_card
    assert previous and previous.last_play.card_id=='mental' and previous.current_turn==2
    assert previous.boundary_input_sequence==2
    assert rounds.accept_skip(frame('back',19,49),returned,resolution,inputs.snapshot())
    current=rounds.bind(frame('back',19,49),inputs.snapshot())
    before=state_for(current)
    inputs.begin('card',current,'active')
    inputs.bind_card_state(before,'active',category='ProduceCardCategory_ActiveSkill')
    assert inputs.previous_round_card is None
    after=same_turn(current,'after-active')
    assert inputs.finish_card(PendingPlay('active',before,0,0),confirmed,state_for(after),after)
    assert inputs.last_owned_card_play.card_id=='active'
    assert inputs.previous_round_card==previous
    from types import SimpleNamespace
    from kaa.idol_road.item_recovery_rule import ItemRecoveryDeclaration,check_item_recovery_categories
    rule=ItemRecoveryDeclaration('item','ProduceCardCategory_ActiveSkill','ProduceCardCategory_MentalSkill',6,
        cross_round_semantics='previous_turn_last_card',timing_source='user_confirmed_previous_turn_last_card')
    candidate=SimpleNamespace(_id='candidate',category='ProduceCardCategory_ActiveSkill')
    state=replace(state_for(after),hand_card_ids=('candidate',))
    check=check_item_recovery_categories(rule,inputs,after,state,candidate)
    assert check.verdict=='satisfied' and check.previous_round.last_play.card_id=='mental'
    assert not check.recovery_guaranteed and check.scope=='category_clause_only'
    mismatch=check_item_recovery_categories(replace(rule,previous_category='ProduceCardCategory_ActiveSkill'),
        inputs,after,state,candidate)
    assert mismatch.verdict=='not_satisfied'
    assert check_item_recovery_categories(replace(rule,timing_source='unverified'),inputs,after,state,candidate).verdict=='unknown'
    assert check_item_recovery_categories(rule,inputs,after,replace(state,hand_card_ids=()),candidate).verdict=='unknown'
    inputs.begin('card',after,'uncertain')
    inputs.bind_card_state(state,'uncertain',category='ProduceCardCategory_ActiveSkill')
    assert check_item_recovery_categories(rule,inputs,after,state,candidate).verdict=='unknown'
    advanced=replace(after,values=replace(after.values,frame_id='advanced',remaining_turns=18),
        observation=replace(after.observation,frame_id='advanced',turn=3),
        absolute_turn=replace(after.absolute_turn,frame_id='advanced',turn=3,remaining_turns=18))
    unknown=PlayFeedback(PlayFeedbackStatus.UNCONFIRMED,'turn_advanced_without_play_observation')
    assert inputs.finish_card(PendingPlay('uncertain',state,0,0),unknown,state_for(advanced),advanced)
    assert inputs.previous_round_card is None and inputs.last_owned_card_play is None


@pytest.mark.parametrize('remaining',[19,20,21])
def test_next_title_resolves_skip_using_absolute_round_even_when_remaining_does_not_decrease(remaining):
    rounds,inputs,permit,binding,_=opened(remaining)
    log=view(2,remaining)
    inspected=binding.inspect(log,SESSION)
    binding.close(inspected)
    returned=binding.observe_hud(frame('back',remaining,49),entry('back'))
    assert returned is not None and returned.input_sequence_consistent
    resolution=inputs.finish_skip_log(returned,log,frame('back',remaining,49))
    assert resolution.status=='next_round_confirmed' and inputs.pending is None
    assert rounds.accept_skip(frame('back',remaining,49),returned,resolution,inputs.snapshot())
    bound=rounds.bind(frame('back',remaining,49),inputs.snapshot())
    assert current_absolute_turn(bound).turn==2 and current_absolute_turn(bound).remaining_turns==remaining
    assert bound.observation.turn_source=='owned_skip_log_return'
    assert rounds.bind(frame('later',remaining,49),inputs.snapshot()).absolute_turn is None


@pytest.mark.parametrize('turn,status',[(1,'same_or_older_header'),(3,'header_gap')])
def test_old_or_skipped_title_does_not_settle_or_predict_a_round(turn,status):
    rounds,inputs,_,binding,_=opened(20)
    log=view(turn,20)
    binding.close(binding.inspect(log,SESSION))
    returned=binding.observe_hud(frame('back',20,49),entry('back'))
    resolution=inputs.finish_skip_log(returned,log,frame('back',20,49))
    assert resolution.status==status and inputs.pending is not None
    assert not rounds.accept_skip(frame('back',20,49),returned,resolution,inputs.snapshot())


def test_window_never_authorizes_another_input_or_an_unanchored_card():
    _,seed,inputs=seeded()
    with pytest.raises(ValueError):
        inputs.begin('card',frame('unanchored'),'card')
    assert inputs.sequence==0
    inputs.begin('skip',seed)
    for kind,card_id in (('skip',None),('card','card')):
        with pytest.raises(ValueError):
            inputs.begin(kind,seed,card_id)
    assert inputs.sequence==1


def test_expired_window_closes_without_resetting_skip_deadline():
    _,inputs,permit,binding,now=opened()
    log=view(2,19)
    binding.inspect(log,SESSION)
    now[0]=20.
    binding.update_inputs(1,True,skip_probe=permit)
    inspected=binding.inspect(log,SESSION)
    assert inspected.expired and inspected.tail_header is None
    assert log_inspection_action(inspected,SESSION.battle_id)=='close'
    assert inputs.pending.started==0


@pytest.mark.parametrize('kind',['sequence','card','historical','unanchored','expired'])
def test_invalid_pending_metadata_cannot_create_read_only_permit(kind):
    _,seed,inputs=seeded()
    inputs.begin('skip',seed)
    snapshot=inputs.snapshot()
    if kind=='sequence':
        snapshot=replace(snapshot,sequence=2)
    elif kind=='card':
        snapshot=replace(snapshot,pending=replace(snapshot.pending,kind='card',card_id='card'))
    elif kind=='historical':
        bad=replace(seed,session=replace(SESSION,identity_source='historical_capture_metadata'))
        snapshot=replace(snapshot,pending=replace(snapshot.pending,frame=bad))
    elif kind=='unanchored':
        snapshot=replace(snapshot,pending=replace(snapshot.pending,frame=frame('raw')))
    else:
        snapshot=replace(snapshot,expired=True)
    assert pending_skip_permit(SESSION,snapshot,now=0) is None


@pytest.mark.parametrize('change',[{'hud_unchanged':False},{'input_sequence_consistent':False},
    {'remaining_consistent':False},{'return_frame_id':'old'},{'tail_header':None}])
def test_incomplete_return_keeps_pending_skip(change):
    _,inputs,permit,binding,_=opened()
    log=view(2,19)
    binding.close(binding.inspect(log,SESSION))
    returned=binding.observe_hud(frame('back',19,49),entry('back'))
    outcome=assess_skip_log(permit,replace(returned,**change),log,frame('back',19,49),inputs.snapshot(),now=0)
    assert outcome.status=='unconfirmed' and inputs.pending is not None


def test_input_change_revokes_probe_candidate_and_old_permit_cannot_restore_it():
    _,_,permit,binding,_=opened()
    binding.update_inputs(2,True)
    binding.update_inputs(1,True,skip_probe=permit)
    inspected=binding.inspect(view(2,19),SESSION)
    assert inspected.inputs_invalidated and inspected.tail_header is None


def test_wrong_battle_permit_cannot_be_used_for_current_hud():
    _,_,permit,_,_=opened()
    binding=BattleLogBinding(clock=lambda:0)
    binding.update_inputs(1,True,skip_probe=permit)
    for name in ('a','b'):
        other=bind_battle_frame(replace(SESSION,battle_id='other'),BattleFrameValues(name,True,20,50))
        binding.observe_hud(other,entry(name))
    assert not binding.can_open(other,entry('b'))


def test_page_rejects_old_variable_frame_after_anchor_invalidates():
    from kaa.idol_road.page import IdolRoadPage
    rounds,seed,inputs=seeded()
    page=object.__new__(IdolRoadPage)
    page._battle_session=SESSION
    page._battle_inputs=inputs
    page._round_tracker=rounds
    page._log_binding=BattleLogBinding(clock=lambda:0)
    rounds.invalidate('new_page_observation')
    with pytest.raises(ValueError):
        page.begin_battle_input(seed,'skip')
    assert inputs.sequence==0 and inputs.pending is None


def test_standalone_ai_trial_rejects_variable_budget_even_without_rule_argument():
    from kaa.idol_road.planner_trial import PlannerTrialSpec,UnsupportedTurnBudget
    from kaa.idol_road.planner_adapter import run_planner_trial
    with pytest.raises(UnsupportedTurnBudget):
        run_planner_trial(PlannerTrialSpec('ttmr',20,3,20),None,SESSION,deadline=100,observe=lambda *args:None)


def test_controller_unknown_frames_do_not_renew_pending_skip_deadline():
    from kaa.config.idol_road import IdolRoadConfig
    from kaa.idol_road.controller import IdolRoadController
    from kaa.idol_road.models import Observation,Scene,Action
    _,seed,inputs=seeded()
    inputs.begin('skip',seed)
    now=[0.]
    controller=IdolRoadController(IdolRoadConfig(characters=['ttmr'],mode='manual'),clock=lambda:now[0])
    controller._stage=20
    controller._requested=True
    pending=Observation(Scene.BATTLE,'ttmr',20,battle_state=frame('after').observation,battle_input_state=inputs.snapshot())
    assert controller.step(pending).action==Action.WAIT
    now[0]=19.
    assert controller.step(Observation(Scene.UNKNOWN)).action==Action.WAIT
    now[0]=20.
    assert controller.step(Observation(Scene.UNKNOWN)).reason=='battle_input_unresolved'
