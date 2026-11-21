"""变量卡牌待结→回合证据→原反馈的领域协议，无设备访问。"""
from dataclasses import replace
import pytest
from kaa.idol_road.battle_inputs import BattleInputs
from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame,current_absolute_turn
from kaa.idol_road.battle_log import BattleLogBinding,LogEntryEvidence,LogRequest,LogReturn,LogViewport,LogScrollEvidence
from kaa.idol_road.card_probe import pending_card_permit,assess_card_log,card_feedback_ready
from kaa.idol_road.round_anchor import InitialRoundTracker
from kaa.idol_road.log_turns import LogTurnEvidence,confirm_visible_log_turn
from kaa.idol_road.rules import RuleTextRead,RuleTextLine
from kaa.idol_road.input_policy import InputBoundPolicy
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from kaa.tasks.produce.new.play_cards.planner.feedback import PendingPlay,evaluate_play_feedback,PlayFeedbackStatus

SESSION=BattleObservationSession('synthetic-card-anchor','ttmr',20,20,'confirmed_formation_start_request','variable_budget')


def raw(name,remaining=20,hp=50):
    value=bind_battle_frame(SESSION,BattleFrameValues(name,True,remaining,hp))
    assert value is not None
    return value


def log(turn,remaining):
    def reads(text):
        return tuple(RuleTextRead('log',kind,(RuleTextLine(text,.99),)) for kind in ('original','grayscale'))
    row=confirm_visible_log_turn(LogTurnEvidence('log',(168,436,98,31),.99,(580,440,88,22),.99,
        (138,426,133,48),reads(f'{turn}ターン目'),(63,419,39,47),reads(str(remaining))))
    return LogViewport('log','b'*64,(342,1140,36,38),(row,),
        LogScrollEvidence('log',(684,128,3,1007),(684,717,3,418),1.,589),(1.,1.),True)


def cycle(turn=1,remaining=20):
    now=[0.]
    inputs=BattleInputs(SESSION,clock=lambda:now[0])
    rounds=InitialRoundTracker(SESSION)
    initial=log(1,20)
    returned=LogReturn(LogRequest('initial',SESSION,'open',20,50,'a'*64),'log','seed',True,True,
        initial.headers[0],input_sequence_consistent=True)
    assert rounds.accept(raw('seed'),returned,initial,inputs.snapshot())
    seed=rounds.bind(raw('seed'),inputs.snapshot())
    before=BattleState(20,50,0,hand_size=2,hand_card_ids=('card','other'),absolute_turn_observation=seed.absolute_turn)
    inputs.begin('card',seed,'card')
    inputs.bind_card_state(before,'card')
    rounds.invalidate('battle_input_requested')
    permit=pending_card_permit(SESSION,inputs.snapshot(),now=0)
    binding=BattleLogBinding(clock=lambda:now[0])
    binding.update_inputs(1,True,card_probe=permit)
    def entry(name):
        return LogEntryEvidence(name,(449,1179,30,39),.99,'d'*64)
    for name in ('a','b'):
        binding.observe_hud(raw(name,remaining,48),entry(name))
    binding.open(raw('b',remaining,48),entry('b'),input_sequence=1,card_probe=permit)
    view=log(turn,remaining)
    binding.close(binding.inspect(view,SESSION))
    back=raw('back',remaining,48)
    returned=binding.observe_hud(back,entry('back'))
    resolution=assess_card_log(permit,returned,view,back,inputs.snapshot(),now=0)
    return rounds,inputs,binding,now,before,back,returned,view,resolution


@pytest.mark.parametrize('turn,remaining,changed',[(1,20,True),(1,20,False),
    (2,19,False),(2,20,False),(2,21,False)])
def test_variable_feedback_only_releases_input_after_original_feedback(turn,remaining,changed):
    from kaa.idol_road.page import IdolRoadPage
    rounds,inputs,binding,_,before,back,returned,view,resolution=cycle(turn,remaining)
    assert not card_feedback_ready(SESSION,inputs.snapshot(),back,resolution,now=0)
    assert rounds.accept_card(back,returned,view,resolution,inputs.snapshot(),now=0)
    bound=rounds.bind(back,inputs.snapshot())
    assert current_absolute_turn(bound).turn==turn
    assert bound.observation.turn_source=='owned_card_log_return'
    assert card_feedback_ready(SESSION,inputs.snapshot(),bound,resolution,now=0)
    assert inputs.pending is not None and resolution.play_confirmed is False
    page=object.__new__(IdolRoadPage)
    page._battle_session=SESSION
    page._battle_inputs=inputs
    page._round_tracker=rounds
    page._log_binding=binding
    with pytest.raises(ValueError):
        page.begin_battle_input(bound,'card','other')
    after=replace(before,remaining_turns=remaining,hp=48,hp_known=changed,
        hand_size=1 if changed else 2,hand_card_ids=('other',) if changed else ('card','other'),
        absolute_turn_observation=bound.absolute_turn)
    pending=PendingPlay('card',before,0,0)
    feedback=evaluate_play_feedback(pending,after)
    policy=InputBoundPolicy(inputs,lambda:bound)
    if not changed and turn==1:
        assert feedback.status is PlayFeedbackStatus.UNCONFIRMED
        with pytest.raises(ValueError):
            policy.feedback(pending,feedback,after)
        assert inputs.pending is not None
        return
    assert feedback.status is (PlayFeedbackStatus.CONFIRMED if changed else PlayFeedbackStatus.UNCONFIRMED)
    policy.feedback(pending,feedback,after)
    assert inputs.pending is None
    assert current_absolute_turn(rounds.bind(bound,inputs.snapshot()))==bound.absolute_turn
    page.begin_battle_input(bound,'card','other')
    assert inputs.sequence==2 and inputs.pending.card_id=='other'
    assert rounds.state.anchor is None
    assert not rounds.accept_card(back,returned,view,resolution,inputs.snapshot(),now=0)


@pytest.mark.parametrize('change',['expired','wrong_sequence','wrong_pending','return_hp','return_frame','weak_return',
    'log_frame','header_gap','forged_resolution'])
def test_card_anchor_rejects_invalid_or_stale_return(change):
    rounds,inputs,_,now,_,back,returned,view,resolution=cycle(3 if change=='header_gap' else 1)
    state=inputs.snapshot()
    if change=='expired':
        now[0]=20
        state=inputs.snapshot()
    elif change=='wrong_sequence':
        state=replace(state,sequence=2)
    elif change=='wrong_pending':
        state=replace(state,pending=replace(state.pending,card_id='other'))
    elif change=='return_hp':
        back=raw('back',20,47)
    elif change=='return_frame':
        back=raw('later',20,48)
    elif change=='weak_return':
        returned=replace(returned,hud_unchanged=False)
    elif change=='log_frame':
        view=replace(view,frame_id='other')
    elif change=='forged_resolution':
        resolution=replace(resolution,turn=2)
    assert not rounds.accept_card(back,returned,view,resolution,state,now=now[0])
    assert rounds.bind(back,state).absolute_turn is None and inputs.pending is not None


@pytest.mark.parametrize('change',['new_frame','same_frame_different_hp','changed_epoch','expired','pending_changed'])
def test_pending_card_anchor_is_exact_frame_and_exact_input_only(change):
    rounds,inputs,_,now,_,back,returned,view,resolution=cycle()
    assert rounds.accept_card(back,returned,view,resolution,inputs.snapshot(),now=0)
    bound=rounds.bind(back,inputs.snapshot())
    state=inputs.snapshot()
    if change=='new_frame':
        bound=raw('later',20,48)
    elif change=='same_frame_different_hp':
        bound=raw('back',20,47)
    elif change=='changed_epoch':
        state=replace(state,sequence=2)
    elif change=='expired':
        now[0]=20
        state=inputs.snapshot()
    else:
        state=replace(state,pending=replace(state.pending,card_id='other'))
    assert rounds.bind(bound,state).absolute_turn is None
    assert rounds.state.anchor is None


def test_variable_before_and_feedback_require_exact_absolute_frame_not_just_round():
    _,inputs,_,_,before,back,_,_,_=cycle()
    assert not inputs._matches(inputs.pending.frame,replace(before,absolute_turn_observation=replace(before.absolute_turn_observation,frame_id='other')))
    assert not inputs._matches(back,replace(before,absolute_turn_observation=None))
