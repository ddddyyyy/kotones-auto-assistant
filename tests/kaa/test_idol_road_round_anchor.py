"""初始返回帧锚点纯规则，合成所有权不代表实机生产闭环。"""
from dataclasses import replace
import pytest
from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame,current_absolute_turn
from kaa.idol_road.battle_log import LogRequest,LogReturn,LogViewport,LogScrollEvidence
from kaa.idol_road.battle_inputs import BattleInputState,BattleInputs
from kaa.idol_road.log_turns import LogTurnEvidence,confirm_visible_log_turn
from kaa.idol_road.rules import RuleTextRead,RuleTextLine
from kaa.idol_road.round_anchor import InitialRoundTracker,confirm_initial_log_anchor
from kaa.idol_road.planner_context import IdolRoadBattleContext
from kaa.idol_road.planner_adapter import TrialFrameGuard
from kaa.idol_road.page import PageSnapshot
from kaa.idol_road.models import Observation,Scene
from kaa.idol_road.battle_log import LogEntryEvidence
from kaa.idol_road.battle_trace import BattleTrace

SESSION=BattleObservationSession('battle','ttmr',20,20,'confirmed_formation_start_request','variable_budget')


def frame(name='back'):
    result=bind_battle_frame(SESSION,BattleFrameValues(name,True,20,50))
    assert result is not None
    return result


def records():
    def reads(text):
        return tuple(RuleTextRead('log',kind,(RuleTextLine(text,.99),)) for kind in ('original','grayscale'))
    row=confirm_visible_log_turn(LogTurnEvidence('log',(168,436,98,31),.99,(580,440,88,22),.99,
        (138,426,133,48),reads('1ターン目'),(63,419,39,47),reads('20')))
    view=LogViewport('log','b'*64,(342,1140,36,38),(row,),
        LogScrollEvidence('log',(684,128,3,1007),(684,717,3,418),1.,589),(1.,1.),True)
    request=LogRequest('request',SESSION,'open',20,50,'a'*64)
    returned=LogReturn(request,'log','back',True,True,row,input_sequence_consistent=True)
    inputs=BattleInputState('battle',0,None,False)
    return returned,view,inputs


def anchored():
    tracker=InitialRoundTracker(SESSION)
    returned,view,inputs=records()
    assert tracker.accept(frame(),returned,view,inputs)
    return tracker,tracker.bind(frame(),inputs),inputs


def test_owned_initial_return_produces_actual_turn_only_on_that_exact_frame():
    tracker,bound,inputs=anchored()
    actual=current_absolute_turn(bound)
    assert actual is not None and actual.turn==1 and actual.remaining_turns==20
    assert bound.observation.turn_source=='owned_initial_log_return'
    assert IdolRoadBattleContext(None,bound).fetch_absolute_turn_observation()==actual
    assert tracker.state.anchor.scope=='exact_return_frame_only'
    guard=TrialFrameGuard(SESSION,100,clock=lambda:0)
    assert guard.check(PageSnapshot(Observation(Scene.BATTLE),battle_frame=bound))=='continue'
    later=tracker.bind(frame('later'),inputs)
    assert later.absolute_turn is None and later.observation.turn is None
    assert tracker.state.anchor is None
    assert guard.check(PageSnapshot(Observation(Scene.BATTLE),battle_frame=later))=='wait'


def test_initial_log_return_cannot_anchor_a_core_only_packet():
    returned,view,inputs=records()
    partial=replace(frame(),values=replace(frame().values,detail_mode='core_only'))
    assert confirm_initial_log_anchor(SESSION,partial,returned,view,inputs) is None


@pytest.mark.parametrize('change',[
    {'hud_unchanged':False},{'remaining_consistent':False},{'input_sequence_consistent':False},
    {'return_frame_id':'old'},{'log_frame_id':'old'},{'tail_header':None}])
def test_incomplete_or_cross_frame_return_cannot_anchor(change):
    returned,view,inputs=records()
    assert confirm_initial_log_anchor(SESSION,frame(),replace(returned,**change),view,inputs) is None


@pytest.mark.parametrize('change',[{'sequence':1},{'sequence':False},{'battle_id':'other'},{'expired':True}])
def test_initial_anchor_requires_zero_current_input_sequence(change):
    returned,view,inputs=records()
    assert confirm_initial_log_anchor(SESSION,frame(),returned,view,replace(inputs,**change)) is None


def test_pending_input_historical_session_and_late_title_are_rejected():
    returned,view,inputs=records()
    pending=BattleInputs(replace(SESSION,turn_budget_mode='fixed_budget'))
    fixed=bind_battle_frame(pending.session,BattleFrameValues('input',True,20,50))
    pending.begin('skip',fixed)
    assert confirm_initial_log_anchor(SESSION,frame(),returned,view,replace(inputs,pending=pending.pending)) is None
    assert confirm_initial_log_anchor(replace(SESSION,identity_source='historical_capture_metadata'),frame(),returned,view,inputs) is None
    late=replace(view.headers[0],absolute_turn=2)
    assert confirm_initial_log_anchor(SESSION,frame(),replace(returned,tail_header=late),replace(view,headers=(late,)),inputs) is None


def test_input_invalidation_strips_old_bound_frame_and_old_request_cannot_be_replayed():
    tracker,bound,inputs=anchored()
    tracker.invalidate('battle_input_requested')
    stale=tracker.bind(bound,inputs)
    assert stale.absolute_turn is None and stale.observation.turn is None
    returned,view,_=records()
    assert not tracker.accept(frame(),returned,view,inputs)
    assert tracker.state.anchor is None


def test_same_frame_id_with_changed_hp_or_input_state_does_not_keep_anchor():
    for kind in ('hp','sequence'):
        tracker,bound,inputs=anchored()
        if kind=='hp':
            bound=replace(bound,values=replace(bound.values,hp=49),observation=replace(bound.observation,hp=49))
        else:
            inputs=replace(inputs,sequence=1)
        stale=tracker.bind(bound,inputs)
        assert current_absolute_turn(stale) is None and tracker.state.anchor is None


def test_filling_turn_without_owned_frame_signal_does_not_pass_context_or_guard():
    raw=frame()
    forged=replace(raw,observation=replace(raw.observation,turn=1,turn_source='owned_initial_log_return'))
    assert current_absolute_turn(forged) is None
    with pytest.raises(ValueError):
        IdolRoadBattleContext(None,forged).fetch_absolute_turn_observation()
    assert TrialFrameGuard(SESSION,100,clock=lambda:0).check(PageSnapshot(Observation(Scene.BATTLE),battle_frame=forged))=='wait'


def entry(name,digest='a'*64):
    return LogEntryEvidence(name,(449,1179,30,39),.99,digest)


def test_identical_current_hud_is_reported_but_never_recovers_absolute_turn():
    tracker,_,inputs=anchored()
    later=tracker.bind(frame('later'),inputs,entry=entry('later'))
    check=tracker.state.recheck
    assert check.verdict=='visible_fields_unchanged'
    assert check.hp_matches and check.remaining_matches and check.hand_matches
    assert check.requires_fresh_owned_log and not check.currentness_confirmed
    assert current_absolute_turn(later) is None and later.observation.turn is None
    assert TrialFrameGuard(SESSION,100,clock=lambda:0).check(PageSnapshot(Observation(Scene.BATTLE),battle_frame=later))=='wait'


@pytest.mark.parametrize('kind',['hp','remaining','hand'])
def test_changed_visible_field_is_separate_from_unknown_and_never_advances_round(kind):
    tracker,_,inputs=anchored()
    later=frame('later')
    evidence=entry('later')
    if kind=='hp':
        later=replace(later,values=replace(later.values,hp=49),observation=replace(later.observation,hp=49))
    elif kind=='remaining':
        later=replace(later,values=replace(later.values,remaining_turns=19))
    else:
        evidence=entry('later','b'*64)
    result=tracker.bind(later,inputs,entry=evidence)
    assert tracker.state.recheck.verdict=='visible_fields_changed'
    assert result.observation.turn is None and result.absolute_turn is None


@pytest.mark.parametrize('evidence',[None,entry('old'),replace(entry('later'),score=.96)])
def test_missing_stale_or_weak_hand_stays_unconfirmed(evidence):
    tracker,_,inputs=anchored()
    tracker.bind(frame('later'),inputs,entry=evidence)
    assert tracker.state.recheck.verdict=='insufficient_visible_fields'
    assert tracker.state.recheck.hand_matches is None


def test_input_and_preview_are_not_reported_as_matching_current_hud():
    tracker,_,inputs=anchored()
    tracker.bind(frame('later'),replace(inputs,sequence=1),entry=entry('later'))
    assert tracker.state.recheck.verdict=='input_not_idle'
    tracker,_,inputs=anchored()
    preview=frame('later')
    tracker.bind(replace(preview,values=replace(preview.values,resource_state='selection_preview')),inputs,entry=entry('later'))
    assert tracker.state.recheck.verdict=='context_unconfirmed'


def test_explicit_input_invalidation_discards_historical_recheck_seed():
    tracker,_,inputs=anchored()
    tracker.invalidate('battle_input_requested')
    result=tracker.bind(frame('later'),inputs,entry=entry('later'))
    assert tracker.state.recheck is None and result.absolute_turn is None


def test_only_a_new_owned_log_return_can_reconfirm_initial_round():
    tracker,_,inputs=anchored()
    tracker.bind(frame('later'),inputs,entry=entry('later'))
    returned,view,_=records()
    returned=replace(returned,request=replace(returned.request,request_id='fresh-request'),return_frame_id='new-return')
    assert tracker.accept(frame('new-return'),returned,view,inputs)
    assert current_absolute_turn(tracker.bind(frame('new-return'),inputs)).frame_id=='new-return'


def test_recheck_trace_merges_identical_visible_fields_not_random_frame_ids():
    tracker,_,inputs=anchored()
    trace=BattleTrace()
    for name in ('later','latest'):
        value=tracker.bind(frame(name),inputs,entry=entry(name))
        trace.observe(Observation(Scene.BATTLE,'ttmr',20,battle_state=value.observation,battle_round_anchor=tracker.state),
                      None,value)
    assert len(trace.snapshot().steps)==1 and trace.snapshot().steps[0].repeated_observations==2


def test_invalid_hand_digest_cannot_seed_an_initial_anchor():
    returned,view,inputs=records()
    returned=replace(returned,request=replace(returned.request,hand_fingerprint='invalid'))
    assert confirm_initial_log_anchor(SESSION,frame(),returned,view,inputs) is None


def test_recheck_uses_new_complete_hand_even_when_log_button_is_hidden():
    from kaa.idol_road.log_hand import LogHandCard,confirm_log_hand
    proof=confirm_log_hand('later',1,(LogHandCard('card',0,True,(40,900,100,180),False),))
    returned,view,inputs=records()
    returned=replace(returned,request=replace(returned.request,hand_fingerprint=proof.fingerprint))
    tracker=InitialRoundTracker(SESSION)
    assert tracker.accept(frame(),returned,view,inputs)
    result=tracker.bind(frame('later'),inputs,hand=proof)
    assert tracker.state.recheck.verdict=='visible_fields_unchanged'
    assert tracker.state.recheck.hand_matches and result.absolute_turn is None
