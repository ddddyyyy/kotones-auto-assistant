"""手牌稳定与当前按钮可见性分离；不模拟设备点击。"""
from dataclasses import replace
import pytest
from kaa.idol_road.log_hand import LogHandCard,confirm_log_hand
from kaa.idol_road.battle_log import BattleLogBinding,LogEntryEvidence,LogViewport,LogScrollEvidence
from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame
from kaa.idol_road.log_turns import LogTurnEvidence,confirm_visible_log_turn
from kaa.idol_road.rules import RuleTextRead,RuleTextLine
from kaa.idol_road.models import Observation,Scene
from kaa.idol_road.battle_trace import BattleTrace

SESSION=BattleObservationSession('synthetic','ttmr',20,20,'confirmed_formation_start_request','variable_budget')


def frame(name,hp=50,remaining=20):
    result=bind_battle_frame(SESSION,BattleFrameValues(name,True,remaining,hp))
    assert result is not None
    return result


def hand(name,card='a'):
    result=confirm_log_hand(name,1,(LogHandCard(card,0,True,(40,900,100,180),False),))
    assert result is not None
    return result


def entry(name,proof=None):
    proof=proof or hand(name)
    return LogEntryEvidence(name,(449,1179,30,39),.99,proof.fingerprint,proof.cards)


def test_known_hand_under_hidden_button_then_current_visible_button_can_open():
    binding=BattleLogBinding()
    binding.observe_hud(frame('a'),None,hand=hand('a'))
    assert not binding.can_open(frame('a'),entry('a'))
    binding.observe_hud(frame('b'),entry('b'),hand=hand('b'))
    assert binding.can_open(frame('b'),entry('b'))


def test_hidden_button_never_uses_old_or_forged_fresh_target():
    binding=BattleLogBinding()
    binding.observe_hud(frame('a'),entry('a'),hand=hand('a'))
    binding.observe_hud(frame('b'),None,hand=hand('b'))
    assert not binding.can_open(frame('b'),None)
    assert not binding.can_open(frame('b'),entry('a'))
    assert not binding.can_open(frame('b'),entry('b'))
    binding.observe_hud(frame('c'),entry('c'),hand=hand('c'))
    assert binding.can_open(frame('c'),entry('c'))


def test_unknown_hand_resets_stability_even_if_button_was_known():
    binding=BattleLogBinding()
    binding.observe_hud(frame('a'),entry('a'),hand=hand('a'))
    binding.observe_hud(frame('b'),None)
    binding.observe_hud(frame('c'),entry('c'),hand=hand('c'))
    assert not binding.can_open(frame('c'),entry('c'))


@pytest.mark.parametrize('kind',['stale','digest','count','different_card','hp','remaining'])
def test_bad_or_changed_current_evidence_does_not_reuse_previous_hand(kind):
    binding=BattleLogBinding()
    binding.observe_hud(frame('a'),None,hand=hand('a'))
    current=frame('b',hp=49 if kind=='hp' else 50,remaining=19 if kind=='remaining' else 20)
    proof=hand('old' if kind=='stale' else 'b',card='different' if kind=='different_card' else 'a')
    if kind=='digest':
        proof=replace(proof,fingerprint='f'*64)
    elif kind=='count':
        proof=replace(proof,count=2)
    binding.observe_hud(current,entry('b',proof),hand=proof)
    assert not binding.can_open(current,entry('b',proof))


def test_input_change_requires_two_new_hand_samples_and_does_not_carry_history():
    binding=BattleLogBinding()
    for name in ('a','b'):
        binding.observe_hud(frame(name),entry(name),hand=hand(name))
    binding.update_inputs(1,False)
    binding.observe_hud(frame('c'),entry('c'),hand=hand('c'))
    assert not binding.can_open(frame('c'),entry('c'))
    binding.observe_hud(frame('d'),entry('d'),hand=hand('d'))
    assert binding.can_open(frame('d'),entry('d'))


def test_pending_input_without_permit_cannot_accumulate_readiness():
    binding=BattleLogBinding()
    binding.update_inputs(1,True)
    for name in ('a','b','c'):
        binding.observe_hud(frame(name),entry(name),hand=hand(name))
    assert not binding.can_open(frame('c'),entry('c'))


def test_log_close_can_verify_fresh_hand_without_clicking_a_hidden_button():
    binding=BattleLogBinding()
    for name in ('a','b'):
        binding.observe_hud(frame(name),entry(name),hand=hand(name))
    binding.open(frame('b'),entry('b'))
    def reads(text):
        return tuple(RuleTextRead('log',kind,(RuleTextLine(text,.99),)) for kind in ('original','grayscale'))
    row=confirm_visible_log_turn(LogTurnEvidence('log',(168,436,98,31),.99,(580,440,88,22),.99,
        (138,426,133,48),reads('1ターン目'),(63,419,39,47),reads('20')))
    view=LogViewport('log','b'*64,(342,1140,36,38),(row,),
        LogScrollEvidence('log',(684,128,3,1007),(684,717,3,418),1.,589),(1.,1.),True)
    binding.close(binding.inspect(view,SESSION))
    returned=binding.observe_hud(frame('back'),None,hand=hand('back'))
    assert returned is not None and returned.hud_unchanged and returned.tail_header.absolute_turn==1
    assert not returned.currentness_confirmed and not binding.can_open(frame('back'),entry('back'))


def test_hand_trace_merges_same_semantics_not_frame_identifiers():
    trace=BattleTrace()
    for name in ('a','b'):
        current=frame(name)
        trace.observe(Observation(Scene.BATTLE,'ttmr',20,battle_state=current.observation,battle_log_hand=hand(name)),None,current)
    assert len(trace.snapshot().steps)==1 and trace.snapshot().steps[0].repeated_observations==2
