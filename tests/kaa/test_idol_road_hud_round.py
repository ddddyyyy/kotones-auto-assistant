from dataclasses import replace
from types import SimpleNamespace
import pytest
from kaa.idol_road import hud_round
from kaa.idol_road.battle_inputs import BattleInputs
from kaa.idol_road.hud_feedback import HudFeedbackWindow
from kaa.idol_road.extra_play import confirm_extra_play
from kaa.idol_road.schedule_countdown import confirm_schedule_countdown
from kaa.idol_road.rules import RuleTextRead,RuleTextLine
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from tests.kaa.test_idol_road_hud_feedback import scenario,frame,hand,SESSION


def countdown(name,value):
    reads=tuple(RuleTextRead(name,kind,(RuleTextLine(str(value)+'ターン後',.99),)) for kind in ('original','grayscale'))
    return confirm_schedule_countdown(name,(96,162,157,32),reads)


def extra(name,value):
    if value==2:
        return confirm_extra_play(name,(14,744,80,84),.8,.8,True,1.,1.,two_score=.99,
            two_stroke_coverage=.99,unique_two=True,two_blue_coverage=1.)
    return confirm_extra_play(name,(14,744,80,84),.99 if value else 0.,.99 if value else 0.,True)


def prepared(kind='card',*,before_extra=1,after_count=9,remaining=20,ids=('other',),origin_count=9,after_extra=0,
             before_queue=None,after_queue=None):
    origin,_=scenario();before=origin.pending.frame;now=[10.]
    from kaa.idol_road.queued_turns import confirm_queued_turns
    def with_queue(current,queue):
        if queue is None:return current
        base,queued=queue
        evidence=confirm_queued_turns(current.values.frame_id,base,.99 if queued==1 else 0.,
            .99 if queued==2 else 0.,.9 if queued else 0.,1. if queued else 0.)
        return replace(current,values=replace(current.values,queued_turns=evidence))
    before=with_queue(before,before_queue)
    inputs=BattleInputs(SESSION,clock=lambda:now[0])
    original=hand('before',('card','other'))
    inputs.begin(kind,before,'card' if kind=='card' else None,extra_play=extra('before',before_extra),origin_hand=original)
    if kind=='card':
        state=BattleState(20,48,0,is_exam=True,hand_size=2,hand_card_ids=('card','other'),
            playable_count=1+before_extra,absolute_turn_observation=before.absolute_turn)
        inputs.bind_card_state(state,'card',category='ProduceCardCategory_MentalSkill',upgrade_count=0)
    window=HudFeedbackWindow(inputs.snapshot(),original,origin_countdown=countdown('before',origin_count))
    for index,name in enumerate(('one','two')):
        current=with_queue(frame(name,remaining),after_queue);current=replace(current,values=replace(current.values,detail_mode='feedback_only'))
        evidence=window.observe(inputs.snapshot(),current,hand(name,ids),now=11+index,
            extra=extra(name,after_extra),countdown=countdown(name,after_count))
    now[0]=12.
    return inputs,now,evidence


def test_observed_queue_increment_confirms_preview_rollover_without_guessing_reward(monkeypatch):
    card=mock_root(monkeypatch)
    inputs,_,evidence=prepared(before_extra=0,origin_count=1,after_count=2,
        before_queue=(20,0),after_queue=(19,1))
    result=hud_round.resolve_hud_round(inputs.snapshot(),evidence,now=12,card=card)
    assert result.turn==2 and result.basis=='queued_turn_increment'
    assert inputs.pending is not None


def test_queue_cannot_hide_multiple_rounds_or_unstable_components(monkeypatch):
    from kaa.idol_road.queued_turns import confirm_queued_turns
    card=mock_root(monkeypatch)
    inputs,_,evidence=prepared(before_extra=0,origin_count=1,after_count=2,
        before_queue=(20,0),after_queue=(18,2))
    assert hud_round.resolve_hud_round(inputs.snapshot(),evidence,now=12,card=card) is None
    inputs,_,evidence=prepared(before_extra=0,origin_count=1,after_count=2,
        before_queue=(20,0),after_queue=(19,1))
    different=confirm_queued_turns('one',18,0.,.99,.9,1.)
    unstable=replace(evidence,first_frame=replace(evidence.first_frame,
        values=replace(evidence.first_frame.values,queued_turns=different)))
    assert hud_round.resolve_hud_round(inputs.snapshot(),unstable,now=12,card=card) is None


def test_supported_draw_uses_fresh_hand_and_does_not_need_predicted_draw_identities(monkeypatch):
    from kaa.db.constants import ProduceExamEffectType
    card=mock_root(monkeypatch)
    card.play_effects=(SimpleNamespace(produce_exam_effect=SimpleNamespace(effect_type=ProduceExamEffectType.ExamCardDraw)),)
    inputs,_,evidence=prepared(ids=('new','drawn'))
    assert evidence.change=='hand_changed'
    result=hud_round.resolve_hud_round(inputs.snapshot(),evidence,now=12,card=card)
    assert result is not None and result.turn==1 and result.play_confirmed
    inputs,_,evidence=prepared(ids=('card','drawn'))
    assert hud_round.resolve_hud_round(inputs.snapshot(),evidence,now=12,card=card) is None


def mock_root(monkeypatch):
    monkeypatch.setattr(hud_round,'supports_current_hp_projection',lambda *a,**k:True)
    monkeypatch.setattr(hud_round,'simulate_play',lambda state,card:replace(state,playable_count=state.playable_count-1))
    return SimpleNamespace(_id='card',upgrade_count=0)


def test_current_countdown_and_owned_root_can_confirm_same_round(monkeypatch):
    card=mock_root(monkeypatch);inputs,_,evidence=prepared()
    resolution=hud_round.resolve_hud_round(inputs.snapshot(),evidence,now=12,card=card)
    assert resolution.turn==1 and resolution.play_confirmed and resolution.source=='owned_card_hud_return'
    assert inputs.pending is not None


def test_two_extras_decrement_to_one_keeps_same_owned_round(monkeypatch):
    card=mock_root(monkeypatch)
    inputs,_,evidence=prepared(before_extra=2,after_extra=1)
    assert inputs.pending.before.playable_count==3
    resolution=hud_round.resolve_hud_round(inputs.snapshot(),evidence,now=12,card=card)
    assert resolution is not None and resolution.turn==1
    assert resolution.basis=='observed_remaining_plays'
    assert inputs.pending is not None


def test_root_projection_cannot_override_missing_or_conflicting_current_round_evidence(monkeypatch):
    card=mock_root(monkeypatch);inputs,_,evidence=prepared(after_count=8,remaining=19)
    assert hud_round.resolve_hud_round(inputs.snapshot(),evidence,now=12,card=card) is None
    missing=replace(evidence,first_countdown=None,countdown=None)
    assert hud_round.resolve_hud_round(inputs.snapshot(),missing,now=12,card=card) is None


def test_observed_auto_advance_is_distinct_from_same_turn(monkeypatch):
    card=mock_root(monkeypatch);inputs,_,evidence=prepared(before_extra=0,after_count=8,remaining=19)
    resolution=hud_round.resolve_hud_round(inputs.snapshot(),evidence,now=12,card=card)
    assert resolution.turn==2 and resolution.play_confirmed


def test_skip_with_unchanged_remaining_and_fresh_hand_advances_one_owned_round():
    inputs,_,evidence=prepared('skip',remaining=20,ids=('next',))
    assert evidence.change=='skip_hand_changed'
    resolution=hud_round.resolve_hud_round(inputs.snapshot(),evidence,now=12)
    assert resolution.turn==2 and resolution.play_confirmed is False


def test_original_deadline_is_rechecked_after_projection(monkeypatch):
    card=mock_root(monkeypatch);inputs,now,evidence=prepared()
    resolution=hud_round.resolve_hud_round(inputs.snapshot(),evidence,now=12,card=card)
    def slow(state,card):
        now[0]=31.
        return replace(state,playable_count=1)
    monkeypatch.setattr(hud_round,'simulate_play',slow)
    assert inputs.finish_hud_resolution(resolution,card=card) is False
    assert inputs.pending is not None


def test_bound_origin_hand_and_stable_cues_cannot_be_replaced():
    inputs,_,evidence=prepared()
    wrong=hand('before',('card',))
    with pytest.raises(ValueError):HudFeedbackWindow(inputs.snapshot(),wrong)
    modified=replace(evidence,extra=extra('old',0))
    assert hud_round.validated_hud_evidence(SESSION,inputs.snapshot(),modified,now=12) is False


def test_preview_boundary_same_round_needs_matching_actual_play_control(monkeypatch):
    card=mock_root(monkeypatch)
    inputs,_,evidence=prepared(origin_count=1,after_count=1)
    result=hud_round.resolve_hud_round(inputs.snapshot(),evidence,now=12,card=card)
    assert result.turn==1 and result.basis=='same_preview_boundary'
    missing=replace(evidence,first_extra=None,extra=None)
    assert hud_round.resolve_hud_round(inputs.snapshot(),missing,now=12,card=card) is None


def test_preview_rollover_can_use_an_actual_single_budget_decrement(monkeypatch):
    card=mock_root(monkeypatch)
    inputs,_,evidence=prepared(origin_count=1,after_count=6,before_extra=0,remaining=19)
    result=hud_round.resolve_hud_round(inputs.snapshot(),evidence,now=12,card=card)
    assert result.turn==2 and result.basis=='remaining_turn_decrement'


@pytest.mark.parametrize('remaining',[20,18])
def test_preview_rollover_does_not_predict_an_unobserved_added_or_multiple_turn(monkeypatch,remaining):
    card=mock_root(monkeypatch)
    inputs,_,evidence=prepared(origin_count=1,after_count=6,before_extra=0,remaining=remaining)
    assert hud_round.resolve_hud_round(inputs.snapshot(),evidence,now=12,card=card) is None


def test_actual_remaining_plays_confirm_same_round_when_preview_is_unreadable(monkeypatch):
    card=mock_root(monkeypatch);inputs,_,evidence=prepared()
    missing=replace(evidence,origin_countdown=None,first_countdown=None,countdown=None)
    result=hud_round.resolve_hud_round(inputs.snapshot(),missing,now=12,card=card)
    assert result.turn==1 and result.basis=='observed_remaining_plays'
    missing=replace(missing,first_extra=None,extra=None)
    assert hud_round.resolve_hud_round(inputs.snapshot(),missing,now=12,card=card) is None


def test_single_observed_budget_decrement_confirms_advance_without_preview(monkeypatch):
    card=mock_root(monkeypatch);inputs,_,evidence=prepared(before_extra=0,remaining=19,after_count=8)
    missing=replace(evidence,origin_countdown=None,first_countdown=None,countdown=None)
    result=hud_round.resolve_hud_round(inputs.snapshot(),missing,now=12,card=card)
    assert result.turn==2 and result.basis=='remaining_turn_decrement'


def test_additional_actual_plays_are_not_replaced_with_a_smaller_root_prediction(monkeypatch):
    card=mock_root(monkeypatch);inputs,_,evidence=prepared(after_extra=1)
    result=hud_round.resolve_hud_round(inputs.snapshot(),evidence,now=12,card=card)
    assert result is not None and result.turn==1 and result.basis=='observed_remaining_plays'
    assert result.evidence.extra.value==1


def test_less_actual_plays_than_the_supported_root_stays_a_conflict(monkeypatch):
    card=mock_root(monkeypatch);inputs,_,evidence=prepared(after_extra=0)
    monkeypatch.setattr(hud_round,'simulate_play',lambda state,card:replace(state,playable_count=2))
    assert hud_round.resolve_hud_round(inputs.snapshot(),evidence,now=12,card=card) is None
