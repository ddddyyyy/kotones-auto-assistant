from dataclasses import replace
import pytest
from kaa.idol_road.status_details import StatusRowEvidence,StatusSplitValueEvidence,BattleStatusDetails,confirm_status_row
from kaa.idol_road.rules import RuleTextRead,RuleTextLine
from kaa.idol_road.models import Observation,Scene,Action
from kaa.idol_road.controller import IdolRoadController
from kaa.config.idol_road import IdolRoadConfig
from kaa.idol_road.battle_trace import BattleTrace


def reads(text,frame='frame',confidence=.99):
    return tuple(RuleTextRead(frame,kind,(RuleTextLine(text,confidence),)) for kind in ('original','grayscale'))


def evidence(**changes):
    return replace(StatusRowEvidence('frame',(145,390,56,29),.99,reads('好調'),
        reads('2ターン'),reads('スコア上昇量を50%増加2ターン')),**changes)


@pytest.mark.parametrize('turns',[1,2,10])
def test_current_explicit_duration_is_not_a_calculated_expiry(turns):
    row=confirm_status_row(evidence(value_reads=reads(f'{turns}ターン'),
        description_reads=reads(f'スコア上昇量を50%増加{turns}ターン')))
    assert row.status=='resolved' and row.kind=='lesson_buff' and row.remaining_turns==turns
    assert row.score_increase_percent==50


@pytest.mark.parametrize('text',['0ターン','-1ターン','2','あと2','2ターン後','2ターンextra'])
def test_partial_zero_negative_and_future_duration_not_current(text):
    assert confirm_status_row(evidence(value_reads=reads(text))).status=='unknown'


@pytest.mark.parametrize('changes', [
    {'value_reads':reads('2ターン',confidence=.97)}, {'value_reads':reads('2ターン',frame='old')},
    {'title_reads':reads('不調')}, {'title_score':.96}, {'frame_id':''},
    {'value_reads':(reads('1ターン')[0],reads('2ターン')[1])},
])
def test_unconfirmed_or_conflicting_evidence_never_resolves(changes):
    assert confirm_status_row(evidence(**changes)).status=='unknown'


def test_low_confidence_body_does_not_invent_score_multiplier():
    row=confirm_status_row(evidence(description_reads=reads('スコア上昇量を50%増加2ターン',confidence=.97)))
    assert row.remaining_turns==2 and row.score_increase_percent is None


def test_close_control_occlusion_vetoes_even_high_confidence_recognition():
    assert confirm_status_row(evidence(occluded=True)).status=='unknown'


def test_ambiguous_separator_in_additional_plays_is_not_repaired_to_a_digit():
    row=confirm_status_row(evidence(title_reads=reads('スキルカード使用数追加'),
        value_reads=reads('1回I1ターン',confidence=.925),description_reads=reads('スキルカードを追加で使用できる')))
    assert row.status=='unknown' and row.available_plays is None


def test_explicit_additional_plays_and_duration_are_separate_quantities():
    row=confirm_status_row(evidence(title_reads=reads('スキルカード使用数追加'),
        value_reads=reads('2回 | 1ターン'),description_reads=reads('スキルカードを追加で使用できる')))
    assert row.available_plays==2 and row.remaining_turns==1


def split(**changes):
    return replace(StatusSplitValueEvidence('frame',(143,607,54,38),(211,607,125,38),
        reads('1回'),reads('1ターン')),**changes)


def extra_evidence(**changes):
    return evidence(title_rect=(145,574,308,29),title_reads=reads('スキルカード使用数追加'),
        value_reads=reads('1回I1ターン',confidence=.925),description_reads=reads('スキルカードを追加で使用できる'),
        **changes)


def test_complete_separate_fields_resolve_low_confidence_separator_without_repairing_it():
    row=confirm_status_row(extra_evidence(split_value=split()))
    assert (row.available_plays,row.remaining_turns,row.value_source)==(1,1,'split_fields')
    assert row.evidence.value_reads[0].lines[0].text=='1回I1ターン'


@pytest.mark.parametrize('text',['2回|1ターン','1回|2ターン','1回I1ターン','謎の効果'])
def test_high_confidence_whole_line_conflicts_veto_split_fields(text):
    value=extra_evidence(split_value=split())
    value=replace(value,value_reads=(reads(text)[0],value.value_reads[1]))
    row=confirm_status_row(value)
    assert row.status=='unknown' and row.available_plays is None and row.remaining_turns is None


@pytest.mark.parametrize('changes', [
    {'frame_id':'old'},{'play_count_reads':reads('1回',frame='old')},
    {'turn_count_reads':reads('1ターン',confidence=.97)}, {'play_count_reads':reads('1')},
    {'turn_count_reads':reads('1')}, {'play_count_reads':reads('0回')},
    {'play_count_reads':(reads('1回')[0],reads('2回')[1])},
    {'play_count_rect':(211,607,54,38)}, {'turn_count_rect':(211,800,125,38)},
    {'play_count_rect':(143,400,54,38)}, {'turn_count_reads':()},
])
def test_stale_partial_conflicting_or_overlapping_fields_are_unknown(changes):
    assert confirm_status_row(extra_evidence(split_value=split(**changes))).status=='unknown'


def test_original_cross_frame_or_missing_representation_cannot_be_bypassed():
    value=extra_evidence(split_value=split())
    for raw in (reads('1回1ターン',frame='old'),(),(value.value_reads[0],)*2):
        assert confirm_status_row(replace(value,value_reads=raw)).status=='unknown'


def test_whole_line_and_split_values_can_agree_but_mismatched_supported_body_is_not_used():
    value=replace(extra_evidence(split_value=split()),value_reads=reads('1回 | 1ターン'))
    assert confirm_status_row(value).value_source=='split_fields'
    assert confirm_status_row(replace(value,description_reads=reads('別の効果'))).status=='unknown'


def test_status_close_is_bounded_and_not_counted_as_starting_a_battle():
    flow=IdolRoadController(IdolRoadConfig(characters=['ttmr'],mode='manual'))
    obs=Observation(Scene.BATTLE_STATUS_DETAILS)
    assert flow.step(obs).action==Action.CLOSE_BATTLE_STATUS
    assert flow.step(obs).action==Action.WAIT
    assert flow.step(obs).action==Action.WAIT
    assert flow.step(obs).reason=='navigation_not_confirmed'
    assert flow.battles_started==0


def test_status_details_never_override_cancellation():
    flow=IdolRoadController(IdolRoadConfig(characters=['ttmr'],mode='manual'))
    assert flow.step(Observation(Scene.BATTLE_STATUS_DETAILS),cancelled=True).reason=='cancelled'


def test_trace_preserves_visible_count_and_source_without_carrying_into_next_frame():
    details=BattleStatusDetails('frame','battle','ttmr',16,(confirm_status_row(evidence()),),
        identity_source='historical_capture_metadata')
    trace=BattleTrace()
    trace.observe(Observation(Scene.BATTLE_STATUS_DETAILS,'ttmr',16,battle_status_details=details),None)
    trace.observe(Observation(Scene.BATTLE,'ttmr',16),None)
    first,last=trace.snapshot().steps
    assert first.status_details.rows[0].remaining_turns==2
    assert first.status_details.identity_source=='historical_capture_metadata'
    assert not first.status_details.trigger_attribution_confirmed and not first.status_details.complete
    assert last.status_details is None and last.battle_frame is None
