from dataclasses import replace
import pytest
from kaa.idol_road.good_condition import GoodConditionDigits,confirm_good_condition
from kaa.idol_road.rules import RuleTextRead, RuleTextLine
from kaa.idol_road.battle_observation import BattleObservationSession, BattleFrameValues, bind_battle_frame
from kaa.idol_road.planner_context import bind_good_condition, IdolRoadBattleContext
from kaa.tasks.produce.new.play_cards.planner.hud_effects import EffectIconObservation
from kaa.tasks.produce.new.play_cards.planner.memory import BattleMemory
from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState


def reads(value=21, confidence=.99):
    return tuple(RuleTextRead('frame', kind, (RuleTextLine(f'{value}ターン',confidence),))
                 for kind in ('original','grayscale'))


def evidence(body=None, scaled=(), **changes):
    args = dict(frame_id='frame', icon_rect=(18,359,40,43), icon_score=.99, active_blue_fraction=.4,
                number_rect=(58,383,130,33), reads=reads() if body is None else body, scaled_reads=scaled)
    return confirm_good_condition(**(args|changes))


def frame(good=None, **changes):
    session = BattleObservationSession('battle','ttmr',19,20,'confirmed_formation_start_request')
    values = BattleFrameValues('frame',True,15,38,good_condition=good)
    result = bind_battle_frame(session, replace(values, **changes))
    assert result is not None
    return result


def numeric_reads(value=6,confidence=.995):
    return tuple(RuleTextRead('frame',kind,(RuleTextLine(str(value),confidence),)) for kind in ('original','grayscale'))


def digits(value=6):
    width=18+8*(len(str(value))-1)
    return GoodConditionDigits('frame',(58,383,width,33),(58+width,388,60,20),.999,numeric_reads(value))


def test_complete_double_size_numeric_pair_keeps_same_crop_and_conflict_guards():
    split=replace(digits(15),mode='whole_prefix_double_size')
    assert evidence(body=reads(15,.9),digits=split).value==15
    assert evidence(body=reads(14),digits=split).status=='conflict'
    assert evidence(body=reads(15,.9),digits=replace(split,frame_id='old')).value is None


@pytest.mark.parametrize('value',[0,6,21,103])
def test_complete_suffix_and_original_grayscale_numeric_region_can_confirm_weak_full_text(value):
    good=evidence(body=reads(value,.8),scaled=reads(value,.97),digits=digits(value))
    assert good.value==value and good.value_source=='matched_suffix_and_digits'
    bound=frame(good)
    assert bound.observation.good_condition_turns==value
    assert next(item.value for item in bind_good_condition([],bound) if item.effect=='parameter_buff')==value


@pytest.mark.parametrize('change',['no_old_proposal','no_matching_old_proposal'])
def test_numeric_recheck_cannot_invent_or_choose_old_full_text_proposal(change):
    body=tuple(replace(read,lines=()) for read in reads()) if change=='no_old_proposal' else (reads(7,.8)[0],reads(8,.8)[1])
    good=evidence(body=body,digits=digits())
    assert good.value is None


@pytest.mark.parametrize('change',['old_conflict','old_invalid','scaled_conflict','digit_conflict','digit_invalid','multirow'])
def test_high_confidence_conflicts_always_veto_numeric_layout_recheck(change):
    body=reads(6,.8)
    scaled=()
    split=digits()
    if change=='old_conflict':
        body=reads(7)
    elif change=='old_invalid':
        body=(replace(body[0],lines=(RuleTextLine('6',.99),)),body[1])
    elif change=='scaled_conflict':
        scaled=reads(7)
    elif change=='digit_conflict':
        split=replace(split,reads=(split.reads[0],numeric_reads(7)[1]))
    elif change=='digit_invalid':
        split=replace(split,reads=(replace(split.reads[0],lines=(RuleTextLine('6ターン',.99),)),split.reads[1]))
    else:
        split=replace(split,reads=(replace(split.reads[0],lines=(RuleTextLine('6',.99),RuleTextLine('1',.99))),split.reads[1]))
    good=evidence(body=body,scaled=scaled,digits=split)
    assert good.status=='conflict' and good.value is None


@pytest.mark.parametrize('change',['frame','suffix_score','suffix_outside','zero_prefix','digit_origin','digit_height',
    'edge','weak','missing','duplicate','read_frame','wrong_mode'])
def test_numeric_layout_preserves_frame_geometry_completeness_and_threshold_guards(change):
    split=digits()
    if change=='frame':
        split=replace(split,frame_id='old')
    elif change=='suffix_score':
        split=replace(split,suffix_score=.969)
    elif change=='suffix_outside':
        split=replace(split,suffix_rect=(76,420,60,20))
    elif change=='zero_prefix':
        split=replace(split,suffix_rect=(58,388,60,20),digit_rect=(58,383,0,33))
    elif change=='digit_origin':
        split=replace(split,digit_rect=(59,383,17,33))
    elif change=='digit_height':
        split=replace(split,digit_rect=(58,388,18,20))
    elif change=='edge':
        split=replace(split,right_edge_ink=True)
    elif change=='weak':
        split=replace(split,reads=numeric_reads(confidence=.9799))
    elif change=='missing':
        split=replace(split,reads=split.reads[:1])
    elif change=='duplicate':
        split=replace(split,reads=(split.reads[0],split.reads[0]))
    elif change=='read_frame':
        split=replace(split,reads=(replace(split.reads[0],frame_id='old'),split.reads[1]))
    else:
        split=replace(split,mode='generic_detect')
    assert evidence(body=reads(6,.8),digits=split).value is None


def test_raw_low_confidence_malformed_variant_can_be_rechecked_without_trimming_it():
    body=(reads(6,.8)[0],replace(reads(6,.8)[1],lines=(RuleTextLine('ōターン',.69),)))
    good=evidence(body=body,scaled=reads(6,.97),digits=digits())
    assert good.value==6 and good.reads==body


def test_weak_full_text_disagreement_requires_new_strong_numeric_pair_not_a_vote():
    body=(reads(6,.70)[0],reads(8,.74)[1])
    good=evidence(body=body,scaled=reads(6,.96),digits=digits())
    assert good.value==6 and good.reads==body
    weak=replace(digits(),reads=numeric_reads(confidence=.979))
    assert evidence(body=body,digits=weak).value is None


def test_forged_or_removed_numeric_evidence_cannot_enter_frame_or_planner():
    good=evidence(body=reads(6,.8),digits=digits())
    for forged in (replace(good,value=99),replace(good,digits=None)):
        bound=frame(forged)
        assert bound.observation.good_condition_turns is None
        assert not any(item.value is not None for item in bind_good_condition([],bound) if item.effect=='parameter_buff')


def test_direct_recognition_capability_is_optional_and_checked_before_use():
    from kaa.idol_road.good_condition_page import _supports_only_rec
    class Legacy:
        def ocr(self,pixels,*,rect=None):
            raise AssertionError('capability check must not OCR')
    class Modern:
        def ocr(self,pixels,*,only_rec=False):
            raise AssertionError('capability check must not OCR')
    assert not _supports_only_rec(Legacy())
    assert _supports_only_rec(Modern())


@pytest.mark.parametrize('value', [0,1,21,103])
def test_full_duration_not_a_bare_or_clamped_number(value):
    good = evidence(body=reads(value))
    assert good.status == 'resolved' and good.value == value
    assert frame(good).observation.good_condition_turns == value


def test_same_crop_enlargement_can_recheck_agreeing_low_confidence_proposal():
    good = evidence(body=reads(17,.975),scaled=reads(17,.995))
    assert good.status == 'resolved' and good.value == 17
    assert good.value_source == 'same_crop_double_size'


@pytest.mark.parametrize('body,scaled', [
    (reads(21), reads(22)), (reads(21),reads('2l')),
    ((reads(21)[0],reads(22)[1]),()),
    ((reads(21)[0],replace(reads(21)[1],lines=(RuleTextLine('21',.99),))),()),
])
def test_high_confidence_conflict_cannot_be_outvoted(body, scaled):
    good = evidence(body=body, scaled=scaled)
    assert good.status == 'conflict'
    assert frame(good).observation.good_condition_turns is None
    assert 'good_condition' in frame(good).observation.conflicted_fields


@pytest.mark.parametrize('changes', [
    {'reads': ()}, {'reads': reads(21,.97)}, {'icon_score':.969},
    {'active_blue_fraction':.1}, {'icon_rect':None}, {'number_rect':None},
    {'reads':(reads()[0],replace(reads()[1],frame_id='old'))},
    {'scaled_reads':reads(21,.97),'reads':reads(21,.97)},
])
def test_unknown_is_not_zero_or_known_absent(changes):
    good = evidence(**changes)
    assert good.value is None
    assert frame(good).observation.good_condition_turns is None


def test_enlargement_does_not_invent_a_missing_original_proposal():
    good = evidence(body=(), scaled=reads(21))
    assert good.status == 'number_unknown' and good.value is None


def test_forged_stale_and_overlay_values_are_not_bound():
    for good in (replace(evidence(),value=99), replace(evidence(),frame_id='old')):
        current = frame(good)
        assert current.observation.good_condition_turns is None
        assert 'good_condition' in current.observation.conflicted_fields
    preview = frame(evidence(),resource_state='selection_preview')
    assert preview.observation.good_condition_turns is None
    with pytest.raises(ValueError):
        bind_good_condition([], preview)


def test_generic_vote_cannot_fill_missing_good_condition_evidence():
    generic = [EffectIconObservation('parameter_buff',.9,18,359,40,43,211)]
    assert bind_good_condition(generic,frame()) == []
    current = bind_good_condition(generic,frame(evidence()))
    assert len(current) == 1 and current[0].value == 21
    partial = bind_good_condition(generic,frame(evidence(body=reads(21,.97))))
    assert len(partial) == 1 and partial[0].value is None


def test_unknown_channel_prevents_false_expiry_but_default_behavior_is_preserved():
    memory = BattleMemory(buffs=ExamBuffState(parameter_buff_turns=21))
    for _ in range(3):
        memory.reconcile_effect_values({},unknown_effects=frozenset({'parameter_buff'}))
    assert memory.buffs.parameter_buff_turns == 21 and memory.effect_misses['parameter_buff'] == 0
    memory.reconcile_effect_values({'parameter_buff':18})
    assert memory.buffs.parameter_buff_turns == 18
    memory.reconcile_effect_values({})
    memory.reconcile_effect_values({})
    assert memory.buffs.parameter_buff_turns == 0


def test_road_context_reports_uncertain_absence_to_memory(monkeypatch):
    import numpy as np
    from kaa.tasks.produce.new.play_cards.page import LessonBattleContext
    monkeypatch.setattr(LessonBattleContext,'fetch_effect_observations',lambda ctx: [])
    pixels = np.zeros((1280,720,3),dtype='uint8')
    assert IdolRoadBattleContext(pixels,frame()).fetch_unknown_effects() == frozenset({'lesson_buff','parameter_buff'})
    assert IdolRoadBattleContext(pixels,frame(evidence())).fetch_unknown_effects() == frozenset({'lesson_buff'})


def neutral_digits():
    from kaa.idol_road.good_condition import GoodConditionNeutralDigits
    base=digits()
    raw=(numeric_reads()[0],numeric_reads(confidence=.979)[1])
    return replace(base,reads=raw,neutral=GoodConditionNeutralDigits('frame',base.digit_rect,numeric_reads(),False))


def test_neutral_whole_prefix_rechecks_strong_original_without_lowering_gray_threshold():
    proof=neutral_digits()
    good=evidence(body=reads(6,.8),digits=proof)
    assert good.value==6 and good.value_source=='matched_suffix_and_neutral_digits'
    assert frame(good).observation.good_condition_turns==6
    assert evidence(body=reads(6,.8),digits=replace(proof,neutral=None)).value is None


@pytest.mark.parametrize('change',[
    {'frame_id':'old'}, {'rect':(59,383,17,33)}, {'mode':'trimmed'},
    {'right_edge_ink':True}, {'right_edge_ink':None}, {'reads':numeric_reads(confidence=.979)},
    {'reads':numeric_reads()[:1]},
])
def test_neutral_prefix_requires_original_geometry_identity_and_complete_strong_pair(change):
    proof=neutral_digits()
    assert evidence(body=reads(6,.8),digits=replace(proof,neutral=replace(proof.neutral,**change))).value is None


@pytest.mark.parametrize('change',['weak_original','strong_raw_conflict','strong_full_conflict','strong_neutral_conflict','strong_neutral_invalid'])
def test_neutral_prefix_cannot_supply_a_missing_seed_or_override_strong_conflicts(change):
    proof=neutral_digits();body=reads(6,.8)
    if change=='weak_original':proof=replace(proof,reads=numeric_reads(confidence=.979))
    elif change=='strong_raw_conflict':proof=replace(proof,reads=(proof.reads[0],numeric_reads(7)[1]))
    elif change=='strong_full_conflict':body=reads(7)
    elif change=='strong_neutral_conflict':proof=replace(proof,neutral=replace(proof.neutral,reads=numeric_reads(7)))
    else:proof=replace(proof,neutral=replace(proof.neutral,reads=numeric_reads('6x')))
    assert evidence(body=body,digits=proof).value is None
