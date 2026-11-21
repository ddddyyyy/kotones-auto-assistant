from dataclasses import replace
import pytest
from kaa.idol_road.rules import RuleTextRead,RuleTextLine
from kaa.idol_road.log_turns import LogTurnEvidence,confirm_visible_log_turn


def reads(text,confidence=.99):
    return tuple(RuleTextRead('frame',kind,(RuleTextLine(text,confidence),)) for kind in ('original','grayscale'))


def evidence():
    return LogTurnEvidence('frame',(186,436,98,31),.99,(580,440,88,22),.99,
        (138,426,151,48),reads('17ターン目'),(63,419,39,47),reads('2'))


def test_absolute_and_remaining_roles_stay_distinct_without_claiming_currentness():
    result=confirm_visible_log_turn(evidence())
    assert result.absolute_turn==17 and result.remaining_turns==2 and result.status=='resolved'
    assert result.scope=='visible_log_row_only' and not result.currentness_confirmed


@pytest.mark.parametrize('absolute,remaining',[(1,20),(16,5),(17,5),(22,1)])
def test_visible_header_does_not_rederive_absolute_turn_from_remaining(absolute,remaining):
    result=confirm_visible_log_turn(replace(evidence(),turn_reads=reads(f'{absolute}ターン目'),remaining_reads=reads(str(remaining))))
    assert (result.absolute_turn,result.remaining_turns)==(absolute,remaining)


@pytest.mark.parametrize('changes',[{'suffix_score':.96},{'hand_info_score':.96},
    {'turn_rect':(180,426,80,48)},{'remaining_rect':(150,419,39,47)},{'hand_info_rect':(580,600,88,22)}])
def test_missing_labels_clipped_title_and_unrelated_row_keep_unknown(changes):
    result=confirm_visible_log_turn(replace(evidence(),**changes))
    assert result.status=='evidence_unknown' and result.absolute_turn is None and result.remaining_turns is None


def test_high_confidence_conflict_is_not_voted_or_coerced():
    base=evidence()
    result=confirm_visible_log_turn(replace(base,turn_reads=(reads('17ターン目')[0],reads('18ターン目')[1])))
    assert result.absolute_turn is None and result.status=='conflict'
    assert result.remaining_turns==2


@pytest.mark.parametrize('text',['17','017ターン目','-1ターン目','17ターン目18ターン目'])
def test_incomplete_or_multiple_titles_are_not_trimmed(text):
    assert confirm_visible_log_turn(replace(evidence(),turn_reads=reads(text))).absolute_turn is None


def test_low_confidence_and_stale_values_are_not_current_or_zero():
    result=confirm_visible_log_turn(replace(evidence(),remaining_reads=reads('2',.97)))
    assert result.remaining_turns is None and result.absolute_turn==17 and result.status=='partial'
    stale=tuple(replace(read,frame_id='old') for read in reads('17ターン目'))
    assert confirm_visible_log_turn(replace(evidence(),turn_reads=stale)).absolute_turn is None


def absolute_direct_evidence():
    from kaa.idol_road.log_turns import LogAbsoluteDirect
    base=evidence()
    tx,ty,_,th=base.turn_rect
    return replace(base,turn_reads=reads('17ターン目',.93),
        absolute_direct=LogAbsoluteDirect('frame',(tx,ty,base.suffix_rect[0]-tx,th),reads('17'),False))


def test_full_prefix_read_and_confirmed_suffix_can_resolve_weak_whole_title():
    result=confirm_visible_log_turn(absolute_direct_evidence())
    assert result.status=='resolved' and result.absolute_turn==17 and result.remaining_turns==2
    assert result.evidence.turn_reads[0].lines[0].confidence==.93
    assert result.currentness_confirmed is False


@pytest.mark.parametrize('change',[
    {'frame_id':'old'},{'edge_clipped':True},{'rect':(140,426,46,48)},
    {'reads':reads('17',.97)},{'reads':reads('017')},{'reads':reads('17ターン目')},
    {'reads':tuple(replace(read,frame_id='old') for read in reads('17'))},
])
def test_prefix_fallback_cannot_use_clipped_stale_trimmed_or_weak_digits(change):
    base=absolute_direct_evidence()
    result=confirm_visible_log_turn(replace(base,absolute_direct=replace(base.absolute_direct,**change)))
    assert result.absolute_turn is None


@pytest.mark.parametrize('raw',[
    reads('18ターン目',.93),reads('17',.93),(),reads('17ターン目18ターン目',.93),
    tuple(replace(read,frame_id='other') for read in reads('17ターン目',.93)),
])
def test_prefix_fallback_requires_complete_same_frame_literal_title_agreement(raw):
    assert confirm_visible_log_turn(replace(absolute_direct_evidence(),turn_reads=raw)).absolute_turn is None


def test_strong_whole_title_conflict_is_not_overridden_by_prefix():
    result=confirm_visible_log_turn(replace(absolute_direct_evidence(),turn_reads=reads('18ターン目')))
    assert result.status=='conflict' and result.absolute_turn is None


def test_same_crop_scaled_read_can_confirm_unknown_grayscale_without_trimming_it():
    raw=(reads('19')[0],reads('19]',.96)[1])
    result=confirm_visible_log_turn(replace(evidence(),remaining_reads=raw,remaining_scaled_reads=reads('19')))
    assert result.remaining_turns==19 and result.status=='resolved'
    assert not result.currentness_confirmed and result.evidence.remaining_reads[1].lines[0].text=='19]'


@pytest.mark.parametrize('raw,scaled',[
    ((reads('19')[0],reads('18')[1]),reads('19')),
    ((reads('19')[0],reads('19]',.99)[1]),reads('19')),
    ((reads('19')[0],reads('19]',.96)[1]),reads('18')),
    ((reads('19',.96)[0],reads('19]',.96)[1]),reads('19')),
    ((reads('19')[0],reads('19]',.96)[1]),reads('19',.97)),
])
def test_enlargement_does_not_override_conflict_unknown_seed_or_low_confidence(raw,scaled):
    result=confirm_visible_log_turn(replace(evidence(),remaining_reads=raw,remaining_scaled_reads=scaled))
    assert result.remaining_turns is None


def test_scaled_reads_from_another_frame_cannot_confirm_current_remaining():
    raw=(reads('19')[0],reads('19]',.96)[1])
    scaled=tuple(replace(read,frame_id='old') for read in reads('19'))
    assert confirm_visible_log_turn(replace(evidence(),remaining_reads=raw,remaining_scaled_reads=scaled)).remaining_turns is None


def direct_evidence():
    from kaa.idol_road.log_turns import LogRemainingDirect
    base=evidence()
    return replace(base,remaining_reads=reads('19]',.96),remaining_scaled_reads=reads('19'),
        remaining_direct=LogRemainingDirect('frame',base.remaining_rect,reads('19'),False))


def mixed_glyph_evidence():
    from kaa.idol_road.log_remaining_glyphs import LogRemainingGlyphs
    base=direct_evidence()
    raw=(reads('17')[0],replace(reads('17')[1],lines=(RuleTextLine('17',.99),RuleTextLine('AATA',.51))))
    parts=((3,9,13,20,88),(21,9,15,20,91))
    return replace(base,remaining_reads=raw,remaining_scaled_reads=reads('17'),
        remaining_direct=replace(base.remaining_direct,reads=reads('17')),
        remaining_glyphs=LogRemainingGlyphs('frame',base.remaining_rect,parts,parts,179,179))


def test_two_full_masks_and_full_crop_readings_can_recheck_weak_extra_non_numeric_text():
    result=confirm_visible_log_turn(mixed_glyph_evidence())
    assert result.status=='resolved' and result.remaining_turns==17
    assert result.evidence.remaining_reads[1].lines[1].text=='AATA'
    assert not result.currentness_confirmed


@pytest.mark.parametrize('change',[
    {'frame_id':'other'},{'rect':(64,419,39,47)},{'neutral_pixels':178},
    {'grayscale_components':((3,9,13,20,88),(21,9,15,20,90))},
    {'neutral_components':((0,9,13,20,88),(21,9,15,20,91)),
     'grayscale_components':((0,9,13,20,88),(21,9,15,20,91))},
    {'neutral_components':((3,9,13,20,88),(21,18,15,20,91)),
     'grayscale_components':((3,9,13,20,88),(21,18,15,20,91))},
    {'neutral_components':((3,9,13,20,88),(21,9,15,20,91),(1,1,1,1,1)),
     'grayscale_components':((3,9,13,20,88),(21,9,15,20,91),(1,1,1,1,1))},
])
def test_missing_pixels_different_masks_extra_components_or_bad_geometry_do_not_repair_ocr(change):
    base=mixed_glyph_evidence()
    result=confirm_visible_log_turn(replace(base,remaining_glyphs=replace(base.remaining_glyphs,**change)))
    assert result.remaining_turns is None


@pytest.mark.parametrize('text,confidence',[('8',.51),('AATA',.99),('A8A',.51),('',.51)])
def test_extra_numeric_or_high_confidence_text_stays_a_conflict(text,confidence):
    base=mixed_glyph_evidence()
    raw=(base.remaining_reads[0],replace(base.remaining_reads[1],lines=(RuleTextLine('17',.99),RuleTextLine(text,confidence))))
    assert confirm_visible_log_turn(replace(base,remaining_reads=raw)).remaining_turns is None


def test_mixed_text_recheck_does_not_override_strong_numeric_disagreement():
    base=mixed_glyph_evidence()
    for changed in (replace(base,remaining_scaled_reads=reads('18')),
                    replace(base,remaining_direct=replace(base.remaining_direct,reads=reads('18'))),
                    replace(base,remaining_reads=(reads('18')[0],base.remaining_reads[1])),
                    replace(base,remaining_glyphs=None)):
        assert confirm_visible_log_turn(changed).remaining_turns is None


def test_direct_full_crop_and_enlarged_pair_confirm_without_trimming_weak_detector_text():
    result=confirm_visible_log_turn(direct_evidence())
    assert result.remaining_turns==19 and result.status=='resolved'
    assert result.evidence.remaining_reads[0].lines[0].text=='19]'
    assert not result.currentness_confirmed


@pytest.mark.parametrize('changes',[
    {'frame_id':'other'}, {'rect':(64,419,38,47)}, {'mode':'trimmed'}, {'right_edge_clipped':True},
    {'reads':reads('19',.97)}, {'reads':reads('19')[:1]},
    {'reads':tuple(replace(read,frame_id='old') for read in reads('19'))},
])
def test_direct_evidence_requires_current_complete_same_crop_and_strong_pair(changes):
    base=direct_evidence()
    result=confirm_visible_log_turn(replace(base,remaining_direct=replace(base.remaining_direct,**changes)))
    assert result.remaining_turns is None


@pytest.mark.parametrize('changes',[
    {'remaining_reads':reads('18')}, {'remaining_reads':reads('19]',.99)},
    {'remaining_scaled_reads':reads('18')}, {'remaining_scaled_reads':reads('19]',.99)},
])
def test_direct_pair_cannot_override_any_high_confidence_old_conflict(changes):
    result=confirm_visible_log_turn(replace(direct_evidence(),**changes))
    assert result.remaining_turns is None and result.status=='conflict'


@pytest.mark.parametrize('reads_value',[reads('18'),reads('19]',.99),reads('0'),reads('31')])
def test_conflicting_or_invalid_strong_direct_text_is_rejected(reads_value):
    base=direct_evidence()
    result=confirm_visible_log_turn(replace(base,remaining_direct=replace(base.remaining_direct,reads=reads_value)))
    assert result.remaining_turns is None and result.status=='conflict'


@pytest.mark.parametrize('changes',[
    {'remaining_scaled_reads':()}, {'remaining_scaled_reads':reads('19',.97)},
    {'remaining_reads':()}, {'remaining_reads':tuple(replace(read,frame_id='old') for read in reads('19]',.96))},
])
def test_direct_pair_needs_full_raw_capture_and_independent_enlarged_agreement(changes):
    assert confirm_visible_log_turn(replace(direct_evidence(),**changes)).remaining_turns is None


def split_evidence():
    base=direct_evidence()
    original,gray=base.remaining_reads
    return replace(base,remaining_reads=(original,replace(gray,lines=(*gray.lines,RuleTextLine('ATA',.62)))))


def test_weak_split_detection_stays_unknown_until_full_crop_independent_reads_agree():
    base=split_evidence()
    assert confirm_visible_log_turn(replace(base,remaining_direct=None)).remaining_turns is None
    result=confirm_visible_log_turn(base)
    assert result.status=='resolved' and result.remaining_turns==19
    assert len(result.evidence.remaining_reads[1].lines)==2
    assert not result.currentness_confirmed


@pytest.mark.parametrize('lines',[
    (RuleTextLine('19',.99),RuleTextLine('ATA',.62)),
    (RuleTextLine('19]',.96),RuleTextLine('18',.98)),
    (RuleTextLine('19]',.96),RuleTextLine('ATA',.62),RuleTextLine('noise',.3)),
])
def test_split_strong_or_excess_lines_remain_conflicts(lines):
    base=split_evidence()
    raw=(base.remaining_reads[0],replace(base.remaining_reads[1],lines=lines))
    result=confirm_visible_log_turn(replace(base,remaining_reads=raw))
    assert result.remaining_turns is None and result.status=='conflict'


@pytest.mark.parametrize('change',[
    {'remaining_scaled_reads':reads('18')}, {'remaining_scaled_reads':reads('19',.97)},
    {'remaining_scaled_reads':()}, {'remaining_direct':None},
])
def test_weak_split_does_not_supply_a_numeric_seed_or_override_other_reads(change):
    assert confirm_visible_log_turn(replace(split_evidence(),**change)).remaining_turns is None


@pytest.mark.parametrize('text', ['18', '19]', '0'])
def test_first_weak_split_does_not_hide_strong_conflict_in_second_representation(text):
    base=split_evidence()
    weak=replace(base.remaining_reads[1],representation='original')
    strong=reads(text)[1]
    result=confirm_visible_log_turn(replace(base,remaining_reads=(weak,strong)))
    assert result.remaining_turns is None and result.status=='conflict'
