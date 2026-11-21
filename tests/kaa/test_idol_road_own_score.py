from dataclasses import replace
import pytest
from kaa.idol_road.own_score import OwnScoreMaskDigits,OwnScoreMarginDigits,confirm_own_score
from kaa.idol_road.rules import RuleTextRead,RuleTextLine


def reads(text='11000',confidence=.99,frame='current'):
    return tuple(RuleTextRead(frame,kind,(RuleTextLine(text,confidence),)) for kind in ('original','grayscale'))


def mask_digits(text='0'):
    return OwnScoreMaskDigits('current',(226,125,124,40),(226,126,45,39),reads(text),300)


@pytest.mark.parametrize('text',['0','928','11000'])
def test_all_foreground_mask_pair_can_resolve_a_number_with_empty_detector_rows(text):
    raw=tuple(replace(read,lines=()) for read in reads())
    result=confirm_own_score('current',(226,125,124,40),raw,reads('O',.25),mask_digits=mask_digits(text))
    assert result.status=='resolved' and result.value==int(text)


@pytest.mark.parametrize('kind',['old','wrong_score_rect','outside','too_short','no_pixels','bad_mode','clipped','missing_raw'])
def test_mask_supplement_requires_matching_complete_crop_provenance(kind):
    digits=mask_digits()
    raw=reads('O',.25)
    if kind=='old':digits=replace(digits,frame_id='old')
    elif kind=='wrong_score_rect':digits=replace(digits,score_rect=(593,125,124,40))
    elif kind=='outside':digits=replace(digits,content_rect=(225,126,45,39))
    elif kind=='too_short':digits=replace(digits,content_rect=(226,126,45,10))
    elif kind=='no_pixels':digits=replace(digits,foreground_pixels=0)
    elif kind=='bad_mode':digits=replace(digits,mode='guessed_digit')
    elif kind=='clipped':digits=replace(digits,edge_ink=True)
    else:raw=()
    assert confirm_own_score('current',(226,125,124,40),raw,mask_digits=digits).value is None


@pytest.mark.parametrize('group',['raw','direct','mask'])
def test_isolated_strong_conflict_cannot_be_overruled_by_any_pair(group):
    raw=reads('0')
    direct=reads('O',.25)
    digits=mask_digits()
    conflict=(reads('1')[0],reads('O',.25)[1])
    if group=='raw':raw=conflict
    elif group=='direct':direct=conflict
    else:digits=replace(digits,reads=conflict)
    result=confirm_own_score('current',(226,125,124,40),raw,direct,mask_digits=digits)
    assert result.status=='conflict' and result.value is None


def test_mask_weak_or_nonliteral_zero_does_not_become_zero():
    for values in (reads('0',.97),reads('O'),reads('00')):
        assert confirm_own_score('current',(226,125,124,40),reads('O',.25),
                                 mask_digits=replace(mask_digits(),reads=values)).value is None


def test_context_revalidates_mask_geometry_instead_of_trusting_its_resolved_zero():
    import numpy as np
    from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame
    from kaa.idol_road.planner_context import IdolRoadBattleContext
    owner=BattleObservationSession('battle','ttmr',20,20,'confirmed_formation_start_request')
    frame=bind_battle_frame(owner,BattleFrameValues('current',True,20,50))
    context=IdolRoadBattleContext(np.zeros((1280,720,3),dtype='uint8'),frame)
    valid=confirm_own_score('current',(226,125,124,40),reads('O',.25),mask_digits=mask_digits())
    context._road_score=valid
    assert context.fetch_exam_current_score()==0
    context._road_score=replace(valid,mask_digits=replace(valid.mask_digits,frame_id='old'))
    assert context.fetch_exam_current_score() is None


@pytest.mark.parametrize('number',[0,1,10999,11000,999999999])
def test_own_score_accepts_only_literal_strong_current_pair(number):
    result=confirm_own_score('current',(226,125,124,40),reads(str(number)))
    assert result.status=='resolved' and result.value==number


@pytest.mark.parametrize('text',['','o','00','01','11,000','11000点','-1','1.0','1000000000',' 11000'])
def test_noncanonical_score_is_not_trimmed_or_coerced(text):
    result=confirm_own_score('current',(226,125,124,40),reads(text))
    assert result.value is None and result.status=='conflict'


@pytest.mark.parametrize('raw',[(),reads('11000',.97),reads(frame='old'),reads()[:1],(reads()[0],reads()[0])])
def test_missing_stale_low_or_duplicate_evidence_stays_unknown_without_zero_fallback(raw):
    assert confirm_own_score('current',(226,125,124,40),raw).value is None


def test_strong_direct_pair_can_resolve_complete_weak_detector_text_without_slicing():
    result=confirm_own_score('current',(226,125,124,40),reads('11000]',.9),reads('11000'))
    assert result.value==11000 and result.reads[0].lines[0].text=='11000]'


@pytest.mark.parametrize('raw,direct',[
    (reads('10999'),reads('11000')),(reads('11000]',.99),reads('11000')),
    ((reads('10999')[0],reads('11000]',.9)[1]),reads('11000')),
    (reads('11000'),reads('10999')),(reads('11000'),reads('11000]',.99)),
])
def test_any_strong_conflict_wins_over_direct_pair(raw,direct):
    result=confirm_own_score('current',(226,125,124,40),raw,direct)
    assert result.value is None and result.status=='conflict'


@pytest.mark.parametrize('rect',[None,(-1,125,124,40),(700,125,124,40),(226,125,0,40)])
def test_missing_owner_or_invalid_crop_cannot_supply_a_score(rect):
    assert confirm_own_score('current',rect,reads()).status=='owner_unknown'


def test_clipping_or_missing_raw_capture_cannot_be_repaired_by_direct_pair():
    assert confirm_own_score('current',(226,125,124,40),reads(),right_edge_clipped=True).value is None
    assert confirm_own_score('current',(226,125,124,40),(),reads()).value is None


def test_context_revalidates_score_without_generic_standings_fallback():
    import numpy as np
    from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame
    from kaa.idol_road.planner_context import IdolRoadBattleContext
    session=BattleObservationSession('battle','ttmr',20,20,'confirmed_formation_start_request')
    frame=bind_battle_frame(session,BattleFrameValues('current',True,19,49))
    context=IdolRoadBattleContext(np.zeros((1280,720,3),dtype='uint8'),frame)
    score=confirm_own_score('current',(226,125,124,40),reads())
    context._road_score=score
    assert context.fetch_exam_current_score()==11000
    for forged in (replace(score,value=99999),replace(score,frame_id='old'),replace(score,reads=())):
        context._road_score=forged
        assert context.fetch_exam_current_score() is None
        assert context.fetch_exam_standings() is None


def test_probe_only_context_cannot_read_score():
    import numpy as np
    from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame
    from kaa.idol_road.planner_context import IdolRoadBattleContext
    session=BattleObservationSession('battle','ttmr',20,20,'confirmed_formation_start_request')
    frame=bind_battle_frame(session,BattleFrameValues('current',True,19,49,detail_mode='core_only'))
    context=IdolRoadBattleContext(np.zeros((1280,720,3),dtype='uint8'),frame)
    with pytest.raises(ValueError,match='日志探针'):
        context.fetch_exam_current_score()


def margin_packet(text='0'):
    parent=replace(mask_digits(text),reads=reads(text,.95))
    line=OwnScoreMaskDigits('current',(226,133,124,32),(226,134,23,31),reads(text),297)
    margin=OwnScoreMarginDigits('current',parent.score_rect,8,((11,4,1,1,1),(43,1,2,2,2)),line)
    raw=tuple(replace(read,lines=()) for read in reads())
    return raw,parent,margin


@pytest.mark.parametrize('text',['0','428'])
def test_bounded_top_margin_corrobates_the_complete_parent_proposal(text):
    raw,parent,margin=margin_packet(text)
    value=confirm_own_score('current',parent.score_rect,raw,reads('0',.25),mask_digits=parent,margin_digits=margin)
    assert value.value==int(text) and value.margin_digits==margin and value.mask_digits==parent


@pytest.mark.parametrize('kind',['frame','parent_rect','margin','bool_margin','mode','cut_boundary','large_piece',
    'outside_margin','duplicate_piece','pixel_loss','line_rect','line_frame','line_edge','weak_line','nonliteral_line',
    'missing_parent','different_parent_proposal'])
def test_margin_supplement_cannot_change_region_or_discard_unaccounted_ink(kind):
    raw,parent,margin=margin_packet()
    if kind=='frame':margin=replace(margin,frame_id='old')
    elif kind=='parent_rect':margin=replace(margin,score_rect=(593,125,124,40))
    elif kind=='margin':margin=replace(margin,top_margin=9)
    elif kind=='bool_margin':margin=replace(margin,top_margin=True)
    elif kind=='mode':margin=replace(margin,mode='selected_digit')
    elif kind=='cut_boundary':margin=replace(margin,boundary_ink=True)
    elif kind=='large_piece':margin=replace(margin,excluded_components=((10,1,3,2,3),))
    elif kind=='outside_margin':margin=replace(margin,excluded_components=((10,7,2,2,3),))
    elif kind=='duplicate_piece':margin=replace(margin,excluded_components=((10,1,1,1,1),)*3)
    elif kind=='pixel_loss':margin=replace(margin,digits=replace(margin.digits,foreground_pixels=296))
    elif kind=='line_rect':margin=replace(margin,digits=replace(margin.digits,score_rect=(227,133,123,32)))
    elif kind=='line_frame':margin=replace(margin,digits=replace(margin.digits,frame_id='old'))
    elif kind=='line_edge':margin=replace(margin,digits=replace(margin.digits,edge_ink=True))
    elif kind=='weak_line':margin=replace(margin,digits=replace(margin.digits,reads=reads('0',.97)))
    elif kind=='nonliteral_line':margin=replace(margin,digits=replace(margin.digits,reads=reads('O')))
    elif kind=='missing_parent':parent=None
    elif kind=='different_parent_proposal':parent=replace(parent,reads=reads('10',.95))
    assert confirm_own_score('current',(226,125,124,40),raw,mask_digits=parent,margin_digits=margin).value is None


@pytest.mark.parametrize('group',['raw','direct','parent'])
def test_margin_pair_never_overrides_an_isolated_strong_conflict(group):
    raw,parent,margin=margin_packet()
    direct=reads('0',.25)
    conflicting=(reads('1')[0],reads('0',.25)[1])
    if group=='raw':raw=conflicting
    elif group=='direct':direct=conflicting
    else:parent=replace(parent,reads=conflicting)
    result=confirm_own_score('current',parent.score_rect,raw,direct,mask_digits=parent,margin_digits=margin)
    assert result.status=='conflict' and result.value is None


def test_context_revalidates_margin_pixel_accounting():
    import numpy as np
    from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame
    from kaa.idol_road.planner_context import IdolRoadBattleContext
    session=BattleObservationSession('battle','ttmr',20,20,'confirmed_formation_start_request')
    frame=bind_battle_frame(session,BattleFrameValues('current',True,20,50))
    context=IdolRoadBattleContext(np.zeros((1280,720,3),dtype='uint8'),frame)
    raw,parent,margin=margin_packet()
    valid=confirm_own_score('current',parent.score_rect,raw,mask_digits=parent,margin_digits=margin)
    context._road_score=valid
    assert context.fetch_exam_current_score()==0
    context._road_score=replace(valid,margin_digits=replace(margin,digits=replace(margin.digits,foreground_pixels=296)))
    assert context.fetch_exam_current_score() is None
