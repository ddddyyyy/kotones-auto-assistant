from dataclasses import replace
import pytest
from kaa.idol_road.score_percent import confirm_score_percent
from kaa.idol_road.rules import RuleTextRead, RuleTextLine


def reads(text='3053%', confidence=.99, frame='current'):
    return tuple(RuleTextRead(frame, kind, (RuleTextLine(text, confidence),))
                 for kind in ('original', 'grayscale'))


@pytest.mark.parametrize('text,value', [('0%',0), ('100%',100), ('1019%',1019), ('3053 %',3053), ('9999%',9999)])
def test_full_current_percent_pair(text, value):
    result=confirm_score_percent('current',(108,82,114,39),reads(text))
    assert result.status=='resolved' and result.value==value


@pytest.mark.parametrize('text', ['3053', '03053%', '30.53%', '-100%', '10000%', 'O%', '3053%extra', '３０５３％'])
def test_no_value_coercion_or_missing_suffix(text):
    result=confirm_score_percent('current',(108,82,114,39),reads(text))
    assert result.status=='conflict' and result.value is None


@pytest.mark.parametrize('case', ['old', 'weak', 'missing', 'same_channel', 'clipped', 'out_of_bounds', 'conflict', 'multiple_lines'])
def test_incomplete_or_conflicting_evidence_cannot_supply_multiplier(case):
    pair=reads();rect=(108,82,114,39)
    if case=='old':pair=reads(frame='old')
    elif case=='weak':pair=(pair[0],reads(confidence=.979)[1])
    elif case=='missing':pair=pair[:1]
    elif case=='same_channel':pair=(pair[0],pair[0])
    elif case=='out_of_bounds':rect=(700,82,114,39)
    elif case=='conflict':pair=(pair[0],reads('1019%')[1])
    elif case=='multiple_lines':pair=(replace(pair[0],lines=pair[0].lines*2),pair[1])
    assert confirm_score_percent('current',rect,pair,case=='clipped').value is None


def test_context_rechecks_source_and_rejects_partial_frame():
    import numpy as np
    from kaa.idol_road.battle_observation import BattleObservationSession, BattleFrameValues, bind_battle_frame
    from kaa.idol_road.planner_context import IdolRoadBattleContext
    owner=BattleObservationSession('battle','ttmr',20,20,'confirmed_formation_start_request')
    frame=bind_battle_frame(owner,BattleFrameValues('current',True,20,50))
    assert frame is not None
    context=IdolRoadBattleContext(np.zeros((1280,720,3),dtype='uint8'),frame)
    valid=confirm_score_percent('current',(108,82,114,39),reads())
    context._road_percent=valid
    assert context.fetch_exam_score_percent()==3053
    context._road_percent=replace(valid,reads=reads(frame='old'))
    assert context.fetch_exam_score_percent() is None
    context._road_percent=valid
    context._road_frame=replace(frame,values=replace(frame.values,detail_mode='feedback_only'))
    with pytest.raises(ValueError):context.fetch_exam_score_percent()
