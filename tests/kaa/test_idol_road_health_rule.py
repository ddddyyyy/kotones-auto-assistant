from dataclasses import replace
import pytest
from kaa.idol_road.health_rule import HealthBlockEvidence, interpret_health_block
from kaa.idol_road.rule_symbols import RuleSymbolEvidence
from kaa.idol_road.rules import RuleTextRead, RuleTextLine, RuleViewport, StageRules, VisibleStageRule
from kaa.idol_road.rule_catalog import build_rule_catalog


def reads(text, confidence=.99, frame='frame'):
    return tuple(RuleTextRead(frame, kind, (RuleTextLine(text, confidence),)) for kind in ('original', 'grayscale'))


def original():
    return reads('体力が20%以下の場合、メンタルスキルカード使用不可（2ターン）')


def evidence(**changes):
    symbols = tuple(RuleSymbolEvidence('frame', kind, rect, .99) for kind, rect in (
        ('mental', (281, 0, 37, 35)), ('mental_skill_label', (323, 2, 171, 29)),
        ('mental_block_two_turns', (3, 37, 294, 31))))
    return replace(HealthBlockEvidence('frame', symbols, reads('体力が20%以下の場合、'), reads('2')), **changes)


def row():
    analysis = interpret_health_block(7, evidence(), original())
    return VisibleStageRule((40, 838, 637, 185), 7, reads('7ターン目'), original(), health_block=analysis)


def test_condition_timeline_does_not_turn_satisfied_into_an_active_or_expired_effect():
    from kaa.idol_road.battle_trace import BattleTrace
    from kaa.idol_road.battle_observation import BattleFrameValues, BattleObservationSession, bind_battle_frame
    from kaa.idol_road.rule_conditions import evaluate_rule_conditions,ConditionVerdict
    from kaa.idol_road.models import Observation,Scene
    trace=BattleTrace()
    catalog=build_rule_catalog('ttmr',16,StageRules('ttmr',16,(RuleViewport('a',(row(),row())),)))
    session=BattleObservationSession('battle','ttmr',16,10,'confirmed_formation_start_request')
    for index,(remaining,hp,max_hp) in enumerate(((4,10,48),(4,9,48),(3,9,None))):
        # 此测试提供显式领域输入，不伪造图像读取或实际触发时机。
        frame=bind_battle_frame(session,BattleFrameValues(str(index),True,remaining,hp))
        frame=replace(frame,observation=replace(frame.observation,max_hp=max_hp))
        report=evaluate_rule_conditions(catalog,frame.observation,battle_id=session.battle_id)
        trace.observe(Observation(Scene.BATTLE,'ttmr',16),None,frame,report)
    steps=trace.snapshot().steps
    assert [s.conditions.checks[0].verdict for s in steps]==[
        ConditionVerdict.NOT_SATISFIED,ConditionVerdict.SATISFIED,ConditionVerdict.UNKNOWN]
    assert all(len(s.conditions.checks)==2 and not s.conditions.executable for s in steps)
    assert steps[-1].battle_frame.observation.max_hp is None
    assert steps[-1].conditions.checks[0].reason=='outside_declared_turn'


@pytest.mark.parametrize('change,inspect', [({},True),({'turn':6},False),({'max_hp':48},False),
    ({'source':'historical_or_unconfirmed_battle_observation'},False),
    ({'source':'unconfirmed_resource_preview'},False),({'conflicted_fields':('turn',)},False),
    ({'stage':15},False),({'battle_id':''},False)])
def test_hp_inspection_only_once_at_required_turn_in_confirmed_battle(change,inspect):
    from kaa.config.idol_road import IdolRoadConfig
    from kaa.idol_road.controller import IdolRoadController
    from kaa.idol_road.models import Action,Observation,Scene,Stage
    from kaa.idol_road.rule_conditions import RuleConditionObservation
    flow=IdolRoadController(IdolRoadConfig(characters=['ttmr'],mode='manual'))
    flow.step(Observation(Scene.STAGES,'ttmr',16,stages=(Stage(16,True,0),),stages_complete=True))
    assert flow.step(Observation(Scene.CONFIRM,'ttmr',16,stars=0,formation_ready=True)).action==Action.START_BATTLE
    flow._rules[('ttmr',16)]=StageRules('ttmr',16,(RuleViewport('a',(row(),)),))
    state=replace(RuleConditionObservation('ttmr',16,'battle','frame',7,hp=10,
        source='confirmed_current_battle_observation'),**change)
    observation=Observation(Scene.BATTLE,'ttmr',16,battle_state=state,battle_hp_info_available=True)
    assert flow.step(observation).action==(Action.INSPECT_BATTLE_HP if inspect else Action.WAIT)
    assert flow.step(replace(observation,battle_state=replace(state,frame_id='next'))).action==Action.WAIT
    assert flow.battles_started==1


def test_visible_mental_block_preserves_inclusive_percentage_and_fixed_duration():
    result = interpret_health_block(7, evidence(), original())
    assert result.status == 'confirmed_visible_declaration' and result.declaration is not None
    assert result.declaration.start_turn == 7 and result.declaration.hp_at_most_percent == 20
    assert result.declaration.blocked_category == 'mental' and result.declaration.duration_turns == 2
    assert result.declaration.executable is False


@pytest.mark.parametrize('change,available,expected', [({},True,True),({'turn':6},True,False),
    ({'frame_id':''},True,False),({'conflicted_fields':('turn',)},True,False),
    ({'source':'historical_or_unconfirmed_battle_observation'},True,False),({},False,False)])
def test_status_inspection_requires_relevant_turn_and_closes_without_reopening(change,available,expected):
    from kaa.config.idol_road import IdolRoadConfig
    from kaa.idol_road.controller import IdolRoadController
    from kaa.idol_road.models import Action,Observation,Scene,Stage
    from kaa.idol_road.rule_conditions import RuleConditionObservation
    flow=IdolRoadController(IdolRoadConfig(characters=['ttmr'],mode='manual'))
    flow.step(Observation(Scene.STAGES,'ttmr',16,stages=(Stage(16,True,0),),stages_complete=True))
    flow.step(Observation(Scene.CONFIRM,'ttmr',16,stars=0,formation_ready=True))
    flow._rules[('ttmr',16)]=StageRules('ttmr',16,(RuleViewport('a',(row(),)),))
    state=replace(RuleConditionObservation('ttmr',16,'battle','frame',7,hp=10,max_hp=48,
        source='confirmed_current_battle_observation'),**change)
    battle=Observation(Scene.BATTLE,'ttmr',16,battle_state=state,battle_status_info_available=available)
    assert flow.step(battle).action==(Action.INSPECT_BATTLE_STATUS if expected else Action.WAIT)
    if expected:
        assert flow.step(battle).reason=='await_navigation'
        assert flow.step(Observation(Scene.BATTLE_STATUS_DETAILS,'ttmr',16)).action==Action.CLOSE_BATTLE_STATUS
    assert flow.step(battle).action==Action.WAIT
    assert flow.battles_started==1


def test_hp_is_inspected_before_status_and_stalled_status_open_is_not_retried():
    from kaa.config.idol_road import IdolRoadConfig
    from kaa.idol_road.controller import IdolRoadController
    from kaa.idol_road.models import Action,Observation,Scene,Stage
    from kaa.idol_road.rule_conditions import RuleConditionObservation
    flow=IdolRoadController(IdolRoadConfig(characters=['ttmr'],mode='manual'))
    flow.step(Observation(Scene.STAGES,'ttmr',16,stages=(Stage(16,True,0),),stages_complete=True))
    flow.step(Observation(Scene.CONFIRM,'ttmr',16,stars=0,formation_ready=True))
    flow._rules[('ttmr',16)]=StageRules('ttmr',16,(RuleViewport('a',(row(),)),))
    state=RuleConditionObservation('ttmr',16,'battle','frame',7,hp=10,source='confirmed_current_battle_observation')
    battle=Observation(Scene.BATTLE,'ttmr',16,battle_state=state,battle_status_info_available=True,battle_hp_info_available=True)
    assert flow.step(battle).action==Action.INSPECT_BATTLE_HP
    assert flow.step(battle).action==Action.INSPECT_BATTLE_STATUS
    assert flow.step(battle).action==Action.WAIT
    assert flow.step(battle).action==Action.WAIT
    assert flow.step(battle).reason=='navigation_not_confirmed'


@pytest.mark.parametrize('text', ['体力が20%以上の場合、', '体力が20以下の場合、',
    '体力が101%以下の場合、', 'スコアが20%以下の場合、', '体力が20%以下の場合、以降、'])
def test_different_or_partial_condition_is_not_guessed(text):
    assert interpret_health_block(7, evidence(condition_reads=reads(text)), original()).declaration is None


@pytest.mark.parametrize('turn', [None, True, 0])
def test_unknown_turn_cannot_form_block(turn):
    assert interpret_health_block(turn, evidence(), original()).status == 'turn_unknown'


def test_all_three_same_frame_symbols_are_required():
    symbols = evidence().symbols
    for changed in ((), symbols[:2], (*symbols, symbols[0]),
                    (replace(symbols[0], score=.96), *symbols[1:]),
                    (replace(symbols[0], frame_id='old'), *symbols[1:]),
                    (symbols[0], symbols[1], replace(symbols[2], rect=(3, 0, 294, 31)))):
        assert interpret_health_block(7, evidence(symbols=changed), original()).status == 'symbols_unknown'


@pytest.mark.parametrize('confidence', [.97, float('nan'), 1.1])
def test_uncertain_hp_condition_stays_unknown(confidence):
    assert interpret_health_block(7, evidence(condition_reads=reads('体力が20%以下の場合、', confidence)), original()).declaration is None


@pytest.mark.parametrize('text', ['体力が30%以下の場合、', '体力が20%以上の場合、',
    'カード使用不可（1ターン）', 'カード使用可', 'アクティブスキルカード使用不可'])
def test_high_confidence_original_conflict_blocks_templates(text):
    result = interpret_health_block(7, evidence(), reads(text))
    assert result.status == 'high_confidence_conflict' and result.declaration is None


def test_original_or_condition_cross_frame_and_conflicts_are_unknown():
    left, right = reads('体力が20%以下の場合、')
    assert interpret_health_block(7, evidence(condition_reads=(left, replace(right, frame_id='old'))), original()).declaration is None
    assert interpret_health_block(7, evidence(condition_reads=(left, reads('体力が30%以下の場合、')[1])), original()).declaration is None
    assert interpret_health_block(7, evidence(), reads('体力が20%以下の場合、', frame='old')).declaration is None


@pytest.mark.parametrize('duration', [(), reads('1'), reads('2', .97), reads('2', frame='old')])
def test_matching_phrase_alone_cannot_force_duration_two(duration):
    assert interpret_health_block(7, evidence(duration_reads=duration), original()).status == 'duration_unknown'


def test_catalog_keeps_separate_sightings_and_unknown_rows_without_event_schedule():
    rules = StageRules('ttmr', 16, (RuleViewport('a', (row(), replace(row(), health_block=None))),
                                   RuleViewport('b', (row(),)), RuleViewport(None, None)))
    catalog = build_rule_catalog('ttmr', 16, rules)
    assert catalog.status == 'partial' and len(catalog.sightings) == 2
    assert [(s.viewport_index, s.rule_index) for s in catalog.sightings] == [(0, 0), (1, 0)]
    assert {s.reason for s in catalog.unknown_sightings} == {'unsupported_visible_rule', 'viewport_unknown'}
    assert not catalog.complete and not catalog.executable


@pytest.mark.parametrize('change,status', [({'character_id': 'hski'}, 'identity_conflict'),
    ({'stage': 13}, 'identity_conflict'), ({'source': 'database'}, 'source_unknown'),
    ({'identity_source': 'history'}, 'source_unknown')])
def test_catalog_requires_current_role_stage_and_open_request_source(change, status):
    catalog = build_rule_catalog('ttmr', 16, replace(StageRules('ttmr', 16, (RuleViewport('a', (row(),)),)), **change))
    assert catalog.status == status and not catalog.sightings


def test_catalog_recomputes_declaration_and_refuses_changed_values_or_old_turn_frame():
    current = row()
    assert current.health_block is not None and current.health_block.declaration is not None
    forged = replace(current, health_block=replace(current.health_block,
                     declaration=replace(current.health_block.declaration, hp_at_most_percent=30)))
    stale = replace(current, turn_reads=reads('7ターン目', frame='old'))
    wrong_turn = replace(current, turn=5)
    for bad in (forged, stale, wrong_turn):
        catalog = build_rule_catalog('ttmr', 16, StageRules('ttmr', 16, (RuleViewport('a', (bad,)),)))
        assert not catalog.sightings and len(catalog.unknown_sightings) == 1


def test_no_rule_viewport_is_unknown_not_a_complete_empty_rule_set():
    catalog = build_rule_catalog('ttmr', 16, StageRules('ttmr', 16, ()))
    assert not catalog.sightings and len(catalog.unknown_sightings) == 1
    assert catalog.unknown_sightings[0].viewport_index is None and not catalog.complete


def test_preparation_exposes_catalog_without_authorizing_battle():
    from kaa.idol_road.models import Observation, Scene
    from kaa.idol_road.preparation import prepare_battle
    rules = StageRules('ttmr', 16, (RuleViewport('a', (row(),)),))
    report = prepare_battle('ttmr', 16, Observation(Scene.CONFIRM, 'ttmr', 16, stars=0, formation_ready=True), stage_rules=rules)
    assert report.rule_declarations is not None and len(report.rule_declarations.sightings) == 1
    assert not report.rule_declarations.complete and not report.automatic_battle_ready
