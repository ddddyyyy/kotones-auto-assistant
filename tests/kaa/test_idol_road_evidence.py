import json

import cv2
import numpy as np

from kaa.idol_road.evidence import EVIDENCE_ENV, RunEvidence
from kaa.idol_road.models import Action, Decision, Observation, Scene
from kaa.idol_road.objectives import StageObjectives, read_rank_requirement
from kaa.idol_road.items import ItemInventory
from kaa.idol_road.preparation import prepare_battle
from kaa.idol_road.rules import RuleViewport, StageRules


def test_evidence_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(EVIDENCE_ENV, raising=False)
    assert RunEvidence.from_environment() is None


def test_environment_selects_directory_without_creating_it(monkeypatch, tmp_path):
    directory = tmp_path / '证据'
    monkeypatch.setenv(EVIDENCE_ENV, str(directory))
    evidence = RunEvidence.from_environment()
    assert evidence is not None and evidence.directory.parent == directory
    assert not directory.exists()


def test_only_last_three_original_frames_are_saved_once(tmp_path):
    evidence = RunEvidence(tmp_path / '中文路径')
    for value in range(5):
        frame = np.full((4, 4, 3), value, dtype=np.uint8)
        evidence.observe(frame)
        frame[:] = 99
    observation = Observation(Scene.CONFIRM, character_id='ttmr', stage=16, stars=0)
    decision = Decision(Action.READY, 'preview_complete', 'ttmr', 16)
    evidence.finish('preview_complete', observation, decision, 0)
    report = json.loads((evidence.directory / 'stop.json').read_text(encoding='utf-8'))
    assert report['reason'] == 'preview_complete' and report['battles_started'] == 0
    assert report['observation']['stars'] == 0 and report['decision']['action'] == 'ready'
    assert report['frames'] == ['frame-0.png', 'frame-1.png', 'frame-2.png']
    for index, expected in enumerate((2, 3, 4)):
        encoded = np.frombuffer((evidence.directory / f'frame-{index}.png').read_bytes(), dtype=np.uint8)
        assert np.all(cv2.imdecode(encoded, cv2.IMREAD_COLOR) == expected)
    evidence.observe(np.zeros((4, 4, 3), dtype=np.uint8))
    evidence.finish('wrong', None, None, 99)
    assert json.loads((evidence.directory / 'stop.json').read_text(encoding='utf-8')) == report


def test_no_observation_is_not_filled_from_history(tmp_path):
    evidence = RunEvidence(tmp_path)
    evidence.finish('interrupted_or_error', None, None, 0)
    report = json.loads((evidence.directory / 'stop.json').read_text(encoding='utf-8'))
    assert report['observation'] is None and report['decision'] is None and report['frames'] == []
    assert report['battle_frame'] is None and report['rule_conditions'] is None


def test_planner_trial_enum_is_serialized_without_masking_stop_reason(tmp_path):
    from kaa.idol_road.planner_trial import PlannerTrialSpec
    evidence=RunEvidence(tmp_path)
    evidence.finish('planner_trial_stopped',None,None,1,planner_trial=PlannerTrialSpec('ttmr',16,3,10))
    report=json.loads((evidence.directory/'stop.json').read_text(encoding='utf-8'))
    assert report['reason']=='planner_trial_stopped'
    assert report['planner_trial']['archetype']=='ProduceExamEffectType_ExamLessonBuff'
    assert report['planner_trial']['rank_at_most']==3


def test_trace_survives_result_without_polluting_latest_stop_frame(tmp_path):
    from kaa.idol_road.battle_observation import BattleFrameValues, BattleObservationSession, bind_battle_frame
    frame=bind_battle_frame(BattleObservationSession('battle','ttmr',16,10,'confirmed_formation_start_request'),
                            BattleFrameValues('current',True,4,10))
    evidence=RunEvidence(tmp_path)
    evidence.observe_step(Observation(Scene.BATTLE,'ttmr',16),Decision(Action.WAIT,'manual','ttmr',16),frame)
    result=Observation(Scene.RESULT,'ttmr',16)
    evidence.observe_step(result,Decision(Action.WAIT,'unknown_result','ttmr',16))
    evidence.finish('unknown_result',result,None,1)
    report=json.loads((evidence.directory/'stop.json').read_text(encoding='utf-8'))
    trace=json.loads((evidence.directory/report['battle_trace']).read_text(encoding='utf-8'))
    assert report['battle_frame'] is None and report['rule_conditions'] is None
    assert trace['steps'][0]['battle_frame']['observation']['hp']==10
    assert trace['steps'][1]['battle_frame'] is None and not trace['replay_ready']
    evidence.observe_step(None,None)
    evidence.finish('again',None,None,99)
    assert json.loads((evidence.directory/'battle-trace.json').read_text(encoding='utf-8'))==trace


def test_battle_frame_keeps_historical_provenance_and_unknown_max_hp(tmp_path):
    from kaa.idol_road.battle_observation import BattleFrameValues, BattleObservationSession, bind_battle_frame
    frame = bind_battle_frame(BattleObservationSession('replay','hski',13,18,'historical_capture_metadata'),
                              BattleFrameValues('current-frame',True,18,49))
    evidence = RunEvidence(tmp_path)
    evidence.finish('replay_only', None, None, 0, battle_frame=frame)
    report = json.loads((evidence.directory / 'stop.json').read_text(encoding='utf-8'))
    assert report['battle_frame']['observation']['source'] == 'historical_or_unconfirmed_battle_observation'
    assert report['battle_frame']['observation']['max_hp'] is None
    assert report['rule_conditions'] is None


def test_write_failure_does_not_mask_original_stop(tmp_path, caplog):
    blocker = tmp_path / 'file'
    blocker.write_text('do not overwrite', encoding='utf-8')
    evidence = RunEvidence(blocker)
    evidence.finish('stage_stars_unknown', None, None, 0)
    assert blocker.read_text(encoding='utf-8') == 'do not overwrite'
    assert '无法保存偶像之路停止证据' in caplog.text


def test_condition_goals_are_serialized_separately_from_actual_stars(tmp_path):
    evidence = RunEvidence(tmp_path)
    goals = StageObjectives('ttmr', 16, (read_rank_requirement(1, '3位以上でクリア'),))
    evidence.finish('preview_complete', Observation(Scene.CONFIRM, stars=0),
                    Decision(Action.READY, 'preview_complete', 'ttmr', 16), 0, goals)
    report = json.loads((evidence.directory / 'stop.json').read_text(encoding='utf-8'))
    assert report['observation']['stars'] == 0
    assert report['objectives']['requirements'][0]['stars'] == 1
    assert report['objectives']['identity_source'] == 'opened_from_confirmed_formation'


def test_visible_inventory_stays_explicitly_incomplete_in_evidence(tmp_path):
    evidence = RunEvidence(tmp_path)
    inventory = ItemInventory('hski', 13, None)
    evidence.finish('preview_complete', Observation(Scene.CONFIRM, stars=0),
                    Decision(Action.READY, 'preview_complete', 'hski', 13), 0,
                    item_inventory=inventory)
    report = json.loads((evidence.directory / 'stop.json').read_text(encoding='utf-8'))
    assert report['item_inventory']['complete'] is False
    assert report['item_inventory']['visible_items'] is None


def test_preparation_serializes_target_without_authorizing_automatic_battle(tmp_path):
    evidence = RunEvidence(tmp_path)
    observation = Observation(Scene.CONFIRM, character_id='ttmr', stage=16, stars=0,
                              formation_ready=True)
    goals = StageObjectives('ttmr', 16, tuple(
        read_rank_requirement(stars, f'{4-stars}位以上でクリア') for stars in (3, 2, 1)
    ))
    preparation = prepare_battle('ttmr', 16, observation, objectives=goals)
    evidence.finish('preview_complete', observation,
                    Decision(Action.READY, 'preview_complete', 'ttmr', 16), 0,
                    goals, preparation=preparation)
    report = json.loads((evidence.directory / 'stop.json').read_text(encoding='utf-8'))
    assert report['observation']['stars'] == 0
    assert report['battle_preparation']['formation_confirmed'] is True
    assert report['battle_preparation']['spec']['rank_at_most'] == 3
    assert report['battle_preparation']['spec']['target_stars'] == 1
    assert report['battle_preparation']['automatic_battle_ready'] is False
    assert 'battle_adapter_unimplemented' in {
        issue['code'] for issue in report['battle_preparation']['issues']
    }


def test_rule_text_observations_are_not_serialized_as_complete_supported_effects(tmp_path):
    evidence = RunEvidence(tmp_path)
    rules = StageRules('ttmr', 16, (RuleViewport(None, None),))
    evidence.finish('preview_complete', None, None, 0, stage_rules=rules)
    report = json.loads((evidence.directory / 'stop.json').read_text(encoding='utf-8'))
    assert report['stage_rules']['complete'] is False
    assert report['stage_rules']['semantics_confirmed'] is False
    assert report['stage_rules']['viewports'][0]['visible_rules'] is None
    assert report['stage_rules']['identity_source'] == 'opened_from_confirmed_formation'


def test_pending_card_state_enum_and_frozen_effects_are_saved_in_battle_trace(tmp_path):
    from dataclasses import replace
    from kaa.db.constants import ProduceExamEffectType
    from kaa.tasks.produce.new.play_cards.planner.state import BattleState
    from tests.kaa.test_idol_road_hud_feedback import scenario
    inputs,_=scenario()
    before=BattleState(20,48,0,archetype=ProduceExamEffectType.ExamLessonBuff,
        observed_effects=frozenset({'lesson_buff'}))
    inputs=replace(inputs,pending=replace(inputs.pending,before=before))
    observation=Observation(Scene.BATTLE,character_id='ttmr',stage=20,battle_input_state=inputs)
    evidence=RunEvidence(tmp_path)
    evidence.observe_step(observation,Decision(Action.WAIT,'owned_input_pending'))
    evidence.finish('battle_limit',None,None,1)
    stop=json.loads((evidence.directory/'stop.json').read_text(encoding='utf-8'))
    trace=json.loads((evidence.directory/'battle-trace.json').read_text(encoding='utf-8'))
    saved=trace['steps'][0]['input_state']['pending']['before']
    assert stop['reason']=='battle_limit' and saved['archetype']==ProduceExamEffectType.ExamLessonBuff.value
    assert saved['observed_effects']==['lesson_buff']
