"""偶像之路薄任务入口，导航与战斗决策分离。"""

import logging
from time import monotonic

from kotonebot import Loop, device, task
from kotonebot.errors import UserFriendlyError

from kaa.config import conf
from kaa.idol_road.controller import IdolRoadController
from kaa.idol_road.models import Action, Decision, Observation, Scene
from kaa.idol_road.preparation import BattlePreparation, prepare_battle
from kaa.idol_road.battle_observation import BattleFrameObservation
from kaa.idol_road.rule_catalog import build_rule_catalog
from kaa.idol_road.rule_conditions import RuleConditionReport, evaluate_rule_conditions
from kaa.idol_road.planner_trial import PlannerTrialSpec,prepare_planner_trial
from kaa.idol_road.initialization import BattleInitializationError,initialize_variable_battle
from kaa.idol_road.battle_inputs import BattleInputs


logger = logging.getLogger(__name__)


@task('偶像之路（实验性）', screenshot_mode='manual')
def idol_road():
    from kaa.idol_road.evidence import RunEvidence
    from kaa.idol_road.page import IdolRoadPage,PageSnapshot

    config = conf().tasks.idol_road
    if not config.characters:
        logger.warning('偶像之路未选择角色，跳过。')
        return
    controller = IdolRoadController(config)
    page = IdolRoadPage(feedback_only_returns=config.mode=='auto')
    evidence = RunEvidence.from_environment()
    unknown_count = 0
    finish_reason = 'interrupted_or_error'
    observation: Observation | None = None
    decision: Decision | None = None
    preparation: BattlePreparation | None = None
    battle_frame: BattleFrameObservation | None = None
    condition_report: RuleConditionReport | None = None
    trial_spec: PlannerTrialSpec | None = None
    trial_started=False
    deadline=monotonic()+config.timeout_seconds
    auto_inputs: BattleInputs | None=None

    def observe_auto(screen,snapshot: PageSnapshot,current_decision: Decision):
        nonlocal observation,decision,battle_frame,condition_report
        controller.observe_requested_route(auto_inputs)
        observation,decision=snapshot.observation,current_decision
        battle_frame=snapshot.battle_frame
        condition_report=None
        if evidence is not None:
            evidence.observe(screen)
            evidence.observe_step(observation,decision,battle_frame,None)

    def observe_initial(screen,snapshot: PageSnapshot,initial_decision: Decision):
        nonlocal observation,decision,battle_frame,condition_report
        current=snapshot.observation
        observation,decision=current,initial_decision
        battle_frame=snapshot.battle_frame
        condition_report=None
        if (initial_decision.action!=Action.STOP and current.scene==Scene.BATTLE and battle_frame is not None
            and battle_frame.values.resource_state=='current'):
            controller.observe_initial_battle(current,ready=initial_decision.action==Action.READY)
        if evidence is not None:
            evidence.observe(screen)
            evidence.observe_step(observation,decision,battle_frame,None)

    def initial_frames():
        for _ in Loop(interval=.3):
            yield device.screenshot()

    def observe_trial(screen, snapshot):
        nonlocal observation,decision,battle_frame,condition_report
        observation=decision=battle_frame=condition_report=None
        if snapshot is None:
            if evidence is not None:
                evidence.observe(screen)
            return
        current=snapshot.observation
        observation=current
        battle_frame=snapshot.battle_frame
        decision=Decision(Action.WAIT,'planner_trial_observation',controller.character_id,current.stage)
        if battle_frame is not None and controller.stage_rules is not None:
            state=battle_frame.observation
            condition_report=evaluate_rule_conditions(build_rule_catalog(state.character_id,state.stage,controller.stage_rules),
                state,battle_id=battle_frame.session.battle_id)
        if evidence is not None:
            evidence.observe_step(observation,decision,battle_frame,condition_report)
    try:
        for _ in Loop(interval=1):
            screen = device.screenshot()
            # 清掉上一帧解析，异常时不把旧身份附到新的未识别画面上。
            observation = None
            decision = None
            preparation = None
            battle_frame = None
            condition_report = None
            if evidence is not None:
                evidence.observe(screen)
            controller.observe_requested_route(page.battle_inputs)
            snapshot = page.observe(screen, controller.character_id)
            observation = snapshot.observation
            unknown_count = unknown_count + 1 if observation.scene == Scene.UNKNOWN else 0
            # 已观测战斗后的名次演出明显长于导航切页；只等待，不点击未知画面。
            unknown_limit = 20 if controller.battles_started or controller.awaiting_battle_start else 3
            if unknown_count >= unknown_limit:
                finish_reason = 'unknown_scene_limit'
                raise UserFriendlyError('偶像之路页面在有限刷新后仍无法确认，已停止；请回到主页或角色页重试。')
            decision = controller.step(observation)
            battle_frame = snapshot.battle_frame
            session = page.battle_session
            if battle_frame is not None and session is not None and decision.action in (
                Action.WAIT, Action.INSPECT_BATTLE_HP, Action.INSPECT_BATTLE_STATUS,
            ):
                state = battle_frame.observation
                if controller.stage_rules is not None:
                    catalog = build_rule_catalog(state.character_id, state.stage, controller.stage_rules)
                    condition_report = evaluate_rule_conditions(catalog, state, battle_id=session.battle_id)
                    logger.info('Idol road battle conditions (not executed): turn=%s hp=%s checks=%s',
                                state.turn, state.hp,
                                [(check.sighting.viewport_index, check.sighting.rule_index, check.verdict.value, check.reason)
                                 for check in condition_report.checks])
            if evidence is not None:
                evidence.observe_step(observation, decision, battle_frame, condition_report)
            if decision.action in (Action.READY, Action.START_BATTLE):
                if config.mode=='auto' and decision.action==Action.START_BATTLE:
                    from kaa.idol_road.planner_adapter import prepare_planner_assets
                    prepare_planner_assets()
                if config.mode=='planner_trial' and decision.action==Action.START_BATTLE:
                    trial_spec=prepare_planner_trial(observation,controller.objectives,controller.stage_rules)
                    from kaa.idol_road.planner_adapter import prepare_planner_assets
                    prepare_planner_assets()
                preparation = prepare_battle(decision.character_id, decision.stage, observation,
                                             objectives=controller.objectives, inventory=controller.item_inventory,
                                             stage_rules=controller.stage_rules)
                logger.info('完整自动爬塔前置缺口（单场实验不代表完整就绪）：%s',
                            '；'.join(issue.message for issue in preparation.issues))
            focused = observation.focused_stage
            observed_stage = focused.number if focused is not None else observation.stage
            observed_stars = focused.stars if focused is not None else observation.stars
            logger.info('Idol road: scene=%s action=%s reason=%s character=%s observed_character=%s observed_stage=%s stars=%s target_stage=%s',
                        observation.scene.value, decision.action.value, decision.reason,
                        decision.character_id, observation.character_id, observed_stage, observed_stars, decision.stage)
            if observation.scene == Scene.CONDITIONS:
                logger.info('Idol road conditions (goals, not result): %s', observation.requirements)
            if observation.scene == Scene.SUPPORT_DETAILS:
                viewport = observation.rule_viewport
                logger.info('Idol road support (partial text only): turns=%s',
                            [rule.turn for rule in viewport.visible_rules]
                            if viewport is not None and viewport.visible_rules is not None else None)
                if viewport is not None and viewport.visible_rules:
                    logger.info('Idol road rule references (partial candidates only): %s',
                                [(rule.turn, rule.description_match.status, len(rule.description_match.candidates))
                                 for rule in viewport.visible_rules if rule.description_match is not None])
                    logger.info('Idol road declared draw (candidate definitions only): %s',
                                [(rule.turn, rule.program_analysis.common_draw_declaration)
                                 for rule in viewport.visible_rules if rule.program_analysis is not None])
                    logger.info('Idol road visible conditional plays (not executable): %s',
                                [(rule.turn, rule.conditional_play.status, rule.conditional_play.declaration)
                                 for rule in viewport.visible_rules if rule.conditional_play is not None])
                    logger.info('Idol road visible mental block (not executable): %s',
                                [(rule.turn, rule.health_block.status, rule.health_block.declaration)
                                 for rule in viewport.visible_rules if rule.health_block is not None])
            if observation.scene == Scene.ITEM_DETAILS:
                logger.info('Idol road visible items (incomplete inventory): %s',
                            [(item.section, item.reference.confirmed_name, item.reference.status.value,
                              len(item.reference.candidate_ids), item.reference.item_id)
                             for item in observation.visible_items] if observation.visible_items is not None else None)
                inventory = controller.item_inventory
                if inventory is not None and inventory.overlaps and decision.action != Action.WAIT:
                    overlap = inventory.overlaps[-1]
                    logger.info('Idol road overlap (candidate only): status=%s links=%s shift=%s',
                                overlap.status.value,
                                [(link.previous_index, link.current_index) for link in overlap.links], overlap.content_shift)
            if decision.action == Action.READY:
                finish_reason = decision.reason
                logger.info('已停在编成页，预览完成；未点击挑战。')
                return
            if (config.mode=='planner_trial' and not trial_started and trial_spec is not None
                and observation.scene==Scene.BATTLE and decision.action==Action.WAIT
                and decision.reason=='manual_battle_in_progress' and page.battle_session is not None):
                from kaa.idol_road.planner_adapter import run_planner_trial
                trial_started=True
                logger.info('启动显式单场Planner试运行：%s；道具／规则收益尚未模拟。',trial_spec)
                run_planner_trial(trial_spec,page,page.battle_session,deadline=deadline,observe=observe_trial,
                                  stage_rules=controller.stage_rules)
                continue
            if decision.action == Action.STOP:
                finish_reason = decision.reason
                logger.info('偶像之路停止：%s（本轮挑战 %d 场）', decision.reason, controller.battles_started)
                return
            if decision.action==Action.START_BATTLE:
                auto_spec=None
                if config.mode=='auto':
                    from kaa.idol_road.battle_start import prepare_variable_battle
                    auto_spec=prepare_variable_battle(observation,controller.objectives,controller.stage_rules)
                page.execute(snapshot,decision,stage_rules=controller.stage_rules,observed_turns=config.mode=='auto')
                owner=page.battle_session
                if config.mode=='auto' and (owner is None or owner.turn_budget_mode!='variable_budget'):
                    raise ValueError('automatic_variable_start_owner_missing')
                if config.mode in ('manual','auto') and owner is not None and owner.turn_budget_mode=='variable_budget':
                    first_screen,first_snapshot=initialize_variable_battle(
                        page,owner,initial_frames(),deadline=deadline,observe=observe_initial,
                        feedback_mode='hud' if config.mode=='auto' else 'log')
                    if config.mode=='auto':
                        from kaa.idol_road.battle_runner import run_variable_battle
                        rules=controller.stage_rules
                        if auto_spec is None or rules is None:
                            raise ValueError('automatic_battle_preflight_missing')
                        auto_inputs=page.battle_inputs
                        result=run_variable_battle(page,owner,first_screen,first_snapshot,initial_frames(),
                            stage_rules=rules,deadline=deadline,observe=observe_auto,
                            retry_policy=controller.retry_policy,rank_at_most=auto_spec.rank_at_most,
                            feedback_mode='hud',allow_audited_redraw=True)
                        controller.observe_requested_route(auto_inputs)
                        auto_inputs=None
                        if result.reason!='owned_battle_finished':
                            finish_reason='automatic_battle_'+result.reason
                            logger.info('偶像之路自动战斗停止：%s；本场不是确认失败。',finish_reason)
                            return
            else:
                page.execute(snapshot, decision)
    except BattleInitializationError as exc:
        finish_reason=str(exc)
        raise UserFriendlyError(f'偶像之路初始战斗观察停止：{exc}') from exc
    except ValueError as exc:
        finish_reason = 'planner_trial_stopped' if trial_started else 'page_or_action_not_confirmed'
        raise UserFriendlyError(str(exc)) from exc
    finally:
        if evidence is not None:
            evidence.finish(finish_reason, observation, decision, controller.battles_started,
                            controller.objectives, controller.item_inventory, preparation, controller.stage_rules,
                            battle_frame, condition_report,trial_spec)
