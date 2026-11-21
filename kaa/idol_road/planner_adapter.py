"""独立单场Planner接入；共享识牌／出牌循环，不伪造育成会话或复制评分。"""
from time import monotonic
from dataclasses import asdict
import logging
from typing import Callable,Literal,TYPE_CHECKING
from cv2.typing import MatLike
from .models import Scene,Action,Decision
from .battle_observation import BattleObservationSession,current_absolute_turn
from .planner_trial import PlannerTrialSpec
from .turn_end import UnplayableTurnGate, PlannedTurnWait
if TYPE_CHECKING:
    from .page import IdolRoadPage,PageSnapshot
    from .rules import StageRules

logger = logging.getLogger(__name__)


def prepare_planner_assets() -> None:
    """在发出挑战前初始化现有识牌库；缺失素材不消耗挑战。"""
    from kaa.tasks.produce.new.play_cards.ui import skill_cards_db
    try:
        skill_cards_db()
    except (OSError,ValueError) as exc:
        raise ValueError('AI识牌素材库未就绪，请先完成游戏数据准备；未发出挑战') from exc


class TrialFrameGuard:
    def __init__(self,session: BattleObservationSession,deadline: float,*,clock: Callable[[],float]=monotonic):
        if session.turn_budget_mode not in ('fixed_budget','variable_budget'):
            raise ValueError('AI试跑的回合预算类型未知')
        self.session=session
        self.deadline=deadline
        self.clock=clock
        self.unknown_since: float | None=None
        self.last_turn_seen=False
        self.log_since: float | None=None

    def check(self,snapshot: 'PageSnapshot',*,move_dialog: bool=False) -> Literal['continue','wait','stop']:
        now=self.clock()
        if now>=self.deadline:
            raise ValueError('AI单场试运行达到总时限，已停止输入')
        obs=snapshot.observation
        if obs.scene in (Scene.JUDGING,Scene.RESULT):
            if (obs.character_id,obs.stage)!=(self.session.character_id,self.session.stage):
                raise ValueError('AI试运行结果身份无法确认')
            return 'stop'
        frame=snapshot.battle_frame
        if obs.scene==Scene.BATTLE_LOG:
            inspection=obs.battle_log_inspection
            if inspection is None or inspection.request is None or inspection.request.session!=self.session:
                raise ValueError('AI试运行遇到未绑定当前战斗的日志')
            if self.log_since is None:
                self.log_since=now
            if now-self.log_since>=35:
                raise ValueError('AI试运行日志未确认关闭，已停止输入')
            return 'wait'
        if obs.scene==Scene.BATTLE and obs.battle_log_pending:
            if frame is None or frame.session!=self.session:
                raise ValueError('AI试运行日志返回会话不匹配')
            if self.log_since is None:
                self.log_since=now
            if now-self.log_since>=35:
                raise ValueError('AI试运行日志返回持续未知，已停止输入')
            return 'wait'
        if obs.scene==Scene.BATTLE and not obs.battle_log_pending:
            self.log_since=None
        elif self.log_since is not None and now-self.log_since>=35:
            raise ValueError('AI试运行日志关闭后的画面持续未知，已停止输入')
        if obs.scene==Scene.BATTLE and frame is not None:
            if frame.session!=self.session:
                raise ValueError('AI试运行战斗会话发生变化')
            if frame.values.detail_mode!='complete':
                self.unknown_since=None
                return 'wait'
            dynamic_turn_unknown=(self.session.turn_budget_mode=='variable_budget'
                and current_absolute_turn(frame) is None)
            if frame.values.resource_state in ('current','selection_preview') and not dynamic_turn_unknown:
                self.unknown_since=None
                if frame.values.resource_state=='current':
                    self.last_turn_seen=(self.session.turn_budget_mode=='fixed_budget' and frame.values.remaining_turns==1
                        and type(self.session.total_turns) is int
                        and frame.observation.turn==self.session.total_turns
                        and frame.observation.source=='confirmed_current_battle_observation'
                        and frame.observation.frame_id==frame.values.frame_id
                        and 'turn' not in frame.observation.conflicted_fields)
                return 'continue'
            # 未确认遮挡不能让共享循环越过面板去解析／点击后方手牌。
        if move_dialog and obs.scene==Scene.UNKNOWN:
            # 共享移牌处理器在排序缺失时仍有旧顺序后备；单场实验不越过严格停止边界。
            raise ValueError('AI单场试运行尚未支持移牌选择，已停止输入')
        if obs.scene not in (Scene.UNKNOWN,Scene.BATTLE):
            raise ValueError(f'AI试运行遇到未支持的交互：{obs.scene.value}')
        if self.unknown_since is None:
            self.unknown_since=now
        # 仅当前会话明确走到末回合后，给未知名次演出60秒无输入等待。
        # 不是结果识别，不授权点击；普通未知／遮挡仍保留20秒界限。
        limit=60 if self.last_turn_seen and obs.scene==Scene.UNKNOWN else 20
        if now-self.unknown_since>=limit:
            raise ValueError('AI试运行页面持续未知，已停止输入')
        return 'wait'


def run_planner_trial(spec: PlannerTrialSpec,page: 'IdolRoadPage',session: BattleObservationSession,
                      *,deadline: float,observe: Callable[[MatLike,'PageSnapshot | None'],None],
                      stage_rules: 'StageRules | None' = None) -> None:
    if session.turn_budget_mode=='variable_budget':
        from .planner_trial import UnsupportedTurnBudget
        raise UnsupportedTurnBudget('变量回合的完整出牌流程尚未验收，不能启动AI试跑')
    from kotonebot.backend.image import find
    from kaa.tasks import R
    from kaa.tasks.produce.shared.cards import do_cards
    from kaa.tasks.produce.new.play_cards.planner.strategy import PlannerStrategy
    from .turn_budget import fixed_turn_budget_conflicts
    from .planner_trial import UnsupportedTurnBudget
    if fixed_turn_budget_conflicts(spec.character_id,spec.stage,stage_rules):
        raise UnsupportedTurnBudget('当前场地含追加回合，固定总回合试跑尚未适配')
    from kaa.db.constants import ProduceExamEffectType
    if (spec.scope,spec.archetype) not in (
        ('single_concentration_trial',ProduceExamEffectType.ExamLessonBuff),
        ('single_good_condition_trial',ProduceExamEffectType.ExamParameterBuff)):
        raise ValueError('AI试跑流派与单场范围不匹配')
    if (session.identity_source!='confirmed_formation_start_request'
        or (session.character_id,session.stage,session.total_turns)!=(spec.character_id,spec.stage,spec.total_turns)):
        raise ValueError('AI单场试运行缺少匹配的已确认开战会话')
    from .category_rule import category_objectives
    from .concentration_priority import concentration_objectives
    from .field_priority import FieldPriority
    from .good_condition_rule import good_condition_planning_requirements
    from .score_priority import score_preparation_objectives
    from .hp_preparation import hp_preparation_budget
    objectives = category_objectives(spec.character_id, spec.stage, stage_rules)
    concentration_goals = concentration_objectives(spec.character_id, spec.stage, stage_rules)
    good_goals = good_condition_planning_requirements(spec.character_id,spec.stage,stage_rules)
    score_goals=score_preparation_objectives(spec.character_id,spec.stage,stage_rules)
    hp_budget=hp_preparation_budget(spec.character_id,spec.stage,stage_rules)
    from .hp_safety import HpSafetyPolicy
    hp_guard=HpSafetyPolicy(spec.total_turns,hp_budget) if hp_budget is not None and (hp_budget.goals or hp_budget.recurring_known) else None
    logger.info('Idol road field objectives (planning only): %s', [asdict(goal) for goal in objectives])
    logger.info('Idol road concentration preparation (phase/reward unconfirmed): %s', [asdict(goal) for goal in concentration_goals])
    logger.info('Idol road good-condition preparation (phase/reward unconfirmed): %s', [asdict(goal) for goal in good_goals])
    logger.info('Idol road score preparation (reward unconfirmed): %s',[asdict(goal) for goal in score_goals])
    logger.info('Idol road HP preparation (supported declarations only): %s',asdict(hp_budget) if hp_budget else None)
    guard=TrialFrameGuard(session,deadline)
    inputs=page.battle_inputs
    if inputs is None or inputs.session!=session:
        raise ValueError('AI单场试运行缺少本次开战的输入记录')
    turn_gate=UnplayableTurnGate()
    planned_wait=PlannedTurnWait(clock=inputs.clock)
    latest_screen: MatLike | None=None
    latest_snapshot: 'PageSnapshot | None'=None

    def frame_policy(screen: MatLike) -> Literal['continue','wait','stop']:
        nonlocal latest_screen,latest_snapshot
        observe(screen,None)
        snapshot=page.observe(screen,spec.character_id)
        latest_screen,latest_snapshot=screen,snapshot
        observe(screen,snapshot)
        move=(snapshot.observation.scene==Scene.UNKNOWN and find(
            screen,R.InProduce.IconTitleSkillCardMove.template,threshold=.97) is not None)
        verdict = guard.check(snapshot,move_dialog=move)
        resolution=snapshot.observation.battle_card_resolution
        if (inputs.pending is not None and inputs.pending.kind=='card' and resolution is not None
            and resolution.status=='unconfirmed' and resolution.unconfirmed_reason=='tail_header_unavailable'):
            from kaa.tasks.produce.new.play_cards.planner.selection_policy import SelectionPolicyStop
            raise SelectionPolicyStop('本次日志尾部没有可核验的回合标题，不能结算出牌；长日志跨视口证据尚未支持')
        if verdict!='stop' and snapshot.observation.scene!=Scene.BATTLE_LOG and inputs.snapshot().expired:
            raise ValueError('爬塔输入20秒未确认结算，已停止输入')
        if (snapshot.observation.scene==Scene.BATTLE and snapshot.observation.battle_log_available
            and inputs.pending is not None):
            page.execute(snapshot,Decision(Action.INSPECT_BATTLE_LOG,'read_only_pending_input_probe',session.character_id,session.stage))
            return 'wait'
        if snapshot.observation.scene==Scene.BATTLE_LOG:
            from .battle_log import log_inspection_action
            inspection=snapshot.observation.battle_log_inspection
            if inspection is not None:
                action=log_inspection_action(inspection,session.battle_id)
                page.execute(snapshot,Decision({'wait':Action.WAIT,'scroll':Action.SCROLL_BATTLE_LOG,'scroll_header':Action.SCROLL_BATTLE_LOG_HEADER,
                    'close':Action.CLOSE_BATTLE_LOG}[action],'owned_log_lifecycle',session.character_id,session.stage))
        if verdict != 'continue':
            return verdict
        frame = snapshot.battle_frame
        if inputs.pending is not None and inputs.pending.kind=='card':
            from .card_probe import card_feedback_ready
            if not card_feedback_ready(session,inputs.snapshot(),frame,snapshot.observation.battle_card_resolution,now=inputs.clock()):
                return 'wait'
        turn = (frame.observation.turn if frame is not None and frame.values.resource_state == 'current'
                and not frame.observation.conflicted_fields else None)
        planned = planned_wait.check(turn)
        if planned == 'stop':
            raise ValueError('场地末手结束回合后未确认新回合，已停止输入')
        return planned

    def result_observer(screen: MatLike) -> None:
        frame_policy(screen)

    def battle_context_factory(screen: MatLike):
        from .planner_context import IdolRoadBattleContext
        snapshot=latest_snapshot
        if (screen is not latest_screen or snapshot is None or snapshot.battle_frame is None
            or snapshot.battle_frame.session!=session or snapshot.battle_frame.values.resource_state!='current'):
            raise ValueError('Planner缺少同一截图的当前爬塔状态')
        return IdolRoadBattleContext(screen,snapshot.battle_frame,
            before_commit=lambda frame,card_id:page.begin_battle_input(frame,'card',card_id),
            hand_context=snapshot.hand_context)

    def request_turn_end(screen: MatLike, *, planned: bool = False) -> bool:
        from kotonebot import device
        from .turn_end_page import read_turn_end
        snapshot=latest_snapshot
        if screen is not latest_screen or snapshot is None or snapshot.battle_frame is None:
            return False
        frame=snapshot.battle_frame
        state=frame.observation
        if (snapshot.observation.scene!=Scene.BATTLE or frame.session!=session
            or frame.values.resource_state!='current'
            or state.source!='confirmed_current_battle_observation'
            or state.conflicted_fields or type(state.turn) is not int):
            return False
        blocked,enabled,cards,count=read_turn_end(screen)
        action=turn_gate.step(session.battle_id,state.turn,blocked or planned,enabled)
        if action=='stop':
            raise ValueError('AI结束回合后未确认新回合，已停止输入')
        if action=='click':
            if hp_guard is not None:
                from kaa.tasks.produce.new.play_cards.planner.state import BattleState
                from .planner_context import IdolRoadBattleContext
                from kaa.tasks.produce.new.play_cards.planner.selection_policy import SelectionPolicyStop
                context=IdolRoadBattleContext(screen,frame,hand_context=snapshot.hand_context)
                end_state=BattleState(frame.values.remaining_turns or 0,state.hp if state.hp is not None else 0,0,
                    hp_known=state.hp is not None,max_hp=context.fetch_exam_max_hp(),
                    remaining_turns_known=frame.values.remaining_turns is not None,
                    absolute_turn_observation=frame.absolute_turn)
                safety=hp_guard.end_turn(end_state)
                if safety.stop:
                    raise SelectionPolicyStop(safety.reason)
            logger.info('Idol road end-turn request: reason=%s battle_id=%s turn=%s frame=%s count=%s cards=%s',
                        'preserve_confirmed_field_last_card' if planned else 'empty_hand' if count==0 else 'blocked_hand',
                        session.battle_id,state.turn,frame.values.frame_id,count,[asdict(row) for row in cards])
            page.begin_battle_input(frame,'skip')
            pending=inputs.pending
            if pending is None or pending.kind!='skip':
                raise ValueError('结束回合输入未登记，已停止输入')
            planned_wait.requested(state.turn,started=pending.started)
            device.click(R.IdolRoad.BoxBattleSkipLabel)
        return action in ('click','wait')

    from .input_policy import InputBoundPolicy
    field=FieldPriority(spec.total_turns,objectives,concentration_goals,good_goals,score_goals,hp_budget) if objectives or concentration_goals or good_goals or score_goals else None
    strategy=PlannerStrategy(is_exam=True,archetype=spec.archetype,rank_at_most=spec.rank_at_most,
                             selection_policy=InputBoundPolicy(inputs,lambda:latest_snapshot.battle_frame if latest_snapshot else None,field,hp_guard),
                             end_turn=lambda ctx: request_turn_end(ctx.fetch_screen(),planned=True))
    try:
        do_cards(True,lambda count,result: False,lambda: False,battle_strategy=strategy,
                 result_observer=result_observer,enable_drink=False,allow_recommended_fallback=False,
                 frame_policy=frame_policy,turn_end_handler=request_turn_end,
                 battle_context_factory=battle_context_factory)
    finally:
        strategy.on_battle_end()
