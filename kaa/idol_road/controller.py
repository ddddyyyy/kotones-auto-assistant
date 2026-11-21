"""基于页面观察的导航决策。调用方负责取图、执行动作、取消和限频。

此模块不操作设备，也不导入育成会话或 Planner。每次调用只处理一帧，
便于后续页面适配器与战斗适配器独立演进。
"""

from time import monotonic
from typing import Callable

from kaa.config.idol_road import IdolRoadConfig

from .models import Action, Decision, Observation, Outcome, Scene
from .navigation import NavigationGuard
from .objectives import StageObjectives
from .items import ItemInventory
from .item_scan import ItemScan
from .item_overlap import associate_item_viewports
from .rules import RuleViewport, StageRules
from .viewport_scan import ViewportScan
from .rule_catalog import build_rule_catalog
from .status_details import BattleStatusDetails
from .status_scan import status_viewport_key
from .planner_trial import prepare_planner_trial, UnsupportedTurnBudget
from .battle_log import log_inspection_action
from .turn_budget import fixed_turn_budget_conflicts
from .retry_policy import RetryDecision, StageRetryPolicy
from .memory_replacement import MemoryReplacement
from .battle_inputs import BattleInputs
from .recurring_hp_cost import recurring_hp_amount


class IdolRoadController:
    def __init__(
        self, config: IdolRoadConfig, *, clock: Callable[[], float] = monotonic,
    ) -> None:
        self.config = config.model_copy(deep=True)
        self._clock = clock
        self._started: float | None = None
        self._character_index = 0
        self._stage: int | None = None
        self._requested = False
        self._battle_seen = False
        self._result_seen = False
        self._completed: set[tuple[str, int]] = set()
        self._terminal: Decision | None = None
        self._returning = False
        self._pending_next: int | None = None
        self._next_waits = 0
        self._map_star_waits = 0
        self._navigation = NavigationGuard()
        self._character_selection_requested = False
        self._map_reentries = 0
        self._return_reason = 'next_character'
        self._conditions_requested: set[tuple[str, int]] = set()
        self._condition_read_waits=0
        self._objectives: dict[tuple[str, int], StageObjectives] = {}
        self._items_requested: set[tuple[str, int]] = set()
        self._inventories: dict[tuple[str, int], ItemInventory] = {}
        self._item_scans: dict[tuple[str, int], ItemScan] = {}
        self._support_requested: set[tuple[str, int]] = set()
        self._support_identity_waits=0
        self._support_rule_read_waits=0
        self._auto_attempted: set[tuple[str,int]] = set()
        self._auto_pending: tuple[str,int] | None = None
        self._auto_applied=False
        self._auto_capacity: int | None=None
        self._auto_waits=0
        self._formation_wait_key: tuple[str,int] | None=None
        self._formation_waits=0
        self._rules: dict[tuple[str, int], StageRules] = {}
        self._rule_scans: dict[tuple[str, int], ViewportScan[RuleViewport]] = {}
        self.battles_started = 0
        self._hp_inspections: set[tuple[str,int]] = set()
        self._status_inspections: set[tuple[str,int]] = set()
        self._pending_status: tuple[str,int] | None = None
        self._log_inspections: set[str] = set()
        self._hud_initial_observations: set[str] = set()
        self._pending_log: str | None = None
        self._pending_input_wait: tuple[str,int,float] | None=None
        self._status_scans: dict[tuple[str,int],ViewportScan[BattleStatusDetails]] = {}
        self._retries: dict[tuple[str,int],StageRetryPolicy] = {}
        self._retry_return: tuple[str,int] | None = None
        self._retry_focus_waits = 0
        self._judging_started: float | None = None
        self._replacement_request: RetryDecision | None = None
        self._replacement: MemoryReplacement | None = None
        self._requested_route: tuple[str,...] = ()

    def observe_requested_route(self,inputs: BattleInputs | None) -> None:
        """Retain the owned ledger before Page clears its session on result."""
        if (inputs is not None and self._requested and not self._result_seen
            and (inputs.session.character_id,inputs.session.stage)==(self.character_id,self._stage)):
            self._requested_route=inputs.requested_route

    @property
    def retry_policy(self) -> StageRetryPolicy | None:
        if self.character_id is None or self._stage is None:return None
        return self._retries.get((self.character_id,self._stage))

    @property
    def awaiting_battle_start(self) -> bool:
        """只表示请求已经发出，不证明游戏已开始或挑战已计数。"""
        return self._requested and not self._battle_seen

    def observe_initial_battle(self,observation: Observation,*,ready: bool=False) -> None:
        """Record owned HUD evidence from the delegated initialization flow."""
        state=observation.battle_state
        if (not self._requested or observation.scene!=Scene.BATTLE or state is None
            or (state.character_id,state.stage)!=(self.character_id,self._stage)
            or not state.battle_id or state.source!='confirmed_current_battle_observation'):
            raise ValueError('初始战斗观察不属于本次挑战')
        if ready and (state.turn!=1 or state.turn_source not in ('owned_initial_log_return','owned_initial_hud')):
            raise ValueError('初始观察尚未确认第1回合')
        if not self._battle_seen:
            self.battles_started+=1
            self._battle_seen=True
        self._navigation.poll(observation)
        if ready:
            if state.turn_source=='owned_initial_hud':
                self._hud_initial_observations.add(state.battle_id)
            else:
                self._log_inspections.add(state.battle_id)

    @property
    def objectives(self) -> StageObjectives | None:
        if self.character_id is None or self._stage is None:
            return None
        return self._objectives.get((self.character_id, self._stage))

    @property
    def item_inventory(self) -> ItemInventory | None:
        if self.character_id is None or self._stage is None:
            return None
        return self._inventories.get((self.character_id, self._stage))

    @property
    def stage_rules(self) -> StageRules | None:
        if self.character_id is None or self._stage is None:
            return None
        return self._rules.get((self.character_id, self._stage))

    @property
    def character_id(self) -> str | None:
        if self._character_index >= len(self.config.characters):
            return None
        return self.config.characters[self._character_index]

    def _decision(self, action: Action, reason: str) -> Decision:
        return Decision(action, reason, self.character_id, self._stage)

    def _stop(self, reason: str) -> Decision:
        self._terminal = self._decision(Action.STOP, reason)
        return self._terminal

    def _next_character(self) -> Decision:
        self._character_index += 1
        self._stage = None
        self._requested = self._battle_seen = self._result_seen = False
        self._pending_status = None
        self._pending_next = None
        self._next_waits = 0
        self._character_selection_requested = False
        self._map_reentries = 0
        self._return_reason = 'next_character'
        if self.character_id is None:
            return self._stop('characters_finished')
        self._returning = True
        return self._decision(Action.BACK_TO_CHARACTERS, 'next_character')

    def step(self, observation: Observation, *, cancelled: bool = False) -> Decision:
        decision = self._step(observation, cancelled=cancelled)
        self._navigation.record(observation, decision)
        if decision.action == Action.SELECT_CHARACTER and observation.character_selectable:
            self._character_selection_requested = True
        return decision

    def _step(self, observation: Observation, *, cancelled: bool) -> Decision:
        if self._terminal is not None:
            return self._terminal
        now = self._clock()
        if self._started is None:
            self._started = now
        if cancelled:
            return self._stop('cancelled')
        if observation.scene==Scene.RESUME_PROMPT:
            return self._stop('unfinished_stage_requires_native_recovery')
        if observation.scene==Scene.SUB_MEMORY_SELECTOR:
            if self._replacement is None:
                return self._stop('unowned_memory_selector')
        if now - self._started >= self.config.timeout_seconds:
            return self._stop('timeout')
        input_state=observation.battle_input_state
        if observation.scene in (Scene.JUDGING,Scene.RESULT):
            self._pending_input_wait=None
        elif input_state is not None:
            pending=input_state.pending
            if pending is None:
                self._pending_input_wait=None
            else:
                key=(input_state.battle_id,pending.sequence)
                if self._pending_input_wait is None or self._pending_input_wait[:2]!=key:
                    self._pending_input_wait=(key[0],key[1],pending.started+20)
        if (self._pending_input_wait is not None and now>=self._pending_input_wait[2]
            and observation.scene!=Scene.BATTLE_LOG):
            return self._stop('battle_input_unresolved')
        if self.character_id is None:
            return self._stop('no_characters')
        if self.config.mode=='planner_trial' and len(self.config.characters)!=1:
            return self._stop('planner_trial_requires_one_character')

        navigation = self._navigation.poll(observation)
        if navigation == 'stalled':
            return self._stop('navigation_not_confirmed')
        if navigation == 'waiting':
            return self._decision(Action.WAIT, 'await_navigation')

        scene = observation.scene
        if scene==Scene.SUB_MEMORY_SELECTOR:
            view=observation.memory_selector
            if view is None or self._replacement is None:
                return self._stop('memory_selector_missing')
            try:
                command=self._replacement.observe_selector(view,observation.memory_replacement_token)
            except ValueError:
                return self._stop('memory_replacement_not_confirmed')
            if command is None:return self._decision(Action.WAIT,'await_native_memory_selection')
            return Decision(Action.REPLACE_SUB_MEMORY,'owned_sub_replacement',self.character_id,self._stage,command)
        if scene==Scene.BATTLE_LOG:
            inspection=observation.battle_log_inspection
            if inspection is None:
                return self._stop('battle_log_evidence_missing')
            request=inspection.request
            owned=(self._requested and request is not None and request.session.battle_id==self._pending_log
                   and (request.session.character_id,request.session.stage)==(self.character_id,self._stage))
            action=log_inspection_action(inspection,self._pending_log if owned else None)
            return self._decision({'wait':Action.WAIT,'scroll':Action.SCROLL_BATTLE_LOG,'scroll_header':Action.SCROLL_BATTLE_LOG_HEADER,
                                   'close':Action.CLOSE_BATTLE_LOG}[action],'inspect_owned_log' if owned else 'close_unbound_log')
        if scene not in (Scene.STAGES,Scene.UNKNOWN):
            self._map_star_waits=0
        if scene==Scene.AUTO_FORMATION_RESULT:
            return self._decision(Action.CLOSE_AUTO_FORMATION_RESULT,'close_auto_formation_result_then_verify')
        if scene==Scene.AUTO_FORMATION:
            if (self._auto_pending is None or (observation.character_id,observation.stage)!=self._auto_pending
                or self.config.mode=='preview'):
                return self._decision(Action.CANCEL_AUTO_FORMATION,'cancel_unbound_auto_formation')
            if self._auto_applied:
                return self._decision(Action.WAIT,'await_auto_formation_close')
            if not observation.auto_formation_confirm_available:
                return self._stop('auto_formation_confirm_unknown')
            self._auto_applied=True
            key=self._auto_pending
            for store in (self._objectives,self._inventories,self._item_scans,self._rules,self._rule_scans):
                store.pop(key,None)
            for requested in (self._conditions_requested,self._items_requested,self._support_requested):
                requested.discard(key)
            return self._decision(Action.APPLY_AUTO_FORMATION,'replace_incomplete_lineup_invalidate_preparation')
        if scene == Scene.BATTLE_STATUS_DETAILS:
            details=observation.battle_status_details
            key=self._pending_status
            if (key is not None and details is not None and details.battle_id==key[0]
                and details.identity_source=='confirmed_formation_start_request'
                and details.source=='visible_battle_status_details'
                and (observation.character_id,observation.stage)==(details.character_id,details.stage)
                and (details.character_id,details.stage)==(self.character_id,self._stage)
                and status_viewport_key(details) is not None):
                scan=self._status_scans.get(key)
                if scan is None and observation.battle_status_scroll_available and any(r.evidence.occluded for r in details.rows):
                    scan=ViewportScan(status_viewport_key,lambda value: bool(value.rows),prefix='battle_status',max_scrolls=1)
                    self._status_scans[key]=scan
                if scan is not None:
                    action,reason=scan.observe(details)
                    if action=='scroll':
                        return self._decision(Action.SCROLL_BATTLE_STATUS,reason)
                    if action=='wait':
                        return self._decision(Action.WAIT,reason)
                    self._pending_status=None
                    return self._decision(Action.CLOSE_BATTLE_STATUS,reason)
            self._pending_status=None
            return self._decision(Action.CLOSE_BATTLE_STATUS,'close_visible_battle_status')
        # 奖励说明不证明领取或通关；只关闭已确认的说明页再重读底层。
        if scene == Scene.REWARD_INFO:
            return self._decision(Action.CLOSE_INFO, 'close_reward_information')
        if scene == Scene.CONDITIONS and observation.character_id is None and observation.stage is None:
            # 启动前人工打开的说明只关闭，不追认到某角色或关卡。
            return self._decision(Action.CLOSE_CONDITIONS, 'close_unbound_conditions')
        if scene == Scene.ITEM_DETAILS and observation.character_id is None and observation.stage is None:
            return self._decision(Action.CLOSE_ITEM_DETAILS, 'close_unbound_item_details')
        if scene == Scene.SUPPORT_DETAILS and observation.character_id is None and observation.stage is None:
            return self._decision(Action.CLOSE_SUPPORT, 'close_unbound_support')
        if self._returning:
            if scene == Scene.CHARACTERS:
                self._returning = False
            elif scene in (Scene.STAGES, Scene.MAP_OVERVIEW, Scene.RESULT):
                return self._decision(Action.BACK_TO_CHARACTERS, self._return_reason)
            else:
                return self._decision(Action.WAIT, 'await_character_list')
        if self._requested and not self._result_seen and scene in (
            Scene.HOME, Scene.CONTEST_HUB, Scene.CHARACTERS,
            Scene.DETAIL, Scene.FORMATION,
        ):
            return self._stop('unexpected_battle_exit')
        if scene == Scene.HOME:
            return self._decision(Action.OPEN_CONTEST, 'enter_contest_hub')
        if scene == Scene.CONTEST_HUB:
            return self._decision(Action.OPEN_ROAD, 'enter_road')
        if scene == Scene.CHARACTERS:
            return self._decision(Action.SELECT_CHARACTER, 'configured_character')
        if scene == Scene.UNKNOWN:
            return self._decision(Action.WAIT, 'unknown_scene')
        if observation.character_id != self.character_id:
            if (observation.character_id is not None
                and self._stage is None and not self._requested
                and not self._character_selection_requested and self.battles_started == 0
                and scene in (Scene.CONFIRM, Scene.STAGES, Scene.MAP_OVERVIEW)):
                # 仅校准任务起点；选过配置角色或开始挑战后，身份冲突仍停止。
                return self._decision(
                    Action.BACK_TO_STAGES if scene == Scene.CONFIRM else Action.BACK_TO_CHARACTERS,
                    'align_configured_character',
                )
            return self._stop('character_not_confirmed')

        if scene == Scene.MAP_OVERVIEW:
            if self._retry_return is not None:
                # A native failed result was already observed and dismissed.
                # The loading map does not yet identify the retry stage.
                self._retry_focus_waits += 1
                if self._retry_focus_waits >= 3:
                    return self._stop('retry_stage_focus_missing')
                return self._decision(Action.WAIT, 'await_retry_stage_focus')
            if self._requested and not self._result_seen:
                return self._stop('battle_result_missing')
            if self._stage is not None or self._pending_next is not None:
                return self._stop('map_focus_missing')
            if self._map_reentries >= 1:
                return self._stop('map_reentry_exhausted')
            # 实测重进当前角色可恢复游戏默认焦点，不猜灰色节点的可挑战性。
            self._map_reentries += 1
            self._return_reason = 'restore_map_focus'
            self._returning = True
            return self._decision(Action.BACK_TO_CHARACTERS, self._return_reason)

        if scene == Scene.STAGES:
            if self._requested and not self._result_seen and self._retry_return is None:
                return self._stop('battle_result_missing')
            if self.battles_started >= self.config.max_battles:
                return self._stop('battle_limit')
            if observation.focused_stage is not None:
                focused = observation.focused_stage
                if not focused.unlocked:
                    return self._stop('focused_stage_locked')
                if focused.stars is None:
                    self._map_star_waits+=1
                    if self._map_star_waits>=3:
                        return self._stop('stage_stars_unknown')
                    return self._decision(Action.WAIT,'await_map_stars')
                self._map_star_waits=0
                if self._retry_return is not None and (self.character_id,focused.number)!=self._retry_return:
                    return self._stop('retry_stage_not_confirmed')
                if self._pending_next is not None:
                    if focused.number + 1 == self._pending_next:
                        self._next_waits += 1
                        if self._next_waits >= 3:
                            return self._stop('next_stage_not_confirmed')
                        return self._decision(Action.WAIT, 'await_next_stage')
                    if focused.number != self._pending_next:
                        return self._stop('unexpected_next_stage')
                    self._pending_next = None
                    self._next_waits = 0
                if focused.stars >= 1:
                    self._retry_return=None
                    self._replacement_request=None
                    # 只浏览下一节点，不启动挑战，不计入战斗预算。
                    self._pending_next = focused.number + 1
                    return Decision(
                        Action.NEXT_STAGE,
                        'three_star_skip' if focused.stars == 3 else 'stage_already_cleared',
                        self.character_id, focused.number + 1,
                    )
                if (self.character_id, focused.number) in self._completed:
                    return self._decision(Action.WAIT, 'await_confirmed_star_update')
                self._stage = focused.number
                self._requested = self._battle_seen = self._result_seen = False
                self._retry_return=None
                return self._decision(Action.SELECT_STAGE, 'focused_uncleared_stage')
            if self._retry_return is not None:
                return self._stop('retry_stage_focus_missing')
            if not observation.stages_complete:
                return self._decision(Action.SCAN_STAGES, 'require_complete_stage_list')
            if self.battles_started >= self.config.max_battles:
                return self._stop('battle_limit')
            if any(stage.stars is None for stage in observation.stages if stage.unlocked):
                return self._stop('stage_stars_unknown')
            candidates = [
                stage.number for stage in observation.stages
                if stage.unlocked and stage.stars == 0
                and (self.character_id, stage.number) not in self._completed
            ]
            if not candidates:
                return self._next_character()
            self._stage = min(candidates)
            self._requested = self._battle_seen = self._result_seen = False
            return self._decision(Action.SELECT_STAGE, 'next_uncleared')

        # 支持从现有编成页启动预览／手动流程，但不接管未知的战斗或结算。
        # 身份由本帧姓名及关卡号提供，星级仍须在下面独立检查。
        if self._stage is None and scene == Scene.CONFIRM:
            if observation.stage is None or observation.stage < 1:
                return self._stop('stage_not_confirmed')
            self._stage = observation.stage
        if self._stage is None or observation.stage != self._stage:
            if (scene==Scene.SUPPORT_DETAILS and self._stage is not None and observation.stage is None
                and observation.rule_viewport is None and not self._requested
                and (self.character_id,self._stage) in self._support_requested):
                self._support_identity_waits+=1
                if self._support_identity_waits<3:
                    return self._decision(Action.WAIT,'await_support_stage_identity')
            return self._stop('stage_not_confirmed')
        if scene == Scene.SUPPORT_DETAILS:
            self._support_identity_waits=0
            key = (self.character_id, self._stage)
            if key not in self._support_requested or self._requested:
                return self._stop('unexpected_support_details')
            viewport=observation.rule_viewport
            if (self.config.mode in ('auto','planner_trial') and viewport is not None and viewport.visible_rules
                and self._support_rule_read_waits<2 and any(
                    row.turn==1 and row.recurring_hp_timing is not None
                    and recurring_hp_amount(row.description_reads,row.card_rect,row.recurring_hp_timing) is None
                    for row in viewport.visible_rules)):
                self._support_rule_read_waits+=1
                return self._decision(Action.WAIT,'await_current_turn_start_cost_rule')
            scan = self._rule_scans.setdefault(key, ViewportScan(
                lambda page: page.key, lambda page: bool(page.visible_rules), prefix='support',max_scrolls=6))
            action, reason = scan.observe(observation.rule_viewport or RuleViewport(None, None))
            self._rules[key] = StageRules(self.character_id, self._stage, scan.viewports)
            return self._decision({'scroll': Action.SCROLL_SUPPORT, 'wait': Action.WAIT,
                                   'close': Action.CLOSE_SUPPORT}[action], reason)
        if scene == Scene.ITEM_DETAILS:
            key = (self.character_id, self._stage)
            if key not in self._items_requested or self._requested:
                return self._stop('unexpected_item_details')
            if observation.item_viewport is not None:
                scan = self._item_scans.setdefault(key, ItemScan())
                action, reason = scan.observe(observation.item_viewport)
                pages = scan.viewports
                first_items = pages[0].visible_items if pages else observation.visible_items
                overlaps = tuple(associate_item_viewports(left, right) for left, right in zip(pages, pages[1:]))
                self._inventories[key] = ItemInventory(self.character_id, self._stage, first_items,
                                                       viewports=pages, overlaps=overlaps)
                return self._decision({'scroll': Action.SCROLL_ITEMS, 'wait': Action.WAIT,
                                       'close': Action.CLOSE_ITEM_DETAILS}[action], reason)
            self._inventories[key] = ItemInventory(self.character_id, self._stage, observation.visible_items)
            return self._decision(Action.CLOSE_ITEM_DETAILS, 'visible_items_observed')
        if scene == Scene.CONDITIONS:
            key = (self.character_id, self._stage)
            if key not in self._conditions_requested or self._requested:
                return self._stop('unexpected_conditions')
            self._objectives[key] = StageObjectives(self.character_id, self._stage, observation.requirements)
            if self.config.mode in ('auto','planner_trial') and self._objectives[key].rank_limit(1) is None:
                self._condition_read_waits+=1
                if self._condition_read_waits<3:
                    return self._decision(Action.WAIT,'await_current_one_star_conditions')
            else:
                self._condition_read_waits=0
            return self._decision(Action.CLOSE_CONDITIONS, 'stage_conditions_observed')
        if scene == Scene.DETAIL:
            return self._decision(Action.OPEN_FORMATION, 'keep_saved_formation')
        if scene == Scene.FORMATION:
            if not observation.formation_ready:
                return self._stop('formation_not_ready')
            return self._decision(Action.CONFIRM_FORMATION, 'saved_formation_ready')
        if scene == Scene.CONFIRM:
            if self._requested:
                return self._decision(Action.WAIT, 'await_battle_start')
            if observation.stars is None:
                return self._stop('stage_stars_unknown')
            if observation.stars >= 1:
                return self._decision(
                    Action.BACK_TO_STAGES,
                    'three_star_skip' if observation.stars == 3 else 'stage_already_cleared',
                )
            slots=observation.formation_slots
            key=(self.character_id,self._stage)
            if self._replacement_request is not None:
                if self._replacement is None:
                    if slots is None or not observation.formation_ready:
                        return self._stop('replacement_formation_missing')
                    try:
                        self._replacement=MemoryReplacement(slots)
                        command=self._replacement.open()
                    except ValueError:
                        return self._stop('replacement_formation_missing')
                    return Decision(Action.REPLACE_SUB_MEMORY,'repeated_failures_change_lineup',self.character_id,self._stage,command)
                try:
                    verified=self._replacement.verify(slots,observation.memory_replacement_receipt)
                except ValueError:
                    return self._stop('memory_replacement_not_confirmed')
                if not verified:return self._decision(Action.WAIT,'verify_changed_formation')
                policy=self.retry_policy
                if policy is None:return self._stop('replacement_retry_owner_missing')
                policy.confirm_lineup_changed(self._replacement_request)
                self._replacement=None
                self._replacement_request=None
                for store in (self._objectives,self._inventories,self._item_scans,self._rules,self._rule_scans):
                    store.pop(key,None)
                for requested in (self._conditions_requested,self._items_requested,self._support_requested):
                    requested.discard(key)
            if self._auto_pending is not None and self._auto_applied:
                if observation.auto_formation_result_pending:
                    self._auto_waits+=1
                    if self._auto_waits>=3:
                        return self._stop('auto_formation_result_not_confirmed')
                    return self._decision(Action.WAIT,'await_auto_formation_result')
                if slots is not None and slots.full and slots.capacity==self._auto_capacity:
                    self._auto_pending=None
                else:
                    self._auto_waits+=1
                    if self._auto_waits>=3:
                        return self._stop('auto_formation_did_not_fill_current_capacity')
                    return self._decision(Action.WAIT,'verify_auto_formation_slots')
            if slots is not None and not slots.known:
                if self._formation_wait_key!=key:
                    self._formation_wait_key=key
                    self._formation_waits=0
                self._formation_waits+=1
                if self._formation_waits<=3:
                    return self._decision(Action.WAIT,'await_current_formation_slots')
                return self._stop('formation_slots_unknown')
            self._formation_wait_key=None
            self._formation_waits=0
            if slots is not None and slots.empty_count and self.config.mode!='preview':
                if not self.config.auto_fill_missing_formation:
                    return self._stop('formation_missing_slots')
                if key in self._auto_attempted or not observation.auto_formation_available:
                    return self._stop('auto_formation_unavailable_or_exhausted')
                self._auto_attempted.add(key)
                self._auto_pending=key
                self._auto_applied=False
                self._auto_waits=0
                self._auto_capacity=slots.capacity
                return self._decision(Action.OPEN_AUTO_FORMATION,'fill_current_stage_missing_slots')
            if not observation.formation_ready:
                return self._stop('formation_not_ready')
            key = (self.character_id, self._stage)
            if observation.conditions_available and key not in self._conditions_requested:
                self._conditions_requested.add(key)
                self._condition_read_waits=0
                return self._decision(Action.OPEN_CONDITIONS, 'inspect_stage_conditions')
            if observation.items_available and key not in self._items_requested:
                self._items_requested.add(key)
                return self._decision(Action.OPEN_ITEM_DETAILS, 'inspect_visible_items')
            if observation.support_available and key not in self._support_requested:
                self._support_requested.add(key)
                self._support_identity_waits=0
                self._support_rule_read_waits=0
                return self._decision(Action.OPEN_SUPPORT, 'inspect_support_rules')
            if self.config.mode == 'preview':
                self._terminal = self._decision(Action.READY, 'preview_missing_formation_slots'
                                               if slots is not None and slots.empty_count else 'preview_complete')
                return self._terminal
            if self.config.mode=='planner_trial':
                if self.battles_started>=1:
                    return self._stop('planner_trial_complete')
                try:
                    prepare_planner_trial(observation,self.objectives,self.stage_rules)
                except UnsupportedTurnBudget:
                    return self._stop('planner_trial_dynamic_turns_unsupported')
                except ValueError:
                    return self._stop('planner_trial_inputs_unconfirmed')
            if self.config.mode=='auto':
                from .battle_start import prepare_variable_battle
                try:
                    prepare_variable_battle(observation,self.objectives,self.stage_rules)
                except ValueError as error:
                    return self._stop(str(error))
                self._retries.setdefault(key,StageRetryPolicy(
                    self.config.attempts_per_lineup,self.config.max_lineup_changes))
            if self.battles_started >= self.config.max_battles:
                return self._stop('battle_limit')
            self._requested = True
            self._judging_started=None
            self._requested_route=()
            return self._decision(Action.START_BATTLE, 'manual_battle')
        if scene in (Scene.BATTLE, Scene.RESULT, Scene.JUDGING):
            if not self._requested:
                return self._stop('unexpected_battle')
            if not self._battle_seen:
                self.battles_started += 1
                self._battle_seen = True
        if scene == Scene.BATTLE:
            if self._result_seen:
                return self._stop('battle_after_confirmed_result')
            state=observation.battle_state
            input_state=observation.battle_input_state
            if input_state is not None and input_state.expired:
                return self._stop('battle_input_unresolved')
            if observation.battle_log_pending:
                return self._decision(Action.WAIT,'await_battle_log_return')
            self._pending_log=None
            if (input_state is not None and input_state.pending is not None
                and observation.battle_log_available and state is not None
                and (state.character_id,state.stage,state.battle_id)==(self.character_id,self._stage,input_state.battle_id)):
                self._pending_log=state.battle_id
                return self._decision(Action.INSPECT_BATTLE_LOG,'inspect_pending_input_once')
            if (self.config.mode=='manual' and state is not None and state.turn is None
                and state.source=='confirmed_current_battle_observation' and state.battle_id
                and state.battle_id not in self._log_inspections
                and state.battle_id not in self._hud_initial_observations
                and (state.character_id,state.stage)==(self.character_id,self._stage)
                and observation.battle_log_available
                and fixed_turn_budget_conflicts(state.character_id,state.stage,self.stage_rules)):
                self._log_inspections.add(state.battle_id)
                self._pending_log=state.battle_id
                return self._decision(Action.INSPECT_BATTLE_LOG,'inspect_variable_turn_log_once')
            if (state is not None and state.source=='confirmed_current_battle_observation'
                and (state.character_id,state.stage)==(self.character_id,self._stage)
                and state.battle_id and type(state.turn) is int and state.turn>0
                and state.frame_id
                and not {'identity','turn'} & set(state.conflicted_fields)
                and self.stage_rules is not None):
                key=(state.battle_id,state.turn)
                catalog=build_rule_catalog(state.character_id,state.stage,self.stage_rules)
                if (state.max_hp is None and observation.battle_hp_info_available
                    and key not in self._hp_inspections and any(row.kind=='hp_mental_block'
                    and row.declaration.start_turn==state.turn for row in catalog.sightings)):
                    self._hp_inspections.add(key)
                    return self._decision(Action.INSPECT_BATTLE_HP,'inspect_required_hp_ratio_once')
                if (observation.battle_status_info_available and key not in self._status_inspections
                    and any(row.declaration.start_turn==state.turn for row in catalog.sightings)):
                    self._status_inspections.add(key)
                    self._pending_status=key
                    return self._decision(Action.INSPECT_BATTLE_STATUS,'inspect_required_status_once')
            return self._decision(Action.WAIT, 'manual_battle_in_progress')
        if scene == Scene.JUDGING:
            if not observation.result_next_available:
                if self._judging_started is None:self._judging_started=now
                if now-self._judging_started>=30:return self._stop('result_controls_not_confirmed')
                return self._decision(Action.WAIT,'await_result_controls')
            return self._decision(Action.ADVANCE_RESULT, 'ranking_is_not_clear_evidence')
        if scene == Scene.RESULT:
            if observation.outcome == Outcome.UNKNOWN:
                return self._stop('result_unknown')
            if observation.outcome == Outcome.FAILURE:
                if observation.stars is not None and observation.stars >= 1:
                    return self._stop('result_stars_conflict')
                if self.config.mode=='planner_trial':
                    return self._stop('planner_trial_failed')
                if self.config.mode=='auto' or self.config.retry_failed_stage:
                    key=(self.character_id,self._stage)
                    policy=self._retries.setdefault(key,StageRetryPolicy(
                        self.config.attempts_per_lineup,self.config.max_lineup_changes))
                    retry=policy.failed(self.battles_started,request_route=self._requested_route)
                    if retry.action=='stop':return self._stop(retry.reason)
                    self._retry_return=key
                    self._retry_focus_waits=0
                    self._replacement_request=retry if retry.action=='change_lineup' else None
                    return self._decision(Action.DISMISS_RESULT,retry.reason)
                if self.config.stop_on_failure:
                    return self._stop('battle_failed')
                return self._next_character()
            if observation.stars is None:
                return self._stop('result_stars_unknown')
            if observation.stars == 0:
                return self._stop('result_has_no_star')
            if self.config.mode=='planner_trial':
                return self._stop('planner_trial_succeeded')
            self._result_seen = True
            self._completed.add((self.character_id, self._stage))
            return self._decision(Action.DISMISS_RESULT, 'success_confirmed')
        if scene == Scene.REWARDS:
            if not self._result_seen:
                return self._stop('reward_without_confirmed_result')
            return self._decision(Action.DISMISS_REWARDS, 'confirmed_battle_rewards')
        return self._decision(Action.WAIT, 'unhandled_scene')
