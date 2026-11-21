"""爬塔当前动作的硬体力约束；远期缺口不是已发生伤害或保证收益。"""
from dataclasses import dataclass,replace
from typing import Literal
from kaa.db.skill_card import SkillCard
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from kaa.tasks.produce.new.play_cards.planner.selection_policy import SelectionConstraint
from kaa.tasks.produce.new.play_cards.planner.simulator import can_afford_play,simulate_play
from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict,evaluate_trigger_id
from kaa.tasks.produce.new.play_cards.planner.costs import special_cost_affordability
from .current_hp_projection import supports_current_hp_projection
from .hp_preparation import HpPreparationBudget,HpRouteAssessment,assess_hp_route
from .planning_turn import PlanningTurnTracker
from .current_skip import CurrentSkipAssessment


@dataclass(frozen=True)
class HpCandidateReview:
    index: int
    card_id: str
    reason: str
    effect_types: tuple[str,...]
    affordable: bool | None = None
    projected_hp: int | None = None
    projected_hp_known: bool = False
    route: HpRouteAssessment | None = None
    projection_scope: Literal['current_supported_effects','audited_redraw_roots_only'] | None = None


@dataclass(frozen=True)
class HpSelectionReview:
    frame_id: str | None
    turn: int | None
    current_route: HpRouteAssessment | None
    candidates: tuple[HpCandidateReview,...]
    allowed_indices: tuple[int,...]


class HpSafetyPolicy:
    def __init__(self,total_turns: int,budget: HpPreparationBudget,*,allow_audited_redraw: bool=False,
                 reserve_mode: Literal['strict','prefer']='strict'):
        if type(allow_audited_redraw) is not bool:raise ValueError('Invalid current redraw gate')
        if reserve_mode not in ('strict','prefer'):raise ValueError('Invalid future HP reserve mode')
        self.total_turns=total_turns
        self.budget=budget
        self._turns=PlanningTurnTracker()
        self.review: HpSelectionReview | None=None
        self.allow_audited_redraw=allow_audited_redraw
        self.reserve_mode=reserve_mode
        self.current_skip: CurrentSkipAssessment | None=None

    def select(self,state: BattleState,cards: tuple[SkillCard,...]) -> SelectionConstraint:
        turn=self._turns.observe(state,self.total_turns)
        frame_id=state.absolute_turn_observation.frame_id if state.absolute_turn_observation is not None else None
        self.review=HpSelectionReview(frame_id,turn.turn if turn else None,None,(),())
        if turn is None or not state.hp_known or type(state.hp) is not int or state.hp<0:
            return SelectionConstraint(reason='hp_safety:current_state_unknown',allowed_indices=(),stop=True)
        before=assess_hp_route(state,state,turn,self.budget)
        allowed=[]
        reviews=[]
        for index,card in enumerate(cards):
            effects=tuple(str(row.produce_exam_effect.effect_type) if row.produce_exam_effect else 'unknown'
                for row in card.play_effects)
            affordable=can_afford_play(state,card)
            ordinary=supports_current_hp_projection(card,state)
            supported=ordinary or (self.allow_audited_redraw and supports_current_hp_projection(
                card,state,allow_audited_redraw=True))
            # select() receives recognized, currently native-available cards.
            # HP zero is legal. A future optional field target cannot prove
            # a current fee or forbid a native-available current action.
            # This grants no numeric payment or follow-up resource balance.
            special=special_cost_affordability(state,card)
            native_current=(self.reserve_mode=='prefer'
                and state.projected_effect_values is None and special is not False)
            observed_payment=(affordable is None and native_current
                and ('stamina_consumption_modifier' in state.observed_effects or special is None
                     or card._id in state.native_payment_conflict_card_ids))
            trigger=evaluate_trigger_id(card._play_produce_exam_trigger_id or '',state)
            observed_trigger=native_current and trigger is TriggerVerdict.UNKNOWN
            reason=('payment_unconfirmed' if affordable is not True and not observed_payment else
                'current_effects_unsupported' if not supported else
                'move_choice_unsupported' if card.move_produce_exam_effects else
                'card_status_unsupported' if card._produce_card_status_enchant_id else
                'play_trigger_unconfirmed' if trigger is not TriggerVerdict.ACTIVE and not observed_trigger else None)
            if reason is not None:
                reviews.append(HpCandidateReview(index,card._id,reason,effects,affordable))
                continue
            if observed_payment or observed_trigger:
                allowed.append(index)
                reason=('native_available_unknown_trigger_current_only' if observed_trigger
                    else 'native_available_current_only')
                reason+=('_field_goal_unconfirmed' if self.budget.goals else '_no_hp_target')
                if card._id in state.native_payment_conflict_card_ids:
                    reason='native_available_payment_model_conflict_current_only_no_numeric_fee'
                reviews.append(HpCandidateReview(index,card._id,reason,effects,
                    affordable=affordable,projected_hp=None,projected_hp_known=False))
                continue
            after=simulate_play(state,card)
            if not after.hp_known or type(after.hp) is not int or after.hp<0:
                reviews.append(HpCandidateReview(index,card._id,'projected_hp_unconfirmed',effects,affordable,
                    after.hp if type(after.hp) is int else None,after.hp_known))
                continue
            if not after.remaining_turns_known or after.remaining_turns!=state.remaining_turns:
                reviews.append(HpCandidateReview(index,card._id,'projected_turn_budget_changed',effects,affordable,after.hp,True))
                continue
            result=assess_hp_route(state,after,turn,self.budget)
            if result.status in ('safe','not_applicable'):
                allowed.append(index)
                reason='allowed'
            elif self.reserve_mode=='prefer' and result.status=='below_reserve':
                allowed.append(index)
                reason='allowed_current_payment_future_reserve_unsecured'
            elif (self.reserve_mode=='prefer' and result.status=='unknown' and not self.budget.recurring_known
                and type(state.max_hp) is int and state.max_hp>0 and after.max_hp==state.max_hp
                and 0<=state.hp<=state.max_hp and 0<=after.hp<=state.max_hp):
                allowed.append(index)
                reason='allowed_current_payment_future_charge_unknown_field_goal_unconfirmed'
            elif before.status=='below_reserve' and result.status=='below_reserve' and after.hp>=state.hp:
                # The existing long-horizon deficit may require future recovery; it is
                # not permission to spend more HP, nor a claim that recovery will occur.
                allowed.append(index)
                reason='allowed_without_worsening_existing_deficit'
            else:
                reason='reserve_unknown' if result.status=='unknown' else 'reserve_spending_rejected'
            reviews.append(HpCandidateReview(index,card._id,reason,effects,affordable,after.hp,True,result,
                'current_supported_effects' if ordinary else 'audited_redraw_roots_only'))
        imminent=[goal for goal in self.budget.goals if goal.declared_turn==turn.turn+1]
        cost=self.budget.recurring
        if (self.reserve_mode=='prefer' and imminent and type(state.max_hp) is int and state.max_hp>0):
            threshold=max((state.max_hp*goal.at_least_percent+99)//100 for goal in imminent)
            charge=(cost.amount if turn.turn+1>=cost.start_turn else 0
                ) if self.budget.recurring_known and cost is not None else None
            # Without a future charge proof, rank by the current balance
            # only. The route assessment stays UNKNOWN, never safe.
            current_target=threshold if charge is None else threshold+charge
            preserving=[row.index for row in reviews if row.index in allowed
                and type(row.projected_hp) is int and row.projected_hp>=current_target]
            skip=self.current_skip
            if preserving:
                reviews=[replace(row,reason=('imminent_field_current_hp_preferred_future_charge_unknown'
                    if charge is None else 'imminent_field_hp_preferred_alternative'))
                    if row.index in allowed and row.index not in preserving else row for row in reviews]
                allowed=preserving
            elif (charge is not None and skip is not None and skip.frame_id==frame_id and skip.status=='known_operations'
                and skip.declared_next_charge==charge and type(skip.known_operations_min_hp) is int
                and skip.known_operations_min_hp>=threshold):
                self.review=HpSelectionReview(frame_id,turn.turn,before,tuple(reviews),())
                return SelectionConstraint(end_turn=True,allowed_indices=(),
                    reason='hp_safety:prefer_current_skip_for_imminent_field,recovery_estimate_not_observed_reward')
        self.review=HpSelectionReview(frame_id,turn.turn,before,tuple(reviews),tuple(allowed))
        if not allowed and reviews and all(row.affordable is None and row.reason=='payment_unconfirmed' for row in reviews):
            return SelectionConstraint(allowed_indices=(),stop=True,reason='hp_safety:current_payment_unknown')
        return SelectionConstraint(allowed_indices=tuple(allowed),stop=not allowed,
            reason=f'hp_safety:allowed={allowed},budget={before.status},reserve_mode={self.reserve_mode},no_future_healing_assumed')

    def end_turn(self,state: BattleState) -> SelectionConstraint:
        turn=self._turns.observe(state,self.total_turns)
        if turn is None or not state.hp_known or type(state.hp) is not int or state.hp<0:
            return SelectionConstraint(reason='hp_safety:end_turn_state_unknown',stop=True)
        if turn.turn>=turn.last_scheduled_turn:
            return SelectionConstraint(reason='hp_safety:end_turn_final_observed_round,extra_turn_unconfirmed')
        cost=self.budget.recurring
        if (not self.budget.recurring_known or cost is None or type(cost.amount) is not int or cost.amount<1
            or type(cost.start_turn) is not int or cost.start_turn<1):
            if self.reserve_mode=='prefer':
                # HP zero does not end this mode's battle. A legal observed
                # SKIP needs no invented future charge. Optional field goals
                # remain unconfirmed until their actual trigger is observed.
                return SelectionConstraint(reason=('hp_safety:end_turn_field_goal_unconfirmed,future_charge_unknown'
                    if self.budget.goals else 'hp_safety:end_turn_no_declared_hp_goal,future_charge_unknown'))
            return SelectionConstraint(reason='hp_safety:end_turn_cost_unknown',stop=True)
        charge=cost.amount if turn.turn+1>=cost.start_turn else 0
        if type(charge) is not int or charge<0:
            return SelectionConstraint(reason='hp_safety:end_turn_charge_unknown',stop=True)
        imminent=[goal for goal in self.budget.goals if goal.declared_turn==turn.turn+1]
        if imminent:
            maximum=state.max_hp
            if type(maximum) is not int or maximum<=0 or state.hp>maximum:
                return SelectionConstraint(reason='hp_safety:end_turn_maximum_unknown',stop=True)
            required=max((maximum*goal.at_least_percent+99)//100+charge for goal in imminent)
            if state.hp<required:
                return SelectionConstraint(reason=f'hp_safety:end_turn_threshold_unmet,required={required}',
                    stop=self.reserve_mode=='strict')
        return SelectionConstraint(reason=f'hp_safety:end_turn_next_charge={charge},future_goals_unconfirmed')
