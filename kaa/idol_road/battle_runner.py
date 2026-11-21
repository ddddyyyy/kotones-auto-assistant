"""Single-input variable battle execution using shared card reading and scoring."""
from dataclasses import dataclass,replace
from time import monotonic
from typing import Callable, Iterable, Literal

from cv2.typing import MatLike

from kaa.tasks.produce.new.play_cards.planner.feedback import PendingPlay
from kaa.tasks.produce.new.play_cards.planner.strategy import PlannerStrategy
from kaa.tasks.produce.new.play_cards.planner.simulator import can_afford_play
from .native_payment import supports_native_payment_conflict
from .battle_observation import BattleObservationSession, current_absolute_turn
from .card_completion import complete_card_return
from .card_feedback import settle_card_hand_feedback
from .skip_completion import complete_skip_return
from .category_rule import category_objectives
from .concentration_priority import concentration_objectives
from .field_priority import FieldPriority
from .good_condition_rule import good_condition_planning_requirements
from .hp_preparation import hp_preparation_budget
from .hp_safety import HpSafetyPolicy
from .input_observation import observe_pending_input
from .input_policy import InputBoundPolicy
from .models import Action, Decision, Scene
from .maximum_hp import confirmed_maximum_hp
from .page import IdolRoadPage, PageSnapshot
from .planner_context import IdolRoadBattleContext
from .ranked_field_policy import RankedFieldPolicy,CurrentCardPreview
from kaa.tasks.produce.new.play_cards.planner.visible_output import supports_current_preview
from .retry_policy import StageRetryPolicy
from .retry_route import RetryRoutePreference
from .rules import StageRules
from .score_priority import score_preparation_objectives
from .turn_end_page import read_enabled_skip, read_turn_end
from .screened_turn_end import screened_turn_end
from .extra_play import bind_sense_play_count
from .hud_input_observation import observe_pending_hud
from .hud_completion import complete_hud_return
from .schedule_countdown_page import read_schedule_countdown
from .skip_recovery_page import read_skip_recovery
from .current_skip import assess_current_skip
from .score_suppression import SuppressionAwareEvaluator
from .score_suppression_page import read_score_suppression


@dataclass(frozen=True)
class BattleRunResult:
    screen: MatLike
    snapshot: PageSnapshot
    reason: Literal['owned_battle_finished','input_limit']
    card_requests: int
    skip_requests: int
    requested_route: tuple[str,...]


def run_variable_battle(page: IdolRoadPage,session: BattleObservationSession,
                        screen: MatLike,snapshot: PageSnapshot,frames: Iterable[MatLike],*,
                        stage_rules: StageRules,deadline: float,
                        observe: Callable[[MatLike,PageSnapshot,Decision],None],
                        retry_policy: StageRetryPolicy | None=None,max_inputs: int=200,
                        rank_at_most: int=3,
                        allow_audited_redraw: bool=False,
                        feedback_mode: Literal['log','hud']='log',
                        clock: Callable[[],float]=monotonic) -> BattleRunResult:
    """Start only from this START's initialized frame; never adopt an old HUD.

    Every request finishes its owned feedback cycle before another selection.
    The runner does not dismiss results, consume drinks, or retry a battle.
    """
    inputs=page.battle_inputs
    profile=page.battle_sense_profile
    if (inputs is None or inputs.session!=session or page.battle_session!=session
        or session.turn_budget_mode!='variable_budget' or profile is None or profile.session!=session
        or inputs.sequence!=0 or inputs.pending is not None
        or snapshot.battle_frame is None or snapshot.battle_frame.session!=session
        or current_absolute_turn(snapshot.battle_frame) is None
        or snapshot.battle_frame.observation.turn!=1
        or snapshot.observation.scene!=Scene.BATTLE
        or feedback_mode not in ('log','hud') or page.feedback_mode!=feedback_mode
        or snapshot.battle_frame.observation.turn_source!=('owned_initial_hud' if feedback_mode=='hud' else 'owned_initial_log_return')
        or confirmed_maximum_hp(snapshot.battle_frame) is None
        or type(max_inputs) is not int or not 1<=max_inputs<=500
        or type(rank_at_most) is not int or not 1<=rank_at_most<=6
        or type(allow_audited_redraw) is not bool
        or type(session.total_turns) is not int or not 1<=session.total_turns<=30
        or (stage_rules.character_id,stage_rules.stage)!=(session.character_id,session.stage)
        or stage_rules.source!='visible_support_details'
        or stage_rules.identity_source!='opened_from_confirmed_formation'):
        raise ValueError('variable_battle_start_unconfirmed')
    total=session.total_turns
    budget=hp_preparation_budget(session.character_id,session.stage,stage_rules)
    if budget is None or (feedback_mode!='hud' and budget.goals and not budget.recurring_known):
        raise ValueError('variable_battle_hp_budget_unknown')
    guard=HpSafetyPolicy(total,budget,allow_audited_redraw=allow_audited_redraw,
        reserve_mode='prefer' if feedback_mode=='hud' else 'strict')
    evaluator=SuppressionAwareEvaluator()
    field=FieldPriority(total,category_objectives(session.character_id,session.stage,stage_rules),
        concentration_objectives(session.character_id,session.stage,stage_rules),
        good_condition_planning_requirements(session.character_id,session.stage,stage_rules),
        score_preparation_objectives(session.character_id,session.stage,stage_rules),budget,
        strict_hp_reserve=feedback_mode!='hud',allow_unknown_good_condition=feedback_mode=='hud',evaluator=evaluator)
    ranked=RankedFieldPolicy(field,evaluator=evaluator)
    policy=InputBoundPolicy(inputs,lambda:snapshot.battle_frame,delegate=ranked,guard=guard)
    reader=PlannerStrategy(is_exam=True,archetype=profile.archetype,rank_at_most=rank_at_most)
    stream=iter(frames)
    cards_played=skips=0
    awaiting_feedback=None

    def result(reason: Literal['owned_battle_finished','input_limit']) -> BattleRunResult:
        return BattleRunResult(screen,snapshot,reason,cards_played,skips,inputs.requested_route)

    def report(reason: str) -> None:
        observe(screen,snapshot,Decision(Action.WAIT,reason,session.character_id,session.stage))

    try:
        while True:
            if clock()>=deadline:raise ValueError('variable_battle_timeout')
            obs=snapshot.observation
            if obs.scene in (Scene.JUDGING,Scene.RESULT):
                if (obs.character_id,obs.stage)!=(session.character_id,session.stage):
                    raise ValueError('variable_battle_result_identity_changed')
                return result('owned_battle_finished')
            frame=snapshot.battle_frame
            if (page.battle_session!=session or page.battle_inputs is not inputs
                or obs.scene!=Scene.BATTLE or frame is None or frame.session!=session
                or frame.values.resource_state!='current' or frame.values.detail_mode!='complete'
                or current_absolute_turn(frame) is None or inputs.pending is not None):
                raise ValueError('variable_battle_current_frame_unconfirmed')
            if cards_played+skips>=max_inputs:return result('input_limit')
            extra=None
            context=IdolRoadBattleContext(screen,frame,hand_context=snapshot.hand_context,
                before_commit=lambda current,card_id:page.begin_battle_input(current,'card',card_id,extra_play=extra))
            hands=context.fetch_hands()
            state=reader._build_observed_state(context,hands)
            extra=context.fetch_extra_play_observation()
            observed_state=bind_sense_play_count(state,frame,extra,profile)
            if observed_state is None:raise ValueError('variable_battle_current_play_count_unknown')
            state=observed_state
            evaluator.current=read_score_suppression(screen,frame.values.frame_id) if feedback_mode=='hud' else None
            from .trouble_hand import TroubleHandContext
            captured=snapshot.hand_context
            if captured is not None and isinstance(captured.context,TroubleHandContext):
                state=replace(state,observed_card_payments=captured.context.payments)
            guard.current_skip=None
            absolute=current_absolute_turn(frame)
            assert absolute is not None
            if feedback_mode=='hud' and any(goal.declared_turn==absolute.turn+1 for goal in budget.goals):
                guard.current_skip=assess_current_skip(inputs,frame,state,
                    read_skip_recovery(screen,frame.values.frame_id),budget)
            if awaiting_feedback is not None:
                ranked.feedback(awaiting_feedback[0],awaiting_feedback[1],state)
                awaiting_feedback=None
            available=[h for h in hands if h is not None and h.card is not None and h.available]
            conflicts=frozenset(h.card._id for h in available if h.card is not None
                and h.card._id not in state.unresolved_growth_card_ids
                and can_afford_play(state,h.card) is False and supports_native_payment_conflict(h.card))
            if conflicts:
                state=replace(state,native_payment_conflict_card_ids=conflicts)
            ranked.current_previews=()
            if feedback_mode=='hud':
                previews=[]
                preview_state=replace(state,observed_card_output_is_lower_bound=True)
                for hand in available:
                    card=hand.card
                    if (card is None or sum(h.card is not None and h.card._id==card._id for h in hands if h is not None)!=1
                        or not supports_current_preview(card,preview_state)):continue
                    preview=hand.wide_output_preview
                    if preview is not None:
                        previews.append(CurrentCardPreview(frame.values.frame_id,card._id,card.upgrade_count,
                            preview.per_hit,preview.count))
                ranked.current_previews=tuple(previews)
            ranked.retry_route=RetryRoutePreference(retry_policy,inputs,frame) if retry_policy is not None else None
            blocked=False
            if not available:
                blocked,_,_,_=read_turn_end(screen)
                if not blocked:raise ValueError('variable_battle_hand_unknown')
                choice=field.select(state,())
                if (not choice.end_turn and captured is not None and isinstance(captured.context,TroubleHandContext)
                    and captured.context.unresolved_sleep_payments):
                    # Gray Trouble styling is not proof of unavailability.
                    # Preserve an explicit field last-card end-turn choice,
                    # otherwise don't silently skip an unpriced Sleep.
                    raise ValueError('variable_battle_trouble_payment_unknown')
            else:
                choice=policy.select(state,tuple(h.card for h in available if h.card is not None))
                if choice.stop:
                    proposal=screened_turn_end(inputs,frame,state,
                        tuple(h.card for h in available if h.card is not None),choice,guard.review,snapshot.log_hand)
                    if proposal is not None:choice=proposal
            report('variable_battle_selection:'+choice.reason)
            if choice.stop:raise ValueError('variable_battle_selection_stopped:'+choice.reason)
            pending_play: PendingPlay | None=None
            requested_card=None
            origin_hand=snapshot.log_hand
            countdown=read_schedule_countdown(screen,frame.values.frame_id) if feedback_mode=='hud' else None
            if feedback_mode=='hud' and origin_hand is None:raise ValueError('variable_battle_origin_hand_unknown')
            if blocked or choice.end_turn:
                safety=guard.end_turn(state)
                if safety.stop:raise ValueError('variable_battle_skip_stopped:'+safety.reason)
                if not read_enabled_skip(screen):raise ValueError('variable_battle_skip_disabled')
                if clock()>=deadline:raise ValueError('variable_battle_timeout')
                from kotonebot import device
                from kaa.tasks import R
                page.begin_battle_input(frame,'skip',extra_play=extra)
                skips+=1
                device.click(R.IdolRoad.BoxBattleSkipLabel)
            else:
                indices=choice.preferred_indices
                if not indices:raise ValueError('variable_battle_no_ranked_candidate')
                index=indices[0]
                if (type(index) is not int or not 0<=index<len(available)
                    or choice.allowed_indices is None or index not in choice.allowed_indices):
                    raise ValueError('variable_battle_candidate_mapping_invalid')
                hand=available[index]
                card=hand.card
                if card is None:raise ValueError('variable_battle_card_missing')
                if clock()>=deadline:raise ValueError('variable_battle_timeout')
                pending_play=PendingPlay(card._id,state,0,0)
                requested_card=card
                context.commit(hand)
                cards_played+=1
                policy.committed(state,card)
            report('variable_battle_input_requested')
            if feedback_mode=='hud':
                assert origin_hand is not None
                returned=observe_pending_hud(page,session,origin_hand,stream,deadline=deadline,observe=observe,
                    clock=clock,origin_countdown=countdown,require_round=True,card=requested_card,
                    allow_audited_redraw=allow_audited_redraw)
                screen,snapshot=returned.screen,returned.snapshot
                if returned.battle_finished:continue
                if returned.evidence is None:raise ValueError('variable_battle_hud_evidence_missing')
                completed=complete_hud_return(page,screen,snapshot,returned.evidence,deadline=deadline,
                    card=requested_card,allow_audited_redraw=allow_audited_redraw)
                snapshot=completed.snapshot
                if pending_play is not None and completed.feedback is not None:
                    awaiting_feedback=(pending_play,completed.feedback)
                report('variable_battle_next_planning_frame')
                continue
            screen,snapshot=observe_pending_input(page,session,stream,deadline=deadline,observe=observe,clock=clock)
            if snapshot.observation.scene in (Scene.JUDGING,Scene.RESULT):
                continue
            if pending_play is not None:
                receipt=settle_card_hand_feedback(policy,snapshot,pending_play,allow_round_advance=True)
                if receipt is None:raise ValueError('variable_battle_card_feedback_unconfirmed')
                snapshot=complete_card_return(page,screen,snapshot,receipt,deadline=deadline).snapshot
            elif snapshot.battle_frame is not None and snapshot.battle_frame.values.detail_mode=='feedback_only':
                snapshot=complete_skip_return(page,screen,snapshot,deadline=deadline).snapshot
            report('variable_battle_next_planning_frame')
    finally:
        reader.on_battle_end()
