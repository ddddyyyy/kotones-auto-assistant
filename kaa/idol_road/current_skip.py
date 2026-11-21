"""Local SKIP recovery and declared charge arithmetic; not action authorization."""
from dataclasses import dataclass,replace
from typing import Literal
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from .battle_inputs import BattleInputs
from .battle_observation import BattleFrameObservation
from .hp_preparation import HpPreparationBudget
from .maximum_hp import confirmed_maximum_hp
from .planning_turn import read_planning_turn
from .skip_recovery import SkipRecoveryObservation,confirm_skip_recovery


@dataclass(frozen=True)
class CurrentSkipAssessment:
    frame_id: str
    input_sequence: int
    status: Literal['unknown','recovery_only','known_operations'] = 'unknown'
    reason: str = 'current_frame_unbound'
    recovery: int | None = None
    recovery_root_hp: int | None = None
    declared_next_charge: int | None = None
    hp_if_recovery_first: int | None = None
    hp_if_charge_first: int | None = None
    known_operations_min_hp: int | None = None
    known_operations_max_hp: int | None = None
    scope: Literal['skip_recovery_and_declared_charge_only'] = 'skip_recovery_and_declared_charge_only'
    unresolved_events: tuple[str,...] = ('event_order','recovery_restrictions','equipped_item_turn_events','other_field_turn_events')
    survival_guaranteed: Literal[False] = False


def assess_current_skip(inputs: BattleInputs,frame: BattleFrameObservation,state: BattleState,
                        recovery: SkipRecoveryObservation | None,
                        budget: HpPreparationBudget) -> CurrentSkipAssessment:
    result=CurrentSkipAssessment(frame.values.frame_id,inputs.sequence)
    if (not inputs.can_select(frame,state) or not state.hp_known or type(state.hp) is not int or state.hp<0
        or state.projected_resource_restrictions or state.projected_resource_gain_modifiers):
        return result
    maximum=confirmed_maximum_hp(frame)
    bound=frame.maximum_hp_parameter
    if (maximum is None or bound is None or bound.input_sequence!=inputs.sequence
        or type(state.max_hp) is not int or state.max_hp!=maximum or not state.hp<=maximum):
        return replace(result,reason='maximum_hp_unconfirmed')
    if (recovery is None or recovery.evidence.frame_id!=frame.values.frame_id
        or confirm_skip_recovery(recovery.evidence)!=recovery or recovery.status!='resolved'
        or type(recovery.value) is not int):
        return replace(result,reason='current_skip_recovery_unknown')
    amount=recovery.value
    root=min(maximum,state.hp+amount)
    result=replace(result,status='recovery_only',reason='next_charge_unknown',recovery=amount,recovery_root_hp=root)
    turn=read_planning_turn(state,inputs.session.initial_total_turns or 0)
    if turn is None or turn.turn>=turn.last_scheduled_turn:
        return replace(result,reason='next_round_unconfirmed')
    cost=budget.recurring
    if (not budget.recurring_known or cost is None or type(cost.start_turn) is not int or cost.start_turn<1
        or type(cost.amount) is not int or cost.amount<1 or not cost.evidence_frames
        or any(not isinstance(name,str) or not name for name in cost.evidence_frames)):
        return result
    charge=cost.amount if turn.turn+1>=cost.start_turn else 0
    recovery_first=max(0,root-charge)
    # User-confirmed Idol Road rule: zero HP does not end the battle. A later
    # SKIP recovery is still possible; a charge clamps at zero first.
    charge_first=min(maximum,max(0,state.hp-charge)+amount)
    return replace(result,status='known_operations',reason='order_unverified_external_events_unmodeled',
        declared_next_charge=charge,hp_if_recovery_first=recovery_first,hp_if_charge_first=charge_first,
        known_operations_min_hp=min(recovery_first,charge_first),known_operations_max_hp=max(recovery_first,charge_first))
