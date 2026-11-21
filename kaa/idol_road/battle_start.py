"""Preflight for the variable Sense runner; no full deck or stable lineup ID."""
from dataclasses import dataclass
from .models import Observation, Scene
from .objectives import StageObjectives
from .rules import StageRules
from .hp_preparation import hp_preparation_budget


@dataclass(frozen=True)
class VariableBattleSpec:
    character_id: str
    stage: int
    total_turns: int
    rank_at_most: int


def prepare_variable_battle(observation: Observation,objectives: StageObjectives | None,
                            rules: StageRules | None) -> VariableBattleSpec:
    character,stage=observation.character_id,observation.stage
    if (observation.scene!=Scene.CONFIRM or not character or type(stage) is not int or stage<1
        or observation.stars!=0 or not observation.formation_ready
        or observation.formation_slots is None or not observation.formation_slots.full
        or observation.formation_recommended_effect not in ('concentration','good_condition')
        or type(observation.total_turns) is not int or not 1<=observation.total_turns<=30):
        raise ValueError('automatic_variable_formation_unconfirmed')
    if (objectives is None or (objectives.character_id,objectives.stage)!=(character,stage)
        or objectives.source!='conditions_dialog' or objectives.identity_source!='opened_from_confirmed_formation'):
        raise ValueError('automatic_one_star_objective_unconfirmed')
    rank=objectives.rank_limit(1)
    if type(rank) is not int or not 1<=rank<=6:
        raise ValueError('automatic_one_star_rank_unknown')
    if (rules is None or (rules.character_id,rules.stage)!=(character,stage)
        or rules.source!='visible_support_details' or rules.identity_source!='opened_from_confirmed_formation'
        or not any(view.visible_rules for view in rules.viewports)):
        raise ValueError('automatic_variable_rules_unconfirmed')
    budget=hp_preparation_budget(character,stage,rules)
    if budget is None:
        raise ValueError('automatic_hp_budget_unknown')
    return VariableBattleSpec(character,stage,observation.total_turns,rank)
