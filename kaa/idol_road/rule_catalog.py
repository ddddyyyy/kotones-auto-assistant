"""统一的局部规则声明视图；保留分屏来源与未知项，不是事件计划。"""
from dataclasses import dataclass
from typing import Literal
from .conditional_rule import ConditionalPlayDeclaration, interpret_conditional_play
from .health_rule import HealthMentalBlockDeclaration, interpret_health_block
from .concentration_rule import ConcentrationDoubleDeclaration, interpret_concentration_rule
from .threshold_turn_end import ThresholdTurnEndDeclaration, interpret_threshold_turn_end
from .rules import StageRules, confirmed_rule_turn_with_number


@dataclass(frozen=True)
class RuleDeclarationSighting:
    viewport_index: int
    rule_index: int
    kind: Literal['concentration_extra_play', 'hp_mental_block', 'concentration_double', 'concentration_turn_end']
    declaration: ConditionalPlayDeclaration | HealthMentalBlockDeclaration | ConcentrationDoubleDeclaration | ThresholdTurnEndDeclaration


@dataclass(frozen=True)
class UnknownRuleSighting:
    viewport_index: int | None
    rule_index: int | None
    reason: Literal['viewport_unknown', 'turn_unconfirmed', 'unsupported_visible_rule', 'conflicting_declarations']


@dataclass(frozen=True)
class RuleDeclarationCatalog:
    character_id: str
    stage: int
    status: Literal['partial', 'identity_conflict', 'source_unknown']
    sightings: tuple[RuleDeclarationSighting, ...] = ()
    unknown_sightings: tuple[UnknownRuleSighting, ...] = ()
    complete: Literal[False] = False
    executable: Literal[False] = False


def build_rule_catalog(character_id: str, stage: int, rules: StageRules) -> RuleDeclarationCatalog:
    if (not character_id or type(stage) is not int or stage < 1
        or (rules.character_id, rules.stage) != (character_id, stage)):
        return RuleDeclarationCatalog(character_id, stage, 'identity_conflict')
    if rules.source != 'visible_support_details' or rules.identity_source != 'opened_from_confirmed_formation':
        return RuleDeclarationCatalog(character_id, stage, 'source_unknown')
    if not rules.viewports:
        return RuleDeclarationCatalog(character_id, stage, 'partial',
                                      unknown_sightings=(UnknownRuleSighting(None, None, 'viewport_unknown'),))
    sightings = []
    unknown = []
    for page_index, page in enumerate(rules.viewports):
        if not page.visible_rules:
            unknown.append(UnknownRuleSighting(page_index, None, 'viewport_unknown'))
            continue
        for index, row in enumerate(page.visible_rules):
            if (row.turn is None or confirmed_rule_turn_with_number(row.turn_reads, row.turn_number_evidence) != row.turn):
                unknown.append(UnknownRuleSighting(page_index, index, 'turn_unconfirmed'))
                continue
            candidates = []
            frame_id = row.turn_reads[0].frame_id
            if row.threshold_turn_end is not None and row.threshold_turn_end.evidence.frame_id == frame_id:
                checked_end = interpret_threshold_turn_end(row.turn,row.threshold_turn_end.evidence,row.description_reads)
                if checked_end.declaration is not None and checked_end == row.threshold_turn_end:
                    candidates.append(RuleDeclarationSighting(page_index,index,'concentration_turn_end',checked_end.declaration))
            if row.concentration_rule is not None and row.concentration_rule.evidence.frame_id == frame_id:
                checked_double = interpret_concentration_rule(row.turn, row.concentration_rule.evidence, row.description_reads)
                if checked_double.declaration is not None and checked_double == row.concentration_rule:
                    candidates.append(RuleDeclarationSighting(page_index, index, 'concentration_double', checked_double.declaration))
            if row.conditional_play is not None and row.conditional_play.evidence.frame_id == frame_id:
                checked = interpret_conditional_play(row.turn, row.conditional_play.evidence, row.description_reads)
                if checked.declaration is not None and checked == row.conditional_play:
                    candidates.append(RuleDeclarationSighting(page_index, index, 'concentration_extra_play', checked.declaration))
            if row.health_block is not None and row.health_block.evidence.frame_id == frame_id:
                checked_health = interpret_health_block(row.turn, row.health_block.evidence, row.description_reads)
                if checked_health.declaration is not None and checked_health == row.health_block:
                    candidates.append(RuleDeclarationSighting(page_index, index, 'hp_mental_block', checked_health.declaration))
            if len(candidates) == 1:
                sightings.extend(candidates)
            else:
                unknown.append(UnknownRuleSighting(page_index, index,
                               'conflicting_declarations' if candidates else 'unsupported_visible_rule'))
    return RuleDeclarationCatalog(character_id, stage, 'partial', tuple(sightings), tuple(unknown))
