"""Current-play HP support is narrower than general simulation, independent of lookahead."""
from kaa.db.constants import ProduceExamEffectType
from kaa.db.skill_card import SkillCard, ProduceExamEffect, load_exam_effects
from kaa.db.exam_status_enchant import get_exam_status_enchant
from kaa.tasks.produce.new.play_cards.planner.simulator import can_afford_play
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from .category_priority import supports_visible_followup


def _plain(effect: ProduceExamEffect, *, enchant: bool=False, chain: bool=False,search: bool=False) -> bool:
    return (all(type(value) is int for value in
            (effect.effect_turn,effect.effect_count,effect.effect_value1,effect.effect_value2))
        and type(effect.effect_value1) is int and effect.effect_value1>=0 and effect.effect_value2==0
        and (chain or not effect.chain_effect_ids)
        and (enchant or not effect._produce_exam_status_enchant_id)
        and not effect._produce_card_status_enchant_id
        and effect._produce_card_grow_effect_ids in (None,'','[]')
        and (search or not effect._produce_card_search_id) and not effect._target_produce_card_id)


def _resource_neutral_installation(effect: ProduceExamEffect) -> bool:
    """Installing these effects cannot spend HP; future gains are not credited.

    Draw identities, installed trigger timing and later resource balances still
    require fresh observations. This does not make a visible follow-up safe.
    """
    kind=effect.effect_type
    if kind==ProduceExamEffectType.ExamEffectTimer:
        if (not _plain(effect,chain=True) or effect.effect_value1!=1
            or effect.effect_count!=1 or effect.effect_turn!=0 or len(effect.chain_effect_ids)!=1):
            return False
        effects=load_exam_effects(set(effect.chain_effect_ids))
        delayed=effects.get(effect.chain_effect_ids[0])
        return (delayed is not None and _plain(delayed)
            and delayed.effect_type==ProduceExamEffectType.ExamCardDraw
            and delayed.effect_turn==delayed.effect_count==0
            and type(delayed.effect_value1) is int and delayed.effect_value1>=1)
    if (kind!=ProduceExamEffectType.ExamStatusEnchant or not _plain(effect,enchant=True)
        or effect.effect_turn!=-1 or effect.effect_count!=0 or effect.effect_value1!=0):
        return False
    enchant=get_exam_status_enchant(effect._produce_exam_status_enchant_id or '')
    if enchant is None or not enchant.root_effects_complete or not enchant.effects or enchant.chained_effects:
        return False
    for nested in enchant.effects:
        if not _plain(nested):return False
        if nested.effect_type==ProduceExamEffectType.ExamLesson:
            shaped=nested.effect_turn==0 and nested.effect_count==1
        elif nested.effect_type in (ProduceExamEffectType.ExamLessonBuff,ProduceExamEffectType.ExamBlockFix):
            shaped=nested.effect_turn==nested.effect_count==0
        elif nested.effect_type==ProduceExamEffectType.ExamParameterBuff:
            shaped=type(nested.effect_turn) is int and nested.effect_turn>0 and nested.effect_count==nested.effect_value1==0
        else:shaped=False
        if not shaped:return False
    return True


def _next_card_repeat(effect: ProduceExamEffect) -> bool:
    # Installing one repetition cannot spend HP or change this turn's budget.
    # Its later trigger and duplicated results are observed, never projected
    # into a second card's resource budget or a visible follow-up sequence.
    return (effect.effect_type==ProduceExamEffectType.ExamCardSearchEffectPlayCountBuff
        and effect._produce_card_search_id=='p_card_search-n-r-sr-ssr-playing'
        and _plain(effect,search=True) and effect.effect_turn==-1
        and effect.effect_value1==effect.effect_count==1)


def supports_current_hp_projection(card: SkillCard, state: BattleState, *, allow_audited_redraw: bool=False) -> bool:
    if state.projected_effect_values is not None or card._id in state.unresolved_growth_card_ids:
        return False
    from kaa.tasks.produce.new.play_cards.planner.costs import observed_play_cost
    if observed_play_cost(state,card) is not None:
        return True
    if supports_visible_followup(card, state):
        return True
    if allow_audited_redraw is True:
        from .current_action import assess_current_action
        # Only the exact audited current roots. This says nothing about the
        # identities/effects of the replaced hand or equipped-item triggers.
        current=assess_current_action(state,card)
        if current.root_projection_available:return True
        if (current.root_effects_supported and state.hp_known and state.genki_known
            and state.playable_count>0 and 'stamina_consumption_modifier' in state.observed_effects
            and can_afford_play(state,card) is None):
            return True
    # Score/buff primitives and a modeled green-cost reduction cannot change
    # the already paid HP cost. Unknown score triggers remain unknown. This
    # grants only a current HP projection, never a future two-card prefix.
    if (not card.play_effects or card._id in state.unresolved_growth_card_ids or state.projected_card_growth):
        return False
    pure_score={ProduceExamEffectType.ExamLesson,ProduceExamEffectType.ExamLessonAddMultipleParameterBuff,
        ProduceExamEffectType.ExamMultipleLessonBuffLesson}
    for segment in card.play_effects:
        effect=segment.produce_exam_effect
        if effect is not None and (_resource_neutral_installation(effect) or _next_card_repeat(effect)):
            continue
        if not (effect is not None and not effect.chain_effect_ids and type(effect.effect_turn) is int
            and type(effect.effect_count) is int
            and type(effect.effect_value1) is int and effect.effect_value1>=0
            and type(effect.effect_value2) is int and effect.effect_value2>=0
            and not effect._produce_exam_status_enchant_id and not effect._produce_card_status_enchant_id
            and effect._produce_card_grow_effect_ids in (None,'','[]')
            and not effect._produce_card_search_id and not effect._target_produce_card_id):
            return False
        kind=effect.effect_type
        if kind in pure_score:
            shaped=effect.effect_turn==0 and effect.effect_count==1
        elif kind==ProduceExamEffectType.ExamBlock:
            shaped=effect.effect_turn==0 and effect.effect_count==0 and effect.effect_value2==0
        elif kind==ProduceExamEffectType.ExamStaminaConsumptionDownFix:
            shaped=effect.effect_turn==-1 and effect.effect_count==0 and effect.effect_value2==0
        elif kind==ProduceExamEffectType.ExamParameterBuffMultiplePerTurn:
            shaped=effect.effect_turn>0 and effect.effect_count==0 and effect.effect_value1==effect.effect_value2==0
        elif kind==ProduceExamEffectType.ExamCardDraw:
            shaped=effect.effect_turn==effect.effect_count==effect.effect_value2==0 and effect.effect_value1>0
        elif kind in (ProduceExamEffectType.ExamAntiDebuff,ProduceExamEffectType.ExamPlayableValueAdd):
            shaped=effect.effect_turn==effect.effect_value1==effect.effect_value2==0 and effect.effect_count>0
        elif kind==ProduceExamEffectType.ExamStaminaConsumptionDown:
            shaped=effect.effect_turn>0 and effect.effect_count==effect.effect_value1==effect.effect_value2==0
        elif kind==ProduceExamEffectType.ExamLessonBuff:
            shaped=effect.effect_turn==effect.effect_count==effect.effect_value2==0
        else:shaped=False
        if not shaped:return False
    return True
