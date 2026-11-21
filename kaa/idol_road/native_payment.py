"""Audit action-count invariants before accepting a native/model fee conflict."""
from kaa.db.constants import ProduceExamEffectType as Effect
from kaa.db.skill_card import SkillCard


_NO_ACTION_CHANGE = frozenset({
    Effect.ExamLesson,Effect.ExamMultipleLessonBuffLesson,Effect.ExamLessonAddMultipleParameterBuff,
    Effect.ExamLessonBuff,Effect.ExamParameterBuff,Effect.ExamParameterBuffMultiplePerTurn,
    Effect.ExamBlock,Effect.ExamStaminaRecoverFix,Effect.ExamStaminaConsumptionDownFix,Effect.ExamAntiDebuff,
})


def supports_native_payment_conflict(card: SkillCard) -> bool:
    """All supported DB upgrades spend one action; no fee/HP amount is proved.

    A partially hidden upgrade may change a fee. Keep the original version
    reading and numeric payment uncertain instead of inventing an upgrade.
    Draws, extra actions, chains and installed triggers need other contracts.
    """
    if not card._asset_id or type(card.upgrade_count) is not int or not 0<=card.upgrade_count<=3:
        return False
    canonical=SkillCard.from_asset_id(card._asset_id,card.upgrade_count)
    if canonical is None or canonical!=card:return False
    for upgrade in range(4):
        variant=SkillCard.from_asset_id(card._asset_id,upgrade)
        # The loader can fall back to the base when a version is absent.
        if variant is None or variant.upgrade_count!=upgrade:continue
        if (variant.force_stamina!=0 or variant.cost_value!=0 or variant.cost_type!='ExamCostType_Unknown'
            or variant._play_produce_exam_trigger_id or variant._produce_card_status_enchant_id
            or variant.move_produce_exam_effects or not variant.play_effects
            or variant.play_move_position_type!=card.play_move_position_type):
            return False
        for segment in variant.play_effects:
            effect=segment.produce_exam_effect
            if (effect is None or effect.effect_type not in _NO_ACTION_CHANGE
                or effect.chain_effect_ids or effect._chain_produce_exam_effect_id
                or effect._produce_exam_status_enchant_id or effect._produce_card_status_enchant_id
                or effect._produce_card_grow_effect_ids not in (None,'','[]')
                or effect._produce_card_search_id or effect._target_produce_card_id):
                return False
    return True
