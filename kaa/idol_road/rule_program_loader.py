"""将描述候选的显式 DB 引用交给纯解释器；不按命名挑选当前关卡。"""

import json

from kaa.db.tower_rules import exam_rule_definition, produce_status_enchant
from .rule_program import CandidateDrawProgram, DrawRuleInputs, RuleProgramAnalysis, interpret_draw_declaration, summarize_rule_programs
from .rule_references import RuleDescriptionMatch


def analyze_rule_programs(match: RuleDescriptionMatch) -> RuleProgramAnalysis:
    programs = []
    for reference in match.candidates:
        root = exam_rule_definition('effect', reference.exam_effect_id) if reference.exam_effect_id else None
        # 原始根记录由解释器再次校验，这里只提取显式附魔 ID。
        fields = json.loads(root.fields_json) if root is not None else {}
        enchant_id = fields.get('produceExamStatusEnchantId')
        enchant = produce_status_enchant(enchant_id) if isinstance(enchant_id, str) and enchant_id else None
        trigger = exam_rule_definition('trigger', enchant.trigger_id) if enchant is not None and enchant.trigger_id else None
        ids = enchant.exam_effect_ids if enchant is not None else None
        children = tuple(exam_rule_definition('effect', effect_id) for effect_id in ids) if ids is not None and len(ids) == 1 else ()
        inputs = DrawRuleInputs(reference.exam_effect_id or '', root.fields_json if root else None,
                                enchant.id if enchant else None, enchant.trigger_id if enchant else None,
                                trigger.fields_json if trigger else None, ids,
                                tuple(child.fields_json if child else None for child in children))
        declaration, reason = interpret_draw_declaration(inputs)
        programs.append(CandidateDrawProgram(reference, declaration, reason))
    return summarize_rule_programs(tuple(programs))
