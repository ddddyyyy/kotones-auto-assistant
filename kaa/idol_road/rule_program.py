"""解释明确附魔引用中的抽牌声明；不是当前关卡事件或执行器。"""

from dataclasses import dataclass
import json
from typing import Literal

from .rule_references import RuleEffectReference


@dataclass(frozen=True)
class DrawRuleInputs:
    root_id: str
    root_json: str | None
    enchant_id: str | None
    trigger_id: str | None
    trigger_json: str | None
    effect_ids: tuple[str, ...] | None
    effect_jsons: tuple[str | None, ...]


@dataclass(frozen=True)
class DrawDeclaration:
    phase: Literal['ProduceExamPhaseType_StartPlay']
    count: int
    enchant_effect_turn: int
    """原始期限字段，不把 -1 或 0 翻译成已确认的无限次数。"""


@dataclass(frozen=True)
class CandidateDrawProgram:
    reference: RuleEffectReference
    declaration: DrawDeclaration | None
    reason: Literal['supported_draw_declaration', 'missing_definition', 'unsupported_definition']


@dataclass(frozen=True)
class RuleProgramAnalysis:
    candidates: tuple[CandidateDrawProgram, ...]
    common_draw_declaration: DrawDeclaration | None
    candidate_scope: Literal['matched_plain_text_candidates_only'] = 'matched_plain_text_candidates_only'
    stage_binding_confirmed: Literal[False] = False
    """共同声明不证明效果组的应用条件或全部行为等价。"""
    executable: Literal[False] = False


_EFFECT_DEFAULTS = {
    'effectValue1': 0, 'effectValue2': 0, 'effectCount': 0, 'effectTurn': 0,
    'targetProduceCardId': '', 'targetUpgradeCount': 0,
    'targetExamEffectType': 'ProduceExamEffectType_Unknown', 'produceCardSearchId': '',
    'movePositionType': 'ProduceCardMovePositionType_Unknown',
    'pickRangeType': 'ProducePickRangeType_Unknown', 'pickCountReferenceProduceCardSearchId': '',
    'pickCountType': 'ProducePickCountType_Unknown', 'pickCountMin': 0, 'pickCountMax': 0,
    'produceCardSearchId2': '', 'pickRangeType2': 'ProducePickRangeType_Unknown',
    'pickCountReferenceProduceCardSearchId2': '', 'pickCountType2': 'ProducePickCountType_Unknown',
    'pickCountMin2': 0, 'pickCountMax2': 0, 'chainProduceExamEffectId': '',
    'chainProduceExamEffectIds': '[]', 'produceExamStatusEnchantId': '',
    'produceCardStatusEnchantId': '', 'produceCardGrowEffectIds': '[]',
}
_TRIGGER_DEFAULTS = {
    'phaseTypes': '["ProduceExamPhaseType_StartPlay"]', 'phaseValues': '[]',
    'fieldStatusCheckTypes': '[]', 'fieldStatusTypes': '[]', 'fieldStatusValues': '[]',
    'fieldStatusProduceCardSearchIds': '[]', 'produceCardSearchId': '',
    'upperSearchCount': 0, 'lowerSearchCount': 0,
    'cardMovePositionType': 'ProduceCardMovePositionType_Unknown', 'effectTypes': '[]',
    'lessonType': 'ProduceStepLessonType_Unknown',
}
_METADATA = {'produceDescriptions', 'customizeProduceDescriptions',
             'playProduceDescriptions', 'playEffectProduceDescriptions', 'effectGroupIds'}
_ARRAYS = {'chainProduceExamEffectIds', 'produceCardGrowEffectIds', 'phaseTypes', 'phaseValues',
           'fieldStatusCheckTypes', 'fieldStatusTypes', 'fieldStatusValues',
           'fieldStatusProduceCardSearchIds', 'effectTypes'}


def _record(raw: str | None, expected_id: str | None) -> dict | None:
    if raw is None or not expected_id:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and data.get('id') == expected_id else None


def _matches(data: dict, expected: dict) -> bool:
    if set(data) - set(expected) - _METADATA - {'id'}:
        return False
    for key, value in expected.items():
        actual = data.get(key)
        if key in _ARRAYS:
            try:
                if not isinstance(actual, str) or json.loads(actual) != json.loads(value):
                    return False
            except (TypeError, json.JSONDecodeError):
                return False
        elif type(actual) is not type(value) or actual != value:
            return False
    return True


def interpret_draw_declaration(inputs: DrawRuleInputs) -> tuple[DrawDeclaration | None, Literal['supported_draw_declaration', 'missing_definition', 'unsupported_definition']]:
    root = _record(inputs.root_json, inputs.root_id)
    trigger = _record(inputs.trigger_json, inputs.trigger_id)
    if root is None or trigger is None or inputs.effect_ids is None or any(raw is None for raw in inputs.effect_jsons):
        return None, 'missing_definition'
    # 仅支持一层附魔到单一抽牌效果；分支、重复、链式效果都显式未支持。
    if len(inputs.effect_ids) != 1 or len(inputs.effect_jsons) != 1:
        return None, 'unsupported_definition'
    child = _record(inputs.effect_jsons[0], inputs.effect_ids[0])
    if child is None:
        return None, 'missing_definition'
    count, duration = child.get('effectValue1'), root.get('effectTurn')
    if type(count) is not int or count < 1 or type(duration) is not int or not inputs.enchant_id:
        return None, 'unsupported_definition'
    root_fields = {**_EFFECT_DEFAULTS, 'effectType': 'ProduceExamEffectType_ExamStatusEnchant',
                   'effectTurn': duration, 'produceExamStatusEnchantId': inputs.enchant_id}
    draw_fields = {**_EFFECT_DEFAULTS, 'effectType': 'ProduceExamEffectType_ExamCardDraw', 'effectValue1': count}
    if not (_matches(root, root_fields) and _matches(child, draw_fields) and _matches(trigger, _TRIGGER_DEFAULTS)):
        return None, 'unsupported_definition'
    return DrawDeclaration('ProduceExamPhaseType_StartPlay', count, duration), 'supported_draw_declaration'


def summarize_rule_programs(candidates: tuple[CandidateDrawProgram, ...]) -> RuleProgramAnalysis:
    common = candidates[0].declaration if candidates else None
    if any(row.declaration != common or row.reason != 'supported_draw_declaration' for row in candidates):
        common = None
    return RuleProgramAnalysis(candidates, common)
