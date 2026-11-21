"""只匹配纯文本描述的候选引用；不声明效果已绑定或候选完整。"""

from dataclasses import dataclass
import json
import re
from typing import Literal
import unicodedata

from .rules import RuleTextRead


def _normalize(text: str) -> str:
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))


@dataclass(frozen=True)
class RuleDescriptionCandidate:
    group_id: str
    priority: int
    start_turn: int
    exam_effect_id: str | None
    description_json: str | None


@dataclass(frozen=True)
class RuleEffectReference:
    group_id: str
    priority: int
    start_turn: int
    exam_effect_id: str | None


@dataclass(frozen=True)
class RuleDescriptionMatch:
    text: str | None
    candidates: tuple[RuleEffectReference, ...]
    status: Literal['text_unknown', 'source_unavailable', 'plain_text_candidates', 'no_plain_text_match']
    candidate_scope: Literal['plain_text_descriptions_only'] = 'plain_text_descriptions_only'
    complete: Literal[False] = False
    """带图标、动态描述和未知格式没有参与比较，不能当作唯一候选集合。"""


def match_rule_description(
    turn: int | None, reads: tuple[RuleTextRead, ...], candidates: tuple[RuleDescriptionCandidate, ...] | None,
) -> RuleDescriptionMatch:
    if (type(turn) is not int or turn < 1 or len(reads) != 2 or not reads[0].frame_id
        or reads[0].frame_id != reads[1].frame_id
        or {read.representation for read in reads} != {'original', 'grayscale'}):
        return RuleDescriptionMatch(None, (), 'text_unknown')
    texts = []
    for read in reads:
        # 不拼接多行碎片；图标可能被漏掉，文字匹配也不能证明效果身份。
        if len(read.lines) != 1 or not .98 <= read.lines[0].confidence <= 1:
            return RuleDescriptionMatch(None, (), 'text_unknown')
        texts.append(_normalize(read.lines[0].text))
    if not texts[0] or texts[0] != texts[1]:
        return RuleDescriptionMatch(None, (), 'text_unknown')
    text = texts[0]
    if candidates is None:
        return RuleDescriptionMatch(text, (), 'source_unavailable')
    matched = []
    for candidate in candidates:
        if candidate.start_turn != turn or not candidate.description_json:
            continue
        try:
            parts = json.loads(candidate.description_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if (not isinstance(parts, list) or not parts
            or any(not isinstance(part, dict) or part.get('produceDescriptionType') != 'ProduceDescriptionType_PlainText'
                   or not isinstance(part.get('text'), str) or '<' in part['text'] or '>' in part['text'] for part in parts)):
            continue
        if _normalize(''.join(part['text'] for part in parts)) == text:
            matched.append(RuleEffectReference(candidate.group_id, candidate.priority,
                                                candidate.start_turn, candidate.exam_effect_id))
    return RuleDescriptionMatch(text, tuple(matched), 'plain_text_candidates' if matched else 'no_plain_text_match')
