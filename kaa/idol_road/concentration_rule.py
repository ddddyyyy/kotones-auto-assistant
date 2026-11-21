"""集中阈值翻倍的可见声明，不推定结算阶段或实际获得倍率。"""
from dataclasses import dataclass
import re
import unicodedata
from typing import Literal
from .rules import RuleTextRead
from .rule_symbols import RuleSymbolEvidence


@dataclass(frozen=True)
class ConcentrationRuleEvidence:
    frame_id: str
    icons: tuple[RuleSymbolEvidence, ...]
    reads: tuple[RuleTextRead, ...]


@dataclass(frozen=True)
class ConcentrationDoubleDeclaration:
    start_turn: int
    concentration_at_least: int
    multiplier: Literal[2] = 2
    executable: Literal[False] = False


@dataclass(frozen=True)
class ConcentrationRuleAnalysis:
    evidence: ConcentrationRuleEvidence
    declaration: ConcentrationDoubleDeclaration | None
    status: Literal['confirmed_visible_declaration', 'evidence_unknown', 'unsupported_text', 'high_confidence_conflict']


def compact(text: str) -> str:
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))


def interpret_concentration_rule(turn: int | None, evidence: ConcentrationRuleEvidence,
                                 original: tuple[RuleTextRead, ...]) -> ConcentrationRuleAnalysis:
    def result(status, declaration=None):
        return ConcentrationRuleAnalysis(evidence, declaration, status)
    icons = evidence.icons
    if (type(turn) is not int or turn < 2 or not evidence.frame_id or len(icons) != 2
        or any(icon.kind != 'concentration' or icon.frame_id != evidence.frame_id
               or not .97 <= icon.score <= 1 or min(icon.rect) < 0 or min(icon.rect[2:]) <= 0 for icon in icons)):
        return result('evidence_unknown')
    first, second = sorted(icons, key=lambda icon: (icon.rect[1], icon.rect[0]))
    if (first.rect[0] < second.rect[0]+second.rect[2] and second.rect[0] < first.rect[0]+first.rect[2]
        and first.rect[1] < second.rect[1]+second.rect[3] and second.rect[1] < first.rect[1]+first.rect[3]):
        return result('evidence_unknown')
    reads = evidence.reads
    if (len(reads) != 2 or {read.representation for read in reads} != {'original', 'grayscale'}
        or any(read.frame_id != evidence.frame_id or not read.lines
               or any(not .98 <= line.confidence <= 1 for line in read.lines) for read in reads)
        or len(original) != 2 or {read.representation for read in original} != {'original', 'grayscale'}
        or any(read.frame_id != evidence.frame_id for read in original)):
        return result('evidence_unknown')
    texts = [compact(''.join(line.text for line in read.lines)) for read in reads]
    if texts[0] != texts[1]:
        return result('high_confidence_conflict')
    match = re.fullmatch(r'集中が([1-9][0-9]*)以上の場合、集中2倍', texts[0])
    if match is None:
        return result('unsupported_text')
    threshold = int(match[1])
    for read in original:
        for line in read.lines:
            if not .98 <= line.confidence <= 1:
                continue
            text = compact(line.text)
            condition = re.search(r'集中が([0-9]+)(以上|以下|未満|超)', text)
            multiplier = re.search(r'集中([0-9]+)倍', text)
            if ((condition is not None and (int(condition[1]) != threshold or condition[2] != '以上'))
                or (multiplier is not None and multiplier[1] != '2')):
                return result('high_confidence_conflict')
    return result('confirmed_visible_declaration', ConcentrationDoubleDeclaration(turn, threshold))
