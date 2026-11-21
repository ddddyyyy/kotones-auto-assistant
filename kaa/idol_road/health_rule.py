"""体力百分比条件下禁用 mental 卡两回合的可见声明；不执行。"""
from dataclasses import dataclass
import re
from typing import Literal
import unicodedata
from .rules import RuleTextRead
from .rule_symbols import RuleSymbolEvidence


@dataclass(frozen=True)
class HealthBlockEvidence:
    frame_id: str
    symbols: tuple[RuleSymbolEvidence, ...]
    condition_reads: tuple[RuleTextRead, ...]
    duration_reads: tuple[RuleTextRead, ...] = ()


@dataclass(frozen=True)
class HealthMentalBlockDeclaration:
    start_turn: int
    hp_at_most_percent: int
    duration_turns: Literal[2] = 2
    blocked_category: Literal['mental'] = 'mental'
    executable: Literal[False] = False


@dataclass(frozen=True)
class HealthBlockAnalysis:
    evidence: HealthBlockEvidence
    declaration: HealthMentalBlockDeclaration | None
    status: Literal['confirmed_visible_declaration', 'turn_unknown', 'symbols_unknown', 'condition_unknown',
                    'duration_unknown', 'unsupported_condition', 'high_confidence_conflict']


def _compact(text: str) -> str:
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))


def interpret_health_block(turn: int | None, evidence: HealthBlockEvidence,
                           original_reads: tuple[RuleTextRead, ...]) -> HealthBlockAnalysis:
    def result(status, declaration=None):
        return HealthBlockAnalysis(evidence, declaration, status)
    if type(turn) is not int or turn < 1:
        return result('turn_unknown')
    symbols = {symbol.kind: symbol for symbol in evidence.symbols}
    if (not evidence.frame_id or len(evidence.symbols) != 3
        or set(symbols) != {'mental', 'mental_skill_label', 'mental_block_two_turns'}
        or any(symbol.frame_id != evidence.frame_id or not .97 <= symbol.score <= 1
               or min(symbol.rect) < 0 or min(symbol.rect[2:]) <= 0 for symbol in evidence.symbols)):
        return result('symbols_unknown')
    icon, label, block = (symbols[k].rect for k in ('mental', 'mental_skill_label', 'mental_block_two_turns'))
    if (label[0] < icon[0]+icon[2] or abs(label[1]-icon[1]) > 4
        or block[1] < icon[1]+icon[3]-2 or block[0]+block[2] > label[0]):
        return result('symbols_unknown')
    duration_reads = evidence.duration_reads
    if (len(duration_reads) != 2 or any(read.frame_id != evidence.frame_id for read in duration_reads)
        or {read.representation for read in duration_reads} != {'original', 'grayscale'}
        or any(len(read.lines) != 1 or not .98 <= read.lines[0].confidence <= 1
               or _compact(read.lines[0].text) != '2' for read in duration_reads)):
        return result('duration_unknown')
    reads = evidence.condition_reads
    if (len(reads) != 2 or any(read.frame_id != evidence.frame_id for read in (*reads, *original_reads))
        or len(original_reads) != 2
        or {read.representation for read in reads} != {'original', 'grayscale'}
        or {read.representation for read in original_reads} != {'original', 'grayscale'}):
        return result('condition_unknown')
    texts = []
    for read in reads:
        if len(read.lines) != 1 or not .98 <= read.lines[0].confidence <= 1:
            return result('condition_unknown')
        texts.append(_compact(read.lines[0].text))
    if texts[0] != texts[1]:
        return result('condition_unknown')
    match = re.fullmatch(r'体力が([1-9][0-9]?|100)%以下の場合、', texts[0])
    if match is None:
        return result('unsupported_condition')
    threshold = int(match[1])
    for read in original_reads:
        for line in read.lines:
            if not .98 <= line.confidence <= 1:
                continue
            text = _compact(line.text)
            condition = re.search(r'体力が([0-9]+)%(以下|以上|未満|超)', text)
            duration = re.search(r'\(([0-9]+)ターン\)', text)
            if ((condition is not None and (int(condition[1]) != threshold or condition[2] != '以下'))
                or (duration is not None and int(duration[1]) != 2)
                or 'アクティブスキル' in text or 'カード使用可' in text):
                return result('high_confidence_conflict')
    return result('confirmed_visible_declaration', HealthMentalBlockDeclaration(turn, threshold))
