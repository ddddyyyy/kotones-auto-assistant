"""正文内多模板匹配、同位置去重；不同位置的重复符号保持独立。"""
import cv2
from cv2.typing import MatLike
from .rule_symbols import RuleSymbolEvidence, RuleSymbolKind


def find_rule_symbols(body: MatLike, frame_id: str,
                      kinds: tuple[tuple[RuleSymbolKind, tuple], ...]) -> tuple[RuleSymbolEvidence, ...]:
    found = []
    for kind, templates in kinds:
        candidates = []
        for template in templates:
            pixels = template.template.pixels
            if body.shape[0] < pixels.shape[0] or body.shape[1] < pixels.shape[1]:
                continue
            scores = cv2.matchTemplate(body, pixels, cv2.TM_CCOEFF_NORMED)
            for _ in range(3):
                _, score, _, (x, y) = cv2.minMaxLoc(scores)
                if score < .97:
                    break
                candidates.append(RuleSymbolEvidence(frame_id, kind, (x, y, pixels.shape[1], pixels.shape[0]), score))
                scores[max(0,y-pixels.shape[0]//2):y+pixels.shape[0]//2+1,
                       max(0,x-pixels.shape[1]//2):x+pixels.shape[1]//2+1] = -1
        selected = []
        for icon in sorted(candidates, key=lambda icon: icon.score, reverse=True):
            if not any(abs(icon.rect[0]-other.rect[0]) < icon.rect[2]//2
                       and abs(icon.rect[1]-other.rect[1]) < icon.rect[3]//2 for other in selected):
                selected.append(icon)
        found.extend(selected)
    return tuple(found)
