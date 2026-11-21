"""Current field countdown evidence; independent of the remaining-turn budget."""
from dataclasses import dataclass
import re
from .rules import RuleTextRead


@dataclass(frozen=True)
class CountdownDigits:
    frame_id: str
    rect: tuple[int,int,int,int]
    suffix_rect: tuple[int,int,int,int]
    suffix_score: float
    reads: tuple[RuleTextRead,...]
    right_edge_ink: bool = False
    scaled_reads: tuple[RuleTextRead,...] = ()


@dataclass(frozen=True)
class ScheduleCountdown:
    frame_id: str
    rect: tuple[int,int,int,int]
    reads: tuple[RuleTextRead,...]
    turns_after: int | None
    scaled_reads: tuple[RuleTextRead,...] = ()
    digits: CountdownDigits | None = None


def confirm_schedule_countdown(frame_id: str,rect: tuple[int,int,int,int],
                               reads: tuple[RuleTextRead,...],
                               scaled_reads: tuple[RuleTextRead,...]=(),
                               digits: CountdownDigits | None=None) -> ScheduleCountdown:
    def result(value=None):return ScheduleCountdown(frame_id,rect,reads,value,scaled_reads,digits)
    if not frame_id or min(rect)<0 or min(rect[2:])<=0:return result()
    values=[];confirmed=[]
    pairs=[(reads,r'(0|[1-9][0-9]?)ターン後')]
    if scaled_reads:pairs.append((scaled_reads,r'(0|[1-9][0-9]?)ターン後'))
    if digits is not None:
        x,y,w,h=rect;sx,sy,sw,sh=digits.suffix_rect
        if (digits.frame_id!=frame_id or not .97<=digits.suffix_score<=1 or digits.right_edge_ink
            or not x<sx or not y<=sy or min(sw,sh)<=0 or sx+sw>x+w or sy+sh>y+h
            or digits.rect!=(x,y,sx-x,h)):
            return result()
        pairs.append((digits.reads,r'(0|[1-9][0-9]?)'))
        if digits.scaled_reads:pairs.append((digits.scaled_reads,r'(0|[1-9][0-9]?)'))
    for pair,pattern in pairs:
        if (len(pair)!=2 or {r.representation for r in pair}!={'original','grayscale'}
            or any(r.frame_id!=frame_id or len(r.lines)>1 for r in pair)):
            return result()
        current=[]
        for read in pair:
            if len(read.lines)==1 and .98<=read.lines[0].confidence<=1:
                match=re.fullmatch(pattern,read.lines[0].text)
                if match is None:return result()
                current.append(int(match[1]))
        values.extend(current)
        if len(current)==2 and current[0]==current[1]:confirmed.append(current[0])
    return result(confirmed[0]) if confirmed and len(set(values))==1 else result()


def countdown_turn_candidates(countdown: ScheduleCountdown,scheduled_turns: tuple[int,...],
                               candidates: tuple[int,...]) -> tuple[int,...]:
    """Match only declared possible rounds; ambiguity remains explicit.

    The visible preview need not be the nearest rule. This returns evidence
    candidates, not a permission to bind a round or settle an input.
    """
    if (confirm_schedule_countdown(countdown.frame_id,countdown.rect,countdown.reads,countdown.scaled_reads,countdown.digits)!=countdown
        or countdown.turns_after is None
        or any(type(turn) is not int or turn<1 for turn in (*scheduled_turns,*candidates))):
        return ()
    return tuple(sorted({turn for turn in candidates if turn+countdown.turns_after in scheduled_turns}))
