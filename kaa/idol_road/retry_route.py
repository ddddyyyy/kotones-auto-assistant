"""Root-only exploration of failed requested routes inside existing hard limits."""
from kaa.db.skill_card import SkillCard
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from .battle_inputs import BattleInputs
from .battle_observation import BattleFrameObservation
from .retry_policy import StageRetryPolicy


class RetryRoutePreference:
    def __init__(self,policy: StageRetryPolicy,inputs: BattleInputs,frame: BattleFrameObservation):
        self.policy=policy
        self.inputs=inputs
        self.frame=frame
        self.prefix=inputs.requested_route
        self.sequence=inputs.sequence
        self.revision=policy.lineup_revision

    def failed_count(self,card: SkillCard,state: BattleState) -> int:
        if (self.revision!=self.policy.lineup_revision or self.sequence!=self.inputs.sequence
            or self.prefix!=self.inputs.requested_route or not self.inputs.can_select(self.frame,state)):
            raise ValueError('Retry ranking requires the current idle owned input frame')
        turn=self.frame.observation.turn
        if type(turn) is not int or turn<1:raise ValueError('Missing current turn')
        return self.policy.failed_continuations(self.prefix,f'{turn}:card:{card._id}')
