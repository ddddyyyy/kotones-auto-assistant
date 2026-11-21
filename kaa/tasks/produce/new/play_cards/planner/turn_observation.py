"""Optional adapter contract: absolute round and remaining horizon are distinct.

The adapter must confirm currentness before producing this value. A visible
historical log title, an OCR correction or a predicted reward is insufficient.
"""
from kaa.battle_turn import AbsoluteTurnObservation as AbsoluteTurnObservation
