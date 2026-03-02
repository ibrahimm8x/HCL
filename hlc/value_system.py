"""Value System: hardcoded motivational signals that cannot be unlearned."""
from dataclasses import dataclass
from enum import Enum

from hlc.config import Config


class Signal(Enum):
    PAIN = "pain"
    JOY = "joy"
    FEAR = "fear"
    CURIOSITY = "curiosity"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"


@dataclass
class ValueState:
    """Current motivational/emotional state of the system."""
    pain: float = 0.0        # 0.0 to 1.0
    joy: float = 0.0
    fear: float = 0.0
    curiosity: float = 0.0
    surprise: float = 0.0

    @property
    def dominant_signal(self) -> Signal:
        """Which signal is currently strongest?"""
        signals = {
            Signal.PAIN: self.pain,
            Signal.JOY: self.joy,
            Signal.FEAR: self.fear,
            Signal.CURIOSITY: self.curiosity,
            Signal.SURPRISE: self.surprise,
        }
        dominant = max(signals, key=signals.get)
        if signals[dominant] < 0.1:
            return Signal.NEUTRAL
        return dominant

    def __repr__(self) -> str:
        return (
            f"ValueState(dominant={self.dominant_signal.value}, "
            f"pain={self.pain:.2f}, joy={self.joy:.2f}, "
            f"fear={self.fear:.2f}, curiosity={self.curiosity:.2f}, "
            f"surprise={self.surprise:.2f})"
        )


class ValueSystem:
    """
    Hardcoded motivational signals. These are architectural, not learned.
    They cannot decay. They cannot be overwritten. Like hunger in the brain.

    For v1: signals are computed but do NOT dynamically tune routing
    parameters. They are informational — logged and reported so we
    can observe the model's "emotional state."
    """

    def __init__(self, config: Config):
        self.config = config
        self.state = ValueState()

    def evaluate(
        self,
        prediction_error: float,
        novelty: float,
        match_confidence: float,
    ) -> ValueState:
        """
        Evaluate the current situation and update value signals.

        Args:
            prediction_error: how wrong the prediction was (0.0 = perfect, 1.0 = completely wrong)
            novelty: how novel the input is (0.0 = familiar, 1.0 = completely new)
            match_confidence: confidence of the best column match (0.0 = no match, 1.0 = perfect match)
        """
        # SURPRISE: unexpected result, but not necessarily harmful
        self.state.surprise = (
            max(0.0, prediction_error - 0.3)
            * self.config.surprise_weight
            if match_confidence > 0.3
            else 0.0
        )

        # CURIOSITY: driven by novelty
        self.state.curiosity = novelty * self.config.curiosity_weight

        # PAIN: the model was very wrong (high prediction error)
        self.state.pain = max(0.0, prediction_error - 0.7) * self.config.pain_weight

        # JOY: correct, confident retrieval
        self.state.joy = (
            max(0.0, match_confidence - 0.5)
            * (1.0 - prediction_error)
            * self.config.joy_weight
        )

        # FEAR: patterns that previously led to pain (v1 simplified)
        self.state.fear = self.state.pain * 0.5 * self.config.fear_weight

        return self.state

    def get_routing_modulation(self) -> dict:
        """
        How should the value state affect routing?
        v1: informational only (not applied to routing parameters).
        Returns recommended adjustments for future use.
        """
        return {
            "exploration_boost": self.state.curiosity * 0.3,
            "threshold_increase": self.state.fear * 0.2 + self.state.pain * 0.3,
            "depth_increase": self.state.surprise * 0.4,
            "competition_decrease": self.state.curiosity * 0.2,
        }
