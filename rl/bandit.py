import json
import random
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent.parent))

from config import BANDIT_STATE_PATH, DEFAULT_EPSILON


class EpsilonGreedyBandit:
    def __init__(
        self,
        actions: List[str],
        epsilon: float = DEFAULT_EPSILON,
        epsilon_decay: float = 0.99,
        min_epsilon: float = 0.01
    ):
        self.actions = actions

        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.q_values = {a: 0.0 for a in actions}
        self.counts = {a: 0 for a in actions}
        self.total_updates = 0

    def choose_action(self) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        max_q = max(self.q_values.values())

        best_actions = [
            action
            for action in self.actions
            if self.q_values[action] == max_q
        ]

        return random.choice(best_actions)

    def update(self, action: str, reward: float):
        self.counts[action] += 1

        n = self.counts[action]
        old_q = self.q_values[action]

        self.q_values[action] = (
            old_q + (reward - old_q) / n
        )

        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay
        )
        self.total_updates += 1

    def reset(self):
        """Reset the bandit to a completely fresh experimental state."""
        self.epsilon = self.initial_epsilon
        self.q_values = {a: 0.0 for a in self.actions}
        self.counts = {a: 0 for a in self.actions}
        self.total_updates = 0

    def save(self, path=BANDIT_STATE_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "q_values": self.q_values,
            "counts": self.counts
            ,"total_updates": self.total_updates
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load(self, path=BANDIT_STATE_PATH):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Bandit state not found: {path}"
            )

        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.epsilon = state["epsilon"]
        self.epsilon_decay = state.get(
            "epsilon_decay",
            self.epsilon_decay
        )
        self.min_epsilon = state.get(
            "min_epsilon",
            self.min_epsilon
        )
        self.q_values = state["q_values"]
        self.counts = state["counts"]
        self.total_updates = state.get("total_updates", sum(self.counts.values()))

    def validate_epsilon(self):
        if not (self.min_epsilon <= self.epsilon <= self.initial_epsilon):
            raise ValueError(f"epsilon out of bounds: {self.epsilon}")

    def __repr__(self):
        return (
            f"Bandit("
            f"eps={self.epsilon}, "
            f"Q={self.q_values}, "
            f"N={self.counts})"
        )
