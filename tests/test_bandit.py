from rl.bandit import EpsilonGreedyBandit

def test_bandit_initialization():
    bandit = EpsilonGreedyBandit(["A", "B"], epsilon=0.1)
    assert bandit.q_values == {"A": 0.0, "B": 0.0}
    assert bandit.counts == {"A": 0, "B": 0}

def test_choose_action_exploit():
    # Epsilon 0 ensures exploit
    bandit = EpsilonGreedyBandit(["A", "B"], epsilon=0.0)
    bandit.q_values = {"A": 0.8, "B": 0.2}
    # Should always choose A
    for _ in range(10):
        assert bandit.choose_action() == "A"

def test_update_q_value():
    bandit = EpsilonGreedyBandit(["A"], epsilon=0.0)
    # First update: Q = 0 + (1 - 0)/1 = 1
    bandit.update("A", 1.0)
    assert bandit.counts["A"] == 1
    assert bandit.q_values["A"] == 1.0
    
    # Second update: Q = 1 + (0 - 1)/2 = 0.5
    bandit.update("A", 0.0)
    assert bandit.counts["A"] == 2
    assert bandit.q_values["A"] == 0.5
