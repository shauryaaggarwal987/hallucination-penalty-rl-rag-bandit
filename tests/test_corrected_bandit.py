import unittest
from rl.bandit import EpsilonGreedyBandit


class BanditTests(unittest.TestCase):
    def test_only_selected_arm_updates_and_epsilon_floor(self):
        b=EpsilonGreedyBandit(['A0','A1'],epsilon=.2,epsilon_decay=.5,min_epsilon=.01)
        b.update('A1',1.0)
        self.assertEqual(b.counts,{'A0':0,'A1':1}); self.assertEqual(b.total_updates,1)
        for _ in range(20): b.update('A1',0.0)
        self.assertEqual(b.epsilon,.01)
        b.validate_epsilon()

    def test_fresh_state(self):
        b=EpsilonGreedyBandit(['A0']); self.assertEqual(b.total_updates,0); self.assertEqual(b.q_values['A0'],0.0)


if __name__=='__main__': unittest.main()
