import numpy as np
from scoring.reward import score_answer


class FakeModel:
    def encode(self,texts,normalize_embeddings=True):
        return np.asarray([[1.0,0.0] if 'cat' in x.lower() else [0.0,1.0] for x in texts])


def test_score_answer_logic():
    score=score_answer('The cat is on the mat.','The cat sat on the mat.',model=FakeModel())
    assert score['supported']==1 and score['reward']==1.0


def test_score_hallucination():
    score=score_answer('The grass is green.','The cat sat on the mat.',model=FakeModel())
    assert score['unsupported']==1 and score['reward']==-1.0
