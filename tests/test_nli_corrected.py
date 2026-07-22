import unittest
import numpy as np
from types import SimpleNamespace
from scoring.nli import nli_support_score, score_premise_hypothesis


class Tokenizer:
    def __call__(self, premise, hypothesis, **kwargs):
        self.calls=getattr(self,'calls',[])+[(premise,hypothesis)]
        n=len(premise) if isinstance(premise,list) else 1
        return {'input_ids':np.array([[1,2]]*n)}


class Tensor:
    def __init__(self,a): self.a=np.asarray(a)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.a


class Model:
    config=SimpleNamespace(id2label={0:'contradiction',1:'neutral',2:'entailment'},label2id={})
    def __call__(self,**kwargs):
        n=len(kwargs['input_ids'])
        return SimpleNamespace(logits=Tensor([[0.,0.,4.]]*n))


class NLITests(unittest.TestCase):
    def test_premise_and_hypothesis_are_pair_inputs(self):
        tok=Tokenizer(); score_premise_hypothesis('retrieved evidence','generated claim',tokenizer=tok,model=Model())
        self.assertEqual(tok.calls,[('retrieved evidence','generated claim')])

    def test_chunkwise_best_source_and_abstention(self):
        tok=Tokenizer(); out=nli_support_score('FAISS is a library.',['evidence one','evidence two'],['s1','s2'],tokenizer=tok,model=Model())
        self.assertEqual(tok.calls,[(['evidence one','evidence two'],['FAISS is a library.','FAISS is a library.'])])
        self.assertTrue(out['per_sentence'][0]['nli_supported'])
        self.assertTrue(nli_support_score('Not in context!',[],tokenizer=tok,model=Model())['is_abstention'])


if __name__=='__main__': unittest.main()
