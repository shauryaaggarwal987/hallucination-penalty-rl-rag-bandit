import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.final_nonlatex_workflow import (
    TEST_IDS, TRAIN_IDS, VALIDATION_IDS, make_response_id, reconstruct_context,
    validate_split_integrity,
)
from experiments.plots import generate_plots
from experiments.run_eval import run_training_loop, validate_final_inputs
from rl.bandit import EpsilonGreedyBandit


class MockClient:
    provider = 'gemini'
    def __init__(self, fail_at=None):
        self.calls=[]; self.fail_at=fail_at
    def generate(self, action, question, hits, **kwargs):
        self.calls.append((action,question))
        if self.fail_at is not None and len(self.calls)==self.fail_at:
            raise RuntimeError('synthetic API failure')
        return 'Supported answer [source: s#0].'


def retriever(question, **kwargs):
    return ['Supported answer evidence.'], ['s#0'], [{'source_id':'s#0','text':'Supported answer evidence.','score':1.0}]


def scorer(answer, context):
    return {'scorer_version':'semantic_proxy_clean_v2','reward':1.0,'groundedness':1.0,
            'hallucination_penalty':0.0,'supported':1,'unsupported':0,'is_abstention':False}


class FinalWorkflowAuditTests(unittest.TestCase):
    def test_single_generation_per_episode_and_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            client=MockClient()
            run=run_training_loop(3,42,Path(td),client=client,retriever=retriever,
                                  scorer=scorer,validate_inputs=False)
            self.assertEqual(len(client.calls),3)
            episodes=pd.read_csv(run/'episodes.csv')
            self.assertEqual(len(episodes),3)
            self.assertEqual(episodes.bandit_updates.tolist(),[1,2,3])
            self.assertTrue((episodes.scorer_version=='semantic_proxy_clean_v2').all())
            meta=json.loads((run/'run_metadata.json').read_text())
            self.assertEqual(meta['semantic_scorer_version'],'semantic_proxy_clean_v2')

    def test_api_failure_causes_no_update_for_failed_episode(self):
        holder={}
        def factory(*args):
            holder['bandit']=EpsilonGreedyBandit(*args); return holder['bandit']
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):
                run_training_loop(3,42,Path(td),client=MockClient(fail_at=2),retriever=retriever,
                                  scorer=scorer,bandit_factory=factory,validate_inputs=False)
            self.assertEqual(holder['bandit'].total_updates,1)
            run=next(Path(td).iterdir())
            self.assertTrue((run/'RUN_FAILED.json').exists())
            self.assertFalse((run/'RUN_COMPLETE.json').exists())

    def test_fresh_run_isolation(self):
        with tempfile.TemporaryDirectory() as td:
            a=run_training_loop(1,11,Path(td),client=MockClient(),retriever=retriever,scorer=scorer,validate_inputs=False)
            b=run_training_loop(1,11,Path(td),client=MockClient(),retriever=retriever,scorer=scorer,validate_inputs=False)
            self.assertNotEqual(a,b)
            self.assertEqual(json.loads((a/'bandit_state.json').read_text())['total_updates'],1)
            self.assertEqual(json.loads((b/'bandit_state.json').read_text())['total_updates'],1)

    def test_split_integrity(self):
        rows=validate_split_integrity()
        self.assertEqual((len(TRAIN_IDS),len(VALIDATION_IDS),len(TEST_IDS)),(34,7,9))
        self.assertEqual(len(rows),50)

    def test_nli_response_id_uses_repeat_value(self):
        row=pd.Series({'dataset':'test','policy':'Fixed-A2','question_id':'q014','repeat':3})
        response_id=make_response_id(row)
        self.assertEqual(response_id,'test:Fixed-A2:q014:r3')
        self.assertNotIn('bound method',response_id)

    def test_frozen_hash_guard_and_context_reconstruction(self):
        # Public release omits text-bearing third-party index artifacts.  The
        # frozen guard must reject their absence until users reconstruct them.
        with self.assertRaisesRegex(RuntimeError,'Final input integrity failure'):
            validate_final_inputs()
        chunks,h=reconstruct_context(['x#0','y#1'],{'x#0':{'text':'first'},'y#1':{'text':'second'}})
        self.assertEqual(chunks,['first','second'])
        self.assertEqual(len(h),64)
        with self.assertRaises(KeyError): reconstruct_context(['missing'],{})

    def test_plotting_rejects_duplicate_episodes_and_multiple_runs(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'episodes.csv'
            pd.DataFrame({'run_id':['a','a'],'episode':[0,0],'chosen_reward':[0,1],
                          'chosen_action':['A0','A1']}).to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'Duplicate episode'): generate_plots(p)
            pd.DataFrame({'run_id':['a','b'],'episode':[0,1],'chosen_reward':[0,1],
                          'chosen_action':['A0','A1']}).to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'multiple run_ids'): generate_plots(p)


if __name__=='__main__': unittest.main()
