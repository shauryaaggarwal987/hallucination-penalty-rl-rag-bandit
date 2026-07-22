import unittest
import numpy as np

from scoring.reward import (
    citation_metrics, extract_scoring_units, is_pure_formatting_unit,
    is_strict_abstention, semantic_support_score_v2,
)


class FakeEncoder:
    def encode(self, texts, normalize_embeddings=True):
        # Deterministic lexical signal sufficient to prove cleaned text reaches encoding.
        self.last_texts = list(texts)
        rows=[]
        for t in texts:
            v=np.array([float('faiss' in t.lower()), float('reward' in t.lower()), 1.0])
            rows.append(v/np.linalg.norm(v))
        return np.asarray(rows)


class CorrectedScoringTests(unittest.TestCase):
    def test_strict_abstention(self):
        self.assertTrue(is_strict_abstention('Not in context!'))
        self.assertFalse(is_strict_abstention('This information is not in context but I think...'))

    def test_formatting_detection(self):
        for text in ['', '1', '2.', '3)', '-', '•', '[source: x#0]', '...']:
            self.assertTrue(is_pure_formatting_unit(text), text)
        self.assertFalse(is_pure_formatting_unit('FAISS is a library.'))

    def test_numbering_and_bullets(self):
        self.assertEqual(extract_scoring_units('1.'), [])
        self.assertEqual(extract_scoring_units('-'), [])
        self.assertEqual(extract_scoring_units('1. Reinforcement learning uses reward feedback.')[0].cleaned_unit,
                         'Reinforcement learning uses reward feedback.')
        self.assertEqual(extract_scoring_units('2) Supervised learning uses labelled data.')[0].cleaned_unit,
                         'Supervised learning uses labelled data.')

    def test_citation_stripped_but_raw_retained(self):
        unit=extract_scoring_units('- FAISS supports vector similarity search [source: faiss.txt#0].')[0]
        self.assertNotIn('[source:',unit.cleaned_unit)
        self.assertIn('[source:',unit.raw_unit)
        self.assertEqual(unit.citations,('faiss.txt#0',))

    def test_citation_coverage_ignores_numbering(self):
        m=citation_metrics('1.\n- FAISS is a library [source: faiss.txt#0].',['faiss.txt#0'])
        self.assertEqual(m['substantive_unit_count'],1)
        self.assertEqual(m['citation_sentence_coverage'],1.0)

    def test_multi_source_tag_is_parsed_as_individual_citations(self):
        unit=extract_scoring_units('FAISS is used for vector search [source: faiss.txt#0, rag.txt#1].')[0]
        self.assertEqual(unit.citations,('faiss.txt#0','rag.txt#1'))
        metrics=citation_metrics('FAISS is used for vector search [source: faiss.txt#0, rag.txt#1].',
                                 ['faiss.txt#0','rag.txt#1'])
        self.assertEqual(metrics['citation_count'],2)
        self.assertEqual(metrics['citation_precision'],1.0)

    def test_reward_bounds_and_empty(self):
        model=FakeEncoder()
        self.assertEqual(semantic_support_score_v2('', 'context', model=model)['reward'],-1.0)
        self.assertEqual(semantic_support_score_v2('Not in context.', 'context', model=model)['reward'],0.0)
        r=semantic_support_score_v2('FAISS is a library.', 'FAISS is a library.', model=model)
        self.assertEqual(r['reward'],1.0)
        self.assertEqual(r['scorer_version'],'semantic_proxy_clean_v2')


if __name__=='__main__': unittest.main()
