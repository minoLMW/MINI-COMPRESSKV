import unittest
import numpy as np
from core import *

class CoreTests(unittest.TestCase):
    def test_span_and_token_ids_are_distinct(self):
        a=np.zeros((2,2,10)); a[:,1,3:5]=.25
        np.testing.assert_allclose(srh_scores(a,[100,999],[3,4],{100}),[0,.5])
    def test_recent_and_sorted_selection(self):
        s=np.zeros(20); s[5]=100
        ix=select_indices(s,6,4)
        self.assertIn(5,ix); np.testing.assert_array_equal(ix[-4:],range(16,20))
        self.assertEqual(len(set(ix)),6)
    def test_budget_constraints(self):
        for e in ([0,0,0,0],[100,0,1,1],[1,2,3,4]):
            for total in range(16,81):
                b=allocate_budget(e,total,4,20)
                self.assertEqual(b.sum(),total)
                self.assertTrue(np.all((b>=4)&(b<=20)))
        with self.assertRaises(ValueError): allocate_budget([1,2],3,2,8)
    def test_full_cache_identity_and_gqa(self):
        r=np.random.default_rng(9)
        k=r.normal(size=(1,2,10,4)); v=r.normal(size=k.shape)
        q=r.normal(size=(1,4,1,4)); w=np.eye(16)
        kc,vc=compress_kv(k,v,np.arange(10))
        np.testing.assert_allclose(gqa_output(q,k,v,w),gqa_output(q,kc,vc,w))
        self.assertEqual(relative_error(gqa_output(q,k,v,w),gqa_output(q,kc,vc,w)),0)
    def test_short_sequence_pooling(self):
        self.assertEqual(token_scores(np.ones((1,2,1,2)),[0],kernel=5).shape,(2,))

if __name__=='__main__': unittest.main()
