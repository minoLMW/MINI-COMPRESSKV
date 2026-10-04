"""No model download: synthetic tensors validate algorithm behavior only."""
import json
from pathlib import Path
import numpy as np
from core import *

rng = np.random.default_rng(42)
n, h, d = 64, 4, 8
# Streaming head peaks at boundaries; retrieval head attends evidence in middle.
a = np.full((1,h,8,n), 0.001)
a[0,0,:,0] = .8
a[0,1,:,28:33] = .18
a /= a.sum(-1, keepdims=True)
cal = np.repeat(a[0,:,0,:][None], 3, axis=0)
scores = srh_scores(cal, [101,102,999], range(28,33), {101,102})
heads = np.argsort(-scores)[:1]
importance = token_scores(a, heads)
indices = select_indices(importance, 16, recent=8)
k = rng.normal(size=(1,2,n,d))
v = rng.normal(size=k.shape)
q = rng.normal(size=(1,h,1,d))
w = rng.normal(size=(h*d,h*d))
kc,vc = compress_kv(k,v,indices)
error = relative_error(gqa_output(q,k,v,w),gqa_output(q,kc,vc,w))
result = dict(kind='synthetic_algorithm_demo_not_LLM_benchmark', selected_heads=heads.tolist(),
 retained_positions=indices.tolist(), kv_bytes_before=k.nbytes+v.nbytes,
 kv_bytes_after=kc.nbytes+vc.nbytes, relative_attention_output_error=error,
 layer_budgets=allocate_budget([.1,.4,.2,.3],64,8,32).tolist())
Path('results').mkdir(exist_ok=True)
Path('results/synthetic.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
