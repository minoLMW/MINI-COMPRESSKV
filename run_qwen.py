"""Stage 1 actual-model baseline. SRH/adaptive calibration is not integrated.
Pinned Transformers 4.44.2; batch=1, unpadded, eager attention, Qwen2 only.
"""
import argparse
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core import token_scores, select_indices

p=argparse.ArgumentParser()
p.add_argument('--model',default='Qwen/Qwen2.5-0.5B-Instruct')
p.add_argument('--method',choices=['full','recent','attention'],default='full')
p.add_argument('--budget',type=int,default=128)
p.add_argument('--max-new-tokens',type=int,default=24)
p.add_argument('--device',default='cpu',choices=['cpu','cuda','mps'])
p.add_argument('--repeats',type=int,default=50)
a=p.parse_args()
if a.budget<1 or a.max_new_tokens<1 or a.repeats<1:
    p.error('Positive budget, max-new-tokens and repeats required')
tok=AutoTokenizer.from_pretrained(a.model)
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32,
    attn_implementation='eager').to(a.device).eval()
if model.config.model_type!='qwen2':
    raise ValueError('This educational runner supports Qwen2 only')
# Same prompt for all methods. Synthetic task, not LongBench or paper reproduction.
noise='The town library opens every morning and closes in the evening. '
text=noise*a.repeats+' The secret access code is 928371. '+noise*a.repeats
prompt=tok.apply_chat_template([{'role':'user','content':text+'\nWhat is the secret access code? Return only the code.'}],tokenize=False,add_generation_prompt=True)
ids=tok(prompt,return_tensors='pt').input_ids.to(a.device)
n=ids.shape[1]

def sync():
    if a.device=='cuda': torch.cuda.synchronize()
    elif a.device=='mps': torch.mps.synchronize()

def cache_bytes(cache):
    return sum(x.numel()*x.element_size() for pair in cache for x in pair)

with torch.inference_mode():
    sync(); start=time.perf_counter()
    out=model(ids,use_cache=True,output_attentions=a.method=='attention')
    sync(); prefill=time.perf_counter()-start
    cache=tuple(tuple(x for x in pair) for pair in out.past_key_values)
    before=cache_bytes(cache)
    sync(); start=time.perf_counter()
    if a.method!='full':
        budget=min(a.budget,n)
        compressed=[]
        for layer,(k,v) in enumerate(cache):
            if a.method=='recent':
                ix=torch.arange(n-budget,n,device=k.device)
            else:
                # All-head baseline, NOT semantic retrieval head selection.
                att=out.attentions[layer].float().cpu().numpy()
                scores=token_scores(att,range(att.shape[1]))
                ix=torch.as_tensor(select_indices(scores,budget),device=k.device)
            compressed.append((k.index_select(2,ix),v.index_select(2,ix)))
        cache=tuple(compressed)
    sync(); compression=time.perf_counter()-start
    after=cache_bytes(cache)
    # First predicted token is intentionally from the identical full prefill.
    token=out.logits[:,-1,:].argmax(-1,keepdim=True)
    del out
    generated=[]
    sync(); start=time.perf_counter()
    for step in range(a.max_new_tokens):
        generated.append(int(token.item()))
        if int(token.item())==tok.eos_token_id or step==a.max_new_tokens-1: break
        # Preserve original absolute RoPE positions after dropping cache entries.
        pos=torch.tensor([[n+step]],device=ids.device)
        out=model(input_ids=token,past_key_values=cache,position_ids=pos,
                  cache_position=pos[0],use_cache=True)
        cache=out.past_key_values
        token=out.logits[:,-1,:].argmax(-1,keepdim=True)
    sync(); decode=time.perf_counter()-start
answer=tok.decode(generated,skip_special_tokens=True)
print(json.dumps(dict(method=a.method,prompt_tokens=n,answer=answer,
 exact_match=answer.strip()=='928371',kv_bytes_before=before,kv_bytes_after=after,
 prefill_seconds=prefill,compression_seconds=compression,decode_seconds=decode,
 generated_tokens=len(generated),note='One sample; no benchmark claim. Cache grows during decode.'),indent=2))
