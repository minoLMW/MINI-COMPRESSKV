"""Educational CompressKV primitives. Arrays use [batch, head, query, key]."""
import numpy as np


def srh_scores(attention, generated_ids, answer_positions, answer_token_ids):
    """attention [steps, query_heads, keys]; positions != vocabulary IDs.
    Membership gating is a simplified interpretation of the supplied report.
    Caller must supply attention aligned with each predicted token.
    """
    a = np.asarray(attention)
    if a.ndim != 3 or len(generated_ids) != a.shape[0]:
        raise ValueError('Expected [steps, heads, keys] and aligned generated IDs')
    pos = np.asarray(answer_positions, dtype=int)
    if len(pos) == 0 or np.any(pos < 0) or np.any(pos >= a.shape[-1]):
        raise ValueError('Answer positions must lie inside the source prompt')
    gate = np.isin(generated_ids, list(answer_token_ids))
    return a[:, :, pos].sum(-1)[gate].sum(0)


def token_scores(attention, selected_heads, window=8, kernel=5):
    """Sum observation queries, pool neighboring tokens, average SRHs.
    This prototype explicitly uses average pooling with zero padding.
    Recent observation tokens are protected separately by select_indices.
    """
    a = np.asarray(attention)
    if a.ndim != 4 or a.shape[0] != 1:
        raise ValueError('Only batch=1 [B,H,Q,K] is supported')
    if window < 1 or kernel < 1 or kernel % 2 != 1:
        raise ValueError('Window positive; kernel positive and odd')
    heads = np.asarray(selected_heads, dtype=int)
    if len(heads) == 0 or np.any(heads < 0) or np.any(heads >= a.shape[1]):
        raise ValueError('Invalid selected query heads')
    x = a[0, heads, -window:, :].sum(axis=1)
    pad = kernel // 2
    padded = np.pad(x, ((0, 0), (pad, pad)))
    pooled = sum(padded[:, i:i+x.shape[-1]] for i in range(kernel)) / kernel
    return pooled.mean(0)


def select_indices(scores, budget, recent=8):
    n = len(scores)
    if not 1 <= budget <= n or recent < 0:
        raise ValueError('Require 1 <= budget <= sequence length')
    tail = min(recent, budget, n)
    end = n-tail
    count = budget-tail
    # Stable tie breaking; restore chronological order after top-k.
    top = np.argsort(-np.asarray(scores[:end]), kind='stable')[:count]
    return np.sort(np.concatenate([top, np.arange(end, n)]))


def compress_kv(key, value, indices):
    if key.shape != value.shape or key.ndim != 4:
        raise ValueError('Expected matching [B, KV_heads, sequence, head_dim]')
    # The same positions are shared by every KV head within this layer.
    return key[:, :, indices, :].copy(), value[:, :, indices, :].copy()


def relative_error(full, compressed, epsilon=1e-6):
    return float(np.linalg.norm(full-compressed) / (np.linalg.norm(full)+epsilon))


def allocate_budget(errors, total, minimum, maximum):
    """Bounded proportional allocation with exact integer total.
    Water-fill clipped layers, then distribute integer rounding remainders.
    maximum may be a scalar or per-layer sequence.
    """
    e = np.asarray(errors, dtype=float)
    if e.ndim != 1 or not len(e) or np.any(e < 0) or not np.all(np.isfinite(e)):
        raise ValueError('Errors must be finite nonnegative layer scores')
    caps = np.broadcast_to(np.asarray(maximum, dtype=int), e.shape).copy()
    if minimum < 0 or np.any(caps < minimum) or not len(e)*minimum <= total <= caps.sum():
        raise ValueError('Infeasible total/minimum/maximum')
    out = np.full(len(e), minimum, dtype=int)
    remaining = int(total-out.sum())
    while remaining:
        active = out < caps
        weights = np.where(active, e, 0.)
        if weights.sum() == 0:
            weights = active.astype(float)
        quota = remaining*weights/weights.sum()
        add = np.minimum(np.floor(quota).astype(int), caps-out)
        if add.sum() == 0:
            order = np.argsort(-quota, kind='stable')
            for i in order:
                if active[i] and remaining:
                    out[i] += 1
                    remaining -= 1
            continue
        out += add
        remaining -= int(add.sum())
    return out


def gqa_output(query, key, value, projection):
    """One decode query, no RoPE here: inputs must already be rotated.
    projection [query_heads*head_dim, output_dim].
    """
    hq, hk = query.shape[1], key.shape[1]
    if hq % hk or query.shape[2] != 1:
        raise ValueError('GQA divisibility and one query required')
    k = np.repeat(key, hq//hk, axis=1)
    v = np.repeat(value, hq//hk, axis=1)
    logits = query @ k.swapaxes(-1, -2) / np.sqrt(query.shape[-1])
    logits -= logits.max(-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(-1, keepdims=True)
    heads = probs @ v
    merged = heads.transpose(0, 2, 1, 3).reshape(query.shape[0], 1, -1)
    return merged @ projection
