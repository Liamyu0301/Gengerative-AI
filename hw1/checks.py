"""HW1 numerical sanity checks for claims made in the report. Run: python hw1/checks.py"""
import os, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import GPTConfig, CausalSelfAttention, apply_rope

torch.manual_seed(0)

# Q1.3: dL/dX[p,q] = sum_{a,b} G[a,b] W[p-a,q-b]  ==  autograd  ==  full conv of 2-padded G with 180°-rotated W
X = torch.randn(1, 1, 4, 4, dtype=torch.float64, requires_grad=True)
W = torch.randn(1, 1, 3, 3, dtype=torch.float64)
G = torch.randn(1, 1, 2, 2, dtype=torch.float64)
F.conv2d(X, W).backward(G)
formula = torch.zeros(4, 4, dtype=torch.float64)
for p in range(4):
    for q in range(4):
        for a in range(2):
            for b in range(2):
                if 0 <= p - a <= 2 and 0 <= q - b <= 2:
                    formula[p, q] += G[0, 0, a, b] * W[0, 0, p - a, q - b]
full_conv = F.conv2d(F.pad(G, (2, 2, 2, 2)), W.flip(-1, -2))[0, 0]
print('Q1.3 formula == autograd:', torch.allclose(formula, X.grad[0, 0]))
print('Q1.3 full conv == autograd:', torch.allclose(full_conv, X.grad[0, 0]))

cfg = dict(block_size=256, vocab_size=65, n_layer=1, n_head=6, n_embd=384, dropout=0.0, bias=False)

# Q4.4: after RoPE, q_i . k_j depends only on i - j
attn = CausalSelfAttention(GPTConfig(**cfg, pos_enc='rope'))
q, k = torch.randn(1, 1, 1, 64), torch.randn(1, 1, 1, 64)
score = lambda i, j: (apply_rope(q, attn.rope_cos[i], attn.rope_sin[i]) * apply_rope(k, attn.rope_cos[j], attn.rope_sin[j])).sum()
s = torch.stack([score(10, 3), score(107, 100), score(200, 193)])
print('Q4.4 RoPE score depends only on i-j:', torch.allclose(s, s[0].expand(3), atol=1e-4))

# Q4.5: GQA (3 kv heads) == MHA whose K/V weights duplicate each kv head for its 2 query heads
d, hs = 384, 64
gqa = CausalSelfAttention(GPTConfig(**cfg, n_kv_head=3))
mha = CausalSelfAttention(GPTConfig(**cfg))
Wq, Wk, Wv = gqa.c_attn.weight.split([d, 3 * hs, 3 * hs], 0)
dup = lambda w: w.view(3, hs, d).repeat_interleave(2, 0).reshape(6 * hs, d)
with torch.no_grad():
    mha.c_attn.weight.copy_(torch.cat([Wq, dup(Wk), dup(Wv)]))
    mha.c_proj.weight.copy_(gqa.c_proj.weight)
x = torch.randn(2, 16, d)
print('Q4.5 GQA == MHA with duplicated K/V:', torch.allclose(gqa(x), mha(x), atol=1e-6))
