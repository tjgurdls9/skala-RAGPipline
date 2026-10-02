"""HF molecule-LM embeddings (mean-pooled last hidden state) from SMILES only. usage: embed.py <hf_id> <tag>"""
import sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
hf, tag = sys.argv[1], sys.argv[2]
D = np.load('features.npz', allow_pickle=True)
smi = list(D['smiles_tr']) + list(D['smiles_pr'])
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
mdl = AutoModel.from_pretrained(hf, trust_remote_code=True).eval().to('mps')
out = []
with torch.no_grad():
    for i in range(0, len(smi), 64):
        b = tok(smi[i:i+64], padding=True, truncation=True, max_length=256, return_tensors='pt').to('mps')
        h = mdl(**b).last_hidden_state
        m = b['attention_mask'].unsqueeze(-1).to(h.dtype)
        out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
E = np.vstack(out); n = len(D['smiles_tr'])
np.savez(f'emb_{tag}.npz', tr=E[:n], pr=E[n:]); print(tag, E.shape, np.isnan(E).sum())
