"""
Compute rank-1, rank-5, mAP on raw DINOv2 embeddings — zero training.
Uses same filter as train_ship_reid.py: ≥MIN_PASSES passes, ≥MIN_LENGTH m.
Gallery = first chip per ship. Query = remaining chips.
"""

import numpy as np
from collections import defaultdict
import torch

EMB_PATH  = '/Volumes/Machine Learning/sar_ship_chips/embeddings.npz'
MIN_PASSES  = 4
MIN_LENGTH  = 100

f = np.load(EMB_PATH, allow_pickle=True)
embs    = f['embeddings']   # (N, 384)
mmsis   = f['mmsi']
lengths = f['length_m']

# Filter: length ≥ MIN_LENGTH
mask = lengths >= MIN_LENGTH
embs  = embs[mask]
mmsis = mmsis[mask]

# Filter: ≥ MIN_PASSES chips per ship
by_mmsi = defaultdict(list)
for i, m in enumerate(mmsis):
    by_mmsi[m].append(i)
qualified = {m: idxs for m, idxs in by_mmsi.items() if len(idxs) >= MIN_PASSES}

print(f"Qualified ships (≥{MIN_PASSES} passes, ≥{MIN_LENGTH}m): {len(qualified)}")
total_chips = sum(len(v) for v in qualified.items())
all_idxs = [i for idxs in qualified.values() for i in idxs]
print(f"Total chips: {len(all_idxs)}")

# Gallery = first chip per ship, query = rest
gallery_idx = [idxs[0] for idxs in qualified.values()]
query_idx   = [i for idxs in qualified.values() for i in idxs[1:]]
gallery_ids = [mmsis[i] for i in gallery_idx]
query_ids   = [mmsis[i] for i in query_idx]

print(f"Gallery: {len(gallery_idx)} ships   Query: {len(query_idx)} chips")

g_emb = torch.tensor(embs[gallery_idx])
q_emb = torch.tensor(embs[query_idx])

# Pairwise L2 distance
dist   = torch.cdist(q_emb, g_emb)
ranked = dist.argsort(dim=1)

rank1 = rank5 = 0
aps = []
for qi, q_id in enumerate(query_ids):
    correct = [gallery_ids[j] == q_id for j in ranked[qi].tolist()]
    if correct[0]:  rank1 += 1
    if any(correct[:5]): rank5 += 1
    hits = [i + 1 for i, c in enumerate(correct) if c]
    if hits:
        ap = sum((k + 1) / h for k, h in enumerate(hits)) / len(hits)
        aps.append(ap)

n = len(query_ids)
print(f"\n=== Raw DINOv2 Baseline (zero training) ===")
print(f"  Rank-1:  {rank1/n*100:.1f}%")
print(f"  Rank-5:  {rank5/n*100:.1f}%")
print(f"  mAP:     {sum(aps)/len(aps)*100:.1f}%")
