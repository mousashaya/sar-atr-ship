import torch

state = torch.load(
    'models/saved/exp_108_rot_baseline_full_v1/weights.pt',
    map_location='cpu', weights_only=True)

print(f"Total layers: {len(state)}")
print(f"\nFirst 10 layers:")
for i, (name, tensor) in enumerate(state.items()):
    if i >= 10: break
    print(f"  {name:<55} shape={list(tensor.shape)}  params={tensor.numel():,}")

total = sum(t.numel() for t in state.values())
print(f"\nTotal parameters: {total:,}")
