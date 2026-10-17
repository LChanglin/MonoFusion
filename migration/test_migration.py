"""Focused checks: operator sampling/gradient, strict schema, lossless conversion."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from monofusion.network.utils.torch_ops import deformable_attention
from monofusion.model import MonoFusion
# Two levels, two heads; pixel centers select exact values. Outside = zero.
v=torch.arange(1.,11.).reshape(1,5,2,1).requires_grad_()
shapes=torch.tensor([[2,2],[1,1]])
loc=torch.tensor([[[[[.25,.25]],[[.5,.5]]],[[[.25,.25]],[[.5,.5]]]]]).reshape(1,1,2,2,1,2)
w=torch.full((1,1,2,2,1),.5)
y=deformable_attention(v,shapes,loc,w)
torch.testing.assert_close(y,torch.tensor([[[5.,6.]]]))
y.sum().backward();assert torch.isfinite(v.grad).all()
zero=deformable_attention(v.detach(),shapes,torch.full_like(loc,3.),w)
assert torch.count_nonzero(zero)==0
source=Path(sys.argv[1]);dest=Path(sys.argv[2])
old=torch.load(source,map_location='cpu',weights_only=False)['net']
new=torch.load(dest,map_location='cpu',weights_only=True)['net']
assert len(new)==387
assert all(torch.equal(v,new[k.removeprefix('module.')]) for k,v in old.items())
m=MonoFusion(upsample_factor=4,checkpoint_arch='legacy_scale');m.load_state_dict(new,strict=True)
bad=dict(new);bad.pop(next(iter(bad)))
try:m.load_state_dict(bad,strict=True)
except RuntimeError:pass
else:raise AssertionError('Strict loading did not reject missing parameter')
print('PASS: multilevel/multihead sampling, zero padding, backward, 387 tensors bitwise equal, strict load and missing-key rejection')
