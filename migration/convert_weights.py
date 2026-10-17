"""Lossless state-dict conversion for the supplied 387-tensor release checkpoint.
Requires the legacy_scale compatibility architecture added to this checkout.
Does NOT manufacture missing depth weights or claim distinct training variants.
"""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from monofusion.model import MonoFusion

def convert(source,destination):
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    state=checkpoint.get('net',checkpoint.get('state_dict',checkpoint))
    converted={k.removeprefix('module.'):v for k,v in state.items()}
    if len(converted)!=len(state): raise ValueError('Key collision after prefix removal')
    model=MonoFusion(upsample_factor=4,num_iterations=1,checkpoint_arch='legacy_scale')
    model.load_state_dict(converted,strict=True)
    metadata={'source_name':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'upstream_commit':'10473b3be485c111d13d2f3354dde5231233ee51',
              'checkpoint_arch':'legacy_scale','tensor_count':len(converted),
              'parameter_count':sum(t.numel() for t in converted.values()),
              'num_scales':2,'upsample_factor':4,
              'warning':'Input filenames do not establish training provenance. No trained depth head is present. Scale broadcast semantics inferred; validate against reference outputs.'}
    destination.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'net':converted,'migration':metadata},destination)
    reloaded=torch.load(destination,map_location='cpu',weights_only=True)['net']
    model.load_state_dict(reloaded,strict=True)
    assert all(torch.equal(reloaded[k],v) for k,v in converted.items())
    metadata['output_sha256']=hashlib.sha256(destination.read_bytes()).hexdigest()
    metadata['strict_load']=True;metadata['all_tensors_bitwise_equal']=True
    destination.with_suffix(destination.suffix+'.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('sources',type=Path,nargs='+');p.add_argument('--output-dir',type=Path,default=Path('pretrained/converted'));a=p.parse_args()
    for source in a.sources:convert(source,a.output_dir/source.name)
