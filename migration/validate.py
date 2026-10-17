"""GPU inference and comparison against released colorized predictions (not GT)."""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,cv2
from PIL import Image,ImageDraw
from monofusion.model import MonoFusion
from utils.draw import flow_to_image,disp2rgb
p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='pretrained/monofusion_kitti.pth.tar');p.add_argument('--iterations',type=int,default=1);p.add_argument('--device',default='cuda');p.add_argument('--size',type=int,nargs=2,default=[352,1216]);p.add_argument('--limit',type=int,default=5);a=p.parse_args()
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
c=torch.load(a.checkpoint,map_location='cpu',weights_only=False);s=c.get('net',c.get('state_dict',c));s={k.removeprefix('module.'):v for k,v in s.items()}
m=MonoFusion(num_scales=2,upsample_factor=4,num_iterations=a.iterations,checkpoint_arch='legacy_scale').to(a.device).eval()
print(m.load_state_dict(s,strict=True),flush=True)
out=Path('migration/results')/f'iter{a.iterations}_{a.size[0]}x{a.size[1]}';out.mkdir(parents=True,exist_ok=True);rows=[]
for scene in sorted(Path('migration/samples').iterdir())[:a.limit]:
 imgs=[np.array(Image.open(scene/'image'/f'{scene.name}_{n}.jpg').convert('RGB')) for n in ['10','11']]
 ts=[torch.from_numpy(x.copy()).permute(2,0,1)[None].float().to(a.device) for x in imgs]
 original=ts[0].shape[-2:];ts=[torch.nn.functional.interpolate(x,size=a.size,mode='bilinear',align_corners=True) for x in ts]
 start=time.time()
 with torch.inference_mode():
  scales,_,flows,_=m(*ts,attn_type='swin',attn_splits_list=[2,8],corr_radius_list=[-1,4],prop_radius_list=[-1,1],num_reg_refine=1)
  scale=scales[-1];flow=flows[-1]
  scale=torch.nn.functional.interpolate(scale,size=original,mode='bilinear',align_corners=True)
  flow=torch.nn.functional.interpolate(flow,size=original,mode='bilinear',align_corners=True);flow[:,0]*=original[1]/a.size[1];flow[:,1]*=original[0]/a.size[0]
  assert torch.isfinite(scale).all() and torch.isfinite(flow).all()
  flow=flow[0].permute(1,2,0).cpu().numpy();scale=scale[0,0].cpu().numpy()
 predflow=flow_to_image(flow);predscale=(disp2rgb(np.clip((scale[...,None]-.5),0,1))*255).astype('uint8')[:,:,::-1]
 Image.fromarray(predflow).save(out/f'{scene.name}_flow.png');Image.fromarray(predscale).save(out/f'{scene.name}_scale.png')
 np.savez_compressed(out/f'{scene.name}.npz',flow=flow,scale=scale)
 row={'scene':scene.name,'seconds':time.time()-start,'flow_abs_mean':float(np.abs(flow).mean()),'scale_quantiles':np.quantile(scale,[0,.01,.5,.99,1]).tolist(),'reference_comparison':{}}
 panels=[imgs[0],predflow,predscale]
 for index in ['09','10']:
  refs=[np.array(Image.open(scene/'flow'/f'flow_{scene.name}_{index}.jpg').convert('RGB')),np.array(Image.open(scene/'ttc'/f'scale_{scene.name}_{index}.png').convert('RGB'))]
  refs=[cv2.resize(x,(original[1],original[0])) for x in refs]
  row['reference_comparison'][index]={'flow_rgb_mae':float(np.abs(predflow.astype(float)-refs[0]).mean()),'scale_rgb_mae':float(np.abs(predscale.astype(float)-refs[1]).mean())}
  if index=='10':panels.extend(refs)
 canvas=Image.new('RGB',(original[1],(original[0]+25)*len(panels)),'white');draw=ImageDraw.Draw(canvas)
 for i,(label,im) in enumerate(zip(['Input frame 10','Migrated flow','Migrated expansion','Reference flow index 10','Reference expansion index 10'],panels)):
  canvas.paste(Image.fromarray(im),(0,i*(original[0]+25)+25));draw.text((5,i*(original[0]+25)+4),label,fill='black')
 canvas.save(out/f'{scene.name}_comparison.jpg'); rows.append(row);print(json.dumps(row),flush=True)
(out/'metrics.json').write_text(json.dumps(rows,indent=2))
