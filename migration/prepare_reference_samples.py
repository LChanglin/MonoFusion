"""Extract every fifth scene, frames 10/11 and reference predictions 09/10."""
import argparse,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--output',type=Path,default=Path('migration/samples'));a=p.parse_args()
with zipfile.ZipFile(a.archive) as z:
 for i in range(0,200,5):
  scene=f'{i:06d}'
  groups={'image':[f'{scene}_{n}.jpg' for n in ['10','11']],
          'flow':[f'flow_{scene}_{n}.jpg' for n in ['09','10']],
          'ttc':[f'scale_{scene}_{n}.png' for n in ['09','10']]}
  for group,names in groups.items():
   for name in names:
    target=a.output/scene/group/name;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(z.read(f'monofusion_c/{scene}/{group}/{name}'))
print('Prepared 40 fixed scenes; references are historical predictions, not ground truth.')
