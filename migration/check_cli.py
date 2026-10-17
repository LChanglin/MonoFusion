from pathlib import Path
import json
import numpy as np,cv2
rows=[]
for p in sorted(Path('migration/cli_output').glob('*.npz')):
 a=np.load(p);b=np.load(Path('migration/results/iter1_352x1216')/p.name)
 np.testing.assert_allclose(a['flow'],b['flow'],atol=1e-4,rtol=1e-5)
 np.testing.assert_allclose(a['scale'],b['scale'],atol=1e-5,rtol=1e-5)
 packed=cv2.imread(str(p.parent/'submit/flow'/f'{p.stem}_10.png'),cv2.IMREAD_UNCHANGED)[:,:,::-1]
 assert packed.dtype==np.uint16 and (packed[:,:,2]==1).all()
 decoded=(packed[:,:,:2].astype(np.float32)-32768)/64
 err=float(np.abs(decoded-a['flow']).max());assert err<=1/64+1e-4
 rows.append({'scene':p.stem,'flow_encoding_max_error':err,'cli_matches_validation':True})
assert len(rows)==3,rows
Path('migration/cli_check.json').write_text(json.dumps(rows,indent=2));print(rows)
