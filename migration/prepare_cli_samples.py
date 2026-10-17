from pathlib import Path
from PIL import Image
out=Path('migration/cli_inputs/image_2');out.mkdir(parents=True,exist_ok=True)
for scene in sorted(Path('migration/samples').iterdir())[:3]:
 for n in ['10','11']:
  Image.open(scene/'image'/f'{scene.name}_{n}.jpg').save(out/f'{scene.name}_{n}.png')
