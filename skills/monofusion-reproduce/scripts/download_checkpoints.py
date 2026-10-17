"""Download only manifest-pinned MonoFusion weights; preserve existing mismatches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def download(entry, destination):
    target = destination / entry['name']
    if target.exists():
        if target.stat().st_size != entry['bytes'] or digest(target) != entry['sha256']:
            raise ValueError(f'Existing file differs from verified release: {target}. Preserve it and choose a different --output-dir.')
        print(f'Verified existing file: {target}', flush=True)
        return
    temporary = None
    try:
        url = 'https://drive.google.com/uc?export=download&id=' + entry['file_id']
        request = urllib.request.Request(url, headers={'User-Agent': 'MonoFusion-checkpoint-verifier/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response, tempfile.NamedTemporaryFile(dir=destination, prefix='.checkpoint-', delete=False) as stream:
            temporary = Path(stream.name)
            count = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > entry['bytes']:
                    raise ValueError('Download is larger than the pinned checkpoint')
                stream.write(block)
        if temporary.stat().st_size != entry['bytes'] or digest(temporary) != entry['sha256']:
            raise ValueError('Downloaded content failed verification; it may be a Drive error page or a changed release')
        # Exclusive creation prevents replacing a file created by another process.
        os.link(temporary, target)
        print(f'Downloaded and verified: {target}', flush=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--files', nargs='+', default=['monofusion_kitti.pth.tar', 'monofusion_r_kitti.pth.tar'])
    args = parser.parse_args()
    manifest = json.loads((args.repo / 'pretrained/checkpoints.json').read_text())
    entries = {x['name']: x for x in manifest['files']}
    unknown = set(args.files) - entries.keys()
    if unknown:
        parser.error(f'Unknown checkpoint names: {sorted(unknown)}')
    destination = args.output_dir or args.repo / 'pretrained'
    destination.mkdir(parents=True, exist_ok=True)
    for name in args.files:
        if Path(name).name != name:
            parser.error('Checkpoint name must be a filename')
        download(entries[name], destination)


if __name__ == '__main__':
    main()
