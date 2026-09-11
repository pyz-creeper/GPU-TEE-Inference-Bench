"""Install audited Python-only PP/MTP changes into the dedicated vLLM environment."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tarfile

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / 'patch-manifest.json').read_text())
dist = importlib.metadata.distribution('vllm')
assert dist.version == manifest['base_version'], dist.version
site = Path(dist.locate_file('')).resolve()
assert '/data/envs/glm53-vllm-pp2-mtp/' in str(site), site
backup = root / 'original-files'
with tarfile.open(root / 'patch-files.tar.gz') as archive:
    updates = []
    for name, hashes in manifest['files'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        data = archive.extractfile(name).read()
        assert hashlib.sha256(data).hexdigest() == hashes['after'], name
        if name.startswith('tests/'):
            dest = root / name
        else:
            dest = site / name
            current = dest.read_bytes() if dest.exists() else None
            digest = hashlib.sha256(current).hexdigest() if current is not None else None
            assert digest in (hashes['before'], hashes['after']), (name, digest)
            if current is not None and digest == hashes['before']:
                saved = backup / name
                saved.parent.mkdir(parents=True, exist_ok=True)
                if not saved.exists():
                    saved.write_bytes(current)
        updates.append((dest, data))
    for dest, data in updates:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
print(json.dumps({'installed': len(updates), 'site': str(site), 'manifest': manifest}, indent=2))
