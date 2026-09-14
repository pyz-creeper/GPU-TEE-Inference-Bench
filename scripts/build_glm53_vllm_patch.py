"""Build the audited patch archive from a pristine installed vLLM 0.28.0 wheel."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

patches = Path(__file__).resolve().parent/'patches'
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=False)
manifest = json.loads((patches/'vllm-glm53-mtp-manifest.json').read_text())
dist = importlib.metadata.distribution('vllm')
assert dist.version == '0.28.0'
with tempfile.TemporaryDirectory(prefix='glm53-vllm-patch-') as temp:
    root = Path(temp)
    for name, hashes in manifest['files'].items():
        src = patches/name if hashes['before'] is None else Path(dist.locate_file(name))
        expected = hashes['after'] if hashes['before'] is None else hashes['before']
        assert hashlib.sha256(src.read_bytes()).hexdigest() == expected, ('not pristine',name)
        dest = root/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dest)
    for name in ['vllm-0.28.0-pp-mtp.patch','vllm-0.28.0-glm53-uva.patch']:
        subprocess.run(['patch','--batch','-p1','-i',str(patches/name)],cwd=root,check=True)
    with tarfile.open(out/'patch-files.tar.gz','w:gz') as tar:
        for name, hashes in manifest['files'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == hashes['after'],name
            tar.add(root/name,arcname=name)
    (out/'patch-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(out)
