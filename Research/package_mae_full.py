"""Package a bounded code snapshot only, never data/checkpoints/credentials."""
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parents[1];model=root/'Model';destination=root/'Model/mae_full_code.tar'
paths=[*model.glob('*.py'),*model.glob('*.html'),*model.glob('requirements*.txt'),
       *(model/'tone_model').glob('*.py'),*(model/'tests').glob('*.py'),*(model/'configs').glob('*.json')]
with tarfile.open(destination,'w') as archive:
    for path in sorted(set(paths)):archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
print(destination,destination.stat().st_size,len(paths),'files')
