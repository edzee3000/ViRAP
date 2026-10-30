"""Validate and publish the complete unreviewed ViRAP corpus from Trinity."""
import collections
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path('/tmp/shihangz/photo-tone-transfer/Dataset')
STAGE = Path('/tmp/shihangz/virap_hub_v1')
REPO = 'edzee3000/ViRAP'

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    api = HfApi()
    assert api.whoami()['name'] == 'edzee3000'
    source = ROOT/'Photo_Triplets_v1/triplets.json'
    records = json.loads(source.read_text())['triplets']
    assert len(records) == 60000
    assert len({r['pair_id'] for r in records}) == 60000
    files = {}
    output_rows = []
    for row in records:
        row = dict(row)
        for key in ('content_path', 'reference_path', 'target_path', 'planned_target_path'):
            original = (source.parent/row[key]).resolve()
            relative = original.relative_to(ROOT).as_posix()
            assert original.is_file(), original
            files[relative] = original
            row[key] = relative
        row['human_preference_status'] = 'unreviewed'
        output_rows.append(row)
    inventory = []
    for index, (relative, original) in enumerate(sorted(files.items())):
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        inventory.append(dict(path=relative, bytes=original.stat().st_size, sha256=digest))
        dest = STAGE/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            os.link(original, dest)
        if index % 2000 == 0:
            print(f'validated {index}/{len(files)}', flush=True)
    expected = {r['target_path']:r['output_sha256'] for r in output_rows}
    assert all(r['sha256'] == expected[r['path']] for r in inventory if r['path'] in expected)
    write(STAGE/'triplets.json', dict(path_base='repository_root', quality_state='generated_unreviewed', triplets=output_rows))
    (STAGE/'triplets.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in output_rows), encoding='utf-8')
    pairs = json.loads((ROOT/'Photo_Pairs_60000_v1/pairs.json').read_text())
    for row in pairs['pairs']:
        for key in ('content_path', 'reference_path', 'target_path', 'planned_target_path'):
            if row.get(key):
                row[key] = (ROOT/'Photo_Pairs_60000_v1'/row[key]).resolve().relative_to(ROOT).as_posix()
    pairs['path_base'] = 'repository_root'
    pairs.pop('inherited_source_manifest', None)
    write(STAGE/'pairs.json', pairs)
    write(STAGE/'metadata/file_inventory.json', inventory)
    counts = dict(collections.Counter(r['split'] for r in output_rows))
    write(STAGE/'metadata/summary.json', dict(triplets=len(records), by_split=counts, unique_images=len(files)-len(records), output_images=len(records), bytes=sum(r['bytes'] for r in inventory), human_reviewed=0))
    for name in ('source_images.json', 'duplicate_groups.json'):
        write(STAGE/'metadata'/name, json.loads((ROOT/'Photo_Pairs_v1'/name).read_text()))
    for name in ('triplets.json', 'summary.json', 'failures.json'):
        write(STAGE/'metadata/original_generation'/name, json.loads((ROOT/'Photo_Triplets_v1'/name).read_text()))
    (STAGE/'README.md').write_text('''---
task_categories:
- image-to-image
language:
- en
- zh
size_categories:
- 10K<n<100K
---
# ViRAP

Photography style-transfer research corpus: 60,000 (input, reference, generated output) triplets generated using the released PETAL model. These samples are **unreviewed**; no 5,000-sample human-preference subset has been selected yet.

## Layout

- `Original Data/<source>/images/`: deduplicated source files, shared across triplets.
- `Generated_Outputs/<split>/<bucket>/`: original generated PNGs, unchanged.
- `triplets.json` and `triplets.jsonl`: complete generated records with repository-relative content, reference and target paths, generation settings and output SHA256.
- `pairs.json`: original planned pair assignments; pending status reflects the pre-generation plan. Use `triplets.json` for completed outputs.
- `metadata/file_inventory.json`: byte sizes and SHA256 for every uploaded image.
- `metadata/source_images.json`: source provenance retained verbatim; historical relative paths refer to the original project layout. Use triplet paths for this repository.
- `metadata/original_generation/`: original generation manifests for reproducibility.

Splits: train 59,100; validation 400; test 400; diagnostic 100. Preserve these splits when selecting a preference subset and fine-tuning; keep test data out of selection and training.

Paths in the root manifests resolve from the downloaded repository root, for example `Image.open(root / row["content_path"])`. Download with `hf download edzee3000/ViRAP --type dataset --local-dir ViRAP`.

Source collections include DIV2K, Unsplash Lite, Wikimedia Commons, album covers and personal phone photos. Source provenance is retained in metadata; original source rights and terms apply. No blanket license is asserted for the mixed corpus.
''', encoding='utf-8')
    print('VALIDATION_COMPLETE', json.dumps(counts), len(files), flush=True)
    api.upload_folder(repo_id=REPO, repo_type='dataset', folder_path=str(STAGE), commit_message='Publish 60000 unreviewed photography style-transfer triplets')
    remote = {f.path:f.size for f in api.list_repo_tree(REPO, repo_type='dataset', recursive=True) if hasattr(f, 'size')}
    missing = [r['path'] for r in inventory if remote.get(r['path']) != r['bytes']]
    assert not missing, missing[:20]
    result = dict(status='complete', revision=api.dataset_info(REPO).sha, image_files=len(inventory), by_split=counts)
    write(STAGE.parent/'virap_upload_result.json', result)
    print('UPLOAD_VERIFIED', json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
