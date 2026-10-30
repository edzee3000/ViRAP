import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mae_gallery import fixed_selection,gallery_payload,safe_asset


class GalleryTests(unittest.TestCase):
    def test_fixed_source_diversity_unique_content_and_split(self):
        rows=[]
        for split in ('train','validation','test'):
            for source in ('A','B'):
                for i in range(8):
                    for j in range(2):rows.append({'split':split,'pair_id':f'{split}_{source}_{i}_{j}','content_key':f'{source}/{split}_{i}'})
        selected=fixed_selection(rows,10)
        self.assertEqual(selected,fixed_selection(rows,10))
        mapping={r['pair_id']:r for r in rows}
        for split,ids in selected.items():
            self.assertEqual(len(ids),10);self.assertEqual(len({mapping[i]['content_key'] for i in ids}),10)
            self.assertTrue(all(mapping[i]['split']==split for i in ids))
            self.assertEqual(sum(mapping[i]['content_key'].startswith('A/') for i in ids),5)

    def test_pagination_and_no_test_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'runs/M1_s17';run.mkdir(parents=True)
            (run/'gallery.json').write_text(json.dumps({'checkpoint_step':500,'validation':[{'pair_id':str(i)} for i in range(12)]}))
            payload=gallery_payload(root,'M1_s17','validation',1,6)
            self.assertEqual([r['pair_id'] for r in payload['rows']],list(map(str,range(6,12))))
            with self.assertRaises(ValueError):gallery_payload(root,'M1_s17','test')
            with self.assertRaises(ValueError):gallery_payload(root,'../outside','train')

    def test_image_path_cannot_escape_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);assets=root/'gallery_assets';assets.mkdir()
            (assets/'image.png').write_bytes(b'example');(root/'private.png').write_bytes(b'private')
            self.assertEqual(safe_asset(root,'image.png'),assets/'image.png')
            with self.assertRaises(ValueError):safe_asset(root,'../private.png')


if __name__=='__main__':unittest.main()
