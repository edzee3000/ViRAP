"""Meaningful ordering/resume and safe dashboard behavior without GPU training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_mae_full import EpochOrder,atomic_json,ValidationMemoryCache,SourceMemoryTriplets
from tone_model.data import TripletDataset
from mae_training_dashboard import snapshot


class FullTrainingTests(unittest.TestCase):
    def test_epoch_order_coverage_and_resume(self):
        full=list(EpochOrder(19,3,17));remaining=list(EpochOrder(19,3,17,23))
        self.assertEqual(full[23:],remaining)
        for epoch in range(3):self.assertEqual(sorted(full[epoch*19:(epoch+1)*19]),list(range(19)))

    def test_dashboard_handles_partial_jsonl_and_process_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'runs/M1_s17';run.mkdir(parents=True)
            atomic_json(run/'status.json',{'state':'running','pid':2147483647,'step':20,'total_steps':100})
            (run/'metrics.jsonl').write_text(json.dumps({'step':20,'elapsed_seconds':12,'train':{'total':.1,'reconstruction':.09,'structure':.01}})+'\n{"step":')
            result=snapshot(root)['runs'][0]
            self.assertEqual(result['status']['state'],'interrupted')
            self.assertEqual(len(result['train']),1)

    def test_atomic_status_replaces_complete_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'status.json';atomic_json(path,{'step':1});atomic_json(path,{'step':2})
            self.assertEqual(json.loads(path.read_text()),{'step':2})
            self.assertFalse(path.with_suffix('.json.tmp').exists())

    def test_validation_cache_preserves_values_and_avoids_repeat_reads(self):
        class Data:
            calls=0
            def __len__(self):return 2
            def __getitem__(self,index):self.calls+=1;return {'value':index}
            def fingerprint(self):return 'frozen'
        data=Data();cached=ValidationMemoryCache(data)
        self.assertEqual(cached[1],{'value':1});self.assertEqual(cached[1],{'value':1})
        self.assertEqual(data.calls,1);self.assertEqual(cached.fingerprint(),'frozen')

    def test_source_ram_cache_is_pixel_identical_for_different_targets(self):
        import numpy as np
        import torch
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rng=np.random.default_rng(2)
            for name in ('content','reference','target0','target1'):
                Image.fromarray(rng.integers(0,256,(39,57,3),dtype='uint8')).save(root/(name+'.png'))
            rows=[{'pair_id':str(i),'split':'train','_decision':'keep','generation':{'max_image_edge':768},
                   '_content_path':root/'content.png','_reference_path':root/'reference.png','_target_path':root/('target'+str(i)+'.png')} for i in range(2)]
            original=TripletDataset(rows,long_edge=32,reference_edge=32)
            cached=SourceMemoryTriplets(rows,long_edge=32,reference_edge=32)
            for index in (0,1,0):
                a=original[index];b=cached[index]
                for role in ('content','reference','target'):self.assertTrue(torch.equal(a[role],b[role]))
            self.assertEqual(len(cached.content_memory),1);self.assertEqual(len(cached.reference_memory),1)


if __name__=='__main__':unittest.main()
