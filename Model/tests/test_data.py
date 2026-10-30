"""Selection/isolation and generation-preprocessing regression tests."""
import hashlib,json,sys,tempfile,unittest,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from tone_model.data import load_triplets,TripletDataset


class DatasetProperties(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'images').mkdir();(self.root/'Photo_Triplets_v1').mkdir();(self.root/'Photo_Pairs_v1').mkdir()
        records=[];sources=[]
        for split,offset in [('train',0),('validation',40)]:
            for role,color in [('content',(60+offset,80,120)),('reference',(180,40+offset,90))]:
                name=f'{split}_{role}';path=self.root/'images'/(name+'.png')
                Image.new('RGB',(35,49),color).save(path)
                sources.append({'key':name,'split':split,'duplicate_group':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'dataset_relative_path':'../images/'+name+'.png'})
            target=self.root/'images'/(split+'_target.png');Image.new('RGB',(35,49),(100,80+offset,90)).save(target)
            records.append({'pair_id':split+'_1','split':split,'content_key':split+'_content','reference_key':split+'_reference','content_path':'../images/'+split+'_content.png','reference_path':'../images/'+split+'_reference.png','target_path':'../images/'+split+'_target.png'})
        self.manifest=self.root/'Photo_Triplets_v1/triplets.json';self.sources=self.root/'Photo_Pairs_v1/source_images.json'
        self.records=records;self.source_rows=sources;self.write()

    def tearDown(self):self.temp.cleanup()

    def write(self):
        self.manifest.write_text(json.dumps({'triplets':self.records}))
        self.sources.write_text(json.dumps({'images':self.source_rows}))

    def test_portrait_shapes_and_review_selection(self):
        review=self.root/'review.json';review.write_text(json.dumps({'collection':'nju-triplets-60000-v1','reviewed':[{'pair_id':'train_1','decision':'high_quality'}]}))
        records=load_triplets(self.manifest,reviews=[str(review)]);sample=TripletDataset(records,'train',32,32,True)[0]
        self.assertEqual(sample['content'].shape,sample['target'].shape)
        self.assertEqual(tuple(sample['content'].shape), (3,32,23))
        self.assertEqual(len(TripletDataset(records,'validation',32,32)),1)

    def test_reference_cannot_cross_frozen_split(self):
        self.records[1]['reference_key']='train_reference';self.records[1]['reference_path']='../images/train_reference.png';self.write()
        with self.assertRaises(ValueError):load_triplets(self.manifest)

    def test_key_cannot_alias_a_different_source_path(self):
        self.records[0]['content_path']='../images/train_reference.png';self.write()
        with self.assertRaises(ValueError):load_triplets(self.manifest)

    def test_unknown_review_and_path_escape_are_rejected(self):
        review=self.root/'review.json';review.write_text(json.dumps({'collection':'nju-triplets-60000-v1','reviewed':[{'pair_id':'not-real','decision':'reject'}]}))
        with self.assertRaises(ValueError):load_triplets(self.manifest,reviews=[str(review)])
        self.records[0]['target_path']='../../outside.png';self.write()
        with self.assertRaises(ValueError):load_triplets(self.manifest)

    def test_generation_thumbnail_matches_exact_source_raster(self):
        rng=np.random.default_rng(7)
        pixels=rng.integers(0,256,(1200,1800,3),dtype=np.uint8)
        path=self.root/'images/train_content.png';Image.fromarray(pixels).save(path)
        expected=Image.open(path).convert('RGB');expected.thumbnail((256,256),Image.Resampling.LANCZOS)
        expected.save(self.root/'images/train_target.png')
        self.records[0]['generation']={'max_image_edge':256};self.records[0]['width'],self.records[0]['height']=expected.size
        self.write();records=load_triplets(self.manifest)
        sample=TripletDataset(records,'train',0,32)[0]
        self.assertTrue(torch.equal(sample['content'],sample['target']))

    def test_lossless_cache_and_stat_invalidation(self):
        records=load_triplets(self.manifest);cache=self.root/'cache'
        dataset=TripletDataset(records,'train',32,32,cache_dir=cache)
        cold=dataset[0];warm=dataset[0]
        for role in ['content','reference','target']:self.assertTrue(torch.equal(cold[role],warm[role]))
        Image.new('RGB',(35,49),(220,230,240)).save(self.root/'images/train_reference.png')
        changed=dataset[0]
        self.assertFalse(torch.equal(changed['reference'],warm['reference']))


if __name__=='__main__':unittest.main(verbosity=2)
