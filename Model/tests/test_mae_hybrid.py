import importlib.util
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tone_model.mae_hybrid_lora import LoRALinear,LoRAPatchProjection,MAEHybridLoRAModel,adaptation_counts
from hybrid_batching import ExactShapeBatches


class BatchingAndAdapters(unittest.TestCase):
    def test_exact_buckets_preserve_coverage_and_resume(self):
        rows=[{'width':768,'height':512 if i%2 else 768} for i in range(35)]
        sampler=ExactShapeBatches(rows,batch_size=16,epochs=2)
        previous=0
        for end in sampler.epoch_ends:
            batches=sampler.batches[previous:end]
            self.assertEqual(sorted(i for batch in batches for i in batch),list(range(35)))
            for batch in batches:self.assertEqual(len({rows[i]['height'] for i in batch}),1)
            previous=end
        resumed=ExactShapeBatches(rows,batch_size=16,epochs=2,start_batch=2)
        self.assertEqual(list(resumed),sampler.batches[2:])

    def test_zero_lora_is_identity_and_base_stays_frozen(self):
        torch.manual_seed(1);linear=LoRALinear(torch.nn.Linear(7,11),rank=3,alpha=6)
        x=torch.randn(2,5,7);self.assertTrue(torch.equal(linear(x),linear.base(x)))
        linear(x).square().mean().backward()
        self.assertIsNone(linear.base.weight.grad)
        self.assertGreater(linear.lora_B.grad.abs().max().item(),0)
        self.assertEqual(linear.lora_A.grad.abs().max().item(),0)

    def test_patch_lora_shape_and_identity(self):
        module=LoRAPatchProjection(torch.nn.Conv2d(3,12,4,stride=4),rank=2,alpha=4)
        x=torch.randn(2,3,16,16)
        self.assertEqual(tuple(module(x).shape),(2,12,4,4))
        self.assertTrue(torch.equal(module(x),module.base(x)))


@unittest.skipUnless(importlib.util.find_spec('timm'),'Requires timm')
class HybridArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);torch.manual_seed(17)
        cls.model=MAEHybridLoRAModel.from_pretrained(allow_uninitialized=True)
    def test_counts_and_training_boundary(self):
        model=self.model;values=adaptation_counts(model)
        self.assertEqual(values['total'],92201719);self.assertEqual(values['trainable'],34756087);self.assertEqual(values['lora'],798720)
        self.assertEqual(len(values['targets']),33)
        self.assertTrue(all(p.requires_grad for p in model.encoder.backbone.blocks[8:].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.encoder.backbone.norm.parameters()))
        for name,p in model.encoder.backbone.named_parameters():
            if name.startswith('trunk.blocks.') and int(name.split('.')[2])<8:
                self.assertEqual(p.requires_grad,name.endswith(('lora_A','lora_B')))
        self.assertFalse(model.encoder.backbone.pos_embed.requires_grad)
    def test_gradients_reach_frozen_prefix_adapters(self):
        model=self.model;model.zero_grad(set_to_none=True)
        encoded=model.encoder(torch.rand(1,3,32,48))
        encoded['dense'][...,0].mean().backward()
        prefix=model.encoder.backbone.blocks[0].attn.qkv
        self.assertIsNone(prefix.base.weight.grad)
        self.assertGreater(prefix.lora_B.grad.abs().max().item(),0)
        self.assertGreater(model.encoder.backbone.blocks[11].attn.qkv.weight.grad.abs().max().item(),0)


if __name__=='__main__':unittest.main()
