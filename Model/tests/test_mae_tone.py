"""Explicit random-structure checks; not pretrained image quality tests."""
import importlib.util
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tone_model.mae_tone import MAEToneModel,MAEToneConfig
from tone_model.losses import ToneLoss,LossConfig


@unittest.skipUnless(importlib.util.find_spec('timm'),'Requires pinned timm dependency')
class MAEToneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);torch.manual_seed(17)
        cls.model=MAEToneModel.from_pretrained(allow_uninitialized=True)

    def test_exact_parameters_and_stages(self):
        m=self.model
        self.assertEqual(sum(p.numel() for p in m.parameters()),91402999)
        for stage,expected in [('heads',5604343),('joint',33957367),('full',91251703)]:
            m.set_stage(stage)
            self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad),expected)
            self.assertFalse(m.encoder.backbone.pos_embed.requires_grad)
        m.set_stage('heads')

    def test_mask_free_features_match_timm_forward(self):
        m=self.model.eval();backbone=m.encoder.backbone
        image=torch.rand(1,3,224,224)
        with torch.no_grad():
            stages=backbone.get_intermediate_layers(image)
            direct=backbone.trunk.forward_features(image)
        self.assertTrue(torch.allclose(stages[-1][0],direct[:,1:],atol=1e-5,rtol=1e-5))
        self.assertTrue(torch.allclose(stages[-1][1],direct[:,0],atol=1e-5,rtol=1e-5))

    def test_shapes_identity_and_reference_conditioning(self):
        m=self.model.eval();m.set_stage('heads')
        image=torch.rand(1,3,31,43)
        with torch.no_grad():
            output,a=m(image,torch.rand(1,3,37,29),True)
            _,b=m(image,torch.rand(1,3,37,29),True)
        self.assertEqual(tuple(a['maps'].shape),(1,4,64,64))
        self.assertEqual(a['memory_tokens'],787)
        self.assertEqual(tuple(a['local_luts'].shape),(1,4,3,17,17,17))
        self.assertLess((output-image).abs().max().item(),1e-6)
        self.assertGreater((a['maps']-b['maps']).abs().max().item(),0.)

    def test_actual_loss_interface_and_frozen_backbone(self):
        m=self.model;m.set_stage('heads');m.train();m.zero_grad(set_to_none=True)
        image=torch.rand(1,3,31,43);target=image*.8
        output,aux=m(image,torch.rand(1,3,37,29),True)
        losses=ToneLoss(LossConfig(style_invariance=0.,style_sensitivity=0.))(output,target,image,aux)
        losses['total'].backward()
        self.assertTrue(torch.isfinite(losses['total']))
        self.assertIsNone(m.encoder.backbone.trunk.patch_embed.proj.weight.grad)
        self.assertIsNotNone(m.color_generator.output.weight.grad)
        self.assertFalse(m.encoder.backbone.training)

    def test_no_unsupported_encoder_only_stage_or_implicit_weights(self):
        with self.assertRaises(ValueError):self.model.set_stage('encoder')
        with self.assertRaises(ValueError):MAEToneModel.from_pretrained()
        with self.assertRaises(ValueError):MAEToneModel.from_official('anything')


if __name__=='__main__':unittest.main()
