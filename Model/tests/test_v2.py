"""Structural checks with explicit RANDOM upstream architecture, not style results."""
import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tone_model.v2 import V2Config,ReferenceToneModelV2,SourceContentContext

REPO=Path(__file__).resolve().parents[2]/'Research/External_Code/dinov2'


@unittest.skipUnless(REPO.is_dir(),'Official DINOv2 source checkout is required')
class V2Architecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);torch.manual_seed(11)
        cls.model=ReferenceToneModelV2.from_official(REPO,allow_uninitialized=True)

    def test_exact_parameter_and_stage_counts(self):
        model=self.model
        self.assertEqual(sum(p.numel() for p in model.parameters()),27315437)
        self.assertEqual(tuple(model.encoder.backbone.pos_embed.shape),(1,1370,384))
        expected={'encoder':3106048,'heads':5258861,'joint':12360557,'full':27315437}
        for stage,n in expected.items():
            model.set_stage(stage)
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),n)
        model.set_stage('heads')

    def test_weight_requirement_is_explicit(self):
        with self.assertRaises(ValueError):ReferenceToneModelV2.from_official(REPO)

    def test_shapes_shared_encoder_and_initial_identity(self):
        model=self.model.eval();model.set_stage('heads')
        with torch.inference_mode():
            image=torch.rand(1,3,35,49);ref=torch.rand(1,3,43,37)
            out,aux=model(image,ref,True)
        self.assertEqual(out.shape,image.shape)
        self.assertEqual(tuple(aux['content_tokens'].shape),(1,16,256))
        self.assertEqual(tuple(aux['global_lut'].shape),(1,3,17,17,17))
        self.assertEqual(tuple(aux['maps'].shape),(1,2,64,64))
        self.assertEqual(aux['memory_tokens'],1399)
        self.assertLess((out-image).abs().max().item(),1e-6)
        self.assertFalse(aux['pretrained_loaded'])

    def test_frozen_backbone_stays_eval_after_train(self):
        model=self.model;model.set_stage('heads');model.train()
        self.assertFalse(model.encoder.backbone.training)
        self.assertTrue(model.encoder.photometric.training)
        self.assertFalse(any(p.requires_grad for p in model.encoder.backbone.parameters()))
        model.set_stage('joint')
        self.assertFalse(any(p.requires_grad for p in model.encoder.backbone.blocks[7].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.encoder.backbone.blocks[8].parameters()))
        model.set_stage('heads')

    def test_reference_changes_color_condition_and_spatial_maps(self):
        model=self.model.eval()
        with torch.inference_mode():
            image=torch.rand(1,3,33,41)
            a=model(image,torch.rand(1,3,37,29),True)[1]
            b=model(image,torch.rand(1,3,37,29),True)[1]
        self.assertGreater((a['maps']-b['maps']).abs().max().item(),1e-8)
        self.assertGreater((a['condition']-b['condition']).abs().max().item(),1e-7)

    def test_legacy_maps_ignore_reference_condition_and_keep_original_parameters(self):
        cfg=V2Config(spatial_conditioning='none');context=SourceContentContext(cfg).eval()
        self.assertEqual(sum(p.numel() for p in context.parameters()),276258)
        self.assertFalse(any('film' in name for name,_ in context.named_parameters()))
        image=torch.rand(1,3,31,37)
        encoded={'dense':torch.rand(1,400,384),'photo_dense':torch.rand(1,64,64,64)}
        with torch.no_grad():
            first=context(image,encoded,torch.rand(1,256));second=context(image,encoded,torch.rand(1,256))
        self.assertTrue(torch.equal(first,second))

    def test_spatial_condition_gradient_and_bounded_maps(self):
        model=self.model.eval();model.zero_grad(set_to_none=True)
        _,aux=model(torch.rand(1,3,31,37),torch.rand(1,3,37,31),True)
        aux['maps'].square().mean().backward()
        gradient=model.content_context.pair_film.weight.grad
        self.assertIsNotNone(gradient);self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().max().item(),0.)
        self.assertGreater(model.conditioner.global_pair[0].weight.grad.abs().max().item(),0.)
        self.assertLessEqual(aux['maps'].abs().max().item(),1.)

    def test_spatial_rank_and_condition_contract(self):
        for rank in (2,4,8):
            context=SourceContentContext(V2Config(local_rank=rank))
            image=torch.rand(1,3,31,37)
            encoded={'dense':torch.rand(1,400,384),'photo_dense':torch.rand(1,64,64,64)}
            with self.assertRaises(ValueError):context(image,encoded)
            with torch.no_grad():maps=context(image,encoded,torch.rand(1,256))
            self.assertEqual(tuple(maps.shape),(1,rank,64,64))

    def test_backward_reaches_style_readout(self):
        model=self.model;model.set_stage('heads');model.train();model.zero_grad()
        image=torch.rand(1,3,31,37);ref=torch.rand(1,3,37,31)
        _,aux=model(image,ref,True)
        # Test representation graph directly: zero-initialized color outputs
        # intentionally do not yet send reconstruction gradients to the encoder.
        aux['condition'].square().mean().backward()
        self.assertIsNone(model.encoder.backbone.patch_embed.proj.weight.grad)
        self.assertIsNotNone(model.encoder.queries.grad)
        self.assertTrue(torch.isfinite(model.encoder.queries.grad).all())

    def test_constant_input_does_not_activate_local_geometry(self):
        model=self.model.eval()
        with torch.inference_mode():
            _,aux=model(torch.full((1,3,33,41),.4),torch.rand(1,3,37,29),True)
        self.assertEqual(aux['maps'].abs().max().item(),0.)


if __name__=='__main__':unittest.main(verbosity=2)
