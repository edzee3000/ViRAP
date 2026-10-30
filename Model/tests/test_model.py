"""Behavioral properties, numerical boundaries and gradient checks (CPU)."""
import dataclasses,sys,unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from tone_model.model import (ModelConfig,ToneTransferModel,identity_lut,
                              sample_lut,project_residual,color_basis)
from tone_model.losses import ToneLoss


class ModelProperties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def setUp(self):torch.manual_seed(17)

    def model(self, **kwargs):
        cfg=dict(style_size=32,patch_size=8,token_dim=32,style_dim=32,
                 condition_dim=48,context_size=16,lut_size=5,render_chunk_rows=11)
        return ToneTransferModel(ModelConfig(**{**cfg,**kwargs}))

    def nonzero_heads(self, model):
        with torch.no_grad():
            model.global_head.weight.normal_(std=.04)
            if model.config.local_rank:model.local_head.weight.normal_(std=.06)

    def test_lut_axis_order_and_identity(self):
        image=torch.rand(2,3,13,19);lut=identity_lut(5).expand(2,-1,-1,-1,-1)
        self.assertLess((sample_lut(lut,image)-image).abs().max().item(),3e-7)
        modified=lut.clone();modified[:,0]=1-modified[:,0]
        expected=image.clone();expected[:,0]=1-expected[:,0]
        self.assertLess((sample_lut(modified,image)-expected).abs().max().item(),3e-7)

    def test_untrained_initialization_is_identity(self):
        model=self.model();x=torch.rand(2,3,29,41);reference=torch.rand(2,3,27,33)
        self.assertLess((model(x,reference)-x).abs().max().item(),3e-7)

    def test_style_invariance_to_patch_permutation(self):
        model=self.model();x=torch.rand(2,3,32,32)
        patches=model.style_encoder.patches(x);permutation=torch.randperm(patches.shape[1])
        a=model.style_encoder.encode_patches(patches)
        b=model.style_encoder.encode_patches(patches[:,permutation])
        self.assertLess((a-b).abs().max().item(),1e-6)
        # Test the complete encoder on an RGB patch-rearranged image too.
        rgb=F.unfold(x,8,stride=8)[:,:,permutation]
        shuffled=F.fold(rgb,(32,32),8,stride=8)
        self.assertLess((model.style_encoder(x)-model.style_encoder(shuffled)).abs().max().item(),1e-6)

    def test_maps_do_not_depend_on_reference(self):
        model=self.model();self.nonzero_heads(model);x=torch.rand(1,3,29,37)
        p=model.prepare(x,torch.rand_like(x));q=model.prepare(x,torch.rand_like(x))
        self.assertTrue(torch.equal(p.maps,q.maps))
        self.assertGreater((p.global_lut-q.global_lut).abs().max().item(),1e-7)

    def test_final_rgb_proxy_residual_is_projected(self):
        model=self.model();self.nonzero_heads(model);x=torch.rand(2,3,31,43)
        p=model.prepare(x,torch.rand(2,3,29,37));a=p.aux
        correlation=a['proxy_basis'].transpose(1,2)@a['proxy_local'].flatten(2).transpose(1,2)/256
        self.assertLess(correlation.abs().max().item(),2e-6)

    def test_projection_removes_color_polynomials(self):
        image=torch.rand(2,3,17,21);basis=color_basis(image)
        raw=torch.einsum('bhwm,bmc->bchw',basis,torch.randn(2,10,3))
        result,_,_=project_residual(raw,image)
        self.assertLess(result.abs().max().item(),3e-5)

    def test_projection_idempotence_and_rank_deficiency(self):
        for image in [torch.rand(1,3,13,17),torch.full((1,3,13,17),.5)]:
            raw=torch.randn_like(image);once,_,_=project_residual(raw,image)
            twice,_,_=project_residual(once,image)
            self.assertTrue(torch.isfinite(twice).all())
            self.assertLess((once-twice).abs().max().item(),1e-5)

    def test_flat_field_has_no_new_spatial_pattern(self):
        model=self.model().eval();self.nonzero_heads(model)
        colors=torch.rand(2,3,1,1);x=colors.expand(2,3,35,49)
        y=model(x,torch.rand(2,3,31,41))
        self.assertLess((y-y[:,:,:1,:1]).abs().max().item(),2e-6)

    def test_row_chunk_render_is_consistent(self):
        model=self.model().eval();self.nonzero_heads(model);x=torch.rand(1,3,45,63)
        plan=model.prepare(x,torch.rand(1,3,41,53));a=model.render(x,plan)
        model.config.render_chunk_rows=1000;b=model.render(x,plan)
        self.assertLess((a-b).abs().max().item(),2e-6)

    def test_proxy_projection_matches_proxy_render(self):
        model=self.model();self.nonzero_heads(model);x=torch.rand(1,3,16,16)
        plan=model.prepare(x,torch.rand_like(x))
        global_only=sample_lut(plan.global_lut,x)
        self.assertLess((model._render_raw(x,plan)-global_only-plan.aux['proxy_local']).abs().max().item(),2e-7)

    def test_ablation_modes_and_local_warmup(self):
        x=torch.rand(1,3,21,29)
        for mode in ['none','mean','color']:
            model=self.model(projection=mode);self.nonzero_heads(model)
            y,a=model(x,torch.rand_like(x),True)
            self.assertTrue(torch.isfinite(y).all())
            if mode=='mean':self.assertLess(a['proxy_local'].mean((2,3)).abs().max().item(),1e-7)
            model.set_local_enabled(False);plan=model.prepare(x,torch.rand_like(x))
            self.assertEqual(plan.active,())
            self.assertLess((model.render(x,plan)-sample_lut(plan.global_lut,x).clamp(0,1)).abs().max().item(),2e-7)

    def test_gradients_and_tiny_optimization(self):
        model=self.model();optimizer=torch.optim.Adam(model.parameters(),lr=.003);criterion=ToneLoss()
        x=torch.rand(1,3,24,32);r=torch.rand(1,3,29,23);target=(x*.8+.05).clamp(0,1)
        losses=[]
        for _ in range(5):
            optimizer.zero_grad();y,a=model(x,r,True);pieces=criterion(y,target,x,a)
            self.assertTrue(torch.isfinite(pieces['total']))
            pieces['total'].backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step();losses.append(pieces['total'].item())
        self.assertLess(losses[-1],losses[0])
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.style_encoder.parameters() if p.grad is not None),0)

    def test_optional_eval_gate_skips_render_branches(self):
        model=self.model(eval_gate_threshold=.9).eval();self.nonzero_heads(model)
        p=model.prepare(torch.rand(1,3,19,23),torch.rand(1,3,23,19))
        self.assertEqual(p.active,())
        self.assertEqual(p.gates.abs().sum().item(),0)

    def test_global_baseline_and_variable_resolution(self):
        model=self.model(local_rank=0)
        x=torch.rand(2,3,37,23);y,a=model(x,torch.rand(2,3,21,31),True)
        self.assertEqual(y.shape,x.shape);self.assertEqual(a['active_local_branches'],0)
        self.assertTrue(torch.isfinite(ToneLoss()(y,x,x,a)['total']))

    def test_native_color_extrapolation_is_bounded(self):
        model=self.model().eval();self.nonzero_heads(model)
        narrow=torch.full((1,3,24,32),.5)+torch.rand(1,3,24,32)*.002
        plan=model.prepare(narrow,torch.rand(1,3,21,31))
        # Stress the complete RGB domain, not just colors present in the proxy.
        native=torch.rand(1,3,47,61)
        local=model._render_raw(native,plan)-sample_lut(plan.global_lut,native)
        self.assertLessEqual(local.abs().max().item(),model.config.local_output_limit+1e-6)
        basis=plan.aux['proxy_basis'];residual=plan.aux['proxy_local'].flatten(2).transpose(1,2)
        self.assertLess((basis.transpose(1,2)@residual/basis.shape[1]).abs().max().item(),1e-5)

    def test_streaming_cpu_uint8_matches_dense_render(self):
        model=self.model().eval();self.nonzero_heads(model)
        source=torch.randint(0,256,(1,3,37,51),dtype=torch.uint8)
        image=source.float()/255;plan=model.prepare(image,torch.rand(1,3,29,41))
        expected=model.render(image,plan)
        actual=torch.cat([block for _,_,block in model.iter_render_rows(source,plan)],2)
        self.assertLess((actual-expected).abs().max().item(),1e-6)


if __name__=='__main__':unittest.main(verbosity=2)
