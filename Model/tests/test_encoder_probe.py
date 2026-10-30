import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tone_model.encoder_probe import EncoderProbe,image_metrics,probe_loss
from run_encoder_probe import schedule


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_stats_only_has_no_visual_or_cnn_weights(self):
        m=EncoderProbe('A');self.assertEqual(len(m.encoder.visual_projections),0)
        self.assertFalse(any('stages' in k for k in m.encoder.photo.state_dict()))
        self.assertEqual(len(m.encoder.stat_projections),4)

    def test_no_photo_group_really_removes_both_photo_paths(self):
        m=EncoderProbe('E');self.assertIsNone(m.encoder.photo)
        self.assertEqual(len(m.encoder.stat_projections),0)
        self.assertEqual(len(m.encoder.photo_projections),0)

    def test_shared_heads_same_initialization_across_groups(self):
        torch.manual_seed(17);a=EncoderProbe('A')
        torch.manual_seed(17);b=EncoderProbe('B')
        for name in ('conditioner','renderer'):
            for x,y in zip(getattr(a,name).parameters(),getattr(b,name).parameters()):self.assertTrue(torch.equal(x,y))
        for x,y in zip(a.encoder.readout.parameters(),b.encoder.readout.parameters()):self.assertTrue(torch.equal(x,y))

    def test_initial_identity_and_backward(self):
        m=EncoderProbe('A');image=torch.rand(2,3,32,32);reference=torch.rand_like(image)
        p,aux=m(image,reference,image)
        self.assertLess(float((p-image).abs().max().detach()),1e-6)
        loss=probe_loss(p,image*.8,aux);loss.backward()
        self.assertIsNotNone(m.renderer.output.weight.grad)
        self.assertTrue(torch.isfinite(m.renderer.output.weight.grad).all())

    def test_metrics_identity(self):
        x=torch.rand(2,3,32,32);metrics=image_metrics(x,x)
        self.assertTrue(torch.allclose(metrics['ssim'],torch.ones(2),atol=1e-5))
        self.assertEqual(float(metrics['l1'].max()),0.)
        self.assertEqual(float(metrics['delta_e76'].max()),0.)

    def test_schedule_covers_each_pair_once_per_epoch(self):
        ids=np.arange(19);first=list(schedule(ids,2,17,8));second=list(schedule(ids,2,17,8))
        for epoch in range(2):self.assertEqual(sorted(np.concatenate([x for e,x in first if e==epoch]).tolist()),ids.tolist())
        for (_,x),(_,y) in zip(first,second):self.assertTrue(np.array_equal(x,y))


if __name__=='__main__':unittest.main()
