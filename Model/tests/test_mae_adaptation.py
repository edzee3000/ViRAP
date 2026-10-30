import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tone_model.mae_adaptation import build_adaptation
from tone_model.mae_tone import MAEToneConfig
from tone_model.mae_hybrid_lora import adaptation_counts

class AdaptationTests(unittest.TestCase):
    def test_all_lora_covers_all_matrices_and_freezes_base(self):
        torch.set_num_threads(2)
        model=build_adaptation(None,MAEToneConfig(),'all_lora',allow_uninitialized=True)
        self.assertEqual(len(model.lora_targets),49)
        for name,p in model.encoder.backbone.named_parameters():
            self.assertEqual(p.requires_grad,name.endswith(('lora_A','lora_B')),name)
        model.encoder(torch.rand(1,3,24,32))['dense'][...,0].mean().backward()
        for index in (0,11):
            layer=model.encoder.backbone.blocks[index].attn.qkv
            self.assertIsNone(layer.base.weight.grad)
            self.assertGreater(layer.lora_B.grad.abs().max().item(),0)
        print('all_lora_counts',adaptation_counts(model))
    def test_full_means_every_parameter_including_position(self):
        model=build_adaptation(None,MAEToneConfig(),'full',allow_uninitialized=True)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual(model.lora_targets,[])
        self.assertEqual(adaptation_counts(model)['lora'],0)
        print('full_counts',adaptation_counts(model))

if __name__=='__main__':unittest.main()
