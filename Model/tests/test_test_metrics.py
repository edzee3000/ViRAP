import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_metrics import image_metrics

class MetricTests(unittest.TestCase):
    def test_identical(self):
        image=torch.rand(1,3,19,25);m=image_metrics(image,image)
        self.assertEqual(m['l1'],0);self.assertEqual(m['ssim'],1);self.assertEqual(m['psnr'],120)
    def test_constant_error(self):
        m=image_metrics(torch.zeros(1,3,17,25),torch.ones(1,3,17,25)*.5)
        self.assertAlmostEqual(m['l1'],.5);self.assertAlmostEqual(m['psnr'],6.0206,places=4)
        self.assertLess(m['ssim'],.01);self.assertEqual(m['gradient_l1'],0)
    def test_geometry_rejected(self):
        with self.assertRaises(ValueError):image_metrics(torch.zeros(1,3,20,20),torch.zeros(1,3,20,21))

if __name__=='__main__':unittest.main()
