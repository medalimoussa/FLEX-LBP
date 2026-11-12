"""Unit tests of the FLEX-LBP layer (PyTorch only): dimensions, exact binary inference, gradients, postprocessing,
training loop. Run with `python -m unittest discover tests` (or pytest)."""
import unittest

import torch

import flexlbp
from flexlbp.layer import PAIRS
from flexlbp.train import fit


def images(n=2, size=48, seed=0):
    return torch.rand(n, 3, size, size, generator=torch.Generator().manual_seed(seed))


def perturbed(learn="all_joint", seed=0, **kwargs):
    """A layer whose parameters are moved away from the initialization (to exercise non-trivial geometry)."""
    torch.manual_seed(seed)
    m = flexlbp.FLEXLBP(learn=learn, **kwargs)
    with torch.no_grad():
        for b, br in m.branches.items():
            br.radius_param.normal_(0, 0.5); br.angle_logits.normal_(0, 0.3); br.threshold.normal_(0, 0.02)
            m.color[b].weight.add_(0.1 * torch.randn(3, 3))
    return m


class TestLayer(unittest.TestCase):
    def test_dimensions(self):
        for template in ("wide", "compact"):
            z = flexlbp.FLEXLBP(template=template)(images())
            self.assertEqual(tuple(z.shape), (2, 13824))
            self.assertTrue(torch.allclose(z.norm(dim=1), torch.ones(2), atol=1e-5))
        self.assertEqual(tuple(flexlbp.postprocess(z.detach()).shape), (2, 3186))

    def test_parameter_counts(self):
        expected = {"fixed": 0, "color": 27, "radius": 12, "threshold": 24, "geometry": 60, "mixer_q": 30, "all_joint": 87}
        for preset, n in expected.items():
            m = flexlbp.FLEXLBP(learn=preset)
            self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad), n, preset)
            self.assertEqual(sum(p.numel() for p in m.parameters()), 90, preset)

    def test_exact_binary_inference(self):
        """The forward value, with gradients enabled, is the exact histogram of the hard codes."""
        m = perturbed(); x = images(1)
        for b, br in m.branches.items():
            y = m.color[b](x)
            with torch.no_grad():
                c, nb = br.sample(br._quantize(y)); ref = []
                for i, j in PAIRS:
                    d = nb[:, j] - c[:, i:i + 1]                                    # (1, K, H, W)
                    mag = d.abs()
                    for bits in ((d > br.threshold.view(1, -1, 1, 1)).long(), (mag > mag.mean()).long()):
                        code = (bits * (2 ** torch.arange(br.K)).view(1, -1, 1, 1)).sum(1).flatten()
                        h = torch.bincount(code, minlength=2 ** br.K).double(); ref.append(h / h.sum())
            out = br(y)[0]
            self.assertTrue(out.requires_grad)
            self.assertLess(float((out.double() - torch.cat(ref)).abs().max()), 1e-6, b)

    def test_gradients_follow_the_preset(self):
        m = flexlbp.FLEXLBP(learn="geometry")
        flexlbp.ProxyCosineLoss(3, m.dim)(m(images(3)), torch.tensor([0, 1, 2])).backward()
        for name, p in m.named_parameters():
            if p.requires_grad:
                self.assertTrue(p.grad is not None and float(p.grad.abs().sum()) > 0, name)
            else:
                self.assertIsNone(p.grad, name)

    def test_bitflip_estimator(self):
        """Same descriptor as the adjacent-bin estimator, different (non-zero) surrogate gradient."""
        x, w = images(), torch.randn(13824, generator=torch.Generator().manual_seed(1))
        a = perturbed(learn="geometry"); b = flexlbp.FLEXLBP(learn="geometry", histogram_estimator="bitflip")
        b.load_state_dict(a.state_dict())
        za, zb = a(x), b(x)
        self.assertLess(float((za - zb).abs().max()), 1e-7)
        (za * w).sum().backward(); (zb * w).sum().backward()
        ga, gb = a.branches["fine"].threshold.grad, b.branches["fine"].threshold.grad
        self.assertGreater(float(gb.abs().sum()), 0.0)
        self.assertFalse(torch.allclose(ga, gb))

    def test_uniform_mapping(self):
        mp = flexlbp.uniform_u2_mapping(8)
        self.assertEqual(int(mp.max()) + 1, 59)
        self.assertEqual(int((mp < 58).sum()), 58)
        self.assertEqual((int(mp[0]), int(mp[255]), int(mp[0b101])), (0, 57, 58))
        z = torch.rand(2, 13824)
        self.assertTrue(torch.allclose(flexlbp.uniform_u2(z).sum(1), z.sum(1), rtol=1e-5))

    def test_small_images_and_median_support(self):
        for size, support in ((12, 1), (40, 3)):
            z = flexlbp.FLEXLBP(support_size=support)(images(1, size))
            self.assertTrue(bool(torch.isfinite(z).all()))
            self.assertAlmostEqual(float(z.norm()), 1.0, places=5)

    def test_standardized_1nn(self):
        tr = torch.tensor([[0.0, 0.0], [10.0, 0.0]]); te = torch.tensor([[1.0, 5.0], [9.0, -3.0]])
        self.assertEqual(flexlbp.standardized_1nn(tr, torch.tensor([3, 7]), te).tolist(), [3, 7])

    def test_shrinkage_lda(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        g = torch.Generator().manual_seed(0)
        tr = torch.cat([torch.randn(20, 5, generator=g), torch.randn(20, 5, generator=g) + 4.0])
        te = torch.tensor([[0.0] * 5, [4.0] * 5])
        self.assertEqual(flexlbp.shrinkage_lda(tr, torch.tensor([3] * 20 + [7] * 20), te).tolist(), [3, 7])

    def test_fit(self):
        g = torch.Generator().manual_seed(1)
        stripes = 0.5 + 0.3 * torch.sin(1.3 * torch.arange(32.0)).view(1, 1, 1, 32) + 0.05 * torch.rand(10, 3, 32, 32, generator=g)
        x = torch.cat([stripes, torch.rand(10, 3, 32, 32, generator=g)]).clamp(0, 1); y = torch.tensor([0] * 10 + [1] * 10)
        m = flexlbp.FLEXLBP(learn="geometry")
        r0, w0 = m.branches["coarse"].radii().detach().clone(), m.color["fine"].weight.detach().clone()
        m, history = fit(m, x, y, epochs=2, batch_size=4, device="cpu", verbose=False)
        self.assertEqual(len(history), 2)
        self.assertFalse(torch.allclose(m.branches["coarse"].radii().detach(), r0))   # learned
        self.assertTrue(torch.equal(m.color["fine"].weight.detach(), w0))              # frozen


if __name__ == "__main__":
    unittest.main()
