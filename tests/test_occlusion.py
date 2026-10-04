"""Occluders hide only geometry behind their finite camera-space footprint."""
import os
import unittest

import numpy as np
import torch
import trimesh


@unittest.skipUnless(os.environ.get('RUN_EASYHEC_GPU') == '1', 'opt-in CUDA rendering')
class OcclusionTests(unittest.TestCase):
    def test_depth_order_finite_extent_and_visibility_gradient(self):
        from easyhec.optim.nvdiffrast_renderer import NVDiffrastRenderer

        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device='cuda')
        f = lambda x: torch.as_tensor(x, dtype=torch.int32, device='cuda')
        renderer = NVDiffrastRenderer(100, 100)
        K = t([[80, 0, 50], [0, 80, 50], [0, 0, 1]])
        mesh = trimesh.creation.box(extents=[.5, .5, .1])
        verts, faces = t(mesh.vertices), f(mesh.faces)
        # Occluder only occupies the right half of the image.
        occluder = (t([[0, -.6, 1], [.6, -.6, 1], [.6, .6, 1], [0, .6, 1]]),
                    f([[0, 1, 2], [0, 2, 3]]))
        def render(x, z, occ=occluder):
            T = torch.eye(4, device='cuda')
            T[0, 3], T[2, 3] = x, z
            return renderer.render_mask(verts, faces, K, T, occluder=occ)

        behind = render(0., 2.)
        self.assertEqual(float(behind[45:55, 53:58].sum()), 0)
        self.assertEqual(float(behind[45:55, 42:47].sum()), 50)
        np.testing.assert_allclose(render(0., .6).cpu(), render(0., .6, None).cpu(), atol=1e-6)
        # A rightward move behind the occluder reduces visible robot area.
        x = torch.tensor(.013, device='cuda', requires_grad=True)
        render(x, 2.).sum().backward()
        self.assertTrue(torch.isfinite(x.grad))
        self.assertLess(float(x.grad), -100)
        eps = .0005
        finite_difference = float((render(float(x.detach())+eps, 2.).sum() -
                                   render(float(x.detach())-eps, 2.).sum()) / (2*eps))
        # nvdiffrast uses approximate visibility gradients; verify the physical
        # direction against a finite perturbation, not exact gradient magnitude.
        self.assertLess(finite_difference, -100)
        self.assertLess(float(render(float(x.detach()) + .005, 2.).sum()),
                        float(render(float(x.detach()), 2.).sum()))


if __name__ == '__main__':
    unittest.main()
