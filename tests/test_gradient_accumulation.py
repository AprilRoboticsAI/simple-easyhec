"""Rendering chunks must preserve the full-data objective and Adam updates."""
import os
import unittest
from unittest.mock import patch

import torch


class ToySolver(torch.nn.Module):
    calls = []

    def __init__(self, cfg):
        super().__init__()
        self.dof = torch.nn.Parameter(cfg.initial_extrinsic_guess[0, 3:4].clone())

    def forward(self, data):
        n = len(data['mask'])
        assert data['intrinsic'].shape == (3, 3)
        if 'gt_camera_pose' in data:
            assert data['gt_camera_pose'].shape == (4, 4)
        prediction = self.dof[0] + torch.sin(data['link_poses'][:, 0, 0, 3])
        if 'mount_poses' in data:
            assert data['mount_poses'].shape == (n, 4, 4)
            prediction = prediction + data['mount_poses'][:, 0, 3]
        type(self).calls.append((n, data.get('record_history', True)))
        loss = ((prediction[:, None, None] - data['mask'])**2).sum((1, 2)).mean()
        return dict(mask_loss=loss)

    def get_predicted_extrinsic(self):
        result = torch.eye(4, dtype=self.dof.dtype)
        result[0, 3] = self.dof.detach()[0]
        return result


def kinematics(q):
    poses = torch.eye(4, dtype=q.dtype).repeat(len(q), 1, 1, 1)
    poses[:, 0, 0, 3] = q[:, 0] + .5 * q[:, 1].square()
    return poses


class GradientAccumulationTests(unittest.TestCase):
    def setUp(self):
        self.initial = torch.eye(4, dtype=torch.float64)
        self.initial[0, 3] = -.3
        self.K = torch.eye(3, dtype=torch.float64)
        self.q = torch.tensor([[-.8, .5], [-.1, -.7], [.6, .2], [.3, .1], [1., -.5]], dtype=torch.float64)
        self.masks = torch.arange(30, dtype=torch.float64).reshape(5, 2, 3) / 10 - 1.2

    def test_camera_updates_match_full_batch_with_uneven_chunks_and_mounts(self):
        from easyhec.optim.optimize import optimize

        results = []
        mounts = torch.eye(4, dtype=torch.float64).repeat(5, 1, 1)
        mounts[:, 0, 3] = torch.arange(5, dtype=torch.float64) * .1
        for size in (5, 2, 1):
            ToySolver.calls = []
            with patch('easyhec.optim.optimize.RBSolver', ToySolver):
                result = optimize(self.initial, self.K, self.masks, kinematics(self.q), [], 3, 2,
                    iterations=15, learning_rate=.05, verbose=False, render_batch_size=size,
                    camera_mount_poses=mounts, gt_camera_pose=torch.eye(4))
            results.append(result)
            self.assertEqual(sum(n for n, _ in ToySolver.calls), 5 * 15)
            self.assertEqual(sum(record for _, record in ToySolver.calls), 15)
            self.assertLessEqual(max(n for n, _ in ToySolver.calls), size)
        for result in results[1:]:
            torch.testing.assert_close(result, results[0], rtol=1e-10, atol=1e-10)

    def test_joint_and_camera_updates_match_full_batch_and_apply_prior_once(self):
        from easyhec.optim.joint_offsets import optimize_joint_offsets

        results = []
        for size in (5, 2, 1):
            ToySolver.calls = []
            with patch('easyhec.optim.joint_offsets.RBSolver', ToySolver):
                result = optimize_joint_offsets(self.initial, self.K, self.masks, self.q, kinematics,
                    [], 3, 2, [0, 1], iterations=20, learning_rate=.04, offset_learning_rate=.02,
                    max_offset=.7, prior_weight=.03, prior_sigma=.2, verbose=False, render_batch_size=size)
            results.append(result)
            self.assertEqual(sum(n for n, _ in ToySolver.calls), 5 * 20)
            self.assertEqual(sum(record for _, record in ToySolver.calls), 20)
            self.assertAlmostEqual(result['prior_loss'], float(.03*(result['offsets']/.2).square().mean()))
        for result in results[1:]:
            torch.testing.assert_close(result['T_cam_base'], results[0]['T_cam_base'], rtol=1e-9, atol=1e-9)
            torch.testing.assert_close(result['offsets'], results[0]['offsets'], rtol=1e-9, atol=1e-9)
            self.assertAlmostEqual(result['objective'], results[0]['objective'], places=10)

    def test_legacy_sample_batch_does_not_slice_shared_intrinsics(self):
        from easyhec.optim.optimize import optimize

        ToySolver.calls = []
        with patch('easyhec.optim.optimize.RBSolver', ToySolver):
            optimize(self.initial, self.K, self.masks, kinematics(self.q), [], 3, 2,
                     iterations=2, batch_size=3, render_batch_size=2, verbose=False)
        self.assertEqual(ToySolver.calls, [(2, True), (1, False)] * 2)

    @unittest.skipUnless(os.environ.get('RUN_EASYHEC_GPU') == '1', 'opt-in CUDA rendering')
    def test_renderer_history_counts_steps_and_safely_stops_at_capacity(self):
        import trimesh
        from easyhec.optim.rb_solver import RBSolver, RBSolverConfig

        pose = torch.eye(4, device='cuda'); pose[2, 3] = 1
        masks = torch.zeros(1, 16, 16, device='cuda')
        links = torch.eye(4, device='cuda')[None, None]
        solver = RBSolver(RBSolverConfig(16, 16, masks, links,
            [trimesh.creation.box(extents=[.1, .1, .1])], pose)).cuda()
        data = dict(intrinsic=torch.tensor([[16., 0, 8], [0, 16, 8], [0, 0, 1]], device='cuda'),
                    link_poses=links, mask=masks, record_history=False)
        solver._history_count = len(solver.history_ops)-1
        with torch.no_grad():
            solver(data)
            self.assertEqual(solver._history_count, 9999)
            data['record_history'] = True
            solver(data)
            solver(data)  # More calls must not overrun the fixed history buffer.
        self.assertEqual(solver._history_count, 10000)


if __name__ == '__main__':
    unittest.main()
