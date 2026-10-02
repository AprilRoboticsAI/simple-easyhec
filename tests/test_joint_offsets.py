"""Independent MuJoCo FK/gradient checks and optional GPU offset recovery."""
import os
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import torch

HAS_FK = all(importlib.util.find_spec(name) is not None for name in ('mujoco', 'pytorch_kinematics'))
if HAS_FK:
    import mujoco
    from easyhec.optim.kinematics import MujocoKinematics


@unittest.skipUnless(HAS_FK, 'requires easyhec[joint-offsets]')
class KinematicsTests(unittest.TestCase):
    def test_camera_optimizer_saves_the_pose_that_produced_the_best_loss(self):
        from easyhec.optim.optimize import optimize

        class ToySolver(torch.nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.x = torch.nn.Parameter(torch.tensor(1.))

            def forward(self, data):
                return {'mask_loss': self.x.square()}

            def get_predicted_extrinsic(self):
                result = torch.eye(4)
                result[0, 3] = self.x.detach()
                return result

        initial = torch.eye(4); initial[0, 3] = 1
        with patch('easyhec.optim.optimize.RBSolver', ToySolver):
            fitted = optimize(initial, torch.eye(3), torch.zeros(1, 2, 2),
                              torch.eye(4)[None, None], [], 2, 2,
                              learning_rate=3., iterations=2, verbose=False)
        # Adam's first step overshoots to -2; the evaluated best remains +1.
        self.assertAlmostEqual(float(fitted[0, 3]), 1.)

    def test_nonzero_pivots_reference_angles_branches_and_gradients(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
          <worldbody><body name="root" pos=".1 .2 .3" euler=".2 -.3 .1">
            <body name="hinge" pos=".2 0 .1" euler=".1 .4 -.2">
              <joint name="turn" axis="0 1 0" pos=".1 .02 -.1" ref=".3"/>
              <geom size=".1"/>
              <body name="slide" pos=".3 .1 0" euler=".2 .1 .3">
                <joint name="slide" type="slide" axis="1 0 0" ref=".02"/>
                <geom size=".1"/>
              </body>
            </body>
            <body name="fixed" pos=".3 -.2 .1"/>
          </body></worldbody></mujoco>''')
        names = ['root', 'hinge', 'slide', 'fixed']
        fk = MujocoKinematics(model, names, dtype=torch.float64)
        q = torch.tensor([[.15, -.04], [-.3, .1]], dtype=torch.float64, requires_grad=True)
        predicted = fk(q).detach().numpy()
        for i, row in enumerate(q.detach().numpy()):
            data = mujoco.MjData(model)
            data.qpos[fk.qpos_indices] = row
            mujoco.mj_forward(model, data)
            for j, name in enumerate(names):
                np.testing.assert_allclose(predicted[i, j, :3, :3], data.body(name).xmat.reshape(3, 3), atol=1e-10)
                np.testing.assert_allclose(predicted[i, j, :3, 3], data.body(name).xpos, atol=1e-10)
        self.assertTrue(torch.autograd.gradcheck(fk, (q,), eps=1e-6, atol=1e-5))

    @unittest.skipUnless(os.environ.get('RUN_EASYHEC_GPU') == '1', 'opt-in CUDA recovery')
    def test_recovers_offset_from_independent_mujoco_masks(self):
        import trimesh
        from easyhec.optim.joint_offsets import optimize_joint_offsets

        model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
          <visual><global offwidth="160" offheight="120"/></visual>
          <worldbody><camera name="cam" pos="0 0 2" fovy="45"/>
            <body name="link"><joint name="turn" axis="0 0 1"/>
              <geom type="box" size=".35 .075 .06" pos=".35 0 0"/>
            </body></worldbody></mujoco>''')
        measured = np.array([-.55, -.1, .35, .7])
        truth = .07
        data = mujoco.MjData(model)
        masks = []
        with mujoco.Renderer(model, height=120, width=160) as renderer:
            for q in measured:
                data.qpos[:] = q + truth
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera='cam')
                renderer.enable_segmentation_rendering()
                masks.append((renderer.render()[:, :, 0] == 0).astype(np.float32))
        f = 60 / np.tan(np.deg2rad(45)/2)
        K = [[f, 0, 80], [0, f, 60], [0, 0, 1]]
        T = np.diag([1., -1., -1., 1.]); T[2, 3] = 2.
        mesh = trimesh.creation.box(extents=[.7, .15, .12]); mesh.apply_translation([.35, 0, 0])
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device='cuda')
        fk = MujocoKinematics(model, ['link'], device='cuda')
        result = optimize_joint_offsets(tensor(T), tensor(K), tensor(np.array(masks)),
            tensor(measured[:, None]), fk, [mesh], 160, 120, [0], fix_camera=True,
            iterations=200, offset_learning_rate=.003, prior_weight=0, verbose=False)
        self.assertLess(abs(float(result['offsets'][0]) - truth), .01)
        np.testing.assert_allclose(result['T_cam_base'].cpu(), T, atol=1e-5)
        self.assertLess(result['mask_mse'], .005)

    @unittest.skipUnless(os.environ.get('RUN_EASYHEC_GPU') == '1', 'opt-in CUDA recovery')
    def test_joint_camera_recovery_with_fixed_base_anchor(self):
        import trimesh
        from easyhec.optim.joint_offsets import optimize_joint_offsets

        model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
          <visual><global offwidth="160" offheight="120"/></visual>
          <worldbody><camera name="cam" pos="0 0 2" fovy="45"/>
            <body name="anchor">
              <geom type="box" size=".12 .2 .08" pos="-.45 -.2 .1"/>
              <geom type="box" size=".2 .08 .15" pos=".25 -.4 .15"/>
            </body>
            <body name="link"><joint name="turn" axis="0 0 1"/>
              <geom type="box" size=".3 .06 .06" pos=".3 .1 0"/>
            </body></worldbody></mujoco>''')
        measured = np.array([-.5, -.15, .3, .7, 1.])
        truth = .06
        data = mujoco.MjData(model)
        masks = []
        with mujoco.Renderer(model, height=120, width=160) as renderer:
            for q in measured:
                data.qpos[:] = q + truth
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera='cam')
                renderer.enable_segmentation_rendering()
                masks.append((renderer.render()[:, :, 0] >= 0).astype(np.float32))
        meshes = []
        for names in ([0, 1], [2]):
            pieces = []
            for gid in names:
                mesh = trimesh.creation.box(extents=2*model.geom_size[gid])
                mesh.apply_translation(model.geom_pos[gid])
                pieces.append(mesh)
            meshes.append(trimesh.util.concatenate(pieces))
        f = 60 / np.tan(np.deg2rad(45)/2)
        K = [[f, 0, 80], [0, f, 60], [0, 0, 1]]
        T = np.diag([1., -1., -1., 1.]); T[2, 3] = 2.
        initial = T.copy(); initial[:3, 3] += [.015, -.01, .02]
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device='cuda')
        fk = MujocoKinematics(model, ['anchor', 'link'], device='cuda')
        result = optimize_joint_offsets(tensor(initial), tensor(K), tensor(np.array(masks)[:4]),
            tensor(measured[:4, None]), fk, meshes, 160, 120, [0], iterations=400,
            learning_rate=.001, offset_learning_rate=.001, prior_weight=1e-4, verbose=False)
        self.assertLess(abs(float(result['offsets'][0]) - truth), .015)
        fitted = result['T_cam_base'].cpu().numpy()
        self.assertLess(np.linalg.norm(fitted[:3, 3] - T[:3, 3]), .02)
        self.assertLess(result['mask_mse'], .006)
        # The unseen fifth pose must also agree with the independent renderer.
        from easyhec.optim.nvdiffrast_renderer import NVDiffrastRenderer
        renderer = NVDiffrastRenderer(120, 160)
        with torch.no_grad():
            poses = fk(tensor(measured[4:, None]) + result['offsets'])
            rendered = torch.stack([renderer.render_mask(tensor(mesh.vertices),
                torch.as_tensor(mesh.faces, dtype=torch.int32, device='cuda'), tensor(K),
                result['T_cam_base'] @ poses[0, i]) for i, mesh in enumerate(meshes)]).sum(0).clamp(max=1)
        a = rendered.cpu().numpy() > .5; b = masks[4] > .5
        self.assertGreater(np.logical_and(a, b).sum()/np.logical_or(a, b).sum(), .95)


if __name__ == '__main__':
    unittest.main()
