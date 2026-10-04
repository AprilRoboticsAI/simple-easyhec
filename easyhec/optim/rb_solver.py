import os.path as osp
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import trimesh

from easyhec.optim.nvdiffrast_renderer import NVDiffrastRenderer
from easyhec.utils import utils_3d


@dataclass
class RBSolverConfig:
    camera_height: int
    camera_width: int
    robot_masks: torch.Tensor
    link_poses_dataset: torch.Tensor
    meshes: List[Union[str, trimesh.Trimesh]]
    initial_extrinsic_guess: torch.Tensor
    occluder_mesh: Union[trimesh.Trimesh, None] = None  # fixed camera coordinates


class RBSolver(nn.Module):
    """Rendering based solver for inverse rendering based extrinsic prediction"""

    def __init__(self, cfg: RBSolverConfig):
        super().__init__()
        self.cfg = cfg
        meshes = self.cfg.meshes
        for link_idx, mesh in enumerate(meshes):
            if isinstance(mesh, str):
                mesh = trimesh.load(osp.expanduser(mesh), force="mesh")
            else:
                assert isinstance(mesh, trimesh.Trimesh)
            vertices = torch.from_numpy(mesh.vertices).float()
            faces = torch.from_numpy(mesh.faces).int()
            self.register_buffer(f"vertices_{link_idx}", vertices)
            self.register_buffer(f"faces_{link_idx}", faces)
        self.nlinks = len(meshes)
        self.register_buffer('occluder_vertices', None if cfg.occluder_mesh is None else
                             torch.as_tensor(cfg.occluder_mesh.vertices.copy(), dtype=torch.float32))
        self.register_buffer('occluder_faces', None if cfg.occluder_mesh is None else
                             torch.as_tensor(cfg.occluder_mesh.faces.copy(), dtype=torch.int32))
        # camera parameters
        init_Tc_c2b = self.cfg.initial_extrinsic_guess
        init_dof = utils_3d.se3_log_map(
            torch.as_tensor(init_Tc_c2b, dtype=torch.float32)[None].permute(0, 2, 1),
            eps=1e-5,
            backend="opencv",
        )[0]
        self.dof = nn.Parameter(init_dof, requires_grad=True)
        # setup renderer
        self.H, self.W = self.cfg.camera_height, self.cfg.camera_width
        self.renderer = NVDiffrastRenderer(self.H, self.W)
        self.register_buffer(f"history_ops", torch.zeros(10000, 6))
        self._history_count = 0

    def forward(self, data):
        # An optimization iteration may evaluate many rendering chunks. Record
        # it once, without scanning the GPU buffer or overflowing on large sets.
        if data.get('record_history', True) and self._history_count < len(self.history_ops):
            self.history_ops[self._history_count] = self.dof.detach()
            self._history_count += 1
        Tc_c2b = utils_3d.se3_exp_map(self.dof[None]).permute(0, 2, 1)[0]
        losses = []
        all_frame_all_link_si = []
        masks_ref = data["mask"]
        link_poses = data["link_poses"]
        intrinsic = data["intrinsic"]
        mount_poses = data["mount_poses"] if "mount_poses" in data else None
        assert link_poses.shape[0] == masks_ref.shape[0]
        assert link_poses.shape[1:] == (self.nlinks, 4, 4)
        assert masks_ref.shape[1:] == (self.H, self.W)

        batch_size = masks_ref.shape[0]
        for bid in range(batch_size):
            all_link_si = []
            for link_idx in range(self.nlinks):
                if mount_poses is not None:
                    Tc_c2l = Tc_c2b @ mount_poses[bid] @ link_poses[bid, link_idx]
                else:
                    Tc_c2l = Tc_c2b @ link_poses[bid, link_idx]
                verts, faces = getattr(self, f"vertices_{link_idx}"), getattr(
                    self, f"faces_{link_idx}"
                )
                occluder = ((self.occluder_vertices, self.occluder_faces)
                            if self.occluder_vertices is not None else None)
                si = self.renderer.render_mask(verts, faces, intrinsic, Tc_c2l, occluder=occluder)
                all_link_si.append(si)
            if len(all_link_si) == 1:
                all_link_si = all_link_si[0].reshape(1, self.H, self.W)
            else:
                all_link_si = torch.stack(all_link_si)
            all_link_si = all_link_si.sum(0).clamp(max=1)
            all_frame_all_link_si.append(all_link_si)
            loss = torch.sum((all_link_si - masks_ref[bid].float()) ** 2)
            losses.append(loss)
        loss = torch.stack(losses).mean()
        all_frame_all_link_si = torch.stack(all_frame_all_link_si)
        
        # metrics
        output = {
            "rendered_masks": all_frame_all_link_si,
            "ref_masks": masks_ref,
            "error_maps": (all_frame_all_link_si - masks_ref.float()).abs(),
        }

        if "gt_camera_pose" in data:
            gt_Tc_c2b = data["gt_camera_pose"]
            if not torch.allclose(gt_Tc_c2b, torch.eye(4).to(gt_Tc_c2b.device)):
                gt_dof6 = utils_3d.se3_log_map(
                    gt_Tc_c2b[None].permute(0, 2, 1), backend="opencv"
                )[0]
                trans_err = ((gt_dof6[:3] - self.dof[:3]) * 100).abs()
                rot_err = (gt_dof6[3:] - self.dof[3:]).abs().max() / np.pi * 180

                metrics = {
                    "err_x": trans_err[0].item(),
                    "err_y": trans_err[1].item(),
                    "err_z": trans_err[2].item(),
                    "err_trans": trans_err.norm().item(),
                    "err_rot": rot_err.item(),
                }
                output["metrics"] = metrics
        output["mask_loss"] = loss
        return output

    def get_predicted_extrinsic(self):
        return utils_3d.se3_exp_map(self.dof[None].detach()).permute(0, 2, 1)[0]
