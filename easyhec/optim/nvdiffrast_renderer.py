import nvdiffrast.torch as dr
import torch

from easyhec.utils import utils_3d


class NVDiffrastRenderer:
    def __init__(self, height: int, width: int):
        self.H, self.W = height, width
        self.resolution = (height, width)
        blender2opencv = (
            torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            .float()
            .cuda()
        )
        self.opencv2blender = torch.inverse(blender2opencv)
        self.glctx = dr.RasterizeCudaContext()

    def render_mask(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        intrinsic: torch.Tensor,
        object_pose: torch.Tensor,
        anti_aliasing: bool = True,
        occluder=None,
    ) -> torch.Tensor:
        """
        Differentiable rendering of given vertices and faces given the pose of the object and intrinsics of the camera

        Parameters:
            verts (torch.Tensor, shape (N, 3)): vertices of the object
            faces (torch.Tensor, shape (M, 3)): faces of the object
            intrinsic (torch.Tensor, shape (3, 3)): intrinsic matrix of the camera
            object_pose (torch.Tensor, shape (4, 4)): pose of the object
            anti_aliasing (bool): Default is True. If True, will use antialiasing.
            occluder: Optional (vertices, faces) tensors for a fixed opaque mesh
                in OpenCV camera coordinates (metres). It affects visibility,
                but contributes zero to the returned robot silhouette.

        """
        proj = utils_3d.K_to_projection(intrinsic, self.H, self.W)
        pose = self.opencv2blender @ object_pose
        pos_clip = utils_3d.transform_pos(proj @ pose, verts)
        # Occluder vertices are fixed in OpenCV camera coordinates, not in the
        # robot frame. Zero-valued attributes hide robot surfaces behind them;
        # joint rasterization preserves antialiasing/visibility gradients.
        colors = torch.ones((len(verts), 1), dtype=verts.dtype, device=verts.device)
        if occluder is not None:
            occ_vertices, occ_faces = occluder
            occ_clip = utils_3d.transform_pos(proj @ self.opencv2blender, occ_vertices)
            pos_clip = torch.cat((pos_clip, occ_clip), dim=1)
            faces = torch.cat((faces, occ_faces + len(verts)), dim=0)
            colors = torch.cat((colors, colors.new_zeros((len(occ_vertices), 1))))
        rast_out, _ = dr.rasterize(
            self.glctx, pos_clip, faces, resolution=self.resolution
        )
        if anti_aliasing:
            color, _ = dr.interpolate(colors[None, ...], rast_out, faces)
            color = dr.antialias(color, rast_out, pos_clip, faces)
            mask = color[0, :, :, 0]
        else:
            color, _ = dr.interpolate(colors[None, ...], rast_out, faces)
            mask = color[0, :, :, 0] > .5
        mask = torch.flip(mask, dims=[0])
        return mask
