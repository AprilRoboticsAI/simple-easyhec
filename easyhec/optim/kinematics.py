"""Differentiable body poses from a compiled MuJoCo model, using PyTorch Kinematics."""
import numpy as np
import torch
import pytorch_kinematics as pk


class MujocoKinematics:
    """Fixed-base hinge/slide trees; q uses MuJoCo's absolute joint coordinates.

    MuJoCo supplies model constants only. All pose evaluation and gradients run
    in PyTorch. Splitting each moving body at its joint pivot preserves nonzero
    joint positions, unlike treating the pivot as the body's origin.
    """

    def __init__(self, model, body_names, *, device='cpu', dtype=torch.float32):
        import mujoco

        self.body_names = list(body_names)
        frames = {}

        def transform(pos, quat=None):
            matrix = np.eye(4)
            matrix[:3, 3] = pos
            if quat is not None:
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, quat)
                matrix[:3, :3] = rotation.reshape(3, 3)
            return pk.Transform3d(matrix=torch.as_tensor(matrix, dtype=dtype), dtype=dtype)

        for bid in range(model.nbody):
            name = model.body(bid).name or f'body_{bid}'
            count = int(model.body_jntnum[bid])
            if count > 1 or model.body_mocapid[bid] >= 0:
                raise ValueError('Differentiable FK requires fixed-base bodies with at most one hinge/slide joint')
            offset = transform(model.body_pos[bid], model.body_quat[bid])
            body = pk.Frame(name, link=pk.Link(name, offset=offset))
            attach = body
            if count:
                jid = int(model.body_jntadr[bid])
                kind = {int(mujoco.mjtJoint.mjJNT_HINGE): 'revolute',
                        int(mujoco.mjtJoint.mjJNT_SLIDE): 'prismatic'}.get(int(model.jnt_type[jid]))
                if kind is None:
                    raise ValueError('Floating and ball joints are not supported by differentiable FK')
                pivot = transform(model.jnt_pos[jid])
                joint = pk.Joint(model.joint(jid).name, joint_type=kind,
                                 axis=model.jnt_axis[jid], dtype=dtype)
                attach = pk.Frame(f'__joint_{jid}',
                                  link=pk.Link(offset=offset.compose(pivot)), joint=joint)
                body = pk.Frame(name, link=pk.Link(name, offset=transform(-model.jnt_pos[jid])))
                attach.children.append(body)
            frames[bid] = body
            if bid:
                frames[int(model.body_parentid[bid])].children.append(attach)
        self.chain = pk.Chain(frames[0], dtype=dtype).to(dtype=dtype, device=device)
        self.joint_names = self.chain.get_joint_parameter_names()
        self.qpos_indices = [int(model.joint(name).qposadr[0]) for name in self.joint_names]
        self.reference = torch.as_tensor(model.qpos0[self.qpos_indices], dtype=dtype, device=device)
        self.frame_indices = [self.chain.frame_to_idx[name] for name in self.body_names]

    def __call__(self, joints):
        # Use ordinary autograd through the tensor FK, including rotation entries.
        poses = self.chain.forward_kinematics_tensor(joints - self.reference, analytical_grad=False)
        return poses[self.frame_indices].permute(1, 0, 2, 3)
