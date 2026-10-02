"""Fit additive joint corrections and camera extrinsics to robot silhouettes."""
import math

import torch
from tqdm import tqdm

from easyhec.optim.rb_solver import RBSolver, RBSolverConfig


def optimize_joint_offsets(initial_extrinsic, intrinsic, masks, joint_positions,
                           kinematics, meshes, width, height, offset_indices, *,
                           iterations=1000, learning_rate=.003, offset_learning_rate=.001,
                           max_offset=math.radians(10), prior_sigma=math.radians(5),
                           prior_weight=1e-4, fix_camera=False, early_stopping_steps=200,
                           verbose=True):
    """q_model = q_recorded + delta, shared across frames. Fingers can stay fixed.

    Minimize mean pixel squared error plus prior_weight * mean((delta/sigma)^2).
    Bounds and the prior stabilize weakly observed directions; they do not make
    the underlying problem identifiable. Selection of fixed joints/anchors is
    the caller's responsibility. Nothing is written to a robot configuration.
    """
    indices = list(offset_indices)
    if (not indices or len(set(indices)) != len(indices)
            or min(indices) < 0 or max(indices) >= joint_positions.shape[1]):
        raise ValueError('Select distinct valid joint indices for offset optimization')
    if (not all(math.isfinite(v) for v in [learning_rate, offset_learning_rate, max_offset,
                                         prior_sigma, prior_weight])
            or min(learning_rate, offset_learning_rate, max_offset, prior_sigma) <= 0
            or prior_weight < 0 or not 1 <= iterations < 10000 or early_stopping_steps < 1):
        raise ValueError('Invalid joint-offset optimizer settings')
    initial_poses = kinematics(joint_positions)
    solver = RBSolver(RBSolverConfig(height, width, masks, initial_poses, meshes,
                                    initial_extrinsic)).to(joint_positions.device)
    solver.dof.requires_grad_(not fix_camera)
    delta = torch.nn.Parameter(joint_positions.new_zeros(len(indices)))
    # Constant selector keeps all unselected joints exactly fixed.
    selector = torch.eye(joint_positions.shape[1], device=joint_positions.device,
                         dtype=joint_positions.dtype)[indices]
    parameters = [{'params': [delta], 'lr': offset_learning_rate}]
    if not fix_camera:
        parameters.append({'params': [solver.dof], 'lr': learning_rate})
    optimizer = torch.optim.Adam(parameters)
    best = None
    last_improvement = 0
    progress = tqdm(range(iterations), desc='Camera + joint offsets', disable=not verbose)
    reason = 'iteration_limit'
    try:
        for iteration in progress:
            optimizer.zero_grad()
            poses = kinematics(joint_positions + delta @ selector)
            output = solver(dict(intrinsic=intrinsic, link_poses=poses, mask=masks))
            mse = output['mask_loss'] / (width * height)
            penalty = prior_weight * (delta / prior_sigma).square().mean()
            objective = mse + penalty
            value = float(objective.detach())
            if not math.isfinite(value):
                raise ValueError('Joint-offset optimization produced a nonfinite objective')
            if best is None or value < best['objective']:
                # Save the parameters that produced this loss, before the step.
                best = dict(T_cam_base=solver.get_predicted_extrinsic().clone(),
                            offsets=delta.detach().clone(), objective=value,
                            mask_mse=float(mse.detach()), prior_loss=float(penalty.detach()),
                            best_iteration=iteration)
                last_improvement = iteration
            progress.set_postfix(mse=f'{float(mse.detach()):.5f}', best=f'{best["objective"]:.5f}',
                                 max_deg=f'{float(delta.detach().abs().max()) * 180 / math.pi:.2f}')
            if iteration - last_improvement >= early_stopping_steps:
                reason = 'early_stopping'
                break
            # Avoid stepping to an unevaluated final parameter vector.
            if iteration + 1 < iterations:
                objective.backward()
                if delta.grad is None or not torch.isfinite(delta.grad).all():
                    raise ValueError('Invalid gradient for joint offsets')
                optimizer.step()
                with torch.no_grad():
                    delta.clamp_(-max_offset, max_offset)
    finally:
        progress.close()
    best.update(iterations=iteration + 1, stop_reason=reason)
    return best
