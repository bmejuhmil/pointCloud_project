import torch
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple
import open3d as o3d

def get_dynamic_voxel_size(pcd: o3d.geometry.PointCloud, fraction: float = 0.05) -> float:
    if not pcd.has_points():
        return 0.05 # Fallback if empty
        
    bbox = pcd.get_axis_aligned_bounding_box()
    extents = bbox.get_extent() # Returns [x_length, y_length, z_length]
    max_dimension = max(extents)
    
    return float(max_dimension * fraction)

def quat2mat(quat: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
        2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
        2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2
    ], dim=1).reshape(B, 3, 3)
    return rotMat

def transform_point_cloud(point_cloud: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)

def npmat2euler(mats: np.ndarray, seq: str = 'zyx') -> np.ndarray:
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')

def compute_registration_error(current_transformation: np.ndarray) -> Tuple[float, float]:
    R = current_transformation[:3, :3]
    t = current_transformation[:3, 3]
    t_error = float(np.linalg.norm(t))
    trace_R = np.trace(R)
    cos_theta = np.clip((trace_R - 1.0) / 2.0, -1.0, 1.0)
    r_error_rad = np.arccos(cos_theta)
    r_error_deg = float(np.degrees(r_error_rad))
    return t_error, r_error_deg