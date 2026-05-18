import time
import numpy as np
import open3d as o3d
from typing import Tuple
from ai_wrappers import AIRegistrationModel

AlgoResult = Tuple[np.ndarray, float, float, float]

def run_icp(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, threshold: float, max_iter: int, init_trans: np.ndarray = None) -> AlgoResult:
    if init_trans is None:
        init_trans = np.identity(4)
        
    start_time = time.time()
    try:
        result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, init_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iter))
        )
        return result.transformation, result.fitness, result.inlier_rmse, time.time() - start_time
    except Exception as e:
        print(f"ICP failed: {e}")
        return init_trans, 0.0, 0.0, time.time() - start_time

def run_icp_point_to_plane(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, threshold: float, max_iter: int, init_trans: np.ndarray = None) -> AlgoResult:
    if init_trans is None:
        init_trans = np.identity(4)
        
    start_time = time.time()
    try:
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))
        result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, init_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iter))
        )
        return result.transformation, result.fitness, result.inlier_rmse, time.time() - start_time
    except Exception as e:
        print(f"ICP (Point-to-Plane) failed: {e}")
        return init_trans, 0.0, 0.0, time.time() - start_time

def _preprocess_for_global(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, voxel_size: float):
    s_down = source.voxel_down_sample(voxel_size)
    t_down = target.voxel_down_sample(voxel_size)
    s_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    t_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    s_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        s_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    t_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        t_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return s_down, t_down, s_fpfh, t_fpfh

def run_ransac(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, voxel_size: float) -> AlgoResult:
    start_time = time.time()
    try:
        s_down, t_down, s_fpfh, t_fpfh = _preprocess_for_global(source, target, voxel_size)
        
        # Safeguard: Ensure we have enough points left after aggressive downsampling
        if len(s_down.points) < 4 or len(t_down.points) < 4:
            return np.identity(4), 0.0, 0.0, time.time() - start_time
            
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            s_down, t_down, s_fpfh, t_fpfh, True, voxel_size * 1.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        return result.transformation, result.fitness, result.inlier_rmse, time.time() - start_time
    except Exception as e:
        print(f"RANSAC error caught gracefully: {e}")
        return np.identity(4), 0.0, 0.0, time.time() - start_time

def run_fgr(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, voxel_size: float) -> AlgoResult:
    start_time = time.time()
    try:
        s_down, t_down, s_fpfh, t_fpfh = _preprocess_for_global(source, target, voxel_size)
        
        # Safeguard: Ensure we have enough points left after aggressive downsampling
        if len(s_down.points) < 4 or len(t_down.points) < 4:
            return np.identity(4), 0.0, 0.0, time.time() - start_time
            
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            s_down, t_down, s_fpfh, t_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=voxel_size * 1.5)
        )
        return result.transformation, result.fitness, result.inlier_rmse, time.time() - start_time
    except Exception as e:
        print(f"FGR error caught gracefully: {e}")
        return np.identity(4), 0.0, 0.0, time.time() - start_time

def run_ai_model(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, ai_model: AIRegistrationModel) -> AlgoResult:
    """Runs a generalized AI model inference (DCP, PRNet)."""
    trans, fit, rmse, t = ai_model.run_inference(source, target)
    return trans, fit, rmse, t