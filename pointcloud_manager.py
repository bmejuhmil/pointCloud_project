import copy
import open3d as o3d
import numpy as np
from typing import Optional

class PointCloudManager:
    def __init__(self) -> None:
        self.target_orig: Optional[o3d.geometry.PointCloud] = None
        self.source_orig: Optional[o3d.geometry.PointCloud] = None
        self.target_work: Optional[o3d.geometry.PointCloud] = None
        self.source_work: Optional[o3d.geometry.PointCloud] = None
        self.current_transformation: np.ndarray = np.identity(4)

    def load_target(self, filepath: str) -> o3d.geometry.PointCloud:
        self.target_orig = o3d.io.read_point_cloud(filepath)
        self.target_work = copy.deepcopy(self.target_orig)
        self.target_work.paint_uniform_color([0.0, 0.651, 0.929])
        return self.target_work

    def load_source(self, filepath: str) -> o3d.geometry.PointCloud:
        self.source_orig = o3d.io.read_point_cloud(filepath)
        self.source_work = copy.deepcopy(self.source_orig)
        self.source_work.paint_uniform_color([1.0, 0.706, 0.0])
        self.current_transformation = np.identity(4)
        return self.source_work

    def apply_transformation(self, trans_matrix: np.ndarray) -> None:
        self.current_transformation = trans_matrix @ self.current_transformation
        if self.source_work:
            self.source_work.transform(trans_matrix)

    def reset_source(self) -> Optional[o3d.geometry.PointCloud]:
        if self.source_orig:
            self.source_work = copy.deepcopy(self.source_orig)
            self.source_work.paint_uniform_color([1.0, 0.706, 0.0])
            self.current_transformation = np.identity(4)
            return self.source_work
        return None