import time
import torch
import numpy as np
import open3d as o3d
from typing import Tuple

try:
    from learning3d.models import DCP, PRNet, DGCNN
except ImportError:
    raise ImportError("Learning3D not found. Please ensure it is installed.")

class AIRegistrationModel:
    """A unified wrapper for Learning3D registration models (DCP, PRNet)."""
    
    def __init__(self, model_type: str, model_path: str = None, num_points: int = 1024):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_points = num_points
        self.model_type = model_type.upper()
        self.is_loaded = False
        self.model = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        try:
            # Force models to use DGCNN with 512 dimensions to match your .t7 weights
            custom_backbone = DGCNN(emb_dims=512)
            
            if self.model_type == "DCP":
                self.model = DCP(feature_model=custom_backbone, pointer_='transformer', head='svd')
            elif self.model_type == "PRNET":
                self.model = PRNet()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            self.model = self.model.to(self.device)
            # strict=False allows us to ignore orphaned batchnorm weights
            self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
            self.model.eval()
            self.is_loaded = True
            print(f"Learning3D: {self.model_type} loaded successfully from {model_path}!")
            
        except Exception as e:
            print(f"Failed to load {self.model_type} model. Error: {e}")

    def sample_point_cloud(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        pts = np.asarray(pcd.points)
        if pts.shape[0] == 0:
            raise ValueError("Point cloud contains zero points.")
        if pts.shape[0] < self.num_points:
            pad_idx = np.random.choice(pts.shape[0], self.num_points - pts.shape[0], replace=True)
            pts = np.vstack((pts, pts[pad_idx, :]))
        idx = np.random.choice(pts.shape[0], self.num_points, replace=False)
        return pts[idx, :]

    def run_inference(self, source_pcd: o3d.geometry.PointCloud, target_pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, float, float, float]:
        start_time = time.time()

        if not self.is_loaded or self.model is None:
            print(f"Warning: {self.model_type} is not loaded. Returning Identity.")
            return np.identity(4), 0.0, 0.0, time.time() - start_time

        try:
            src_pts = self.sample_point_cloud(source_pcd)
            tgt_pts = self.sample_point_cloud(target_pcd)

            # --- 1. ZERO-CENTERING ---
            src_centroid = np.mean(src_pts, axis=0)
            tgt_centroid = np.mean(tgt_pts, axis=0)
            src_pts_centered = src_pts - src_centroid
            tgt_pts_centered = tgt_pts - tgt_centroid

            # --- 2. SCALE NORMALIZATION ---
            global_scale = max(np.max(np.linalg.norm(src_pts_centered, axis=1)), 
                               np.max(np.linalg.norm(tgt_pts_centered, axis=1)))
            if global_scale < 1e-6: global_scale = 1.0

            src_pts_normalized = src_pts_centered / global_scale
            tgt_pts_normalized = tgt_pts_centered / global_scale

            src_tensor = torch.tensor(src_pts_normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
            tgt_tensor = torch.tensor(tgt_pts_normalized, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                res = self.model(src_tensor, tgt_tensor)
                # Handle learning3d's output formats (tuple vs dict)
                if isinstance(res, dict):
                    rot, trans = res['est_R'], res['est_t']
                else:
                    rot, trans = res[0], res[1]

            rot_np = rot.squeeze(0).cpu().numpy()
            trans_np = trans.squeeze(0).cpu().numpy()

            # --- 3. REVERSE SCALE AND CENTERING ---
            trans_np_scaled = trans_np * global_scale
            final_trans = tgt_centroid + trans_np_scaled - (rot_np @ src_centroid)

            trans_matrix = np.identity(4)
            trans_matrix[:3, :3] = rot_np
            trans_matrix[:3, 3] = final_trans

            evaluation = o3d.pipelines.registration.evaluate_registration(
                source_pcd, target_pcd, 0.05, trans_matrix
            )

            return trans_matrix, evaluation.fitness, evaluation.inlier_rmse, time.time() - start_time

        except Exception as e:
            print(f"Inference error in {self.model_type}: {e}")
            return np.identity(4), 0.0, 0.0, time.time() - start_time