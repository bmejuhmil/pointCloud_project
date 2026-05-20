import os
import copy
import csv
import time
import glob
from datetime import datetime
import numpy as np
import open3d as o3d

import registration_algos as algos
import util
from ai_wrappers import AIRegistrationModel

class AutomatedTestbench:
    def __init__(self, dataset_dir: str):
        print("Initializing Testbench...")
        self.dataset_dir = dataset_dir
        
        print("Loading AI Models...")
        self.dcp_model = AIRegistrationModel("DCP", "pretrained/exp_dcp/models/best_model.t7")
        self.prnet_model = AIRegistrationModel("PRNET", "pretrained/exp_prnet/models/best_model.t7")

        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Test parameters
        self.noise_levels = [0.0, 0.01, 0.05] 
        self.crop_ratios = [1.0, 0.4, 0.1]
        self.sparsity_levels = [1]     
        self.num_trials_per_scenario = 10

    def _get_ply_files(self) -> list:
        """Recursively finds all .ply files in the dataset directory."""
        search_pattern = os.path.join(self.dataset_dir, "**", "*.ply")
        return glob.glob(search_pattern, recursive=True)

    def _apply_random_transform(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """Applies a random rigid transformation and returns the ground truth matrix."""
        tx, ty, tz = np.random.uniform(-40, 40, 3)
        rx, ry, rz = np.random.uniform(-40.0, 40.0, 3) # Max degree rotation on each axis
        
        R = pcd.get_rotation_matrix_from_xyz((np.radians(rx), np.radians(ry), np.radians(rz)))
        trans = np.identity(4)
        trans[:3, :3] = R
        trans[:3, 3] = [tx, ty, tz]
        
        pcd.transform(trans)
        return trans

    def _apply_noise(self, pcd: o3d.geometry.PointCloud, std_dev: float):
        if std_dev <= 0.0: return
        pts = np.asarray(pcd.points)
        noise = np.random.normal(0, std_dev, pts.shape)
        pcd.points = o3d.utility.Vector3dVector(pts + noise)

    def _apply_crop(self, pcd: o3d.geometry.PointCloud, ratio: float):
        if ratio >= 1.0: return
        pts = np.asarray(pcd.points)
        min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
        threshold = min_x + ratio * (max_x - min_x)
        mask = pts[:, 0] < threshold
        
        pcd.points = o3d.utility.Vector3dVector(pts[mask])
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            pcd.colors = o3d.utility.Vector3dVector(colors[mask])

    def _apply_sparsity(self, pcd: o3d.geometry.PointCloud, ratio: float):
        if ratio >= 1.0: return
        
        # Using Open3D's native downsampling
        downsampled = pcd.random_down_sample(sampling_ratio=ratio)
        
        # Overwrite the original pcd points
        pcd.points = downsampled.points
        if downsampled.has_colors():
            pcd.colors = downsampled.colors
        if downsampled.has_normals():
            pcd.normals = downsampled.normals

    def run_suite(self):
        ply_files = self._get_ply_files()
        if not ply_files:
            print(f"No .ply files found in {self.dataset_dir}!")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.output_dir, f"automated_testbench_{timestamp}.csv")
        
        # Calculate total runs for progress tracking
        runs_per_file = len(self.noise_levels) * len(self.crop_ratios) * len(self.sparsity_levels) * self.num_trials_per_scenario * 6
        total_runs = len(ply_files) * runs_per_file
        current_run = 0

        print(f"\nFound {len(ply_files)} models. Total algorithm executions planned: {total_runs}")
        
        # Open CSV to stream results in case of a crash
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Model_Name and Num_Points to the headers
            writer.writerow(["Model_Name", "Num_Points", "Trial_ID", "Noise_m", "Crop_Ratio", "Algorithm", 
                             "Init_Trans_m", "Init_Rot_deg", 
                             "Time_sec", "Fitness", "RMSE", "Final_Trans_Err_m", "Final_Rot_Err_deg"])

            trial_id = 1
            
            # --- Iterate through all found files ---
            for filepath in ply_files:
                base_model_name = os.path.basename(filepath)
                
                print(f"\n--- Loading Model: {base_model_name} ---")
                try:
                    base_pcd = o3d.io.read_point_cloud(filepath)
                    if not base_pcd.has_points():
                        print(f"Skipping {base_model_name}: No points found.")
                        continue
                except Exception as e:
                    print(f"Failed to load {base_model_name}: {e}")
                    continue

                for sparsity in self.sparsity_levels:

                    sparse_base_pcd = copy.deepcopy(base_pcd)
                    self._apply_sparsity(sparse_base_pcd, sparsity)
                    
                    current_num_points = len(sparse_base_pcd.points)
                    current_model_name = f"{base_model_name}_{sparsity}x"

                    for noise in self.noise_levels:
                        for crop in self.crop_ratios:
                            for _ in range(self.num_trials_per_scenario):
                                
                                # 1. Prepare fresh target and source from the current base model
                                target = copy.deepcopy(sparse_base_pcd)
                                source = copy.deepcopy(sparse_base_pcd)

                                # 2. Apply Ground Truth Perturbation to Source
                                gt_transform = self._apply_random_transform(source)
                                init_t_dist, init_r_dist = util.compute_registration_error(gt_transform)
                                
                                # 3. Apply Scenario Degradations to Source
                                self._apply_crop(source, crop)
                                self._apply_noise(source, noise)
                                
                                # Recalculate normals after modifications
                                source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                                target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

                                # 4. Define Algorithms
                                dyn_voxel = util.get_dynamic_voxel_size(source, fraction=0.05)
                                dyn_icp_thresh = dyn_voxel * 2.0 
                                
                                # Use a slightly finer voxel size than global methods
                                icp_voxel = dyn_voxel * 0.75 
                                if len(source.points) > 5_000_000:
                                    icp_source = source.voxel_down_sample(icp_voxel)
                                    icp_target = target.voxel_down_sample(icp_voxel)
                                else:
                                    icp_source = source
                                    icp_target = target

                                algo_dict = {
                                    "RANSAC": lambda: algos.run_ransac(copy.deepcopy(source), target, dyn_voxel),
                                    "FGR": lambda: algos.run_fgr(copy.deepcopy(source), target, dyn_voxel),
                                    "DCP (AI)": lambda: algos.run_ai_model(copy.deepcopy(source), target, self.dcp_model),
                                    "PRNet (AI)": lambda: algos.run_ai_model(copy.deepcopy(source), target, self.prnet_model),
                                    "ICP (Pt2Pt)": lambda: algos.run_icp(copy.deepcopy(icp_source), icp_target, dyn_icp_thresh, 2000),
                                    "ICP (Pt2Pl)": lambda: algos.run_icp_point_to_plane(copy.deepcopy(icp_source), icp_target, dyn_icp_thresh, 2000)
                                }

                                # 5. Execute each algorithm
                                for algo_name, algo_func in algo_dict.items():
                                    current_run += 1
                                    print(f"[{current_run}/{total_runs}] {current_model_name} ({current_num_points} pts) | {algo_name} ...", end="", flush=True)
                                    
                                    # Run algorithm
                                    est_trans, fitness, rmse, t_sec = algo_func()
                                    
                                    # Calculate absolute error against Ground Truth
                                    error_matrix = est_trans @ gt_transform
                                    t_err, r_err = util.compute_registration_error(error_matrix)
                                    
                                    # Write immediately to CSV using the updated name and point count
                                    writer.writerow([current_model_name, current_num_points, trial_id, noise, crop, algo_name, 
                                                     f"{init_t_dist:.5f}", f"{init_r_dist:.4f}",
                                                     f"{t_sec:.4f}", f"{fitness:.4f}", f"{rmse:.5f}", 
                                                     f"{t_err:.5f}", f"{r_err:.4f}"])
                                    print(" Done.")
                                    
                                trial_id += 1
                            
        print(f"\nTestbench Complete! Data saved to: {csv_path}")

if __name__ == "__main__":
    # Update this to the base folder containing all your subfolders!
    DATASET_DIRECTORY = "pointClouds" 
    
    try:
        bench = AutomatedTestbench(DATASET_DIRECTORY)
        bench.run_suite()
    except Exception as e:
        print(f"Testbench Failed: {e}")
