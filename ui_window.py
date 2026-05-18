import threading
import copy
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import csv, os
from datetime import datetime

import util
from pointcloud_manager import PointCloudManager
from ai_wrappers import AIRegistrationModel
import registration_algos as algos

class RegistrationApp:
    def __init__(self):
        self.manager = PointCloudManager()
        
        # 1. INITIALIZE OPEN3D GUI FIRST
        self.window = gui.Application.instance.create_window("Point Cloud Registration Tool", 1280, 800)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0.1, 0.1, 0.1, 1])
        
        self.benchmark_transforms = {}

        try:
            self.scene.scene.view.set_post_processing(False)
        except Exception:
            pass 

        em = self.window.theme.font_size
        self.panel = gui.Vert(int(0.5 * em), gui.Margins(em, em, em, em))
        
        # Setup UI
        self._setup_data_panel(em)
        self._setup_manual_panel(em)
        self._setup_algo_panel(em)
        self._setup_benchmark_panel(em)

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene)
        self.window.add_child(self.panel)

        # 2. LOAD AI MODELS LAST (Prevents OpenGL Context Crashes)
        print("Initializing AI Models... this may take a moment.")
        # Ensure these filenames match exactly what you downloaded to your folder!
        self.dcp_model = AIRegistrationModel(model_type="DCP", model_path="pretrained/exp_dcp/models/best_model.t7")
        self.prnet_model = AIRegistrationModel(model_type="PRNET", model_path="pretrained/exp_prnet/models/best_model.t7")

    def _setup_data_panel(self, em):
        self.panel.add_child(gui.Label("Load Data"))
        btn_load_target = gui.Button("Load Full Scan (Target)")
        btn_load_target.set_on_clicked(self._on_load_target)
        self.panel.add_child(btn_load_target)

        btn_load_source = gui.Button("Load Partial Scan (Source)")
        btn_load_source.set_on_clicked(self._on_load_source)
        self.panel.add_child(btn_load_source)

    def _setup_manual_panel(self, em):
        self.panel.add_child(gui.Label("\nManual Alignment Controls"))

        def add_slider(name, min_val, max_val):
            self.panel.add_child(gui.Label(name))
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(min_val, max_val)
            slider.set_on_value_changed(self._on_slider_changed)
            self.panel.add_child(slider)
            return slider

        self.sl_tx = add_slider("Translate X", -1.0, 1.0)
        self.sl_ty = add_slider("Translate Y", -1.0, 1.0)
        self.sl_tz = add_slider("Translate Z", -1.0, 1.0)
        self.sl_rx = add_slider("Rotate X (deg)", -180.0, 180.0)
        self.sl_ry = add_slider("Rotate Y (deg)", -180.0, 180.0)
        self.sl_rz = add_slider("Rotate Z (deg)", -180.0, 180.0)

        btn_randomize = gui.Button("Randomize Pose (Perturb)")
        btn_randomize.set_on_clicked(self._on_randomize_pose)
        btn_randomize.background_color = gui.Color(0.8, 0.4, 0.1)
        self.panel.add_child(btn_randomize)

        btn_reset = gui.Button("Reset Sliders & Position")
        btn_reset.set_on_clicked(self._on_reset)
        self.panel.add_child(btn_reset)

        self.panel.add_child(gui.Label("\nRobustness Tests"))
        
        btn_noise = gui.Button("Add Noise (0.5mm Gaussian)")
        btn_noise.set_on_clicked(self._on_add_noise)
        btn_noise.background_color = gui.Color(0.4, 0.4, 0.4)
        self.panel.add_child(btn_noise)

        btn_crop = gui.Button("Crop Source (Simulate Partial Overlap)")
        btn_crop.set_on_clicked(self._on_crop_source)
        btn_crop.background_color = gui.Color(0.4, 0.4, 0.4)
        self.panel.add_child(btn_crop)

    def _setup_algo_panel(self, em):
        self.panel.add_child(gui.Label("\nAlgorithm Parameters"))
        grid = gui.VGrid(2, int(0.25 * em))

        self.num_icp_thresh = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.num_icp_thresh.double_value = 0.2
        grid.add_child(gui.Label("ICP Thresh (m):"))
        grid.add_child(self.num_icp_thresh)

        self.num_icp_iter = gui.NumberEdit(gui.NumberEdit.INT)
        self.num_icp_iter.int_value = 2000
        grid.add_child(gui.Label("ICP Max Iter:"))
        grid.add_child(self.num_icp_iter)

        self.num_voxel_size = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.num_voxel_size.double_value = 0.05
        grid.add_child(gui.Label("Voxel Size (m):"))
        grid.add_child(self.num_voxel_size)

        self.panel.add_child(grid)

        self.panel.add_child(gui.Label("\nRun Algorithms"))
        
        btn_icp = gui.Button("Run ICP (Pt-to-Pt)")
        btn_icp.set_on_clicked(lambda: self._trigger_algo("ICP (Pt-to-Pt)", self._execute_icp))
        self.panel.add_child(btn_icp)

        btn_icp_p2p = gui.Button("Run ICP (Pt-to-Plane)")
        btn_icp_p2p.set_on_clicked(lambda: self._trigger_algo("ICP (Pt-to-Plane)", self._execute_icp_p2p))
        self.panel.add_child(btn_icp_p2p)

        btn_ransac = gui.Button("Run RANSAC (Global)")
        btn_ransac.set_on_clicked(lambda: self._trigger_algo("RANSAC", self._execute_ransac))
        self.panel.add_child(btn_ransac)

        btn_fgr = gui.Button("Run Fast Global (FGR)")
        btn_fgr.set_on_clicked(lambda: self._trigger_algo("FGR", self._execute_fgr))
        self.panel.add_child(btn_fgr)

        btn_dcp = gui.Button("Run Deep Closest Point (DCP)")
        btn_dcp.set_on_clicked(lambda: self._trigger_algo("DCP (AI)", self._execute_dcp))
        btn_dcp.background_color = gui.Color(0.2, 0.4, 0.7)
        self.panel.add_child(btn_dcp)
        
        btn_prnet = gui.Button("Run Partial Registration Net (PRNet)")
        btn_prnet.set_on_clicked(lambda: self._trigger_algo("PRNet (AI)", self._execute_prnet))
        btn_prnet.background_color = gui.Color(0.5, 0.2, 0.7)
        self.panel.add_child(btn_prnet)

    def _setup_benchmark_panel(self, em):
        self.panel.add_child(gui.Label("\nBenchmark"))
        
        btn_run_all = gui.Button("Run All & Compare")
        btn_run_all.set_on_clicked(self._on_run_all)
        btn_run_all.background_color = gui.Color(0.6, 0.2, 0.2)
        self.panel.add_child(btn_run_all)

        self.panel.add_child(gui.Label("View Benchmark Result:"))
        self.combo_results = gui.Combobox()
        self.combo_results.add_item("No Results Yet")
        self.combo_results.enabled = False
        self.combo_results.set_on_selection_changed(self._on_result_selected)
        self.panel.add_child(self.combo_results)

        self.lbl_result = gui.Label("\nResults: N/A")
        self.panel.add_child(self.lbl_result)

    def _on_layout(self, layout_context):
        try:
            r = self.window.content_rect
            panel_width = 340
            scene_width = max(0, r.width - panel_width)
            self.scene.frame = gui.Rect(r.x, r.y, scene_width, r.height)
            self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)
        except Exception:
            pass

    def _on_load_target(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select Target Point Cloud", self.window.theme)
        dlg.add_filter(".ply", "PLY Files")
        dlg.set_on_cancel(lambda: self.window.close_dialog())
        dlg.set_on_done(self._on_target_picked)
        self.window.show_dialog(dlg)

    def _on_target_picked(self, path):
        self.window.close_dialog()
        self.lbl_result.text = "\nResults: Loading Target..."

        def load_worker():
            try:
                pcd = self.manager.load_target(path)
                def update_ui():
                    self._update_geometry("target", pcd)
                    self.scene.setup_camera(60, self.scene.scene.bounding_box, (0, 0, 0))
                    self.lbl_result.text = "\nResults: Target loaded successfully!"
                gui.Application.instance.post_to_main_thread(self.window, update_ui)
            except Exception as e:
                gui.Application.instance.post_to_main_thread(self.window, lambda: setattr(self.lbl_result, 'text', f"\nResults: Error loading file!"))

        threading.Thread(target=load_worker, daemon=True).start()

    def _on_load_source(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select Source Point Cloud", self.window.theme)
        dlg.add_filter(".ply", "PLY Files")
        dlg.set_on_cancel(lambda: self.window.close_dialog())
        dlg.set_on_done(self._on_source_picked)
        self.window.show_dialog(dlg)

    def _on_source_picked(self, path):
        self.window.close_dialog()
        self.lbl_result.text = "\nResults: Loading Source..."

        def load_worker():
            try:
                pcd = self.manager.load_source(path)
                def update_ui():
                    self._update_geometry("source", pcd)
                    self._on_reset()
                    self.lbl_result.text = "\nResults: Source loaded successfully!"
                gui.Application.instance.post_to_main_thread(self.window, update_ui)
            except Exception:
                pass

        threading.Thread(target=load_worker, daemon=True).start()

    def _update_geometry(self, name, pcd):
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 3.0

        if self.scene.scene.has_geometry(name):
            self.scene.scene.remove_geometry(name)
        self.scene.scene.add_geometry(name, pcd, mat)

    def _on_slider_changed(self, val):
        if not self.manager.source_orig:
            return

        tx = self.sl_tx.double_value
        ty = self.sl_ty.double_value
        tz = self.sl_tz.double_value
        rx = np.radians(self.sl_rx.double_value)
        ry = np.radians(self.sl_ry.double_value)
        rz = np.radians(self.sl_rz.double_value)

        temp_source = copy.deepcopy(self.manager.source_orig)
        temp_source.paint_uniform_color([1.0, 0.706, 0.0])

        R = temp_source.get_rotation_matrix_from_xyz((rx, ry, rz))

        trans = np.identity(4)
        trans[:3, :3] = R
        trans[0, 3] = tx
        trans[1, 3] = ty
        trans[2, 3] = tz

        temp_source.transform(trans)
        self.manager.source_work = temp_source
        self.manager.current_transformation = trans

        self._update_geometry("source", self.manager.source_work)
        
    def _on_randomize_pose(self):
        if not self.manager.source_orig:
            return
        self.sl_tx.double_value = np.random.uniform(-2.0, 2.0)
        self.sl_ty.double_value = np.random.uniform(-2.0, 2.0)
        self.sl_tz.double_value = np.random.uniform(-2.0, 2.0)
        self.sl_rx.double_value = np.random.uniform(-45.0, 45.0)
        self.sl_ry.double_value = np.random.uniform(-45.0, 45.0)
        self.sl_rz.double_value = np.random.uniform(-45.0, 45.0)
        self._on_slider_changed(0)
        self.lbl_result.text = "\nResults: Source Pose Randomly Perturbed!"

    def _on_reset(self):
        self.sl_tx.double_value = 0.0
        self.sl_ty.double_value = 0.0
        self.sl_tz.double_value = 0.0
        self.sl_rx.double_value = 0.0
        self.sl_ry.double_value = 0.0
        self.sl_rz.double_value = 0.0

        pcd = self.manager.reset_source()
        if pcd:
            self._update_geometry("source", pcd)
            self.lbl_result.text = "\nResults: Reset to original position."

    def _on_add_noise(self):
        """Gauss-zajt ad a Source pontfelhőhöz (Tartós módosítás)."""
        if not self.manager.source_orig:
            return
            
        pts = np.asarray(self.manager.source_orig.points)
        noise = np.random.normal(0, 0.0001, pts.shape)
        
        # Az EREDETI pontfelhőt módosítjuk, így a benchmarkok is ezt használják majd
        self.manager.source_orig.points = o3d.utility.Vector3dVector(pts + noise)
        self.manager.source_orig.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        # Frissítjük a nézetet a jelenlegi transzformációkkal
        self._on_slider_changed(0)
        self.lbl_result.text = "\nResults: 2mm Gaussian noise permanently added to Source!"

    def _on_crop_source(self):
        """Levágja a Source pontfelhő egy részét (Tartós módosítás)."""
        if not self.manager.source_orig:
            return
            
        pts = np.asarray(self.manager.source_orig.points)
        min_x = np.min(pts[:, 0])
        max_x = np.max(pts[:, 0])
        threshold = min_x + 0.7 * (max_x - min_x)
        mask = pts[:, 0] < threshold
        
        cropped_pts = pts[mask]
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(cropped_pts)
        
        if self.manager.source_orig.has_colors():
            colors = np.asarray(self.manager.source_orig.colors)
            new_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        else:
            new_pcd.paint_uniform_color([1.0, 0.706, 0.0])
            
        # Az EREDETI pontfelhőt írjuk felül
        self.manager.source_orig = new_pcd
        self.manager.source_orig.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        self._on_slider_changed(0)
        self.lbl_result.text = f"\nResults: Source cropped! ({len(pts)} -> {len(cropped_pts)} points remaining)"

    def _trigger_algo(self, name: str, func):
        if not self.manager.source_work or not self.manager.target_work:
            return
        self.lbl_result.text = f"\nResults: Running {name}..."
        gui.Application.instance.post_to_main_thread(self.window, func)

    def _execute_icp(self):
        trans, fit, rmse, t = algos.run_icp(
            self.manager.source_work, self.manager.target_work, 
            self.num_icp_thresh.double_value, self.num_icp_iter.int_value)
        self._apply_and_update(trans, "ICP (Pt-to-Pt)", t, fit, rmse)

    def _execute_icp_p2p(self):
        trans, fit, rmse, t = algos.run_icp_point_to_plane(
            self.manager.source_work, self.manager.target_work, 
            self.num_icp_thresh.double_value, self.num_icp_iter.int_value)
        self._apply_and_update(trans, "ICP (Pt-to-Plane)", t, fit, rmse)

    def _execute_ransac(self):
        trans, fit, rmse, t = algos.run_ransac(
            self.manager.source_work, self.manager.target_work, self.num_voxel_size.double_value)
        self._apply_and_update(trans, "RANSAC", t, fit, rmse)

    def _execute_fgr(self):
        trans, fit, rmse, t = algos.run_fgr(
            self.manager.source_work, self.manager.target_work, self.num_voxel_size.double_value)
        self._apply_and_update(trans, "Fast Global Registration", t, fit, rmse)

    def _execute_dcp(self):
        trans, fit, rmse, t = algos.run_ai_model(
            self.manager.source_work, self.manager.target_work, self.dcp_model)
        self._apply_and_update(trans, "DCP (AI)", t, fit, rmse)
        
    def _execute_prnet(self):
        trans, fit, rmse, t = algos.run_ai_model(
            self.manager.source_work, self.manager.target_work, self.prnet_model)
        self._apply_and_update(trans, "PRNet (AI)", t, fit, rmse)

    def _apply_and_update(self, trans, algo_name, t_sec, fitness, rmse):
        self.manager.apply_transformation(trans)
        self._update_geometry("source", self.manager.source_work)
        
        t_err, r_err = util.compute_registration_error(self.manager.current_transformation)
        
        # Updated UI text to clearly separate real-world metrics from synthetic ones
        self.lbl_result.text = (
            f"\n{algo_name} Result:\n"
            f"Time: {t_sec:.4f} s\n"
            f"--- Real-World Metrics ---\n"
            f"Fitness (Overlap): {fitness:.4f} | Inlier RMSE: {rmse:.5f}\n"
            f"--- Synthetic Benchmark Error ---\n"
            f"(Only valid if original load state was perfectly aligned)\n"
            f"Translation Err: {t_err:.5f} m | Rotation Err: {r_err:.4f}°"
        )

    def _on_run_all(self):
        if not self.manager.source_work or not self.manager.target_work:
            return
        self.lbl_result.text = "\nResults: Running Benchmarks...\n(This will take a moment)"
        gui.Application.instance.post_to_main_thread(self.window, self._execute_run_all)

    def _execute_run_all(self):
        initial_trans = np.copy(self.manager.current_transformation)
        
        self.benchmark_transforms = {
            "Original (Identity)": np.identity(4),
            "Manual (Start)": initial_trans
        }

        def reset_to_initial():
            self.manager.current_transformation = np.copy(initial_trans)
            temp_source = copy.deepcopy(self.manager.source_orig)
            temp_source.paint_uniform_color([1.0, 0.706, 0.0])
            temp_source.transform(initial_trans)
            self.manager.source_work = temp_source

        thresh = self.num_icp_thresh.double_value
        iters = self.num_icp_iter.int_value
        voxel = self.num_voxel_size.double_value

        results = []

        algorithms = [
            ("ICP (Pt2Pt)", lambda: algos.run_icp(self.manager.source_work, self.manager.target_work, thresh, iters)),
            ("ICP (Pt2Pl)", lambda: algos.run_icp_point_to_plane(self.manager.source_work, self.manager.target_work, thresh, iters)),
            ("RANSAC", lambda: algos.run_ransac(self.manager.source_work, self.manager.target_work, voxel)),
            ("FGR", lambda: algos.run_fgr(self.manager.source_work, self.manager.target_work, voxel)),
            ("DCP (AI)", lambda: algos.run_ai_model(self.manager.source_work, self.manager.target_work, self.dcp_model)),
            ("PRNet (AI)", lambda: algos.run_ai_model(self.manager.source_work, self.manager.target_work, self.prnet_model))
        ]

        for name, func in algorithms:
            reset_to_initial()
            trans, fit, rmse, t = func()
            self.manager.apply_transformation(trans)
            t_err, r_err = util.compute_registration_error(self.manager.current_transformation)
            
            results.append((name, t, rmse, t_err, r_err))
            self.benchmark_transforms[name] = np.copy(self.manager.current_transformation)

        reset_to_initial()
        self._update_geometry("source", self.manager.source_work)

        def update_ui_elements():
            self.combo_results.clear_items()
            for key in self.benchmark_transforms.keys():
                self.combo_results.add_item(key)
            self.combo_results.enabled = True

            table = "\n--- Benchmark Complete ---\n"
            for name, time_s, rmse, err_t, err_r in results:
                table += f"[{name}]\nTime: {time_s:.2f}s | RMSE: {rmse:.4f}\n(Synthetic) T-Err: {err_t:.4f}m | R-Err: {err_r:.2f}°\n\n"
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_dir, "logs")
        
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"benchmark_results_{timestamp}.csv")
            
            export_msg = ""
            try:
                with open(filepath, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Algorithm", "Time_sec", "RMSE", "Trans_Err_m", "Rot_Err_deg"])
                    for name, time_s, rmse, err_t, err_r in results:
                        writer.writerow([name, f"{time_s:.4f}", f"{rmse:.5f}", f"{err_t:.5f}", f"{err_r:.4f}"])
                export_msg = f"\n[Data saved to:\n{filepath}]"
            except Exception as e:
                export_msg = f"\n[Warning: Could not save CSV: {e}]"

            self.lbl_result.text = table.strip() + export_msg

        gui.Application.instance.post_to_main_thread(self.window, update_ui_elements)

    def _on_result_selected(self, selected_name: str, index: int):
        if selected_name not in self.benchmark_transforms:
            return
        trans_matrix = self.benchmark_transforms[selected_name]
        self.manager.current_transformation = np.copy(trans_matrix)
        temp_source = copy.deepcopy(self.manager.source_orig)
        temp_source.paint_uniform_color([1.0, 0.706, 0.0])
        temp_source.transform(trans_matrix)
        self.manager.source_work = temp_source
        self._update_geometry("source", self.manager.source_work)