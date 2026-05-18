import os
import argparse
import open3d as o3d

def downsample_ply(input_path: str, ratio: float) -> None:
    # 1. Validate inputs
    if not os.path.isfile(input_path):
        print(f"Error: File '{input_path}' does not exist.")
        return
    
    if not input_path.lower().endswith('.ply'):
        print(f"Error: '{input_path}' is not a .ply file.")
        return

    if not (0.0 < ratio < 1.0):
        print("Error: Ratio must be strictly between 0.0 and 1.0 (e.g., 0.5 to keep 50%).")
        return

    # 2. Load the point cloud
    print(f"Loading: {input_path}")
    try:
        pcd = o3d.io.read_point_cloud(input_path)
    except Exception as e:
        print(f"Failed to read point cloud: {e}")
        return
    
    original_points = len(pcd.points)
    if original_points == 0:
        print("Error: Point cloud contains 0 points.")
        return

    print(f"Original points: {original_points}")
    print(f"Downsampling to {ratio * 100:.1f}%...")

    # 3. Apply the downsample
    downsampled_pcd = pcd.random_down_sample(sampling_ratio=ratio)
    new_points = len(downsampled_pcd.points)

    # 4. Construct the output filepath
    dir_name = os.path.dirname(os.path.abspath(input_path))
    base_name = os.path.basename(input_path)
    name_without_ext, ext = os.path.splitext(base_name)
    
    # Example: "scan_room.ply" with ratio 0.1 -> "scan_room_0.1x.ply"
    new_filename = f"{name_without_ext}_{ratio}x{ext}"
    output_path = os.path.join(dir_name, new_filename)

    # 5. Save the result
    try:
        o3d.io.write_point_cloud(output_path, downsampled_pcd)
        print(f"Success! Saved {new_points} points to:\n -> {output_path}")
    except Exception as e:
        print(f"Failed to write point cloud: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quickly downsample a .ply point cloud file.")
    parser.add_argument("input_file", type=str, help="Path to the original .ply file")
    parser.add_argument("ratio", type=float, help="Ratio of points to keep (e.g., 0.1 for 10%)")
    
    args = parser.parse_args()
    downsample_ply(args.input_file, args.ratio)