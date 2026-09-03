"""Live USB Record3D: unproject depth with per-frame K + the same pose correction as mapping.

Walk around. If walls stay put and the camera frame moves, pose + depth + K are consistent.
If the room smears, pose/K is wrong. Dollhouse/skyscraper = depth scale (mm vs m).

Must import open3d_display before Open3D (Wayland segfault).
"""

from conceptgraph.utils.open3d_display import force_x11_for_open3d

force_x11_for_open3d()

import numpy as np
import open3d as o3d

from conceptgraph.utils.record3d_utils import DemoApp

# --- edit these ---
DEV_IDX = 0
STRIDE = 4
DEPTH_MIN = 0.15
DEPTH_MAX = 8.0
ACCUMULATE = True
VOXEL = 0.03
MAX_ACCUM_POINTS = 250_000
PRINT_EVERY = 15
# --- end edit ---


def unproject_rgb_d(rgb_bgr, depth, K, T_wc, stride):
    """Same pinhole as mapping: x = (u-cx)*z/fx in camera, then T_wc to world."""
    h, w = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[vs, us]
    valid = np.isfinite(z) & (z > DEPTH_MIN) & (z < DEPTH_MAX)
    us = us[valid].astype(np.float32)
    vs = vs[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    if z.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), z
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    pts_cam = np.stack((x, y, z), axis=1)
    pts_world = pts_cam @ T_wc[:3, :3].T + T_wc[:3, 3]
    colors = rgb_bgr[vs.astype(int), us.astype(int)][:, ::-1] / 255.0
    return pts_world.astype(np.float32), colors.astype(np.float32), z


def make_camera_frame(T_wc, size=0.2):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(T_wc)
    return frame


def cap_pcd(pcd, max_points):
    n = np.asarray(pcd.points).shape[0]
    if n <= max_points:
        return pcd
    idx = np.random.choice(n, max_points, replace=False)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[idx])
    out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[idx])
    return out


def main():
    print(
        "Open3D pose+depth check (same K/pose path as r3d_stream_rerun_realtime_mapping).\n"
        "Walk the room. Walls should stay; orange camera axes should move.\n"
        "Smear = bad pose/K. Tiny/huge room = depth units. Ctrl+C to quit."
    )
    app = DemoApp()
    app.connect_to_device(dev_idx=DEV_IDX)

    vis = o3d.visualization.Visualizer()
    vis.create_window("Record3D pose + depth", width=1280, height=800)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(origin)

    live_pcd = o3d.geometry.PointCloud()
    accum_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(live_pcd)
    if ACCUMULATE:
        vis.add_geometry(accum_pcd)

    cam_frame = None
    traj_pts = []
    traj = o3d.geometry.LineSet()
    vis.add_geometry(traj)

    n = 0
    prev_t = None
    reset_view = True
    try:
        while True:
            rgb, depth, K4, T_wc = app.get_frame_data()
            if rgb is None:
                print("Waiting for USB frame...")
                vis.poll_events()
                vis.update_renderer()
                continue
            n += 1
            K = K4.cpu().numpy()[:3, :3]
            T_wc = np.asarray(T_wc, dtype=np.float64)
            pts, cols, z = unproject_rgb_d(rgb, depth, K, T_wc, STRIDE)
            t = T_wc[:3, 3]
            dt = 0.0 if prev_t is None else float(np.linalg.norm(t - prev_t))
            prev_t = t.copy()

            live_pcd.points = o3d.utility.Vector3dVector(pts)
            live_pcd.colors = o3d.utility.Vector3dVector(cols)
            vis.update_geometry(live_pcd)

            if ACCUMULATE and pts.shape[0]:
                chunk = o3d.geometry.PointCloud()
                chunk.points = o3d.utility.Vector3dVector(pts)
                chunk.colors = o3d.utility.Vector3dVector(cols)
                accum_pcd += chunk
                down = accum_pcd.voxel_down_sample(VOXEL)
                down = cap_pcd(down, MAX_ACCUM_POINTS)
                accum_pcd.points = down.points
                accum_pcd.colors = down.colors
                vis.update_geometry(accum_pcd)

            if cam_frame is not None:
                vis.remove_geometry(cam_frame, reset_bounding_box=False)
            cam_frame = make_camera_frame(T_wc)
            vis.add_geometry(cam_frame, reset_bounding_box=False)

            traj_pts.append(t)
            if len(traj_pts) >= 2:
                vis.remove_geometry(traj, reset_bounding_box=False)
                traj = o3d.geometry.LineSet()
                traj.points = o3d.utility.Vector3dVector(np.asarray(traj_pts))
                traj.lines = o3d.utility.Vector2iVector(
                    np.array([[i, i + 1] for i in range(len(traj_pts) - 1)])
                )
                traj.paint_uniform_color([1.0, 0.45, 0.0])
                vis.add_geometry(traj, reset_bounding_box=False)

            vis.poll_events()
            vis.update_renderer()
            if reset_view:
                vis.reset_view_point(True)
                reset_view = False

            if n == 1 or n % PRINT_EVERY == 0:
                z_stats = (
                    f"depth[m] min={z.min():.2f} med={np.median(z):.2f} max={z.max():.2f}"
                    if z.size
                    else "depth[m] (none)"
                )
                print(
                    f"frame {n:4d}  {z_stats}  npts={pts.shape[0]}  "
                    f"t=({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})  |Δt|={dt:.3f} m  "
                    f"fx={K[0,0]:.1f}"
                )
                if z.size and np.median(z) < 0.05:
                    print("  median depth < 5cm — units look like mm, not meters.")
                if z.size and np.median(z) > 20:
                    print("  median depth > 20m — K/depth scale looks too large.")
    except KeyboardInterrupt:
        print("\nstopped")
    vis.destroy_window()


if __name__ == "__main__":
    main()
