#!/usr/bin/env python3

from pathlib import Path
import sys
import argparse
import math

import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tumtraf_dataset import TUMTrafDataset


CLASS_COLORS = {
    "CAR": (0.10, 0.65, 1.00),
    "PEDESTRIAN": (1.00, 0.85, 0.10),
    "TRUCK": (1.00, 0.35, 0.10),
    "TRAILER": (0.80, 0.30, 1.00),
    "BUS": (0.10, 1.00, 0.35),
    "VAN": (0.20, 0.90, 0.90),
    "BICYCLE": (1.00, 0.55, 0.20),
    "MOTORCYCLE": (1.00, 0.20, 0.55),
    "EMERGENCY_VEHICLE": (1.00, 0.10, 0.10),
    "UNKNOWN": (0.90, 0.90, 0.90),
}


def import_open3d():
    try:
        import open3d as o3d
        from open3d.visualization import rendering
        return o3d, rendering
    except Exception as e:
        raise RuntimeError(
            "Open3D is required for this visualization.\n"
            "Install it with:\n"
            "  pip install open3d\n"
            f"Original error: {e}"
        )


def colorize_points(points):
    """
    Color point cloud using height + intensity.
    Returns Nx3 RGB colors in [0, 1].
    """
    z = points[:, 2]
    intensity = points[:, 3] if points.shape[1] > 3 else np.ones_like(z)

    z_norm = np.clip((z - np.percentile(z, 2)) / max(np.percentile(z, 98) - np.percentile(z, 2), 1e-6), 0, 1)
    i_norm = np.clip(intensity, 0, 1)

    colors = np.zeros((points.shape[0], 3), dtype=np.float64)

    # Dark road points, brighter object/high points.
    colors[:, 0] = 0.25 + 0.55 * z_norm
    colors[:, 1] = 0.25 + 0.55 * i_norm
    colors[:, 2] = 0.25 + 0.65 * (1.0 - z_norm)

    return np.clip(colors, 0, 1)


def filter_points(points, xlim, ylim, zlim):
    mask = (
        (points[:, 0] >= xlim[0]) & (points[:, 0] <= xlim[1]) &
        (points[:, 1] >= ylim[0]) & (points[:, 1] <= ylim[1]) &
        (points[:, 2] >= zlim[0]) & (points[:, 2] <= zlim[1])
    )
    return points[mask]


def box_corners_3d(box):
    center = box["center"]
    length, width, height = box["dims_lwh"]
    yaw = box["yaw"]

    local = np.array(
        [
            [ length / 2,  width / 2, -height / 2],
            [ length / 2, -width / 2, -height / 2],
            [-length / 2, -width / 2, -height / 2],
            [-length / 2,  width / 2, -height / 2],
            [ length / 2,  width / 2,  height / 2],
            [ length / 2, -width / 2,  height / 2],
            [-length / 2, -width / 2,  height / 2],
            [-length / 2,  width / 2,  height / 2],
        ],
        dtype=np.float64,
    )

    c = math.cos(yaw)
    s = math.sin(yaw)

    rot = np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    corners = local @ rot.T
    corners += center.reshape(1, 3)
    return corners


def create_box_lineset(o3d, box):
    corners = box_corners_3d(box)

    edges = np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],

            # front direction marker
            [0, 5], [1, 4],
        ],
        dtype=np.int32,
    )

    color = CLASS_COLORS.get(box["class_name"], CLASS_COLORS["UNKNOWN"])
    colors = np.tile(np.array(color, dtype=np.float64), (len(edges), 1))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def create_grid_lineset(o3d, xlim, ylim, z, spacing=10.0):
    points = []
    lines = []
    colors = []

    color = [0.22, 0.22, 0.22]

    xs = np.arange(math.floor(xlim[0] / spacing) * spacing, xlim[1] + spacing, spacing)
    ys = np.arange(math.floor(ylim[0] / spacing) * spacing, ylim[1] + spacing, spacing)

    for x in xs:
        start_idx = len(points)
        points.append([x, ylim[0], z])
        points.append([x, ylim[1], z])
        lines.append([start_idx, start_idx + 1])
        colors.append(color)

    for y in ys:
        start_idx = len(points)
        points.append([xlim[0], y, z])
        points.append([xlim[1], y, z])
        lines.append([start_idx, start_idx + 1])
        colors.append(color)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))

    return line_set


def create_axis_lineset(o3d, origin=(-70, -70, -6.5), length=20.0):
    ox, oy, oz = origin

    points = np.array(
        [
            [ox, oy, oz],
            [ox + length, oy, oz],
            [ox, oy + length, oz],
        ],
        dtype=np.float64,
    )

    lines = np.array([[0, 1], [0, 2]], dtype=np.int32)

    colors = np.array(
        [
            [1.0, 0.2, 0.2],
            [0.2, 1.0, 0.2],
        ],
        dtype=np.float64,
    )

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def add_overlay_text(image_bgr, sample, boxes):
    h, w = image_bgr.shape[:2]

    panel = image_bgr.copy()
    cv2.rectangle(panel, (20, 20), (640, 210), (20, 20, 20), -1)
    image_bgr = cv2.addWeighted(panel, 0.55, image_bgr, 0.45, 0)

    lines = [
        f"Frame: {sample['frame_id']}",
        f"Points: {sample['points'].shape[0]:,}",
        f"Boxes: {len(boxes)}",
        "BEV Open3D render | x-right, y-up",
    ]

    y = 55
    for line in lines:
        cv2.putText(
            image_bgr,
            line,
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        y += 35

    legend_x = w - 360
    legend_y = 40

    cv2.rectangle(image_bgr, (legend_x - 20, legend_y - 25), (w - 30, legend_y + 310), (20, 20, 20), -1)

    for cls, color_rgb in CLASS_COLORS.items():
        if cls == "UNKNOWN":
            continue

        color_bgr = tuple(int(255 * c) for c in color_rgb[::-1])

        cv2.circle(image_bgr, (legend_x, legend_y), 7, color_bgr, -1)
        cv2.putText(
            image_bgr,
            cls,
            (legend_x + 20, legend_y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        legend_y += 32

    return image_bgr


def render_sample(sample, output_path, width, height, xlim, ylim, zlim, point_size, line_width):
    o3d, rendering = import_open3d()

    points = filter_points(sample["points"], xlim, ylim, zlim)
    boxes = sample["boxes"]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colorize_points(points))

    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([0.03, 0.03, 0.035, 1.0])
    renderer.scene.set_lighting(renderer.scene.LightingProfile.NO_SHADOWS, (0, 0, 1))

    point_material = rendering.MaterialRecord()
    point_material.shader = "defaultUnlit"
    point_material.point_size = point_size

    line_material = rendering.MaterialRecord()
    line_material.shader = "unlitLine"
    line_material.line_width = line_width

    renderer.scene.add_geometry("points", pcd, point_material)

    grid = create_grid_lineset(o3d, xlim, ylim, z=zlim[0], spacing=10.0)
    renderer.scene.add_geometry("grid", grid, line_material)

    axis = create_axis_lineset(o3d, origin=(xlim[0] + 10, ylim[0] + 10, zlim[0] + 0.1), length=20.0)
    renderer.scene.add_geometry("axis", axis, line_material)

    for i, box in enumerate(boxes):
        line_set = create_box_lineset(o3d, box)
        renderer.scene.add_geometry(f"box_{i}", line_set, line_material)

    cx = 0.5 * (xlim[0] + xlim[1])
    cy = 0.5 * (ylim[0] + ylim[1])
    cz = 0.5 * (zlim[0] + zlim[1])

    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]

    # Top-down orthographic camera.
    renderer.scene.camera.set_projection(
        rendering.Camera.Projection.Ortho,
        -x_span / 2,
        x_span / 2,
        -y_span / 2,
        y_span / 2,
        0.1,
        500.0,
    )
    renderer.scene.camera.look_at(
        [cx, cy, cz],
        [cx, cy, cz + 200.0],
        [0.0, 1.0, 0.0],
    )

    image_o3d = renderer.render_to_image()
    image_rgb = np.asarray(image_o3d)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image_bgr = add_overlay_text(image_bgr, sample, boxes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_bgr)

    print(f"Saved: {output_path}")


def parse_range(text):
    a, b = text.split(",")
    return float(a), float(b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", default="data/processed/frame_indices/train_frames.csv")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/bev/open3d")
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1800)
    parser.add_argument("--xlim", default="-80,90")
    parser.add_argument("--ylim", default="-80,120")
    parser.add_argument("--zlim", default="-9,4")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--line-width", type=float, default=5.0)
    args = parser.parse_args()

    dataset = TUMTrafDataset(args.index_path)
    sample = dataset[args.sample_idx]

    output_path = Path(args.output_dir) / f"{args.sample_idx:04d}_{sample['frame_id']}_open3d_bev.png"

    print(f"Dataset size: {len(dataset)}")
    print(f"Sample index: {args.sample_idx}")
    print(f"Frame ID: {sample['frame_id']}")
    print(f"Points: {sample['points'].shape}")
    print(f"Boxes: {len(sample['boxes'])}")

    render_sample(
        sample=sample,
        output_path=output_path,
        width=args.width,
        height=args.height,
        xlim=parse_range(args.xlim),
        ylim=parse_range(args.ylim),
        zlim=parse_range(args.zlim),
        point_size=args.point_size,
        line_width=args.line_width,
    )


if __name__ == "__main__":
    main()
