#!/usr/bin/env python3

from pathlib import Path
import sys
import argparse
import math
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tumtraf_dataset import TUMTrafDataset


CLASS_COLORS_BGR = {
    "CAR": (255, 170, 40),
    "PEDESTRIAN": (40, 240, 255),
    "TRUCK": (40, 120, 255),
    "TRAILER": (255, 80, 255),
    "BUS": (80, 255, 80),
    "VAN": (255, 230, 80),
    "BICYCLE": (80, 180, 255),
    "MOTORCYCLE": (180, 80, 255),
    "EMERGENCY_VEHICLE": (40, 40, 255),
    "UNKNOWN": (230, 230, 230),
}


def parse_range(text):
    a, b = text.split(",")
    return float(a), float(b)


def world_to_pixel_xy(x, y, xlim, ylim, width, height):
    xmin, xmax = xlim
    ymin, ymax = ylim

    u = (x - xmin) / (xmax - xmin) * (width - 1)
    v = (ymax - y) / (ymax - ymin) * (height - 1)

    return u, v


def draw_text_with_outline(img, text, org, scale=0.55, color=(255, 255, 255), thickness=1):
    x, y = org
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_grid(img, xlim, ylim, spacing=10.0):
    h, w = img.shape[:2]

    xmin, xmax = xlim
    ymin, ymax = ylim

    grid_color = (42, 42, 42)
    axis_color = (85, 85, 85)

    x_start = math.floor(xmin / spacing) * spacing
    x_end = math.ceil(xmax / spacing) * spacing
    y_start = math.floor(ymin / spacing) * spacing
    y_end = math.ceil(ymax / spacing) * spacing

    x = x_start
    while x <= x_end:
        u1, v1 = world_to_pixel_xy(x, ymin, xlim, ylim, w, h)
        u2, v2 = world_to_pixel_xy(x, ymax, xlim, ylim, w, h)

        color = axis_color if abs(x) < 1e-6 else grid_color
        thickness = 2 if abs(x) < 1e-6 else 1

        cv2.line(img, (int(u1), int(v1)), (int(u2), int(v2)), color, thickness)

        if int(x) % 20 == 0:
            draw_text_with_outline(img, f"x={int(x)}", (int(u1) + 4, h - 12), scale=0.38, color=(150, 150, 150), thickness=1)

        x += spacing

    y = y_start
    while y <= y_end:
        u1, v1 = world_to_pixel_xy(xmin, y, xlim, ylim, w, h)
        u2, v2 = world_to_pixel_xy(xmax, y, xlim, ylim, w, h)

        color = axis_color if abs(y) < 1e-6 else grid_color
        thickness = 2 if abs(y) < 1e-6 else 1

        cv2.line(img, (int(u1), int(v1)), (int(u2), int(v2)), color, thickness)

        if int(y) % 20 == 0:
            draw_text_with_outline(img, f"y={int(y)}", (8, int(v1) - 4), scale=0.38, color=(150, 150, 150), thickness=1)

        y += spacing


def draw_points(img, points, xlim, ylim, zlim, point_size=1):
    h, w = img.shape[:2]

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    mask = (
        (x >= xlim[0]) & (x <= xlim[1]) &
        (y >= ylim[0]) & (y <= ylim[1]) &
        (z >= zlim[0]) & (z <= zlim[1])
    )

    pts = points[mask]
    if len(pts) == 0:
        return 0

    u, v = world_to_pixel_xy(pts[:, 0], pts[:, 1], xlim, ylim, w, h)

    u = np.round(u).astype(np.int32)
    v = np.round(v).astype(np.int32)

    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[valid]
    v = v[valid]
    pts = pts[valid]

    z_vals = pts[:, 2]
    z_norm = np.clip((z_vals - zlim[0]) / max(zlim[1] - zlim[0], 1e-6), 0, 1)

    color_values = (255 * z_norm).astype(np.uint8)
    colors = cv2.applyColorMap(color_values.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)

    if point_size <= 1:
        img[v, u] = colors
    else:
        for px, py, color in zip(u, v, colors):
            cv2.circle(img, (int(px), int(py)), point_size, color.tolist(), -1, cv2.LINE_AA)

    return len(u)


def box_corners_bev(box):
    center = box["center"]
    dims = box["dims_lwh"]
    yaw = box["yaw"]

    x, y = center[0], center[1]
    length, width = dims[0], dims[1]

    local = np.array(
        [
            [ length / 2.0,  width / 2.0],
            [ length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0,  width / 2.0],
        ],
        dtype=np.float64,
    )

    c = math.cos(yaw)
    s = math.sin(yaw)

    rot = np.array(
        [
            [c, -s],
            [s,  c],
        ],
        dtype=np.float64,
    )

    corners = local @ rot.T
    corners[:, 0] += x
    corners[:, 1] += y

    return corners


def draw_boxes(img, boxes, xlim, ylim, thickness=4, draw_labels=True, fill_alpha=0.18):
    h, w = img.shape[:2]

    fill_layer = img.copy()

    # First fill all boxes lightly.
    for box in boxes:
        cls = box["class_name"]
        color = CLASS_COLORS_BGR.get(cls, CLASS_COLORS_BGR["UNKNOWN"])

        corners = box_corners_bev(box)
        u, v = world_to_pixel_xy(corners[:, 0], corners[:, 1], xlim, ylim, w, h)
        poly = np.stack([u, v], axis=1).round().astype(np.int32)

        cv2.fillPoly(fill_layer, [poly], color)

    img[:] = cv2.addWeighted(fill_layer, fill_alpha, img, 1.0 - fill_alpha, 0)

    # Then draw thick outlines and heading markers on top.
    for box in boxes:
        cls = box["class_name"]
        color = CLASS_COLORS_BGR.get(cls, CLASS_COLORS_BGR["UNKNOWN"])

        corners = box_corners_bev(box)
        u, v = world_to_pixel_xy(corners[:, 0], corners[:, 1], xlim, ylim, w, h)
        poly = np.stack([u, v], axis=1).round().astype(np.int32)

        cv2.polylines(img, [poly], isClosed=True, color=(0, 0, 0), thickness=thickness + 4, lineType=cv2.LINE_AA)
        cv2.polylines(img, [poly], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

        # Heading marker: center to front edge.
        center = box["center"]
        length = box["dims_lwh"][0]
        yaw = box["yaw"]

        front_x = center[0] + math.cos(yaw) * length / 2.0
        front_y = center[1] + math.sin(yaw) * length / 2.0

        cu, cv = world_to_pixel_xy(center[0], center[1], xlim, ylim, w, h)
        fu, fv = world_to_pixel_xy(front_x, front_y, xlim, ylim, w, h)

        cv2.arrowedLine(
            img,
            (int(cu), int(cv)),
            (int(fu), int(fv)),
            color,
            max(2, thickness - 1),
            cv2.LINE_AA,
            tipLength=0.35,
        )

        if draw_labels:
            label = cls
            meta = box.get("num_points")
            if meta is not None:
                label += f" n={meta}"

            lx, ly = int(cu) + 5, int(cv) - 5
            draw_text_with_outline(img, label, (lx, ly), scale=0.42, color=color, thickness=1)


def draw_info_panel(img, sample, boxes, points_drawn, xlim, ylim, zlim):
    h, w = img.shape[:2]

    panel_w = 650
    panel_h = 175

    overlay = img.copy()
    cv2.rectangle(overlay, (18, 18), (18 + panel_w, 18 + panel_h), (18, 18, 18), -1)
    img[:] = cv2.addWeighted(overlay, 0.72, img, 0.28, 0)

    class_counts = Counter(box["class_name"] for box in boxes)

    lines = [
        f"Frame: {sample['frame_id']}",
        f"Points drawn: {points_drawn:,} / raw {sample['points'].shape[0]:,}",
        f"Boxes: {len(boxes)} | " + ", ".join([f"{k}:{v}" for k, v in class_counts.most_common()]),
        f"Range x={xlim}, y={ylim}, z={zlim}",
    ]

    y = 52
    for line in lines:
        draw_text_with_outline(img, line, (38, y), scale=0.55, color=(235, 235, 235), thickness=1)
        y += 34


def draw_legend(img):
    h, w = img.shape[:2]

    x0 = w - 330
    y0 = 35

    overlay = img.copy()
    cv2.rectangle(overlay, (x0 - 20, y0 - 25), (w - 25, y0 + 305), (18, 18, 18), -1)
    img[:] = cv2.addWeighted(overlay, 0.72, img, 0.28, 0)

    y = y0
    for cls, color in CLASS_COLORS_BGR.items():
        if cls == "UNKNOWN":
            continue

        cv2.rectangle(img, (x0, y - 10), (x0 + 18, y + 8), color, -1)
        draw_text_with_outline(img, cls, (x0 + 30, y + 7), scale=0.48, color=(240, 240, 240), thickness=1)
        y += 31


def draw_axes(img, xlim, ylim):
    h, w = img.shape[:2]

    ox, oy = world_to_pixel_xy(0, 0, xlim, ylim, w, h)
    ox, oy = int(ox), int(oy)

    x_end = world_to_pixel_xy(20, 0, xlim, ylim, w, h)
    y_end = world_to_pixel_xy(0, 20, xlim, ylim, w, h)

    cv2.arrowedLine(img, (ox, oy), (int(x_end[0]), int(x_end[1])), (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.18)
    cv2.arrowedLine(img, (ox, oy), (int(y_end[0]), int(y_end[1])), (0, 255, 0), 3, cv2.LINE_AA, tipLength=0.18)

    draw_text_with_outline(img, "+x", (int(x_end[0]) + 6, int(x_end[1]) + 6), scale=0.55, color=(0, 0, 255), thickness=1)
    draw_text_with_outline(img, "+y", (int(y_end[0]) + 6, int(y_end[1]) + 6), scale=0.55, color=(0, 255, 0), thickness=1)


def render_sample(dataset, sample_idx, args):
    sample = dataset[sample_idx]

    xlim = parse_range(args.xlim)
    ylim = parse_range(args.ylim)
    zlim = parse_range(args.zlim)

    img = np.full((args.height, args.width, 3), (8, 8, 10), dtype=np.uint8)

    draw_grid(img, xlim, ylim, spacing=args.grid_spacing)

    points_drawn = draw_points(
        img,
        sample["points"],
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        point_size=args.point_size,
    )

    draw_boxes(
        img,
        sample["boxes"],
        xlim=xlim,
        ylim=ylim,
        thickness=args.box_thickness,
        draw_labels=not args.no_labels,
        fill_alpha=args.fill_alpha,
    )

    draw_axes(img, xlim, ylim)
    draw_info_panel(img, sample, sample["boxes"], points_drawn, xlim, ylim, zlim)
    draw_legend(img)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{sample_idx:04d}_{sample['frame_id']}_cv2_bev.png"
    cv2.imwrite(str(output_path), img)

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", default="data/processed/frame_indices/train_frames.csv")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--sample-indices", default=None, help="Comma-separated sample indices, e.g. 0,10,50")
    parser.add_argument("--output-dir", default="outputs/bev/cv2")
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1800)
    parser.add_argument("--xlim", default="-80,90")
    parser.add_argument("--ylim", default="-80,120")
    parser.add_argument("--zlim", default="-9,4")
    parser.add_argument("--point-size", type=int, default=1)
    parser.add_argument("--box-thickness", type=int, default=4)
    parser.add_argument("--fill-alpha", type=float, default=0.16)
    parser.add_argument("--grid-spacing", type=float, default=10.0)
    parser.add_argument("--no-labels", action="store_true")
    args = parser.parse_args()

    dataset = TUMTrafDataset(args.index_path)

    if args.sample_indices is not None:
        indices = [int(x.strip()) for x in args.sample_indices.split(",") if x.strip()]
    else:
        indices = [args.sample_idx]

    print(f"Dataset size: {len(dataset)}")
    print(f"Rendering indices: {indices}")

    for idx in tqdm(indices, desc="Rendering BEV", unit="frame"):
        render_sample(dataset, idx, args)


if __name__ == "__main__":
    main()
