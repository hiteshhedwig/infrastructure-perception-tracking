from pathlib import Path
import csv
import json
import math
from typing import Dict, List, Any

import numpy as np


class TUMTrafDataset:
    """
    Minimal TUMTraf V2X loader for registered LiDAR point clouds and OpenLABEL 3D cuboids.

    This loader assumes you already built frame index CSVs:
      data/processed/frame_indices/train_frames.csv
      data/processed/frame_indices/val_frames.csv
      data/processed/frame_indices/test_frames.csv
    """

    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.rows = self._read_index(self.index_path)

        if len(self.rows) == 0:
            raise ValueError(f"No rows found in index: {self.index_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]

        frame_id = row["frame_id"]
        label_path = Path(row["label_path"])
        point_cloud_path = Path(row["point_cloud_path"])

        points = self.load_ascii_pcd(point_cloud_path)
        boxes = self.parse_openlabel_boxes(label_path)

        return {
            "frame_id": frame_id,
            "points": points,
            "boxes": boxes,
            "label_path": label_path,
            "point_cloud_path": point_cloud_path,
        }

    @staticmethod
    def _read_index(index_path: Path) -> List[Dict[str, str]]:
        rows = []

        with open(index_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        return rows

    @staticmethod
    def load_ascii_pcd(pcd_path: Path) -> np.ndarray:
        """
        Load ASCII PCD with fields:
          x y z intensity

        Returns:
          points: np.ndarray of shape [N, 4]
        """
        data_start_line = None

        with open(pcd_path, "r") as f:
            for line_idx, line in enumerate(f):
                if line.strip().startswith("DATA"):
                    data_start_line = line_idx + 1
                    break

        if data_start_line is None:
            raise ValueError(f"Could not find DATA line in PCD file: {pcd_path}")

        points = np.loadtxt(pcd_path, skiprows=data_start_line)

        if points.ndim == 1:
            points = points.reshape(1, -1)

        if points.shape[1] < 3:
            raise ValueError(f"Expected at least x,y,z fields in: {pcd_path}")

        return points

    @staticmethod
    def _get_first_cuboid(object_data: Dict[str, Any]):
        cuboid = object_data.get("cuboid")

        if cuboid is None:
            return None

        if isinstance(cuboid, list):
            if len(cuboid) == 0:
                return None
            return cuboid[0]

        return cuboid

    @staticmethod
    def _get_attribute(cuboid: Dict[str, Any], attribute_name: str):
        attrs = cuboid.get("attributes", {})

        for attr_type, attr_list in attrs.items():
            if not isinstance(attr_list, list):
                continue

            for attr in attr_list:
                if attr.get("name") == attribute_name:
                    return attr.get("val")

        return None

    @classmethod
    def parse_openlabel_boxes(cls, label_path: Path) -> List[Dict[str, Any]]:
        """
        Parse OpenLABEL cuboids.

        Cuboid format:
          [x, y, z, qx, qy, qz, qw, length, width, height]

        Returns:
          list of boxes with center, dimensions, quaternion, yaw, class, track id.
        """
        with open(label_path, "r") as f:
            data = json.load(f)

        frames = data["openlabel"]["frames"]
        boxes = []

        for _, frame in frames.items():
            objects = frame.get("objects", {})

            for object_id, obj in objects.items():
                object_data = obj.get("object_data", {})
                class_name = object_data.get("type", "UNKNOWN")

                cuboid = cls._get_first_cuboid(object_data)

                if cuboid is None:
                    continue

                val = cuboid.get("val", [])

                if len(val) != 10:
                    continue

                x, y, z = val[0], val[1], val[2]
                qx, qy, qz, qw = val[3], val[4], val[5], val[6]
                length, width, height = val[7], val[8], val[9]

                # For road objects here, qx/qy are usually zero.
                # This gives yaw around vertical axis.
                yaw = 2.0 * math.atan2(qz, qw)

                boxes.append(
                    {
                        "track_id": object_id,
                        "class_name": class_name,
                        "center": np.array([x, y, z], dtype=np.float64),
                        "quaternion": np.array([qx, qy, qz, qw], dtype=np.float64),
                        "dims_lwh": np.array([length, width, height], dtype=np.float64),
                        "yaw": yaw,
                        "num_points": cls._get_attribute(cuboid, "num_points"),
                        "sensor_id": cls._get_attribute(cuboid, "sensor_id"),
                        "occlusion_level": cls._get_attribute(cuboid, "occlusion_level"),
                    }
                )

        return boxes
