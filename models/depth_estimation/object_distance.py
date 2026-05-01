from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""
    confidence: float = 1.0

    @classmethod
    def from_xyxy(cls, xyxy, label: str = "", confidence: float = 1.0) -> "BoundingBox":
        x1, y1, x2, y2 = xyxy
        return cls(float(x1), float(y1), float(x2), float(y2), label, confidence)

    @classmethod
    def from_xywh(cls, xywh, label: str = "", confidence: float = 1.0) -> "BoundingBox":
        x, y, w, h = xywh
        return cls(float(x), float(y), float(x + w), float(y + h), label, confidence)


@dataclass
class LocatedObject:
    label: str
    confidence: float
    bbox: BoundingBox
    depth_m: float
    position_m: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def distance_from_camera_m(self) -> float:
        return float(np.linalg.norm(self.position_m))

    @property
    def lateral_offset_m(self) -> float:
        return float(self.position_m[0])


class CameraIntrinsics:
    def __init__(self, width: int, height: int, hfov_deg: float = 70.0):
        self.width = int(width)
        self.height = int(height)
        self.hfov_deg = float(hfov_deg)
        self.fx = self.width / (2.0 * np.tan(np.radians(self.hfov_deg) / 2.0))
        self.fy = self.fx  # assume square pixels
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def back_project(self, u: float, v: float, z: float) -> np.ndarray:
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([x, y, z], dtype=np.float32)


class ObjectDistanceEstimator:
    def __init__(self, intrinsics: CameraIntrinsics):
        self.intrinsics = intrinsics

    def _sample_object_depth(
        self,
        depth_map: np.ndarray,
        bbox: BoundingBox,
        centre_shrink: float = 0.5,
    ) -> float | None:
        h, w = depth_map.shape[:2]

        cx_box = (bbox.x1 + bbox.x2) / 2.0
        cy_box = (bbox.y1 + bbox.y2) / 2.0
        half_w = (bbox.x2 - bbox.x1) * centre_shrink / 2.0
        half_h = (bbox.y2 - bbox.y1) * centre_shrink / 2.0

        x1 = max(0, int(round(cx_box - half_w)))
        y1 = max(0, int(round(cy_box - half_h)))
        x2 = min(w, int(round(cx_box + half_w)))
        y2 = min(h, int(round(cy_box + half_h)))

        if x2 <= x1 or y2 <= y1:
            return None

        patch = depth_map[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            return None

        return float(np.median(valid))

    def locate(
        self,
        depth_map: np.ndarray,
        bboxes: Sequence[BoundingBox],
    ) -> list[LocatedObject]:
        located: list[LocatedObject] = []
        for bb in bboxes:
            z = self._sample_object_depth(depth_map, bb)
            if z is None or z <= 0:
                continue

            u = (bb.x1 + bb.x2) / 2.0
            v = (bb.y1 + bb.y2) / 2.0
            position = self.intrinsics.back_project(u, v, z)

            located.append(
                LocatedObject(
                    label=bb.label,
                    confidence=bb.confidence,
                    bbox=bb,
                    depth_m=z,
                    position_m=position,
                )
            )
        return located

    def distances_from_camera(self, located: Sequence[LocatedObject]) -> list[dict]:
        return [
            {
                "label": obj.label or "object",
                "distance_m": round(obj.distance_from_camera_m, 2),
                "direction": self._describe_direction(obj),
                "confidence": obj.confidence,
            }
            for obj in located
        ]

    def _describe_direction(self, obj: LocatedObject) -> str:
        if obj.position_m[2] <= 0:
            return "ahead"
        angle_deg = np.degrees(np.arctan2(obj.position_m[0], obj.position_m[2]))
        if angle_deg < -15:
            return "on your left"
        if angle_deg > 15:
            return "on your right"
        return "ahead"