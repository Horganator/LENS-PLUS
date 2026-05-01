from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time
import os
import sys
import argparse

os.environ["XFORMERS_DISABLED"] = "1"
sys.modules["xformers"] = None
sys.modules["xformers.ops"] = None

import json
import re
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPTH_ANYTHING_PATH = os.path.join(BASE_DIR, "Depth-Anything-V2", "metric_depth")
sys.path.insert(0, DEPTH_ANYTHING_PATH)

from depth_anything_v2.dpt import DepthAnythingV2

from object_distance import (
    BoundingBox,
    CameraIntrinsics,
    ObjectDistanceEstimator,
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 360

DEMO_BATCH_SIZE = 3 # merge groups of frames every n groups

def read_and_resize(path: Path) -> np.ndarray | None:
    frame = cv2.imread(str(path))
    if frame is not None:
        return cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    return None

class DepthZoneAnalyser:
    ZONE_THRESHOLDS_M = { # proximity in metres away
        "very_close": 0.8,
        "close":      2.0,
        "mid":        4.0,
        "far":        8.0,
    }

    NEAREST_PERCENTILE = 5.0

    def get_zones(self, depth: np.ndarray) -> dict:
        h, w = depth.shape
        return {
            "left":   depth[:, :w // 3],
            "centre": depth[:, w // 3: 2 * w // 3],
            "right":  depth[:, 2 * w // 3:],
            "floor":  depth[h // 2:, :],
        }

    def _nearest_in(self, zone: np.ndarray) -> float:
        valid = zone[np.isfinite(zone) & (zone > 0)]
        if valid.size == 0:
            return float("inf")
        return float(np.percentile(valid, self.NEAREST_PERCENTILE))

    def _mean_in(self, zone: np.ndarray) -> float:
        valid = zone[np.isfinite(zone) & (zone > 0)]
        if valid.size == 0:
            return float("inf")
        return float(np.mean(valid))

    def analyse_zones(self, depth: np.ndarray) -> dict:
        zones = self.get_zones(depth)

        zone_nearest_m = {k: self._nearest_in(v) for k, v in zones.items()}
        zone_mean_m = {k: self._mean_in(v) for k, v in zones.items()}

        left_mean = zone_mean_m["left"]
        centre_mean = zone_mean_m["centre"]
        right_mean = zone_mean_m["right"]
        centre_near = zone_nearest_m["centre"]
        floor_near = zone_nearest_m["floor"]

        dominant = min(
            ["left", "centre", "right"],
            key=lambda k: zone_mean_m[k]
        )

        buffer_m = 0.15

        if left_mean < centre_mean - buffer_m and left_mean < right_mean:
            direction_warning = "on your left"
        elif right_mean < centre_mean - buffer_m and right_mean < left_mean:
            direction_warning = "on your right"
        else:
            direction_warning = "ahead"

        primary_hazard_m = min(centre_near, floor_near)

        T = self.ZONE_THRESHOLDS_M
        if primary_hazard_m < T["very_close"]:
            proximity_status = "VERY_CLOSE"
            proximity_detail = f"obstacle immediately {direction_warning}"
        elif primary_hazard_m < T["close"]:
            proximity_status = "CLOSE"
            proximity_detail = f"obstacle close {direction_warning}"
        elif primary_hazard_m < T["mid"]:
            proximity_status = "MID"
            proximity_detail = f"obstacle at mid range {direction_warning}"
        elif primary_hazard_m < T["far"]:
            proximity_status = "FAR"
            proximity_detail = f"obstacle in the distance {direction_warning}"
        else:
            proximity_status = "CLEAR"
            proximity_detail = "path appears clear"

        # scene nearest point overall (for logging)
        positive = depth[depth > 0]
        scene_nearest_m = (
            round(float(np.percentile(positive, self.NEAREST_PERCENTILE)), 3)
            if positive.size > 0 else float("inf")
        )

        return {
            "zone_nearest_m": {k: round(v, 3) for k, v in zone_nearest_m.items()},
            "zone_mean_m": {k: round(v, 3) for k, v in zone_mean_m.items()},
            "nearest_m": scene_nearest_m,
            "primary_hazard_m": round(primary_hazard_m, 3),
            "proximity_status": proximity_status,
            "proximity_detail": proximity_detail,
            "dominant_zone": dominant,
            "direction_warning": direction_warning,
        }


class DepthEstimator:
    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    def __init__(
        self,
        checkpoint_path: str,
        frames_root: str,
        variant: str = "vits",
        target_fps: int = 10,
        max_depth: float = 10.0,
        camera_hfov_deg: float = 70.0,
        write_video: bool = True
    ):
        self.frames_root = Path(frames_root)
        self.checkpoint_path = checkpoint_path
        self.variant = variant
        self.target_fps = target_fps
        self.max_depth = max_depth

        self.prev_depth = None

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"DepthEstimator using device: {self.device}")

        self.model = self._load_model()
        self.analyser = DepthZoneAnalyser()
        self.intrinsics = CameraIntrinsics(
            width=OUTPUT_WIDTH,
            height=OUTPUT_HEIGHT,
            hfov_deg=camera_hfov_deg,
        )
        self.distance_estimator = ObjectDistanceEstimator(self.intrinsics)
        self.write_video = write_video

    def _load_model(self) -> DepthAnythingV2:
        config = {**self.MODEL_CONFIGS[self.variant], "max_depth": self.max_depth}
        model = DepthAnythingV2(**config)
        model.load_state_dict(
            torch.load(self.checkpoint_path, map_location="cpu")
        )
        model.to(self.device).eval()
        return model

    def natural_key(self, path: Path):
        return [
            int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", path.name)
        ]

    def find_latest_artifact(self) -> Path:
        artifacts = [p for p in self.frames_root.iterdir() if p.is_dir()]
        if not artifacts:
            raise FileNotFoundError(f"No artifacts found in {self.frames_root}")
        artifacts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return artifacts[0]

    def get_group_folders(self, artifact_dir: Path) -> list[Path]:
        groups = [
            p for p in artifact_dir.iterdir()
            if p.is_dir() and p.name.startswith("group-")
        ]
        groups.sort(key=self.natural_key)
        return groups

    def load_frame_paths(self, folder: Path) -> list[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        frames = [p for p in folder.iterdir() if p.suffix.lower() in exts]
        frames.sort(key=self.natural_key)
        return frames

    def load_detections_for_frame(self, frame_path: Path) -> list[BoundingBox]:
        sidecar = frame_path.with_suffix(".detections.json")
        if not sidecar.exists():
            return []

        try:
            data = json.loads(sidecar.read_text())
            detections = data.get("detections", [])
            bboxes: list[BoundingBox] = []
            for d in detections:
                label = d.get("label", "")
                confidence = d.get("confidence", 1.0)
                if "xyxy" in d:
                    bboxes.append(BoundingBox.from_xyxy(d["xyxy"], label, confidence))
                elif "xywh" in d:
                    bboxes.append(BoundingBox.from_xywh(d["xywh"], label, confidence))
            return bboxes
        except Exception as error:
            print(f"  detection parse failed for {frame_path.name}: {error}")
            return []
        
    def load_navigation_for_frame(self, frame_path: Path) -> dict:
        sidecar = frame_path.with_suffix(".navigation.json")
        if not sidecar.exists():
            return {}

        try:
            data = json.loads(sidecar.read_text())
            return data.get("segmentation", {})
        except Exception as error:
            print(f"  navigation parse failed for {frame_path.name}: {error}")
            return {}

    def write_depth_to_json(
        self,
        frame_path: Path,
        analysis: dict,
        per_object: list,
        consistency: float,
        variance: float,
        edge_density: float,
        valid_ratio: float,
        status_changed: bool,
        zone_stability: float,
        latency_ms: float,
    ):
        sidecar = frame_path.with_suffix(".navigation.json")
        try:
            existing = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        except Exception:
            existing = {}

        existing["frame"] = frame_path.name
        existing["timestamp"] = frame_path.stem.split("-")[-1]

        existing["depth"] = {
            "proximity_status":     analysis["proximity_status"],
            "proximity_detail":     analysis["proximity_detail"],
            "primary_hazard_m":     analysis["primary_hazard_m"],
            "direction_warning":    analysis["direction_warning"],
            "dominant_zone":        analysis["dominant_zone"],
            "zone_nearest_m":       analysis["zone_nearest_m"],
            "zone_mean_m":          analysis["zone_mean_m"],
            "nearest_m":            analysis["nearest_m"],
            "objects":              per_object,
            "temporal_consistency": round(consistency, 4),
            "depth_variance":       variance,
            "depth_edge_density":   edge_density,
            "valid_pixel_ratio":    valid_ratio,
            "status_changed":       status_changed,
            "zone_stability_m":     zone_stability,
            "inference_latency_ms": round(latency_ms, 2),
        }

        sidecar.write_text(json.dumps(existing, indent=2))

    def infer_real_fps(self, frame_paths: list[Path]) -> int:
        if len(frame_paths) < 2:
            return self.target_fps

        def parse_timestamp(path: Path) -> datetime:
            stamp = path.stem.split("-")[-1]
            return datetime.strptime(stamp, "%Y%m%dT%H%M%S%fZ")

        try:
            start = parse_timestamp(frame_paths[0])
            end = parse_timestamp(frame_paths[-1])
            seconds = (end - start).total_seconds()
            if seconds <= 0:
                return self.target_fps
            return max(1, round(len(frame_paths) / seconds))
        except Exception:
            return self.target_fps

    def get_depth_predictions(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            depth = self.model.infer_image(rgb)
        return depth

    def depth_variance(self, depth: np.ndarray) -> float:
        valid = depth[depth > 0]
        return round(float(np.var(valid)), 4) if valid.size > 0 else 0.0

    def depth_edge_density(self, depth: np.ndarray) -> float:
        clipped = np.clip(depth, 0, self.max_depth).astype(np.float32)
        edges = cv2.Sobel(clipped, cv2.CV_32F, 1, 0) + cv2.Sobel(clipped, cv2.CV_32F, 0, 1)
        return round(float(np.mean(np.abs(edges))), 4)
    
    def valid_pixel_ratio(self, depth: np.ndarray) -> float:
        total = depth.size
        valid = np.sum(np.isfinite(depth) & (depth > 0))
        return round(float(valid / total), 4)

    def compute_temporal_consistency(self, a, b) -> float:
        if a is None or b is None:
            return 0.0
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        return float(np.abs(a - b).mean())

    def apply_temporal_smoothing(self, current, previous, alpha=0.75) -> np.ndarray:
        if previous is None:
            return current
        if current.shape != previous.shape:
            previous = cv2.resize(previous, (current.shape[1], current.shape[0]))
        return alpha * current + (1.0 - alpha) * previous

    def create_visualisation(
        self,
        frame: np.ndarray,
        depth: np.ndarray,
        analysis: dict,
        frame_idx: int,
        located: list | None = None,
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        valid_depth = depth[depth > 0]
        if valid_depth.size > 0:
            d_min = np.percentile(valid_depth, 2)
            d_max = np.percentile(valid_depth, 98)
        else:
            d_min, d_max = 0.0, self.max_depth
        
        if d_max - d_min > 1e-3:
            norm = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)
        else:
            norm = np.zeros_like(depth)

        norm = 1.0 - norm

        cmap = plt.get_cmap("Spectral_r")
        colourmap = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
        colourmap = cv2.cvtColor(colourmap, cv2.COLOR_RGB2BGR)
        colourmap = cv2.resize(colourmap, (w, h))

        overlay = cv2.addWeighted(frame, 0.45, colourmap, 0.70, 0)

        # Bounding boxes with metric distance labels
        if located:
            for obj in located:
                bb = obj.bbox
                x1, y1 = int(bb.x1), int(bb.y1)
                x2, y2 = int(bb.x2), int(bb.y2)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
                tag = f"{obj.label or 'obj'} {obj.distance_from_camera_m:.1f}m"
                cv2.putText(
                    overlay, tag, (x1, max(12, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                )

        zm = analysis.get("zone_nearest_m", {})

        lines = [
            f"frame: {frame_idx}",
            f"status: {analysis['proximity_status']}",
            f"detail: {analysis['proximity_detail']}",
            f"nearest: {analysis.get('primary_hazard_m', 0):.2f} m",
            f"L:{zm.get('left', 0):.2f}m  "
            f"C:{zm.get('centre', 0):.2f}m  "
            f"R:{zm.get('right', 0):.2f}m  "
            f"F:{zm.get('floor', 0):.2f}m",
        ]

        status_colours = {
            "VERY_CLOSE": (0, 0, 255),    # red
            "CLOSE":      (0, 165, 255),  # amber
            "MID":        (0, 255, 255),  # yellow
            "FAR":        (0, 255, 0),    # green
            "CLEAR":      (255, 255, 255) # white
        }

        y = 30
        for i, line in enumerate(lines):
            colour = (255, 255, 255)
            if i == 1:
                colour = status_colours.get(analysis["proximity_status"], (255, 255, 255))
            cv2.putText(overlay, line, (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            y += 28

        return overlay

    def process_group(self, frame_paths: list[Path], output_path: Path) -> dict:
        if not frame_paths:
            return {}
        
        print(f"Pre-fetching {len(frame_paths)} frames using threads...")
        t_io = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as executor:
            frames = list(executor.map(read_and_resize, frame_paths))
        print(f"Loaded in {(time.perf_counter() - t_io)*1000:.0f}ms")

        self.prev_depth = None

        fps = self.infer_real_fps(frame_paths)
        out = None
        if self.write_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                str(output_path), fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT)
            )
            if not out.isOpened():
                raise RuntimeError(f"Could not open writer: {output_path}")

        prev_status = None
        prev_zone_nearest = None
        total_frames = 0

        for idx, (frame_path, frame) in enumerate(zip(frame_paths, frames)):
            if frame is None:
                continue

            frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

            t0 = time.perf_counter()
            depth = self.get_depth_predictions(frame)
            latency_ms = (time.perf_counter() - t0) * 1000

            depth = self.apply_temporal_smoothing(depth, self.prev_depth)
            analysis = self.analyser.analyse_zones(depth)
            consistency = self.compute_temporal_consistency(depth, self.prev_depth)

            variance = self.depth_variance(depth)
            edge_density = self.depth_edge_density(depth)
            valid_ratio = self.valid_pixel_ratio(depth)
            bboxes = self.load_detections_for_frame(frame_path)
            located = self.distance_estimator.locate(depth, bboxes)
            per_object = self.distance_estimator.distances_from_camera(located)
            status_changed = (
                prev_status is not None and
                analysis["proximity_status"] != prev_status
            )
            zone_stability = (
                round(float(np.mean([
                    abs(analysis["zone_nearest_m"][k] - prev_zone_nearest[k])
                    for k in ["left", "centre", "right", "floor"]
                ])), 4)
                if prev_zone_nearest is not None else 0.0
            )

            if out is not None:
                viz = self.create_visualisation(frame, depth, analysis, idx, located)
                viz = cv2.resize(viz, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                out.write(viz)

            self.write_depth_to_json(
                frame_path=frame_path,
                analysis=analysis,
                per_object=per_object,
                consistency=consistency,
                variance=variance,
                edge_density=edge_density,
                valid_ratio=valid_ratio,
                status_changed=status_changed,
                zone_stability=zone_stability,
                latency_ms=latency_ms,
            )

            self.prev_depth = depth
            prev_status = analysis["proximity_status"]
            prev_zone_nearest = analysis["zone_nearest_m"].copy()
            total_frames += 1

            distance_summary = ""
            if per_object:
                nearest = min(per_object, key=lambda o: o["distance_m"])
                distance_summary = (
                    f" — {nearest['label']} @ {nearest['distance_m']}m"
                )

            print(
                f"  [{frame_path.name}] {analysis['proximity_status']} — "
                f"{analysis['proximity_detail']}{distance_summary}  ({latency_ms:.0f}ms)"
            )

        if out is not None:
            out.release()

        return {
            "total_frames": total_frames,
            "inferred_fps": fps,
        }

    def merge_n_groups(self, artifact_name: str, group_keys: list[str], batch_num: int):
        output_dir = Path(OUTPUT_DIR)
        mp4s = [output_dir / f"{k}.mp4" for k in group_keys
                if (output_dir / f"{k}.mp4").exists()]
        if not mp4s:
            return

        demo_path = output_dir / f"{artifact_name}_demo_{batch_num}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(demo_path), fourcc, self.target_fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT)
        )

        for mp4 in mp4s:
            cap = cv2.VideoCapture(str(mp4))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                out.write(frame)
            cap.release()

        out.release()
        print(f"  Depth Demos: {demo_path.name}  ({len(mp4s)} groups)")
    
    def run(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        processed_groups = set()
        processed_order = []
        last_new_group_time = time.time()
        demo_batch_num = 1
        last_merged_count = 0

        while True:
            try:
                artifact = self.find_latest_artifact()
            except FileNotFoundError:
                print("Waiting for a session to start...")
                time.sleep(2)
                continue

            groups = self.get_group_folders(artifact)

            is_closed = False
            manifest_path = artifact / "session.json"
            if manifest_path.exists():
                try:
                    manifest_data = json.loads(manifest_path.read_text())
                    if manifest_data.get("closed_at") is not None:
                        is_closed = True
                except Exception:
                    pass

            if is_closed:
                safe_groups = groups
            else:
                safe_groups = groups[:-1] if len(groups) > 1 else []

            new_group_found = False

            for group in safe_groups:
                key = f"{artifact.name}_{group.name}"

                if key in processed_groups:
                    continue

                frame_paths = self.load_frame_paths(group)
                if not frame_paths:
                    continue

                print(f"[Depth] Processing: {group.name}")

                video_path = Path(OUTPUT_DIR) / f"{key}.mp4"

                try:
                    results = self.process_group(frame_paths, video_path)
                    print(f"  Done: {group.name} ({results['total_frames']} frames)")

                    processed_groups.add(key)
                    processed_order.append(key)
                    new_group_found = True

                except Exception as error:
                    print(f"  Group failed: {group.name} — {error}")

            if self.write_video:
                unmerged = len(processed_order) - last_merged_count
                if unmerged >= DEMO_BATCH_SIZE:
                    batch_keys = processed_order[last_merged_count : last_merged_count + DEMO_BATCH_SIZE]
                    self.merge_n_groups(artifact.name, batch_keys, demo_batch_num)
                    last_merged_count += DEMO_BATCH_SIZE
                    demo_batch_num    += 1
                elif is_closed and unmerged > 0:
                    print(f"  Flushing final {unmerged} group(s) into demo...")
                    batch_keys = processed_order[last_merged_count:]
                    self.merge_n_groups(artifact.name, batch_keys, demo_batch_num)
                    last_merged_count += unmerged
                    demo_batch_num += 1

            if new_group_found:
                last_new_group_time = time.time()

            idle = time.time() - last_new_group_time
            if idle > 10 and idle % 30 < 2: # print every 30s
                print(f"  Waiting for new groups... ({idle:.0f}s idle)")

            time.sleep(2)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-video", action="store_true",
                        help="Skip mp4 writing")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()

    model = DepthEstimator(
        checkpoint_path=os.path.join(
            BASE_DIR, "checkpoints", "depth_anything_v2_metric_hypersim_vits.pth"
        ),
        frames_root=os.path.abspath(
            os.path.join(BASE_DIR, "..", "..", "api", "app", "session_artifacts")
        ),
        target_fps=10,
        max_depth=10.0,
        camera_hfov_deg=70.0,
        write_video=not args.no_video,
    )
    model.run()