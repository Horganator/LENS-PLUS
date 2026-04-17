import os
import sys
import json
import re
from pathlib import Path
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEEPLAB_PATH = os.path.join(BASE_DIR, "DeepLabV3Plus-Pytorch")
sys.path.insert(0, DEEPLAB_PATH)

import cv2
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO

import network


class CityscapesAccessibilityMapper:
    def __init__(self):
        self.class_names = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.walkable_classes = [1, 9]
        self.hazard_classes = [0]
        self.dynamic_obstacle_classes = [11, 12, 13, 14, 15, 16, 17, 18]

    def get_walkable_mask(self, preds):
        mask = np.zeros_like(preds, dtype=np.uint8)
        for c in self.walkable_classes:
            mask[preds == c] = 1
        return mask

    def get_hazard_mask(self, preds):
        mask = np.zeros_like(preds, dtype=np.uint8)
        for c in self.hazard_classes:
            mask[preds == c] = 1
        return mask

    def get_dynamic_obstacle_mask(self, preds):
        mask = np.zeros_like(preds, dtype=np.uint8)
        for c in self.dynamic_obstacle_classes:
            mask[preds == c] = 1
        return mask

    def get_traffic_sign_mask(self, preds):
        return (preds == 7).astype(np.uint8)

    def get_class_distribution(self, preds):
        total = preds.size
        result = {}

        for i, name in enumerate(self.class_names):
            count = np.sum(preds == i)
            pct = (count / total) * 100
            if pct > 0:
                result[name] = round(float(pct), 2)

        return result


class ImprovedSegmentation:
    def __init__(
        self,
        frames_root: str,
        yolo_model_path: str,
        deeplab_model_path: str,
        output_video_path: str,
        output_json_path: Optional[str] = None,
        target_fps: int = 10,
        use_yolo: bool = True,
        deeplab_every_n_frames: int = 2,
    ):
        self.frames_root = frames_root
        self.yolo_model_path = yolo_model_path
        self.deeplab_model_path = deeplab_model_path
        self.output_video_path = output_video_path
        self.output_json_path = (
            output_json_path
            or output_video_path.replace(".mp4", "_metrics.json")
        )

        self.target_fps = target_fps
        self.use_yolo = use_yolo
        self.deeplab_every_n_frames = deeplab_every_n_frames

        self.prev_walkable_mask = None
        self.prev_hazard_mask = None
        self.target_size = None
        self.frame_metrics = []

        self.yolo_model = YOLO(self.yolo_model_path) if self.use_yolo else None
        self.deeplab_model = self.load_deeplab()
        self.mapper = CityscapesAccessibilityMapper()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def natural_key(self, path):
        return [
            int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", path.name)
        ]

    def find_frame_folder(self):
        root = Path(self.frames_root)

        if not root.exists():
            raise FileNotFoundError(f"{root} does not exist")

        subfolders = [p for p in root.iterdir() if p.is_dir()]

        if not subfolders:
            raise FileNotFoundError(f"No subfolders found in {root}")

        subfolders.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        return subfolders[0]

    def load_frame_paths(self, folder):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}

        frames = [
            p for p in folder.iterdir()
            if p.suffix.lower() in exts
        ]

        if not frames:
            raise FileNotFoundError(f"No image frames found in {folder}")

        frames.sort(key=self.natural_key)
        return frames

    def load_deeplab(self):
        model = network.modeling.__dict__["deeplabv3plus_mobilenet"](
            num_classes=19,
            output_stride=16,
        )

        checkpoint = torch.load(
            self.deeplab_model_path,
            map_location="cpu",
            weights_only=False,
        )

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

    def get_semantic_predictions(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.deeplab_model(tensor)

        preds = outputs.max(1)[1].cpu().numpy()[0]

        h, w = image.shape[:2]

        preds = cv2.resize(
            preds.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        return preds

    def compute_temporal_iou(self, a, b):
        if a is None or b is None:
            return 0.0

        if a.shape != b.shape:
            a = cv2.resize(
                a.astype(np.uint8),
                (b.shape[1], b.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()

        return float(inter / union) if union > 0 else 0.0

    def apply_temporal_smoothing(
        self,
        current_mask,
        previous_mask,
        alpha=0.7,
    ):
        if previous_mask is None:
            return current_mask.astype(np.uint8)

        if current_mask.shape != previous_mask.shape:
            previous_mask = cv2.resize(
                previous_mask.astype(np.uint8),
                (current_mask.shape[1], current_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        smoothed = (
            alpha * current_mask.astype(np.float32)
            + (1.0 - alpha) * previous_mask.astype(np.float32)
        )

        return (smoothed > 0.5).astype(np.uint8)

    def analyze_safety(self, walkable, hazard, dynamic):
        h, w = walkable.shape
        total = h * w

        walkable_pct = float(np.sum(walkable) / total * 100)
        hazard_pct = float(np.sum(hazard) / total * 100)
        obstacle_pct = float(np.sum(dynamic) / total * 100)

        center_col = walkable[:, w // 2]
        path_clear = bool(np.sum(center_col) / len(center_col) > 0.6)

        if hazard_pct > 30:
            status = "UNWALKABLE"
            detail = "high road presence"
        elif walkable_pct >= 50 and path_clear:
            status = "WALKABLE"
            detail = "clear path"
        elif walkable_pct >= 30:
            status = "WALKABLE"
            detail = "proceed with caution"
        elif walkable_pct >= 15:
            status = "WALKABLE"
            detail = "limited safe area"
        else:
            status = "UNWALKABLE"
            detail = "no safe path"

        if obstacle_pct > 20:
            warning = "multiple obstacles"
        elif obstacle_pct > 5:
            warning = "obstacles present"
        else:
            warning = "path clear"

        return {
            "walkable_percentage": round(walkable_pct, 2),
            "hazard_percentage": round(hazard_pct, 2),
            "obstacle_percentage": round(obstacle_pct, 2),
            "path_clear": path_clear,
            "navigation_status": status,
            "navigation_detail": detail,
            "obstacle_warning": warning,
        }

    def create_visualization(
        self,
        image,
        yolo_results,
        walkable,
        hazard,
        dynamic,
        signs,
        safety,
        frame_idx,
    ):
        if yolo_results is not None:
            overlay = yolo_results[0].plot()

            if overlay.shape[:2] != image.shape[:2]:
                overlay = cv2.resize(
                    overlay,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
        else:
            overlay = image.copy()

        overlay = overlay.astype(np.float32)
        h, w = overlay.shape[:2]

        def fix_mask(mask):
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            return mask.astype(np.uint8)

        walkable = fix_mask(walkable)
        hazard = fix_mask(hazard)
        dynamic = fix_mask(dynamic)
        signs = fix_mask(signs)

        green = np.zeros_like(overlay)
        green[:] = [0, 255, 0]

        red = np.zeros_like(overlay)
        red[:] = [0, 0, 255]

        yellow = np.zeros_like(overlay)
        yellow[:] = [0, 255, 255]

        cyan = np.zeros_like(overlay)
        cyan[:] = [255, 255, 0]

        def blend(base, mask, color, alpha):
            mask3 = np.stack([mask, mask, mask], axis=2)
            return np.where(
                mask3 > 0,
                base * (1 - alpha) + color * alpha,
                base,
            )

        overlay = blend(overlay, walkable, green, 0.40)
        overlay = blend(overlay, hazard, red, 0.50)
        overlay = blend(overlay, dynamic, yellow, 0.50)
        overlay = blend(overlay, signs, cyan, 0.50)

        overlay = overlay.astype(np.uint8)

        lines = [
            f"frame: {frame_idx}",
            f"walkable: {safety['walkable_percentage']}%",
            f"hazard: {safety['hazard_percentage']}%",
            f"obstacles: {safety['obstacle_percentage']}%",
            safety["navigation_status"],
            safety["navigation_detail"],
            safety["obstacle_warning"],
        ]

        y = 30
        for line in lines:
            cv2.putText(
                overlay,
                line,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            y += 28

        return overlay

    def save_metrics(self):
        if not self.frame_metrics:
            return

        avg_walkable = np.mean(
            [f["safety"]["walkable_percentage"] for f in self.frame_metrics]
        )

        avg_hazard = np.mean(
            [f["safety"]["hazard_percentage"] for f in self.frame_metrics]
        )

        avg_iou = np.mean(
            [
                f["temporal_consistency"]["walkable_iou"]
                for f in self.frame_metrics
            ]
        )

        summary = {
            "total_frames": len(self.frame_metrics),
            "average_metrics": {
                "walkable_percentage": round(float(avg_walkable), 2),
                "hazard_percentage": round(float(avg_hazard), 2),
                "temporal_consistency": round(float(avg_iou), 3),
            },
            "frame_details": self.frame_metrics,
        }

        with open(self.output_json_path, "w") as f:
            json.dump(summary, f, indent=2)

    def run(self):
        os.makedirs(os.path.dirname(self.output_video_path), exist_ok=True)

        frame_folder = self.find_frame_folder()
        frame_paths = self.load_frame_paths(frame_folder)

        print("Using:", frame_folder)
        print("Frames:", len(frame_paths))

        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            return

        h, w = first.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            self.target_fps,
            (w, h)
        )

        processed_count = 0
        segmentation_cache = None

        for frame_path in frame_paths:

            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            frame = cv2.resize(frame, (w, h))

            if self.use_yolo:
                yolo_results = self.yolo_model(
                    frame,
                    conf=0.5,
                    verbose=False
                )
            else:
                yolo_results = None

            if processed_count % self.deeplab_every_n_frames == 0:
                semantic = self.get_semantic_predictions(frame)
                segmentation_cache = semantic
            else:
                semantic = segmentation_cache

            walkable = self.mapper.get_walkable_mask(semantic)
            hazard = self.mapper.get_hazard_mask(semantic)
            dynamic = self.mapper.get_dynamic_obstacle_mask(semantic)
            signs = self.mapper.get_traffic_sign_mask(semantic)

            walkable = self.apply_temporal_smoothing(
                walkable,
                self.prev_walkable_mask
            )

            hazard = self.apply_temporal_smoothing(
                hazard,
                self.prev_hazard_mask
            )

            safety = self.analyze_safety(
                walkable,
                hazard,
                dynamic
            )

            viz = self.create_visualization(
                frame,
                yolo_results,
                walkable,
                hazard,
                dynamic,
                signs,
                safety,
                processed_count
            )

            viz = cv2.resize(viz, (w, h))
            out.write(viz)

            self.prev_walkable_mask = walkable
            self.prev_hazard_mask = hazard

            processed_count += 1

        out.release()
        self.save_metrics()

        print("Done:", self.output_video_path)


if __name__ == "__main__":
    model = ImprovedSegmentation(
        frames_root=os.path.abspath(
            os.path.join(
                BASE_DIR,
                "..",
                "..",
                "..",
                "api",
                "app",
                "session_artifacts",
            )
        ),
        yolo_model_path="yolov8n-seg.pt",
        deeplab_model_path=os.path.join(
            BASE_DIR,
            "deeplabv3plus-mobilenet.pth",
        ),
        output_video_path=os.path.join(
            BASE_DIR,
            "..",
            "output",
            "improved_segmented_video.mp4",
        ),
        target_fps=10,
    )

    model.run()