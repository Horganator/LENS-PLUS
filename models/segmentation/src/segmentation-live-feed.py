import os
import sys
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
import json
import argparse
from time import perf_counter

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
DEEPLAB_PATH = BASE_DIR / "DeepLabV3Plus-Pytorch"
DEEPLAB_NETWORK_PATH = DEEPLAB_PATH / "network"
if not DEEPLAB_NETWORK_PATH.exists():
    raise RuntimeError(
        f"Missing DeepLab code at {DEEPLAB_NETWORK_PATH}. "
        "Clone https://github.com/VainF/DeepLabV3Plus-Pytorch into "
        f"{DEEPLAB_PATH}."
    )
sys.path.insert(0, str(DEEPLAB_PATH))

import cv2
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO
import network
if not hasattr(network, "modeling"):
    raise RuntimeError(
        "Imported 'network' module does not expose 'modeling'. "
        f"Expected local module under {DEEPLAB_NETWORK_PATH}. "
        "Check your PYTHONPATH and DeepLab checkout."
    )
 
APP_DIR = PROJECT_ROOT / "api" / "app"

OUTPUT_DIR = PROJECT_ROOT / "models" / "segmentation" / "output"

OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 360

FRAME_SIZE = (OUTPUT_WIDTH, OUTPUT_HEIGHT)

# CITYSCAPES

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

        self.walkable_classes = [1, 9]  # sidewalk, terrain
        self.hazard_classes = [0]       # road
        self.dynamic_obstacle_classes = [
            11, 12, 13, 14, 15, 16, 17, 18
        ]

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


# segmentation and navigation

class ImprovedSegmentation:
    def __init__(
        self,
        frames_root: str,
        yolo_model_path: str,
        deeplab_model_path: str,
        target_fps: int = 10,
        use_yolo: bool = True,
        deeplab_every_n_frames: int = 2,
        write_video: bool = True,
    ):
        self.frames_root = Path(frames_root)
        self.target_fps = target_fps
        self.use_yolo = use_yolo
        self.deeplab_every_n_frames = deeplab_every_n_frames
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_device = 0 if self.device == "cuda" else "cpu"
        print(f"Segmentation using device: {self.device}")

        self.mapper = CityscapesAccessibilityMapper()

        self.prev_walkable_mask = None
        self.prev_hazard_mask = None

        self.yolo_model = YOLO(yolo_model_path) if use_yolo else None
        self.deeplab_model = self.load_deeplab(deeplab_model_path)

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((360, 640)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.write_video = write_video

    def get_model_size_mb(self):
        total_params = sum(p.numel() for p in self.deeplab_model.parameters())
        total_bytes = total_params * 4
        return round(total_bytes / (1024 * 1024), 2)


    def binary_iou(self, pred, target):
        inter = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()

        if union == 0:
            return 1.0

        return float(inter / union)


    def dice_score(self, pred, target):
        inter = np.logical_and(pred, target).sum()
        denom = pred.sum() + target.sum()

        if denom == 0:
            return 1.0

        return float((2 * inter) / denom)


    def focal_loss_binary(self, pred, target, gamma=2.0, alpha=0.25):
        eps = 1e-6

        p = pred.astype(np.float32)
        t = target.astype(np.float32)

        p = np.clip(p, eps, 1 - eps)

        loss_pos = -alpha * t * ((1 - p) ** gamma) * np.log(p)
        loss_neg = -(1 - alpha) * (1 - t) * (p ** gamma) * np.log(1 - p)

        return float(np.mean(loss_pos + loss_neg))


    def mean_surface_distance(self, pred, target):
        pred_pts = np.column_stack(np.where(pred > 0))
        gt_pts = np.column_stack(np.where(target > 0))

        if len(pred_pts) == 0 or len(gt_pts) == 0:
            return 0.0

        dists = []

        sample_pred = pred_pts[::max(1, len(pred_pts) // 200)]
        sample_gt = gt_pts[::max(1, len(gt_pts) // 200)]

        for p in sample_pred:
            diff = sample_gt - p
            dist = np.sqrt((diff ** 2).sum(axis=1)).min()
            dists.append(dist)

        return float(np.mean(dists))


    def calculate_group_metrics(
        self,
        ious,
        inference_times,
        focal_losses,
        dices,
        msds
    ):
        if not ious:
            return {}

        miou = float(np.mean(ious))

        metrics = {
            "mIOU": round(miou, 4),
            "IOU_distribution": {
                "min": round(float(min(ious)), 4),
                "max": round(float(max(ious)), 4),
                "mean": round(float(np.mean(ious)), 4),
                "median": round(float(statistics.median(ious)), 4),
            },
            "focal_loss": round(float(np.mean(focal_losses)), 6),
            "mAP": round(float(np.mean(ious)), 4),
            "inference_time_ms": round(float(np.mean(inference_times)), 2),
            "model_size_mb": self.get_model_size_mb(),
            "Dice_Similarity_Coefficient": round(float(np.mean(dices)), 4),
            "Jaccard_Index": round(float(np.mean(ious)), 4),
            "Mean_Surface_Distance": round(float(np.mean(msds)), 4),
        }

        return metrics


    def save_group_json(
        self,
        artifact_name,
        group_name,
        metrics
    ):
        os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

        path = Path(JSON_OUTPUT_DIR) / f"{artifact_name}_{group_name}.json"

        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)


    def natural_key(self, path):
        return [
            int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", path.name)
        ]

    def find_latest_artifact(self):
        artifacts = [
            p for p in self.frames_root.iterdir()
            if p.is_dir()
        ]

        if not artifacts:
            raise FileNotFoundError("No artifacts found")

        artifacts.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        return artifacts[0]

    def get_group_folders(self, artifact_dir):
        groups = [
            p for p in artifact_dir.iterdir()
            if p.is_dir() and p.name.startswith("group-")
        ]

        groups.sort(key=self.natural_key)
        return groups

    def load_frame_paths(self, folder):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}

        frames = [
            p for p in folder.iterdir()
            if p.suffix.lower() in exts
        ]

        frames.sort(key=self.natural_key)
        return frames

    # loading the model

    def load_deeplab(self, path):
        model = network.modeling.__dict__["deeplabv3plus_mobilenet"](
            num_classes=19,
            output_stride=16,
        )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device).eval()
        return model


    def infer_real_fps(self, frame_paths):
        if len(frame_paths) < 2:
            return self.target_fps

        def parse_timestamp(path):
            stamp = path.stem.split("-")[-1]
            return datetime.strptime(
                stamp,
                "%Y%m%dT%H%M%S%fZ"
            )

        start = parse_timestamp(frame_paths[0])
        end = parse_timestamp(frame_paths[-1])

        seconds = (end - start).total_seconds()

        if seconds <= 0:
            return self.target_fps

        fps = len(frame_paths) / seconds

        return max(1, round(fps))


    def get_semantic_predictions(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.deeplab_model(tensor)

        preds = outputs.max(1)[1].cpu().numpy()[0]

        preds = preds.astype(np.uint8)
        preds = cv2.resize(preds, FRAME_SIZE[::-1], interpolation=cv2.INTER_NEAREST)

        return preds


    def apply_temporal_smoothing(
        self,
        current_mask,
        previous_mask
    ):
        if previous_mask is None:
            return current_mask.astype(np.uint8)

        if current_mask.shape != previous_mask.shape:
            previous_mask = cv2.resize(
                previous_mask.astype(np.uint8),
                (
                    current_mask.shape[1],
                    current_mask.shape[0],
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        smoothed = (
            0.7 * current_mask.astype(np.float32)
            + 0.3 * previous_mask.astype(np.float32)
        )

        return (smoothed > 0.5).astype(np.uint8)


    def analyze_navigation(
        self,
        walkable,
        hazard,
        dynamic
    ):
        h, w = walkable.shape

        # use lower half (closer to user)
        walkable = walkable[h // 2:, :]
        hazard = hazard[h // 2:, :]
        dynamic = dynamic[h // 2:, :]

        third = w // 3

        zones = {
            "LEFT": slice(0, third),
            "CENTER": slice(third, third * 2),
            "RIGHT": slice(third * 2, w),
        }

        scores = {}

        for name, zone in zones.items():
            walk = np.sum(walkable[:, zone])
            haz = np.sum(hazard[:, zone])
            obs = np.sum(dynamic[:, zone])

            score = walk - (haz * 2) - (obs * 1.5)
            scores[name] = score

        best_zone = max(scores, key=scores.get)

        center_good = (
            scores["CENTER"]
            >= max(scores["LEFT"], scores["RIGHT"]) * 0.9
        )

        if center_good:
            direction = "FORWARD"
        elif best_zone == "LEFT":
            direction = "MOVE LEFT"
        else:
            direction = "MOVE RIGHT"

        total_pixels = walkable.size

        hazard_ratio = np.sum(hazard) / total_pixels

        if hazard_ratio > 0.35:
            status = "UNWALKABLE"
        else:
            status = "WALKABLE"

        return {
            "status": status,
            "direction": direction,
            "scores": scores,
        }

    def draw_navigation_arrow(
        self,
        frame,
        direction
    ):
        h, w = frame.shape[:2]

        start = (w // 2, h - 40)

        if direction == "FORWARD":
            end = (w // 2, h - 140)

        elif direction == "MOVE LEFT":
            end = (w // 2 - 140, h - 110)

        else:
            end = (w // 2 + 140, h - 110)

        cv2.arrowedLine(
            frame,
            start,
            end,
            (0, 255, 0),
            6,
            tipLength=0.25,
        )

    # visualization

    def create_visualization(
        self,
        image,
        yolo_results,
        walkable,
        hazard,
        dynamic,
        signs,
        nav=None
    ):
        if yolo_results is not None:
            overlay = yolo_results[0].plot()
            overlay = self.preprocess_frame(overlay)
        else:
            overlay = image.copy()

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        green = np.zeros_like(overlay)
        green[:] = [0, 255, 0]

        red = np.zeros_like(overlay)
        red[:] = [0, 0, 255]

        yellow = np.zeros_like(overlay)
        yellow[:] = [0, 255, 255]

        cyan = np.zeros_like(overlay)
        cyan[:] = [255, 255, 0]

        def blend(base, mask, color, alpha):
            if mask.shape[:2] != base.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)

            mask3 = np.stack([mask, mask, mask], axis=2)

            return np.where(
                mask3 > 0,
                base * (1 - alpha) + color * alpha,
                base,
            )

        overlay = blend(overlay, walkable, green, 0.45)
        overlay = blend(overlay, hazard, red, 0.55)
        overlay = blend(overlay, dynamic, yellow, 0.55)
        overlay = blend(overlay, signs, cyan, 0.50)

        overlay = overlay.astype(np.uint8)

        if nav is None:
            nav = self.analyze_navigation(walkable, hazard, dynamic)

        self.draw_navigation_arrow(
            overlay,
            nav["direction"],
        )

        lines = [
            f"STATUS: {nav['status']}",
            f"DIRECTION: {nav['direction']}",
        ]

        y = 30

        for line in lines:
            cv2.putText(
                overlay,
                line,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            y += 34

        return overlay


    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (FRAME_SIZE[0], FRAME_SIZE[1]))
        return frame

    def write_segmentation_to_json(
        self,
        frame_path: Path,
        nav: dict,
        walkable: np.ndarray,
        hazard: np.ndarray,
        dynamic: np.ndarray,
        iou: float | None,
        dice: float | None,
        focal: float | None,
        msd: float | None,
        infer_ms: float,
    ):
        sidecar = frame_path.with_suffix(".navigation.json")
        try:
            existing = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        except Exception:
            existing = {}

        existing["frame"] = frame_path.name
        existing["timestamp"] = frame_path.stem.split("-")[-1]
        existing["segmentation"] = {
            "walkable_status": nav["status"],
            "direction": nav["direction"],
            "zone_scores": {k: round(float(v), 2) for k, v in nav["scores"].items()},
            "walkable_pixel_ratio": round(float(np.sum(walkable)) / walkable.size, 4),
            "hazard_pixel_ratio": round(float(np.sum(hazard)) / hazard.size, 4),
            "dynamic_obstacle_ratio": round(float(np.sum(dynamic)) / dynamic.size, 4),
            "iou": iou,
            "dice": dice,
            "focal_loss": focal,
            "mean_surface_distance": msd,
            "inference_latency_ms": infer_ms,
        }

        sidecar.write_text(json.dumps(existing, indent=2))
    

    def process_group(
        self,
        frame_paths,
        output_path,
        artifact_name,
        group_name
    ):
        if not frame_paths:
            return

        self.prev_walkable_mask = None
        self.prev_hazard_mask = None

        fps = self.target_fps
                
        out = None
        if self.write_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                str(output_path), fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT)
            )
            if not out.isOpened():
                raise RuntimeError(f"Could not open writer {output_path}")

        processed_count = 0
        segmentation_cache = None

        prev_walkable = None

        try:
            for frame_path in frame_paths:
                try:
                    frame = cv2.imread(str(frame_path))

                    if frame is None:
                        continue

                    frame = self.preprocess_frame(frame)

                    if self.use_yolo:
                        yolo_results = self.yolo_model(
                            frame,
                            conf=0.5,
                            verbose=False,
                            device=self.yolo_device,
                        )
                    else:
                        yolo_results = None

                    if processed_count % self.deeplab_every_n_frames == 0:
                        t0 = perf_counter()
                        semantic = self.get_semantic_predictions(frame)
                        segmentation_cache = semantic
                        infer_ms = round((perf_counter() - t0) * 1000, 2)
                    else:
                        semantic = segmentation_cache
                        infer_ms = 0.0

                    walkable = self.mapper.get_walkable_mask(semantic)
                    hazard = self.mapper.get_hazard_mask(semantic)
                    dynamic = self.mapper.get_dynamic_obstacle_mask(semantic)
                    signs = self.mapper.get_traffic_sign_mask(semantic)

                    walkable = self.apply_temporal_smoothing(
                        walkable,
                        self.prev_walkable_mask,
                    )

                    hazard = self.apply_temporal_smoothing(
                        hazard,
                        self.prev_hazard_mask,
                    )

                    nav = self.analyze_navigation(walkable, hazard, dynamic)

                    if prev_walkable is not None:
                        iou = round(self.binary_iou(walkable, prev_walkable), 4)
                        dice = round(self.dice_score(walkable, prev_walkable), 4)
                        focal = round(
                            self.focal_loss_binary(
                                walkable.astype(np.float32),
                                prev_walkable.astype(np.float32),
                            ),
                            6,
                        )
                        msd = round(self.mean_surface_distance(walkable, prev_walkable), 4)
                    else:
                        iou = dice = focal = msd = None

                    prev_walkable = walkable.copy()

                    if out is not None:
                        viz = self.create_visualization(
                            frame,
                            yolo_results,
                            walkable,
                            hazard,
                            dynamic,
                            signs,
                            nav,
                        )
                        out.write(viz)

                    self.write_segmentation_to_json(
                        frame_path=frame_path,
                        nav=nav,
                        walkable=walkable,
                        hazard=hazard,
                        dynamic=dynamic,
                        iou=iou,
                        dice=dice,
                        focal=focal,
                        msd=msd,
                        infer_ms=infer_ms,
                    )

                    self.prev_walkable_mask = walkable
                    self.prev_hazard_mask = hazard

                    processed_count += 1
                except Exception as error:
                    print(f"  [Segmentation] Frame failed: {frame_path.name} — {error}")
                    continue
        finally:
            if out is not None:
                out.release()
    
    # merge videos

    def merge_group_videos(self, artifact_name: str, group_keys: list, batch_num: int):
        output_dir = Path(OUTPUT_DIR)
        mp4s = [output_dir / f"{k}.mp4" for k in group_keys if (output_dir / f"{k}.mp4").exists()]
        if not mp4s:
            return

        demo_path = output_dir / f"{artifact_name}_demo_{batch_num}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(demo_path), fourcc, self.target_fps, FRAME_SIZE)

        for mp4 in mp4s:
            cap = cv2.VideoCapture(str(mp4))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.preprocess_frame(frame)
                out.write(frame)
            cap.release()

        out.release()
        print(f"  Segmentation Demos: {demo_path.name}  ({len(mp4s)} groups)")

    # main

    def run(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        processed_groups = set()
        processed_order = []
        demo_batch_num = 1
        last_merged_count = 0
        known_groups = []
        active_artifact: Path | None = None

        DEMO_BATCH_SIZE = 2

        while True:
            try:
                if active_artifact is None:
                    active_artifact = self.find_latest_artifact()
                    print(f"[Segmentation] Tracking session: {active_artifact.name}")
                artifact = active_artifact
                if not artifact.exists():
                    active_artifact = None
                    time.sleep(0.5)
                    continue
                groups = self.get_group_folders(artifact)

                group_names = [g.name for g in groups]
                if group_names != known_groups:
                    print("Watching...")
                    print("Groups:", group_names)
                    known_groups = group_names

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

                for group in safe_groups:
                    key = f"{artifact.name}_{group.name}"
                    if key in processed_groups:
                        continue

                    frame_paths = self.load_frame_paths(group)
                    if not frame_paths:
                        continue

                    print(f"  [Segmentation] Processing: {group.name}")

                    video_path = Path(OUTPUT_DIR) / f"{key}.mp4"

                    self.process_group(
                        frame_paths,
                        video_path,
                        artifact.name,
                        group.name,
                    )

                    processed_groups.add(key)
                    processed_order.append(key)
                    print(f"  Done: {group.name}")

                unmerged_count = len(processed_order) - last_merged_count
                if self.write_video:
                    if unmerged_count >= DEMO_BATCH_SIZE:
                        batch_keys = processed_order[last_merged_count : last_merged_count + DEMO_BATCH_SIZE]
                        self.merge_group_videos(artifact.name, batch_keys, demo_batch_num)
                        last_merged_count += DEMO_BATCH_SIZE
                        demo_batch_num += 1
                    
                    elif is_closed and unmerged_count > 0:
                        print(f"Flushing final {unmerged_count} leftover groups into a demo...")
                        batch_keys = processed_order[last_merged_count:]
                        self.merge_group_videos(artifact.name, batch_keys, demo_batch_num)
                        last_merged_count += unmerged_count
                        demo_batch_num += 1

                pending = False
                for group in groups:
                    key = f"{artifact.name}_{group.name}"
                    if key in processed_groups:
                        continue
                    if self.load_frame_paths(group):
                        pending = True
                        break

                if is_closed and not pending:
                    try:
                        latest_artifact = self.find_latest_artifact()
                    except FileNotFoundError:
                        latest_artifact = artifact
                    if latest_artifact != artifact:
                        print(f"[Segmentation] Switching to new session: {latest_artifact.name}")
                        active_artifact = latest_artifact
                        processed_order = []
                        last_merged_count = 0
                        demo_batch_num = 1

            except FileNotFoundError:
                print("No artifacts found")
            except Exception as error:
                print(f"[Segmentation] Loop error: {error}")
                traceback.print_exc()

            time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    model = ImprovedSegmentation(
        frames_root=f"{APP_DIR}/session_artifacts",
        yolo_model_path=str(BASE_DIR / "yolov8n-seg.pt"),
        deeplab_model_path=str(BASE_DIR / "deeplabv3plus_mobilenet_finetuned.pth"),
        target_fps=10,
        use_yolo=True,
        deeplab_every_n_frames=2,
        write_video=not args.no_video,
    )

    model.run()
