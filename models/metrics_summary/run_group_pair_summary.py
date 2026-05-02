import argparse
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
APP_DIR = PROJECT_ROOT / "api" / "app"

DEFAULT_BATCH_SIZE = 2


def natural_key(path: Path) -> list:
    import re

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def numeric_stats(values: list[float | None]) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"count": 0}

    arr = np.array(clean, dtype=float)
    return {
        "count": int(arr.size),
        "mean": round(float(np.mean(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
    }


class GroupPairSummaryRunner:
    def __init__(self, frames_root: str, batch_size: int = DEFAULT_BATCH_SIZE):
        self.frames_root = Path(frames_root)
        self.batch_size = batch_size
        self.pending_pairs_logged: set[str] = set()

    def find_latest_artifact(self) -> Path:
        artifacts = [p for p in self.frames_root.iterdir() if p.is_dir()]
        if not artifacts:
            raise FileNotFoundError(f"No artifacts found in {self.frames_root}")
        artifacts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return artifacts[0]

    def get_group_folders(self, artifact_dir: Path) -> list[Path]:
        groups = [
            p for p in artifact_dir.iterdir() if p.is_dir() and p.name.startswith("group-")
        ]
        groups.sort(key=natural_key)
        return groups

    def is_session_closed(self, artifact_dir: Path) -> bool:
        manifest_path = artifact_dir / "session.json"
        if not manifest_path.exists():
            return False
        try:
            manifest_data = json.loads(manifest_path.read_text())
            return manifest_data.get("closed_at") is not None
        except Exception:
            return False

    def load_frame_paths(self, folder: Path) -> list[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        frames = [p for p in folder.iterdir() if p.suffix.lower() in exts]
        frames.sort(key=natural_key)
        return frames

    def _read_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    def pair_is_complete(self, groups: list[Path]) -> bool:
        for group in groups:
            for frame_path in self.load_frame_paths(group):
                det_path = frame_path.with_suffix(".detections.json")
                nav_path = frame_path.with_suffix(".navigation.json")
                if not det_path.exists() or not nav_path.exists():
                    return False
                nav_data = self._read_json(nav_path)
                if "segmentation" not in nav_data or "depth" not in nav_data:
                    return False
        return True

    def pair_latest_input_mtime(self, groups: list[Path]) -> float:
        latest = 0.0
        for group in groups:
            for frame_path in self.load_frame_paths(group):
                for path in (
                    frame_path,
                    frame_path.with_suffix(".detections.json"),
                    frame_path.with_suffix(".navigation.json"),
                ):
                    if path.exists():
                        latest = max(latest, path.stat().st_mtime)
        return latest

    def collect_pair_data(self, groups: list[Path]) -> dict:
        frame_records = []

        detection_counts = []
        hazard_ratios = []
        detection_persistence = []
        detection_latency_ms = []

        segmentation_walkable_ratio = []
        segmentation_hazard_ratio = []
        segmentation_iou = []
        segmentation_dice = []
        segmentation_latency_ms = []

        depth_primary_hazard_m = []
        depth_latency_ms = []
        depth_status_counter: Counter[str] = Counter()

        for group in groups:
            for frame_path in self.load_frame_paths(group):
                det_path = frame_path.with_suffix(".detections.json")
                nav_path = frame_path.with_suffix(".navigation.json")
                det_data = self._read_json(det_path)
                nav_data = self._read_json(nav_path)
                has_det_sidecar = det_path.exists()

                detections = det_data.get("detections", [])
                det_metrics = det_data.get("metrics", {})
                seg_metrics = nav_data.get("segmentation", {})
                depth_metrics = nav_data.get("depth", {})

                detection_count = det_metrics.get("num_detections")
                if detection_count is None and has_det_sidecar and isinstance(detections, list):
                    detection_count = len(detections)
                detection_count = int(detection_count) if isinstance(detection_count, int) else None

                hazard_ratio = safe_float(det_metrics.get("hazard_detection_ratio"))
                persistence = safe_float(det_metrics.get("detection_persistence_rate"))
                det_latency = safe_float(det_metrics.get("inference_latency_ms"))

                seg_walk = safe_float(seg_metrics.get("walkable_pixel_ratio"))
                seg_hazard = safe_float(seg_metrics.get("hazard_pixel_ratio"))
                seg_iou_val = safe_float(seg_metrics.get("iou"))
                seg_dice_val = safe_float(seg_metrics.get("dice"))
                seg_latency = safe_float(seg_metrics.get("inference_latency_ms"))

                depth_primary = safe_float(depth_metrics.get("primary_hazard_m"))
                depth_latency = safe_float(depth_metrics.get("inference_latency_ms"))
                depth_status = depth_metrics.get("proximity_status")
                if isinstance(depth_status, str):
                    depth_status_counter[depth_status] += 1

                detection_counts.append(float(detection_count) if detection_count is not None else None)
                hazard_ratios.append(hazard_ratio)
                detection_persistence.append(persistence)
                detection_latency_ms.append(det_latency)

                segmentation_walkable_ratio.append(seg_walk)
                segmentation_hazard_ratio.append(seg_hazard)
                segmentation_iou.append(seg_iou_val)
                segmentation_dice.append(seg_dice_val)
                segmentation_latency_ms.append(seg_latency)

                depth_primary_hazard_m.append(depth_primary)
                depth_latency_ms.append(depth_latency)

                frame_records.append(
                    {
                        "group": group.name,
                        "frame": frame_path.name,
                        "timestamp": frame_path.stem.split("-")[-1],
                        "detection": {
                            "num_detections": detection_count,
                            "hazard_detection_ratio": hazard_ratio,
                            "detection_persistence_rate": persistence,
                            "inference_latency_ms": det_latency,
                        },
                        "segmentation": {
                            "walkable_pixel_ratio": seg_walk,
                            "hazard_pixel_ratio": seg_hazard,
                            "iou": seg_iou_val,
                            "dice": seg_dice_val,
                            "inference_latency_ms": seg_latency,
                        },
                        "depth": {
                            "primary_hazard_m": depth_primary,
                            "proximity_status": depth_status,
                            "inference_latency_ms": depth_latency,
                        },
                    }
                )

        return {
            "frames": frame_records,
            "series": {
                "detection_counts": detection_counts,
                "hazard_ratios": hazard_ratios,
                "detection_persistence": detection_persistence,
                "detection_latency_ms": detection_latency_ms,
                "segmentation_walkable_ratio": segmentation_walkable_ratio,
                "segmentation_hazard_ratio": segmentation_hazard_ratio,
                "segmentation_iou": segmentation_iou,
                "segmentation_dice": segmentation_dice,
                "segmentation_latency_ms": segmentation_latency_ms,
                "depth_primary_hazard_m": depth_primary_hazard_m,
                "depth_latency_ms": depth_latency_ms,
            },
            "depth_status_counts": dict(depth_status_counter),
        }

    def _plot_series(self, ax, x_vals, y_vals, label, color):
        points = [(x, y) for x, y in zip(x_vals, y_vals) if y is not None]
        if not points:
            return
        xs, ys = zip(*points)
        ax.plot(xs, ys, label=label, color=color, linewidth=1.8)

    def save_graphical_summary(self, png_path: Path, pair_name: str, pair_data: dict):
        series = pair_data["series"]
        frame_count = len(pair_data["frames"])
        x_vals = list(range(1, frame_count + 1))

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Metrics Summary: {pair_name}", fontsize=14, fontweight="bold")

        ax = axes[0, 0]
        self._plot_series(ax, x_vals, series["detection_counts"], "detections/frame", "#1f77b4")
        self._plot_series(ax, x_vals, series["detection_persistence"], "persistence rate", "#2ca02c")
        ax2 = ax.twinx()
        self._plot_series(ax2, x_vals, series["hazard_ratios"], "hazard ratio", "#d62728")
        ax.set_title("Detection Density and Stability")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Count / Rate")
        ax2.set_ylabel("Hazard ratio")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines1 or lines2:
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)

        ax = axes[0, 1]
        self._plot_series(ax, x_vals, series["detection_latency_ms"], "detection", "#1f77b4")
        self._plot_series(ax, x_vals, series["segmentation_latency_ms"], "segmentation", "#ff7f0e")
        self._plot_series(ax, x_vals, series["depth_latency_ms"], "depth", "#9467bd")
        ax.set_title("Inference Latency per Frame")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Latency (ms)")
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8)

        ax = axes[1, 0]
        self._plot_series(
            ax, x_vals, series["segmentation_walkable_ratio"], "walkable ratio", "#2ca02c"
        )
        self._plot_series(
            ax, x_vals, series["segmentation_hazard_ratio"], "hazard ratio", "#d62728"
        )
        self._plot_series(ax, x_vals, series["segmentation_iou"], "IoU", "#1f77b4")
        self._plot_series(ax, x_vals, series["segmentation_dice"], "Dice", "#17becf")
        ax.set_title("Segmentation Quality and Scene Ratios")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Ratio / Score")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8)

        ax = axes[1, 1]
        self._plot_series(ax, x_vals, series["depth_primary_hazard_m"], "primary hazard (m)", "#9467bd")
        for threshold, color, label in [
            (0.8, "#d62728", "very_close"),
            (2.0, "#ff7f0e", "close"),
            (4.0, "#bcbd22", "mid"),
            (8.0, "#2ca02c", "far"),
        ]:
            ax.axhline(y=threshold, color=color, linestyle="--", linewidth=0.8, alpha=0.6, label=label)
        ax.set_title("Depth Hazard Distance")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Metres")
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=7, ncols=2)

        status_counts = pair_data.get("depth_status_counts", {})
        if status_counts:
            status_text = "\n".join(f"{k}: {v}" for k, v in sorted(status_counts.items()))
            ax.text(
                0.02,
                0.98,
                status_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )

        fig.tight_layout()
        fig.savefig(png_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def summarise_pair(self, artifact: Path, groups: list[Path], pair_index: int):
        summary_dir = artifact / "metrics_summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)

        pair_name = f"group-pair-{pair_index:03d}-{groups[0].name}-to-{groups[-1].name}"
        json_path = summary_dir / f"{pair_name}.summary.json"
        png_path = summary_dir / f"{pair_name}.summary.png"

        pair_data = self.collect_pair_data(groups)
        series = pair_data["series"]

        summary = {
            "artifact": artifact.name,
            "groups": [g.name for g in groups],
            "group_pair_name": pair_name,
            "batch_size": len(groups),
            "generated_at": utc_now_iso(),
            "frame_count": len(pair_data["frames"]),
            "aggregate_metrics": {
                "detection_num_per_frame": numeric_stats(series["detection_counts"]),
                "detection_hazard_ratio": numeric_stats(series["hazard_ratios"]),
                "detection_persistence_rate": numeric_stats(series["detection_persistence"]),
                "detection_latency_ms": numeric_stats(series["detection_latency_ms"]),
                "segmentation_walkable_ratio": numeric_stats(series["segmentation_walkable_ratio"]),
                "segmentation_hazard_ratio": numeric_stats(series["segmentation_hazard_ratio"]),
                "segmentation_iou": numeric_stats(series["segmentation_iou"]),
                "segmentation_dice": numeric_stats(series["segmentation_dice"]),
                "segmentation_latency_ms": numeric_stats(series["segmentation_latency_ms"]),
                "depth_primary_hazard_m": numeric_stats(series["depth_primary_hazard_m"]),
                "depth_latency_ms": numeric_stats(series["depth_latency_ms"]),
                "depth_status_counts": pair_data["depth_status_counts"],
            },
            "frame_metrics": pair_data["frames"],
            "graphical_summary_png": png_path.name,
        }

        json_path.write_text(json.dumps(summary, indent=2) + "\n")
        self.save_graphical_summary(png_path, pair_name, pair_data)

        print(f"[Summary] Wrote {json_path.name} and {png_path.name}")

    def run(self):
        while True:
            try:
                artifact = self.find_latest_artifact()
            except FileNotFoundError:
                print("[Summary] Waiting for a session to start...")
                time.sleep(2)
                continue

            groups = self.get_group_folders(artifact)
            is_closed = self.is_session_closed(artifact)
            safe_groups = groups if is_closed else (groups[:-1] if len(groups) > 1 else [])

            pair_index = 1
            for idx in range(0, len(safe_groups), self.batch_size):
                pair_groups = safe_groups[idx : idx + self.batch_size]
                if len(pair_groups) < self.batch_size:
                    continue

                key = f"{artifact.name}:{pair_groups[0].name}:{pair_groups[-1].name}"
                pair_name = f"group-pair-{pair_index:03d}-{pair_groups[0].name}-to-{pair_groups[-1].name}"
                summary_dir = artifact / "metrics_summaries"
                json_path = summary_dir / f"{pair_name}.summary.json"
                png_path = summary_dir / f"{pair_name}.summary.png"

                try:
                    if not self.pair_is_complete(pair_groups):
                        if key not in self.pending_pairs_logged:
                            print(f"[Summary] Waiting for complete data: {pair_name}")
                            self.pending_pairs_logged.add(key)
                        pair_index += 1
                        continue

                    latest_input_mtime = self.pair_latest_input_mtime(pair_groups)
                    if json_path.exists() and png_path.exists():
                        output_mtime = min(json_path.stat().st_mtime, png_path.stat().st_mtime)
                        if output_mtime >= latest_input_mtime:
                            pair_index += 1
                            continue

                    self.summarise_pair(artifact, pair_groups, pair_index)
                    self.pending_pairs_logged.discard(key)
                except Exception as error:
                    print(
                        f"[Summary] Failed {pair_groups[0].name}-{pair_groups[-1].name}: {error}"
                    )

                pair_index += 1

            time.sleep(2)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="How many groups to aggregate into one summary",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    runner = GroupPairSummaryRunner(
        frames_root=str(APP_DIR / "session_artifacts"),
        batch_size=max(1, int(args.batch_size)),
    )
    runner.run()
