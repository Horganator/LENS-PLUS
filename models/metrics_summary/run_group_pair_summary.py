import argparse
import json
import math
import time
from collections import Counter, defaultdict
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
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
    }


def rolling_mean(values: list[float | None], window: int = 10) -> list[float | None]:
    out: list[float | None] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = [v for v in values[start : i + 1] if v is not None]
        if not window_vals:
            out.append(None)
        else:
            out.append(float(np.mean(window_vals)))
    return out


def rolling_variance(values: list[float | None], window: int = 15) -> list[float | None]:
    out: list[float | None] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = [v for v in values[start : i + 1] if v is not None]
        if len(window_vals) < 2:
            out.append(None)
        else:
            out.append(float(np.var(window_vals)))
    return out


class GroupPairSummaryRunner:
    DEPTH_STATUSES = ["VERY_CLOSE", "CLOSE", "MID", "FAR", "CLEAR"]

    def __init__(self, frames_root: str, batch_size: int = DEFAULT_BATCH_SIZE):
        self.frames_root = Path(frames_root)
        self.batch_size = batch_size
        self.pending_pairs_logged: set[str] = set()

    def get_artifacts(self) -> list[Path]:
        artifacts = [p for p in self.frames_root.iterdir() if p.is_dir()]
        artifacts.sort(key=natural_key)
        return artifacts

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

        detection_counts: list[float | None] = []
        hazard_ratios: list[float | None] = []
        detection_persistence: list[float | None] = []
        detection_latency_ms: list[float | None] = []
        detection_conf_avg: list[float | None] = []
        detection_conf_max: list[float | None] = []
        detection_conf_min: list[float | None] = []

        segmentation_walkable_ratio: list[float | None] = []
        segmentation_hazard_ratio: list[float | None] = []
        segmentation_iou: list[float | None] = []
        segmentation_dice: list[float | None] = []
        segmentation_latency_ms: list[float | None] = []

        depth_primary_hazard_m: list[float | None] = []
        depth_latency_ms: list[float | None] = []
        depth_status_series: list[str | None] = []
        depth_status_counter: Counter[str] = Counter()

        has_det_series: list[float | None] = []
        has_seg_series: list[float | None] = []
        has_depth_series: list[float | None] = []

        frame_label_counts: list[dict[str, int]] = []
        overall_label_counts: Counter[str] = Counter()
        group_frame_counts: dict[str, int] = defaultdict(int)

        for group in groups:
            for frame_path in self.load_frame_paths(group):
                group_frame_counts[group.name] += 1

                det_path = frame_path.with_suffix(".detections.json")
                nav_path = frame_path.with_suffix(".navigation.json")
                det_data = self._read_json(det_path)
                nav_data = self._read_json(nav_path)

                detections = det_data.get("detections", [])
                det_metrics = det_data.get("metrics", {})
                seg_metrics = nav_data.get("segmentation", {})
                depth_metrics = nav_data.get("depth", {})

                label_counts = Counter()
                if isinstance(detections, list):
                    for d in detections:
                        label = d.get("label")
                        if isinstance(label, str):
                            label_counts[label] += 1
                overall_label_counts.update(label_counts)
                frame_label_counts.append(dict(label_counts))

                detection_count = det_metrics.get("num_detections")
                if detection_count is None and det_path.exists() and isinstance(detections, list):
                    detection_count = len(detections)
                detection_count = int(detection_count) if isinstance(detection_count, int) else None

                hazard_ratio = safe_float(det_metrics.get("hazard_detection_ratio"))
                persistence = safe_float(det_metrics.get("detection_persistence_rate"))
                det_latency = safe_float(det_metrics.get("inference_latency_ms"))
                conf_avg = safe_float(det_metrics.get("avg_confidence"))
                conf_max = safe_float(det_metrics.get("max_confidence"))
                conf_min = safe_float(det_metrics.get("min_confidence"))

                if conf_avg is None and isinstance(detections, list) and detections:
                    confs = [
                        safe_float(d.get("confidence"))
                        for d in detections
                        if safe_float(d.get("confidence")) is not None
                    ]
                    if confs:
                        conf_avg = float(np.mean(confs))
                        conf_max = float(np.max(confs))
                        conf_min = float(np.min(confs))

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
                else:
                    depth_status = None

                has_det = detection_count is not None
                has_seg = bool(seg_metrics)
                has_depth = bool(depth_metrics)

                detection_counts.append(float(detection_count) if detection_count is not None else None)
                hazard_ratios.append(hazard_ratio)
                detection_persistence.append(persistence)
                detection_latency_ms.append(det_latency)
                detection_conf_avg.append(conf_avg)
                detection_conf_max.append(conf_max)
                detection_conf_min.append(conf_min)

                segmentation_walkable_ratio.append(seg_walk)
                segmentation_hazard_ratio.append(seg_hazard)
                segmentation_iou.append(seg_iou_val)
                segmentation_dice.append(seg_dice_val)
                segmentation_latency_ms.append(seg_latency)

                depth_primary_hazard_m.append(depth_primary)
                depth_latency_ms.append(depth_latency)
                depth_status_series.append(depth_status)

                has_det_series.append(1.0 if has_det else 0.0)
                has_seg_series.append(1.0 if has_seg else 0.0)
                has_depth_series.append(1.0 if has_depth else 0.0)

                frame_records.append(
                    {
                        "group": group.name,
                        "frame": frame_path.name,
                        "timestamp": frame_path.stem.split("-")[-1],
                        "detection": {
                            "num_detections": detection_count,
                            "avg_confidence": conf_avg,
                            "max_confidence": conf_max,
                            "min_confidence": conf_min,
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

        group_boundaries = []
        current_group = None
        for idx, rec in enumerate(frame_records):
            group_name = rec["group"]
            if current_group is None:
                current_group = group_name
                continue
            if group_name != current_group:
                group_boundaries.append({"x": idx + 0.5, "label": current_group})
                current_group = group_name

        return {
            "frames": frame_records,
            "series": {
                "detection_counts": detection_counts,
                "hazard_ratios": hazard_ratios,
                "detection_persistence": detection_persistence,
                "detection_latency_ms": detection_latency_ms,
                "detection_conf_avg": detection_conf_avg,
                "detection_conf_max": detection_conf_max,
                "detection_conf_min": detection_conf_min,
                "segmentation_walkable_ratio": segmentation_walkable_ratio,
                "segmentation_hazard_ratio": segmentation_hazard_ratio,
                "segmentation_iou": segmentation_iou,
                "segmentation_dice": segmentation_dice,
                "segmentation_latency_ms": segmentation_latency_ms,
                "depth_primary_hazard_m": depth_primary_hazard_m,
                "depth_latency_ms": depth_latency_ms,
                "depth_status_series": depth_status_series,
                "has_det_series": has_det_series,
                "has_seg_series": has_seg_series,
                "has_depth_series": has_depth_series,
            },
            "depth_status_counts": dict(depth_status_counter),
            "frame_label_counts": frame_label_counts,
            "overall_label_counts": dict(overall_label_counts),
            "group_frame_counts": dict(group_frame_counts),
            "group_boundaries": group_boundaries,
        }

    def _plot_series(self, ax, x_vals, y_vals, label, color, linewidth=1.8):
        points = [(x, y) for x, y in zip(x_vals, y_vals) if y is not None]
        if not points:
            return
        xs, ys = zip(*points)
        ax.plot(xs, ys, label=label, color=color, linewidth=linewidth)

    def _add_group_boundaries(self, ax, boundaries: list[dict]):
        for item in boundaries:
            ax.axvline(item["x"], color="#666666", linestyle="--", linewidth=0.8, alpha=0.35)

    def _status_one_hot(self, statuses: list[str | None], label: str) -> list[float]:
        return [1.0 if s == label else 0.0 for s in statuses]

    def _plot_normal_fit_distribution(
        self,
        ax,
        values: list[float | None],
        label: str,
        color: str,
        bins: int = 16,
    ):
        clean = [v for v in values if v is not None]
        if len(clean) < 2:
            return False

        arr = np.array(clean, dtype=float)
        ax.hist(
            arr,
            bins=bins,
            density=True,
            alpha=0.28,
            color=color,
            edgecolor=color,
            linewidth=0.8,
            label=f"{label} hist",
        )

        mu = float(np.mean(arr))
        sigma = float(np.std(arr))
        if sigma <= 1e-9:
            return True

        x_min = max(0.0, float(np.min(arr)) - 0.1)
        x_max = min(1.0, float(np.max(arr)) + 0.1)
        if x_max <= x_min:
            x_max = x_min + 1e-3
        x = np.linspace(x_min, x_max, 200)
        pdf = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax.plot(x, pdf, color=color, linewidth=2.0, label=f"{label} normal fit")
        ax.axvline(mu, color=color, linestyle="--", linewidth=1.0, alpha=0.85, label=f"{label} mean={mu:.3f}")
        return True

    def save_graphical_summary(self, png_path: Path, pair_name: str, pair_data: dict):
        series = pair_data["series"]
        frame_count = len(pair_data["frames"])
        x_vals = list(range(1, frame_count + 1))
        boundaries = pair_data.get("group_boundaries", [])

        fig, axes = plt.subplots(4, 3, figsize=(24, 18))
        fig.suptitle(f"Metrics Summary: {pair_name}", fontsize=15, fontweight="bold")

        ax = axes[0, 0]
        self._plot_series(ax, x_vals, series["detection_counts"], "detections/frame", "#1f77b4")
        self._plot_series(ax, x_vals, series["detection_persistence"], "persistence rate", "#2ca02c")
        ax2 = ax.twinx()
        self._plot_series(ax2, x_vals, series["hazard_ratios"], "hazard ratio", "#d62728")
        self._add_group_boundaries(ax, boundaries)
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
        self._plot_series(ax, x_vals, series["detection_conf_avg"], "avg conf", "#1f77b4")
        self._plot_series(ax, x_vals, series["detection_conf_max"], "max conf", "#2ca02c")
        self._plot_series(ax, x_vals, series["detection_conf_min"], "min conf", "#d62728")
        self._add_group_boundaries(ax, boundaries)
        ax.set_title("Detection Confidence Trend")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Confidence")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8)

        ax = axes[0, 2]
        self._plot_series(
            ax, x_vals, rolling_mean(series["has_det_series"], 10), "det availability", "#1f77b4"
        )
        self._plot_series(
            ax, x_vals, rolling_mean(series["has_seg_series"], 10), "seg availability", "#ff7f0e"
        )
        self._plot_series(
            ax, x_vals, rolling_mean(series["has_depth_series"], 10), "depth availability", "#9467bd"
        )
        self._add_group_boundaries(ax, boundaries)
        ax.set_title("Data Completeness (Rolling 10)")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Availability rate")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower right", fontsize=8)

        ax = axes[1, 0]
        overall_counts = Counter(pair_data.get("overall_label_counts", {}))
        top_labels = [label for label, _ in overall_counts.most_common(5)]
        if top_labels and frame_count > 0:
            stack_data = []
            for label in top_labels:
                stack_data.append(
                    [
                        float(frame_counts.get(label, 0))
                        for frame_counts in pair_data.get("frame_label_counts", [])
                    ]
                )
            ax.stackplot(x_vals, stack_data, labels=top_labels, alpha=0.85)
            self._add_group_boundaries(ax, boundaries)
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Top-Class Composition")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Detections")
        ax.grid(alpha=0.25)

        ax = axes[1, 1]
        self._plot_series(
            ax,
            x_vals,
            rolling_variance(series["detection_persistence"], 15),
            "var(persistence)",
            "#2ca02c",
        )
        self._plot_series(
            ax,
            x_vals,
            rolling_variance(series["segmentation_iou"], 15),
            "var(seg IoU)",
            "#1f77b4",
        )
        self._plot_series(
            ax,
            x_vals,
            rolling_variance(series["depth_primary_hazard_m"], 15),
            "var(depth hazard m)",
            "#9467bd",
        )
        self._add_group_boundaries(ax, boundaries)
        ax.set_title("Rolling Stability Variance (15)")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Variance")
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8)

        ax = axes[1, 2]
        self._plot_series(ax, x_vals, series["detection_latency_ms"], "detection", "#1f77b4")
        self._plot_series(ax, x_vals, series["segmentation_latency_ms"], "segmentation", "#ff7f0e")
        self._plot_series(ax, x_vals, series["depth_latency_ms"], "depth", "#9467bd")
        self._add_group_boundaries(ax, boundaries)
        ax.set_title("Inference Latency per Frame")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Latency (ms)")
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8)

        ax = axes[2, 0]
        latency_sets = []
        latency_labels = []
        for label, vals in [
            ("Detection", series["detection_latency_ms"]),
            ("Segmentation", series["segmentation_latency_ms"]),
            ("Depth", series["depth_latency_ms"]),
        ]:
            clean = [v for v in vals if v is not None]
            if clean:
                latency_sets.append(clean)
                latency_labels.append(label)
        if latency_sets:
            ax.boxplot(latency_sets, labels=latency_labels, patch_artist=True)
        ax.set_title("Latency Distribution")
        ax.set_ylabel("Latency (ms)")
        ax.grid(alpha=0.25, axis="y")

        ax = axes[2, 1]
        self._plot_series(
            ax, x_vals, series["segmentation_walkable_ratio"], "walkable ratio", "#2ca02c"
        )
        self._plot_series(
            ax, x_vals, series["segmentation_hazard_ratio"], "hazard ratio", "#d62728"
        )
        self._plot_series(ax, x_vals, series["segmentation_iou"], "IoU", "#1f77b4")
        self._plot_series(ax, x_vals, series["segmentation_dice"], "Dice", "#17becf")
        self._add_group_boundaries(ax, boundaries)
        ax.set_title("Segmentation Quality and Scene Ratios")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Ratio / Score")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8)

        ax = axes[2, 2]
        self._plot_series(ax, x_vals, series["depth_primary_hazard_m"], "primary hazard (m)", "#9467bd")
        for threshold, color, label in [
            (0.8, "#d62728", "very_close"),
            (2.0, "#ff7f0e", "close"),
            (4.0, "#bcbd22", "mid"),
            (8.0, "#2ca02c", "far"),
        ]:
            ax.axhline(y=threshold, color=color, linestyle="--", linewidth=0.8, alpha=0.6, label=label)
        self._add_group_boundaries(ax, boundaries)
        ax.set_title("Depth Hazard Distance")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Metres")
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=7, ncols=2)

        ax = axes[3, 0]
        status_series = series["depth_status_series"]
        if frame_count > 0:
            stack = [self._status_one_hot(status_series, s) for s in self.DEPTH_STATUSES]
            ax.stackplot(x_vals, stack, labels=self.DEPTH_STATUSES, alpha=0.85)
            self._add_group_boundaries(ax, boundaries)
            ax.legend(loc="upper right", fontsize=8, ncols=3)
        ax.set_title("Depth Safety Timeline")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Status indicator")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)

        ax = axes[3, 1]
        scatter_x = []
        scatter_y = []
        scatter_c = []
        for det_count, depth_m, hz in zip(
            series["detection_counts"], series["depth_primary_hazard_m"], series["hazard_ratios"]
        ):
            if det_count is None or depth_m is None:
                continue
            scatter_x.append(det_count)
            scatter_y.append(depth_m)
            scatter_c.append(hz if hz is not None else 0.0)
        if scatter_x:
            sc = ax.scatter(scatter_x, scatter_y, c=scatter_c, cmap="viridis", alpha=0.75)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Hazard ratio")
        ax.set_title("Cross-Metric: Detections vs Depth Hazard")
        ax.set_xlabel("Detections per frame")
        ax.set_ylabel("Primary hazard (m)")
        ax.grid(alpha=0.25)

        ax = axes[3, 2]
        has_iou = self._plot_normal_fit_distribution(
            ax, series["segmentation_iou"], "IoU", "#1f77b4"
        )
        has_dice = self._plot_normal_fit_distribution(
            ax, series["segmentation_dice"], "Dice", "#17becf"
        )
        if has_iou or has_dice:
            ax.set_xlim(0.0, 1.0)
            ax.set_title("Segmentation Mean Distribution (Normal Fit)")
            ax.set_xlabel("Score")
            ax.set_ylabel("Density")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper left", fontsize=7)
        else:
            ax.set_title("Segmentation Mean Distribution (Normal Fit)")
            ax.text(0.5, 0.5, "Insufficient IoU/Dice data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout(rect=[0, 0, 1, 0.97])
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

        completeness_rates = {
            "detection": round(float(np.mean(series["has_det_series"])), 4) if series["has_det_series"] else 0.0,
            "segmentation": round(float(np.mean(series["has_seg_series"])), 4) if series["has_seg_series"] else 0.0,
            "depth": round(float(np.mean(series["has_depth_series"])), 4) if series["has_depth_series"] else 0.0,
        }

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
                "detection_conf_avg": numeric_stats(series["detection_conf_avg"]),
                "detection_conf_max": numeric_stats(series["detection_conf_max"]),
                "detection_conf_min": numeric_stats(series["detection_conf_min"]),
                "segmentation_walkable_ratio": numeric_stats(series["segmentation_walkable_ratio"]),
                "segmentation_hazard_ratio": numeric_stats(series["segmentation_hazard_ratio"]),
                "segmentation_iou": numeric_stats(series["segmentation_iou"]),
                "segmentation_dice": numeric_stats(series["segmentation_dice"]),
                "segmentation_latency_ms": numeric_stats(series["segmentation_latency_ms"]),
                "depth_primary_hazard_m": numeric_stats(series["depth_primary_hazard_m"]),
                "depth_latency_ms": numeric_stats(series["depth_latency_ms"]),
                "depth_status_counts": pair_data["depth_status_counts"],
                "completeness_rates": completeness_rates,
                "top_label_counts": pair_data.get("overall_label_counts", {}),
                "group_frame_counts": pair_data.get("group_frame_counts", {}),
            },
            "frame_metrics": pair_data["frames"],
            "graphical_summary_png": png_path.name,
        }

        json_path.write_text(json.dumps(summary, indent=2) + "\n")
        self.save_graphical_summary(png_path, pair_name, pair_data)

        print(f"[Summary] Wrote {json_path.name} and {png_path.name}")

    def run(self):
        while True:
            artifacts = self.get_artifacts()
            if not artifacts:
                print("[Summary] Waiting for a session to start...")
                time.sleep(2)
                continue

            for artifact in artifacts:
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
