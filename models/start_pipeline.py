import subprocess
import sys
import time
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

OBJ_DETECTION_SCRIPT = BASE_DIR / "object_detection" / "run_live_detection.py"
SEGMENTATION_SCRIPT = BASE_DIR / "segmentation" / "src" / "segmentation-live-feed.py"
DEPTH_SCRIPT = BASE_DIR / "depth_estimation" / "depth_estimator.py"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip mp4 writing in all scripts",
    )
    args = parser.parse_args()

    extra_args = ["--no-video"] if args.no_video else []

    if args.no_video:
        print("mp4 saving DISABLED")
    else:
        print("mp4 saving ENABLED")

    print("Pipeline Starting...\n")

    print(f"Starting Object Detection: {OBJ_DETECTION_SCRIPT.name}")

    obj_det_process = subprocess.Popen(
        [sys.executable, str(OBJ_DETECTION_SCRIPT)] + extra_args,
        cwd=str(OBJ_DETECTION_SCRIPT.parent),
    )

    time.sleep(2)

    print(f"Starting Segmentation: {SEGMENTATION_SCRIPT.name}")

    seg_process = subprocess.Popen(
        [sys.executable, str(SEGMENTATION_SCRIPT)] + extra_args,
        cwd=str(SEGMENTATION_SCRIPT.parent),
    )

    time.sleep(2)
    print(f"Starting Depth Estimation: {DEPTH_SCRIPT.name}")

    depth_process = subprocess.Popen(
        [sys.executable, str(DEPTH_SCRIPT)] + extra_args,
        cwd=str(DEPTH_SCRIPT.parent),
    )

    print("\nPipeline is live. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
            if obj_det_process.poll() is not None:
                print("WARN: Object Detection script stopped unexpectedly!")
            if seg_process.poll() is not None:
                print("WARN: Segmentation script stopped unexpectedly!")
            if depth_process.poll() is not None:
                print("WARN: Depth script stopped unexpectedly!")
                
    except KeyboardInterrupt:
        print("Shutting down pipeline...")
        obj_det_process.terminate()
        seg_process.terminate()
        depth_process.terminate()
        
        obj_det_process.wait()
        seg_process.wait()
        depth_process.wait()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()