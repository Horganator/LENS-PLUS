import subprocess
import sys
import time
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

OBJ_DETECTION_SCRIPT = BASE_DIR / "object_detection" / "run_live_detection.py"
SEGMENTATION_SCRIPT = BASE_DIR / "segmentation" / "src" / "segmentation-live-feed.py"
DEPTH_SCRIPT = BASE_DIR / "depth_estimation" / "depth_estimator.py"
SUMMARY_SCRIPT = BASE_DIR / "metrics_summary" / "run_group_pair_summary.py"

def launch_process(script_path: Path, extra_args: list[str] | None = None) -> subprocess.Popen:
    args = [sys.executable, str(script_path)]
    if extra_args:
        args.extend(extra_args)
    return subprocess.Popen(args, cwd=str(script_path.parent))

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

    process_specs = {
        "Object Detection": (OBJ_DETECTION_SCRIPT, extra_args),
        "Segmentation": (SEGMENTATION_SCRIPT, extra_args),
        "Depth Estimation": (DEPTH_SCRIPT, extra_args),
        "Metrics Summary": (SUMMARY_SCRIPT, []),
    }

    processes: dict[str, subprocess.Popen] = {}
    for name, (script_path, script_args) in process_specs.items():
        print(f"Starting {name}: {script_path.name}")
        processes[name] = launch_process(script_path, script_args)
        time.sleep(2)

    print("\nPipeline is live. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
            for name, process in list(processes.items()):
                if process.poll() is None:
                    continue
                code = process.returncode
                print(f"WARN: {name} stopped unexpectedly (exit {code}). Restarting...")
                script_path, script_args = process_specs[name]
                processes[name] = launch_process(script_path, script_args)
                
    except KeyboardInterrupt:
        print("Shutting down pipeline...")
        for process in processes.values():
            process.terminate()

        for process in processes.values():
            process.wait()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
