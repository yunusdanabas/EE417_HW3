#!/usr/bin/env python3
import sys
import subprocess
import argparse
import shutil
import time

from match import extract_and_match
import geom, triangulate

PY        = sys.executable
SCRIPTS   = "src"
CALIB     = f"{SCRIPTS}/calibrate.py"
UNDIST    = f"{SCRIPTS}/undistort.py"
COMPARE   = f"{SCRIPTS}/compare_detectors.py"
EPIERR    = f"{SCRIPTS}/epipolar_error.py"
REPROJ    = f"{SCRIPTS}/reprojection_error.py"

def print_header(title):
    width = 50
    bar   = "=" * width
    print(f"\n{bar}")
    print(title.center(width))
    print(f"{bar}")

def run_script(name, script, *args):
    print_header(f"{name}")
    cmd = [PY, script, *args]
    # print("Command:", " ".join(cmd))
    subprocess.check_call(cmd)

def main(detector):
    start = time.time()

    # 1) Calibration
    run_script("1) Calibration", CALIB)

    # 2) Undistort Images
    run_script("2) Undistort Images", UNDIST)

    # 3) Feature Matching
    print_header("3) Feature Matching")
    t0 = time.time()
    extract_and_match(
        "results/undistorted_scene_left.png",
        "results/undistorted_scene_right.png",
        detector
    )
    print(f"-> done in {time.time()-t0:.2f}s")

    # 4) Epipolar Geometry & Pose
    print_header("4) Epipolar Geometry & Pose Estimation")
    t0 = time.time()
    geom.main(detector)
    print(f"-> done in {time.time()-t0:.2f}s")

    # 5) Triangulation
    print_header("5) Triangulation")
    t0 = time.time()
    triangulate.main(detector)
    print(f"-> done in {time.time()-t0:.2f}s")

    # 6) Comparative Detector Study
    run_script("6) Comparative Detector Study", COMPARE)

    # 7) Epipolar Error Analysis
    run_script("7) Epipolar Error Analysis", EPIERR)

    # 8) 3D Reprojection Error Analysis
    run_script("8) 3D Reprojection Error Analysis", REPROJ)

    total = time.time() - start
    print_header(f"Completed pipeline in {total:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full stereo-3D pipeline end-to-end"
    )
    parser.add_argument(
        "--detector", choices=["SIFT", "ORB"],
        default="SIFT",
        help="Keypoint detector for match/pose/triangulate"
    )
    args = parser.parse_args()
    main(args.detector)
