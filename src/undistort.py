#!/usr/bin/env python3
# undistort.py

import os, cv2, numpy as np

K    = np.load("calib/K.npy")
D    = np.load("calib/dist.npy")
os.makedirs("results", exist_ok=True)

for fn in ("images/scene_left.jpg", "images/scene_right.jpg"):
    img = cv2.imread(fn)
    if img is None:
        raise FileNotFoundError(fn)
    h, w = img.shape[:2]
    newK = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 1)[0]
    und  = cv2.undistort(img, K, D, None, newK)
    out  = f"results/undistorted_{os.path.splitext(os.path.basename(fn))[0]}.png"
    cv2.imwrite(out, und)
    print("wrote", out)
