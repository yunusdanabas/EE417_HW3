#!/usr/bin/env python3
import glob, os
import cv2, numpy as np

# Checkerboard settings
CHECKER = (7, 10)          # inner corners per row/col
SQUARE  = 0.025            # square size in meters
CRIT    = (cv2.TERM_CRITERIA_EPS |
           cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

# Prepare a single object‐point pattern
obj0 = np.zeros((CHECKER[0]*CHECKER[1], 3), np.float32)
obj0[:, :2] = np.mgrid[0:CHECKER[0], 0:CHECKER[1]].T.reshape(-1, 2)
obj0 *= SQUARE

obj_pts, img_pts = [], []
for fn in sorted(glob.glob("calib/*.jpg")):
    img  = cv2.imread(fn)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, CHECKER,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        print("skip", os.path.basename(fn))
        continue

    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), CRIT)
    obj_pts.append(obj0)
    img_pts.append(corners)
    print("ok", os.path.basename(fn))

if not obj_pts:
    raise RuntimeError("no corners detected")

# Calibrate camera
h, w = gray.shape
rms, K, D, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None)
print(f"RMS = {rms:.4f}px")

# Save intrinsics
os.makedirs("calib", exist_ok=True)
np.save("calib/K.npy",   K)
np.save("calib/dist.npy", D)
print("saved calib/K.npy and calib/dist.npy")

# Create a calibration preview image
sample = cv2.imread(sorted(glob.glob("calib/*.jpg"))[0])
newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1)
undist  = cv2.undistort(sample, K, D, None, newK)
combo   = np.hstack([sample, undist])

# Scale combo to fit within 1200×800 if needed
h0, w0 = combo.shape[:2]
scale  = min(1200/w0, 800/h0, 1.0)
disp   = cv2.resize(combo, (int(w0*scale), int(h0*scale)))

# Save the preview instead of showing it
os.makedirs("results", exist_ok=True)
cv2.imwrite("results/calib_preview.png", disp)
print("wrote results/calib_preview.png")
