#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Parameters
RATIO     = 0.75       # Lowe’s ratio test
RANSAC_TH = 1.0        # Fundamental matrix RANSAC threshold (px)
BASELINE  = 0.10       # Your measured camera shift in meters

def main(detector="SIFT"):
    # Load & undistort pair
    img1 = cv2.imread("results/undistorted_scene_left.png")
    img2 = cv2.imread("results/undistorted_scene_right.png")
    K    = np.load("calib/K.npy")

    # Grayscale
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detector setup
    if detector.upper() == "ORB":
        feat, norm = cv2.ORB_create(5000), cv2.NORM_HAMMING
    else:
        feat, norm = cv2.SIFT_create(), cv2.NORM_L2

    # Detect & match
    kp1, des1 = feat.detectAndCompute(g1, None)
    kp2, des2 = feat.detectAndCompute(g2, None)
    matches   = cv2.BFMatcher(norm).knnMatch(des1, des2, k=2)
    good      = [m for m,n in matches if m.distance < RATIO*n.distance]

    # Prepare point arrays
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # Fundamental + inliers
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, RANSAC_TH)
    in1 = pts1[mask.ravel()==1]
    in2 = pts2[mask.ravel()==1]

    # Essential + pose (unit translation)
    E = K.T @ F @ K
    _, R, t_unit, _ = cv2.recoverPose(E, in1, in2, K)
    t = t_unit * BASELINE

    # Projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t.reshape(3,1)))

    # Triangulate
    pts4 = cv2.triangulatePoints(P1, P2, in1.T, in2.T)  # 4×N
    pts4 /= pts4[3]                                     # homogeneous → metric
    pts3 = pts4[:3].T                                   # N×3

    # Keep only points in front of the first camera
    pts3 = pts3[pts3[:,2] > 0]

    # Save point-cloud
    os.makedirs("results", exist_ok=True)
    np.savetxt("results/pointcloud.xyz", pts3, fmt="%.6f")
    print(f"triangulated {len(pts3)} points (from {len(good)} matches)")

    # Quick 3D scatter
    fig = plt.figure()
    ax  = fig.add_subplot(projection='3d')
    ax.scatter(pts3[:,0], pts3[:,1], pts3[:,2], s=1)
    plt.tight_layout()
    plt.savefig("results/pointcloud.png")
