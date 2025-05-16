#!/usr/bin/env python3
# compare_detectors.py
import cv2
import numpy as np
from utils import print_table

LEFT, RIGHT = "results/undistorted_scene_left.png", "results/undistorted_scene_right.png"
K           = np.load("calib/K.npy")
BASELINE    = 0.15          # same meter shift you used in triangulate.py

DETECTORS = {
    "SIFT": (cv2.SIFT_create(),    cv2.NORM_L2),
    "ORB" : (cv2.ORB_create(5000), cv2.NORM_HAMMING),
}
RATIO, THR = 0.75, 1.0

def compute_stats(feat, norm):
    img1, img2 = cv2.imread(LEFT), cv2.imread(RIGHT)
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, d1 = feat.detectAndCompute(g1, None)
    kp2, d2 = feat.detectAndCompute(g2, None)
    matches = cv2.BFMatcher(norm).knnMatch(d1, d2, 2)
    good    = [m for m,n in matches if m.distance < RATIO*n.distance]

    p1 = np.float32([kp1[m.queryIdx].pt for m in good])
    p2 = np.float32([kp2[m.trainIdx].pt for m in good])

    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, THR)
    in1 = p1[mask.ravel()==1]
    in2 = p2[mask.ravel()==1]
    inliers = int(mask.sum())

    E = K.T @ F @ K
    _, R, t_unit, _ = cv2.recoverPose(E, in1, in2, K)
    t = t_unit * BASELINE

    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t.reshape(3,1)))

    pts4 = cv2.triangulatePoints(P1, P2, in1.T, in2.T).T
    pts4 /= pts4[:,3:4]
    pts3  = pts4[:,:3]
    pts3  = pts3[pts3[:,2] > 0]

    return {
        "KP1":       len(kp1),
        "KP2":       len(kp2),
        "Good":      len(good),
        "Inliers":   inliers,
        "3DPoints":  len(pts3)
    }

rows = []
for name, (feat, norm) in DETECTORS.items():
    stats = compute_stats(feat, norm)
    stats["Detector"] = name
    rows.append(stats)

print_table(rows, title="Detector Comparison")
