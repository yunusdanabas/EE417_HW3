#!/usr/bin/env python3
import cv2, numpy as np
from utils import print_table

LEFT, RIGHT = "results/undistorted_scene_left.png", "results/undistorted_scene_right.png"
K = np.load("calib/K.npy")
DET = {"SIFT": (cv2.SIFT_create(), cv2.NORM_L2),
       "ORB" : (cv2.ORB_create(5000), cv2.NORM_HAMMING)}
RATIO, THR = 0.75, 1.0

def epi_err(src, dst, F):
    lines = cv2.computeCorrespondEpilines(src.reshape(-1,1,2), 1, F).reshape(-1,3)
    num   = np.abs(np.sum(lines* np.hstack([dst, np.ones((len(dst),1))]),1))
    return num/np.linalg.norm(lines[:,:2], axis=1)

def run(f,norm):
    i1,i2 = cv2.imread(LEFT), cv2.imread(RIGHT)
    g1,g2 = cv2.cvtColor(i1,cv2.COLOR_BGR2GRAY), cv2.cvtColor(i2,cv2.COLOR_BGR2GRAY)
    k1,d1 = f.detectAndCompute(g1,None); k2,d2 = f.detectAndCompute(g2,None)
    good  = [m for m,n in cv2.BFMatcher(norm).knnMatch(d1,d2,2) if m.distance<RATIO*n.distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    F,msk= cv2.findFundamentalMat(p1,p2,cv2.FM_RANSAC,THR)
    p1,p2= p1[msk.ravel()==1], p2[msk.ravel()==1]
    e    = np.hstack([epi_err(p1,p2,F), epi_err(p2,p1,F.T)])
    return len(good), len(p1), e.mean(), np.median(e), e.max()

summary = []
for n,(f,norm) in DET.items():
    m, i, mu, med, mx = run(f,norm)
    summary.append(dict(Detector=n, Matches=m, Inliers=i,
                        Mean=f"{mu:.2f}", Median=f"{med:.2f}", Max=f"{mx:.2f}"))
    
print_table(summary, title="Epipolar Error (px)")
