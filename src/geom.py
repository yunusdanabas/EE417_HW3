#!/usr/bin/env python3
import cv2, numpy as np, os

RATIO = 0.75
THR   = 1.0

def _draw_epi(img, lines, pts):
    w = img.shape[1]
    out = img.copy()
    for (a,b,c), (x,y) in zip(lines, pts):
        p1, p2 = (0, int(-c/b)), (w, int(-(c+a*w)/b))
        cv2.line(out, p1, p2, (0,255,0), 1)
        cv2.circle(out, (int(x),int(y)), 3, (0,0,255), -1)
    return out

def main(det="SIFT"):
    i1 = cv2.imread("results/undistorted_scene_left.png")
    i2 = cv2.imread("results/undistorted_scene_right.png")
    K  = np.load("calib/K.npy")
    g1, g2 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

    if det.upper() == "ORB":
        f, norm = cv2.ORB_create(5000), cv2.NORM_HAMMING
    else:
        f, norm = cv2.SIFT_create(), cv2.NORM_L2

    k1, d1 = f.detectAndCompute(g1, None)
    k2, d2 = f.detectAndCompute(g2, None)
    good   = [m for m,n in cv2.BFMatcher(norm).knnMatch(d1, d2, 2) if m.distance < RATIO*n.distance]

    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    F, msk = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, THR)
    in1, in2 = p1[msk.ravel()==1], p2[msk.ravel()==1]
    print("inliers", len(in1), "/", len(good))

    E = K.T @ F @ K
    cv2.recoverPose(E, in1, in2, K)

    l1 = cv2.computeCorrespondEpilines(in2.reshape(-1,1,2), 2, F).reshape(-1,3)
    l2 = cv2.computeCorrespondEpilines(in1.reshape(-1,1,2), 1, F).reshape(-1,3)

    os.makedirs("results", exist_ok=True)
    cv2.imwrite("results/epi_lines.png",
                np.hstack([_draw_epi(i1, l1, in1), _draw_epi(i2, l2, in2)]))
    print("wrote epi_lines.png")

if __name__ == "__main__":
    main()
