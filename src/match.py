#!/usr/bin/env python3
import cv2, os
from utils import print_table

RATIO = 0.75
ORB_K = 5000

def extract_and_match(img1, img2, det="SIFT"):
    i1, i2 = cv2.imread(img1), cv2.imread(img2)
    g1, g2 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

    if det.upper() == "ORB":
        f, norm = cv2.ORB_create(ORB_K), cv2.NORM_HAMMING
    else:
        f, norm = cv2.SIFT_create(), cv2.NORM_L2

    k1, d1 = f.detectAndCompute(g1, None)
    k2, d2 = f.detectAndCompute(g2, None)
    good   = [m for m,n in cv2.BFMatcher(norm).knnMatch(d1, d2, 2) if m.distance < RATIO*n.distance]

    out = f"results/matches_{det.lower()}.png"
    cv2.imwrite(out, cv2.drawMatches(i1, k1, i2, k2, good, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
    
    print_table([dict(Detector=det,
                      KP1=len(k1), KP2=len(k2), Good=len(good),
                      File=os.path.basename(out))],
                title="Feature-Match Summary")
    return k1, k2, good
