#!/usr/bin/env python3
# triangulate.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Parameters
RATIO     = 0.75       # Stricter ratio test for better matches
RANSAC_TH = 2.0        # Stricter RANSAC threshold for better F-matrix
BASELINE  = 0.15       # Your measured camera shift in meters
REPROJ_TH = 20.0       # Based on the observed mean error
MAX_DEPTH = 3.0        # Maximum depth in meters (reduced from 5.0)
MIN_DEPTH = 0.5        # Minimum depth in meters (increased from 0.1)

def calculate_reprojection_error(pts3, pts2, P):
    """Calculate reprojection error for a set of 3D points."""
    try:
        # Project 3D points to 2D
        pts4 = np.vstack((pts3.T, np.ones(len(pts3))))
        pts2_proj = P @ pts4
        pts2_proj = pts2_proj[:2] / pts2_proj[2]
        
        # Calculate error
        error = np.linalg.norm(pts2 - pts2_proj.T, axis=1)
        return error
    except Exception as e:
        print(f"Error in calculate_reprojection_error: {str(e)}")
        print(f"pts3 shape: {pts3.shape}, pts2 shape: {pts2.shape}, P shape: {P.shape}")
        raise

def draw_camera_pose(ax, R, t, scale=0.1, color='r'):
    """Draw camera pose as a coordinate frame."""
    # Camera center
    C = -R.T @ t.reshape(3, 1)  # Ensure t is a column vector
    C = C.flatten()  # Convert back to 1D array for easier indexing
    # Camera axes
    axes = np.eye(3) * scale
    # Transform axes to world coordinates
    axes_world = (R.T @ axes.T).T
    # Draw axes
    ax.quiver(C[0], C[1], C[2], axes_world[0,0], axes_world[0,1], axes_world[0,2], color='r', label='X')
    ax.quiver(C[0], C[1], C[2], axes_world[1,0], axes_world[1,1], axes_world[1,2], color='g', label='Y')
    ax.quiver(C[0], C[1], C[2], axes_world[2,0], axes_world[2,1], axes_world[2,2], color='b', label='Z')
    # Add text labels
    ax.text(C[0] + axes_world[0,0]*1.2, C[1] + axes_world[0,1]*1.2, C[2] + axes_world[0,2]*1.2, 'X', color='r')
    ax.text(C[0] + axes_world[1,0]*1.2, C[1] + axes_world[1,1]*1.2, C[2] + axes_world[1,2]*1.2, 'Y', color='g')
    ax.text(C[0] + axes_world[2,0]*1.2, C[1] + axes_world[2,1]*1.2, C[2] + axes_world[2,2]*1.2, 'Z', color='b')
    return C  # Return camera center for label placement

def main(detector="SIFT"):
    try:
        # Load & undistort pair
        img1 = cv2.imread("results/undistorted_scene_left.png")
        img2 = cv2.imread("results/undistorted_scene_right.png")
        K    = np.load("calib/K.npy")

        if img1 is None or img2 is None:
            print("Error: Could not load images. Please check if the files exist:")
            print("  - results/undistorted_scene_left.png")
            print("  - results/undistorted_scene_right.png")
            return

        print(f"Image shapes: left={img1.shape}, right={img2.shape}")
        print(f"Camera matrix shape: {K.shape}")
        print(f"Camera matrix:\n{K}")

        # Grayscale
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Detector setup
        if detector.upper() == "ORB":
            feat = cv2.ORB_create(10000)  # Increased max features
            norm = cv2.NORM_HAMMING
        else:
            feat = cv2.SIFT_create(nfeatures=10000)  # Increased max features
            norm = cv2.NORM_L2

        # Detect & match
        kp1, des1 = feat.detectAndCompute(g1, None)
        kp2, des2 = feat.detectAndCompute(g2, None)
        matches   = cv2.BFMatcher(norm).knnMatch(des1, des2, k=2)
        good      = [m for m,n in matches if m.distance < RATIO*n.distance]
        print(f"Initial matches: {len(good)}")

        if len(good) < 8:
            print("Error: Not enough good matches found")
            return

        # Prepare point arrays
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # Fundamental + inliers
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, RANSAC_TH)
        if F is None:
            print("Error: Could not find fundamental matrix")
            return

        print(f"Fundamental matrix:\n{F}")

        in1 = pts1[mask.ravel()==1]
        in2 = pts2[mask.ravel()==1]
        print(f"After F-matrix filtering: {len(in1)} points")

        # Draw feature matches and epipolar lines
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("results/feature_matches.png", match_img)
        
        def draw_epipolar_lines(img1, img2, pts1, pts2, F):
            h, w = img1.shape[:2]
            lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
            lines1 = lines1.reshape(-1,3)
            img1_epi = img1.copy()
            for r,pt1 in zip(lines1, pts1):
                color = tuple(np.random.randint(0,255,3).tolist())
                x0,y0 = map(int, [0, -r[2]/r[1]])
                x1,y1 = map(int, [w, -(r[2]+r[0]*w)/r[1]])
                cv2.line(img1_epi, (x0,y0), (x1,y1), color, 1)
            return img1_epi

        epi_img = draw_epipolar_lines(img1, img2, in1, in2, F)
        cv2.imwrite("results/epipolar_lines.png", epi_img)

        # Essential + pose (unit translation)
        E = K.T @ F @ K
        retval, R, t_unit, mask = cv2.recoverPose(E, in1, in2, K)
        if not retval:
            print("Error: Could not recover pose")
            return

        print(f"Rotation matrix:\n{R}")
        print(f"Translation vector (unit):\n{t_unit}")

        t = t_unit * BASELINE
        print(f"Translation vector (scaled):\n{t}")

        # Projection matrices
        P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
        P2 = K @ np.hstack((R, t.reshape(3,1)))

        print(f"Projection matrix 1:\n{P1}")
        print(f"Projection matrix 2:\n{P2}")

        # Triangulate
        pts4 = cv2.triangulatePoints(P1, P2, in1.T, in2.T)  # 4×N
        pts4 /= pts4[3]                                     # homogeneous → metric
        pts3 = pts4[:3].T                                   # N×3

        # Scale the points to match the baseline
        scale = BASELINE / np.linalg.norm(t)
        pts3 *= scale

        print(f"Point cloud statistics before filtering:")
        print(f"  Mean position: {np.mean(pts3, axis=0)}")
        print(f"  Std position: {np.std(pts3, axis=0)}")
        print(f"  Min position: {np.min(pts3, axis=0)}")
        print(f"  Max position: {np.max(pts3, axis=0)}")

        # Keep only points in front of both cameras and within depth range
        mask = pts3[:,2] > MIN_DEPTH
        mask &= pts3[:,2] < MAX_DEPTH
        
        # Check points in second camera
        pts3_cam2 = (R @ pts3.T).T + t.reshape(1, 3)
        mask &= pts3_cam2[:,2] > MIN_DEPTH
        mask &= pts3_cam2[:,2] < MAX_DEPTH
        
        # Additional filtering: remove points too far from camera centers
        cam1_center = np.zeros(3)
        cam2_center = -R.T @ t.reshape(3)  # Fixed: reshape to 1D array
        dist1 = np.linalg.norm(pts3 - cam1_center, axis=1)
        dist2 = np.linalg.norm(pts3 - cam2_center, axis=1)
        max_dist = MAX_DEPTH * 2  # Maximum distance from either camera
        mask &= (dist1 < max_dist) & (dist2 < max_dist)
        
        # Additional filtering: remove statistical outliers
        mean_pos = np.mean(pts3, axis=0)
        std_pos = np.std(pts3, axis=0)
        z_scores = np.abs((pts3 - mean_pos) / std_pos)
        mask &= np.all(z_scores < 3.0, axis=1)  # Remove points more than 3 std devs away
        
        pts3 = pts3[mask]
        in1 = in1[mask]
        in2 = in2[mask]
        print(f"After depth filtering: {len(pts3)} points")

        if len(pts3) == 0:
            print("Error: No points left after depth filtering")
            return

        # Calculate reprojection errors
        error1 = calculate_reprojection_error(pts3, in1, P1)
        error2 = calculate_reprojection_error(pts3, in2, P2)
        
        # After reprojection error calculation, add error distribution plot
        plt.figure(figsize=(10, 5))
        plt.hist(error1, bins=50, alpha=0.5, label='Left Camera')
        plt.hist(error2, bins=50, alpha=0.5, label='Right Camera')
        plt.axvline(REPROJ_TH, color='r', linestyle='--', label=f'Threshold ({REPROJ_TH}px)')
        plt.xlabel('Reprojection Error (pixels)')
        plt.ylabel('Number of Points')
        plt.title('Reprojection Error Distribution')
        plt.legend()
        plt.grid(True)
        plt.savefig("results/reprojection_errors.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Calculate point cloud metrics
        if len(pts3) > 0:
            # Calculate bounding box
            min_coords = np.min(pts3, axis=0)
            max_coords = np.max(pts3, axis=0)
            bbox_size = max_coords - min_coords
            bbox_volume = np.prod(bbox_size)
            
            # Calculate point cloud metrics
            print("\nReconstruction Quality Metrics:")
            print(f"  Average reprojection error: {np.mean([error1, error2]):.2f} px")
            print(f"  Median reprojection error: {np.median([error1, error2]):.2f} px")
            print(f"  Point cloud density: {len(pts3)/bbox_volume:.2f} points/m³")
            print(f"  Point cloud coverage: {bbox_size} m")
            print(f"  Depth range: {min_coords[2]:.2f} to {max_coords[2]:.2f} m")
            print(f"  Bounding box size: {bbox_size} m")
            print(f"  Total points: {len(pts3)}")
        else:
            print("No points in point cloud to calculate metrics")

        # Filter points based on reprojection error
        mask = (error1 < REPROJ_TH) & (error2 < REPROJ_TH)
        pts3 = pts3[mask]
        in1 = in1[mask]
        print(f"After reprojection error filtering: {len(pts3)} points")
        
        if len(pts3) == 0:
            print("Error: No points left after reprojection error filtering")
            return

        # Sample colors from left image using bilinear interpolation
        colors = np.zeros((len(in1), 3))
        for i, pt in enumerate(in1):
            x, y = pt[0], pt[1]
            x0, y0 = int(x), int(y)
            x1, y1 = min(x0 + 1, img1.shape[1] - 1), min(y0 + 1, img1.shape[0] - 1)
            
            # Bilinear interpolation weights
            wx = x - x0
            wy = y - y0
            
            # Get colors from four nearest pixels
            c00 = img1[y0, x0]
            c10 = img1[y0, x1]
            c01 = img1[y1, x0]
            c11 = img1[y1, x1]
            
            # Interpolate
            c0 = c00 * (1 - wx) + c10 * wx
            c1 = c01 * (1 - wx) + c11 * wx
            color = c0 * (1 - wy) + c1 * wy
            
            colors[i] = color / 255.0  # Normalize to [0,1]

        # Save point-cloud with colors
        os.makedirs("results", exist_ok=True)
        colored_points = np.hstack((pts3, colors))
        np.savetxt("results/pointcloud.xyz", colored_points, fmt="%.6f")
        print(f"\nFinal statistics:")
        print(f"  Total matches: {len(good)}")
        print(f"  Final points: {len(pts3)}")
        print(f"  Filtered points: {len(mask) - np.sum(mask)}")

        # Create two separate visualizations
        
        # 1. Simple point cloud visualization
        fig1 = plt.figure(figsize=(12, 10))
        ax1 = fig1.add_subplot(projection='3d')
        
        # Plot points in black
        scatter1 = ax1.scatter(pts3[:,0], pts3[:,1], pts3[:,2], 
                             c='black', s=1, alpha=0.6)
        
        # Set labels and title
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Point Cloud Reconstruction')
        
        # Set equal aspect ratio
        max_range = np.array([pts3[:,0].max()-pts3[:,0].min(),
                            pts3[:,1].max()-pts3[:,1].min(),
                            pts3[:,2].max()-pts3[:,2].min()]).max() / 2.0
        mid_x = (pts3[:,0].max()+pts3[:,0].min()) * 0.5
        mid_y = (pts3[:,1].max()+pts3[:,1].min()) * 0.5
        mid_z = (pts3[:,2].max()+pts3[:,2].min()) * 0.5
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Set the initial view angle
        ax1.view_init(elev=20, azim=45)
        
        # Add grid for better depth perception
        ax1.grid(True)
        
        plt.tight_layout()
        plt.savefig("results/pointcloud_simple.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Colored point cloud with camera poses
        fig2 = plt.figure(figsize=(12, 10))
        ax2 = fig2.add_subplot(projection='3d')
        
        # Plot colored points
        scatter2 = ax2.scatter(pts3[:,0], pts3[:,1], pts3[:,2], 
                             c=colors, s=2, alpha=0.8)
        
        # Calculate appropriate scale for camera poses
        point_cloud_size = np.max(pts3, axis=0) - np.min(pts3, axis=0)
        pose_scale = np.min(point_cloud_size) * 0.1  # 10% of the smallest dimension
        
        # Draw camera poses with adjusted scale and get camera centers
        cam1_center = draw_camera_pose(ax2, np.eye(3), np.zeros(3), scale=pose_scale, color='r')  # First camera
        cam2_center = draw_camera_pose(ax2, R, t, scale=pose_scale, color='b')  # Second camera
        
        # Add camera labels
        ax2.text(cam1_center[0]-0.1, cam1_center[1], cam1_center[2], 'Right Camera', color='r', fontsize=10)
        ax2.text(cam2_center[0]+0.3, cam2_center[1], cam2_center[2], 'Left Camera', color='b', fontsize=10)
        
        # Set labels and title
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.set_title('3D Reconstruction with Camera Poses and Colors')
        
        # Set equal aspect ratio
        max_range = np.array([pts3[:,0].max()-pts3[:,0].min(),
                            pts3[:,1].max()-pts3[:,1].min(),
                            pts3[:,2].max()-pts3[:,2].min()]).max() / 2.0
        mid_x = (pts3[:,0].max()+pts3[:,0].min()) * 0.5
        mid_y = (pts3[:,1].max()+pts3[:,1].min()) * 0.5
        mid_z = (pts3[:,2].max()+pts3[:,2].min()) * 0.5
        
        # Increase the range to ensure camera poses are visible
        zoom_factor = 1.4  # Increase this value to zoom out more
        ax2.set_xlim(mid_x - max_range * zoom_factor, mid_x + max_range * zoom_factor)
        ax2.set_ylim(mid_y - max_range * zoom_factor, mid_y + max_range * zoom_factor)
        ax2.set_zlim(mid_z - max_range * zoom_factor, mid_z + max_range * zoom_factor)
        
        # Set the initial view angle
        ax2.view_init(elev=20, azim=45)
        
        # Add grid for better depth perception
        ax2.grid(True)
        
        # Add legend for camera axes
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.tight_layout()
        plt.savefig("results/pointcloud_colored.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create interactive visualization
        create_interactive_visualization(pts3, colors, R, t)

        # Save point cloud in PLY format
        save_point_cloud_ply(pts3, colors, "results/pointcloud.ply")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

def create_interactive_visualization(pts3, colors, R, t, scale=0.1):
    """Create an interactive 3D visualization using plotly."""
    # Create figure
    fig = go.Figure()
    
    # Add point cloud
    fig.add_trace(go.Scatter3d(
        x=pts3[:, 0], y=pts3[:, 1], z=pts3[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            opacity=0.8
        ),
        name='Point Cloud'
    ))
    
    # Add camera poses
    def add_camera_pose(R, t, name, color):
        # Camera center
        C = -R.T @ t.reshape(3, 1)
        C = C.flatten()
        
        # Camera axes
        axes = np.eye(3) * scale
        axes_world = (R.T @ axes.T).T
        
        # Add axes
        for i, (axis, label) in enumerate(zip(axes_world, ['X', 'Y', 'Z'])):
            fig.add_trace(go.Scatter3d(
                x=[C[0], C[0] + axis[0]],
                y=[C[1], C[1] + axis[1]],
                z=[C[2], C[2] + axis[2]],
                mode='lines',
                line=dict(color=color, width=3),
                name=f'{name} {label}'
            ))
    
    # Add first camera (identity)
    add_camera_pose(np.eye(3), np.zeros(3), 'Camera 1', 'red')
    # Add second camera
    add_camera_pose(R, t, 'Camera 2', 'blue')
    
    # Update layout
    fig.update_layout(
        title='Interactive 3D Reconstruction',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        showlegend=True
    )
    
    # Save as HTML for interactive viewing
    fig.write_html("results/interactive_pointcloud.html")

def save_point_cloud_ply(pts3, colors, filename):
    """Save point cloud in PLY format with colors."""
    with open(filename, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts3)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write points and colors
        for pt, color in zip(pts3, colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")

if __name__ == "__main__":
    main()
