import numpy as np
import cv2


def calculate_correspondences(pc_1, pc_2):
    sq_dist = lambda p1, p2: (p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2
    correspondences = []
    for pt in pc_1:
        closest_point = min(pc_2, key=lambda x: sq_dist(pt, x))
        correspondences.append(closest_point)
    return np.array(correspondences)


def transform_pc(pc, affine_transform_mat):
    return cv2.transform(np.array([pc]).astype(np.float32), affine_transform_mat)[0]


def procrustes_analysis(points1, points2):
    """
    Perform Procrustes analysis to find the optimal rotation and translation
    that aligns points1 to points2.

    Parameters:
    points1 (np.ndarray): The first set of points (N x 2).
    points2 (np.ndarray): The second set of points (N x 2).

    Returns:
    np.ndarray: The transformed points1.
    np.ndarray: The rotation matrix.
    np.ndarray: The translation vector.
    """
    # Center the points by subtracting the centroid
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)
    
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2

    # Compute the optimal rotation matrix using SVD
    U, _, Vt = np.linalg.svd(np.dot(centered_points2.T, centered_points1))
    rotation_matrix = np.dot(U, Vt)

    # Compute the translation vector
    translation_vector = centroid2 - np.dot(centroid1, rotation_matrix.T)

    # Apply the transformation
    transformed_points1 = np.dot(points1, rotation_matrix.T) + translation_vector

    return transformed_points1, rotation_matrix, translation_vector


def icp(pc_1, pc_2, num_iters=10):
    result_pc, corr_pc_2 = pc_1.astype(np.float32).copy(), pc_2.astype(np.float32)
    for _ in range(num_iters):
        corr_pc_2 = calculate_correspondences(result_pc, pc_2)
        result_pc, _, _ = procrustes_analysis(result_pc, corr_pc_2)

    sse = np.sum((result_pc - corr_pc_2) ** 2)
    return result_pc, sse