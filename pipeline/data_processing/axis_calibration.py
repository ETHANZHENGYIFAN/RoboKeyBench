"""
Cross-view principal-axis calibration.

This module derives a consistent object-centric coordinate frame from top-view
and bottom-view rendered images, avoiding ambiguity from single-view vertical
axis estimation.
"""

import cv2
import numpy as np


def mask_centroid_3d(
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    extrinsic: np.ndarray,
) -> np.ndarray:
    """
    Compute the 3D centroid of the foreground region in a rendered view.

    Args:
        mask: binary foreground mask with shape (H, W).
        depth: depth map with shape (H, W).
        K: camera intrinsic matrix.
        extrinsic: camera-to-world transform.

    Returns:
        World-space centroid of the foreground points.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Empty mask: no foreground pixels found.")

    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    pts_cam = np.stack([x_cam, y_cam, zs, np.ones_like(zs)], axis=1)

    pts_world = (extrinsic @ pts_cam.T).T[:, :3]
    return pts_world.mean(axis=0)


def calibrate_vertical_axis(
    c_top: np.ndarray,
    c_bot: np.ndarray,
) -> np.ndarray:
    """
    Compute the principal vertical axis from top-view and bottom-view centroids.
    """
    diff = c_top - c_bot
    norm = np.linalg.norm(diff)
    if norm < 1e-8:
        raise ValueError(
            "Top and bottom centroids are too close; cannot determine vertical axis."
        )
    return diff / norm


def build_coordinate_frame(v_vert: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a right-handed object-centric coordinate frame from the vertical axis.
    """
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(v_vert, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    v_horiz1 = ref - np.dot(ref, v_vert) * v_vert
    v_horiz1 /= np.linalg.norm(v_horiz1)

    v_horiz2 = np.cross(v_vert, v_horiz1)
    v_horiz2 /= np.linalg.norm(v_horiz2)

    return v_vert, v_horiz1, v_horiz2


def calibrate_from_masks(
    mask_top: np.ndarray,
    mask_bot: np.ndarray,
    depth_top: np.ndarray,
    depth_bot: np.ndarray,
    K_top: np.ndarray,
    K_bot: np.ndarray,
    extrinsic_top: np.ndarray,
    extrinsic_bot: np.ndarray,
) -> dict:
    """
    Calibrate a coordinate frame from top and bottom masks and depth maps.

    Returns a dict with vertical and horizontal axes plus the two view
    centroids used for calibration.
    """
    c_top = mask_centroid_3d(mask_top, depth_top, K_top, extrinsic_top)
    c_bot = mask_centroid_3d(mask_bot, depth_bot, K_bot, extrinsic_bot)
    v_vert = calibrate_vertical_axis(c_top, c_bot)
    v_vert, v_horiz1, v_horiz2 = build_coordinate_frame(v_vert)

    return {
        "v_vert": v_vert,
        "v_horiz1": v_horiz1,
        "v_horiz2": v_horiz2,
        "c_top": c_top,
        "c_bot": c_bot,
    }


def normalize_image_color(image: np.ndarray) -> np.ndarray:
    """
    Apply per-channel histogram equalization to reduce cross-view appearance
    variation before semantic alignment.
    """
    normalized = np.empty_like(image)
    for channel in range(image.shape[2]):
        normalized[..., channel] = cv2.equalizeHist(image[..., channel])
    return normalized
