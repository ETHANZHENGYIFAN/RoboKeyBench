"""
Cross-View Principal Axis Calibration (Section B.4)

Derives a consistent object-centric coordinate frame from top-view and
bottom-view rendered images, avoiding the ambiguity that arises when the
vertical axis is inferred from a single viewpoint.
"""

import numpy as np
import cv2


def mask_centroid_3d(
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    extrinsic: np.ndarray,
) -> np.ndarray:
    """
    Compute the 3-D centroid of the foreground region in a rendered view.

    Args:
        mask:      Binary foreground mask (H, W), uint8 or bool.
        depth:     Depth map (H, W) in metres.
        K:         (3, 3) camera intrinsic matrix.
        extrinsic: (4, 4) camera-to-world transform [R | t].

    Returns:
        centroid_3d: (3,) world-space centroid of the foreground.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Empty mask — no foreground pixels found.")

    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Back-project to camera space
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    pts_cam = np.stack([x_cam, y_cam, zs, np.ones_like(zs)], axis=1)  # (N,4)

    # Transform to world space
    pts_world = (extrinsic @ pts_cam.T).T[:, :3]
    return pts_world.mean(axis=0)


def calibrate_vertical_axis(
    c_top: np.ndarray,
    c_bot: np.ndarray,
) -> np.ndarray:
    """
    Compute the principal vertical axis from top-view and bottom-view centroids.

        v_vert = (c_top - c_bot) / ||c_top - c_bot||

    Args:
        c_top: (3,) 3-D centroid from top-view segmentation.
        c_bot: (3,) 3-D centroid from bottom-view segmentation.

    Returns:
        v_vert: (3,) unit vertical axis.
    """
    diff = c_top - c_bot
    norm = np.linalg.norm(diff)
    if norm < 1e-8:
        raise ValueError("Top and bottom centroids are too close; "
                         "cannot determine vertical axis.")
    return diff / norm


def build_coordinate_frame(v_vert: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a right-handed object-centric coordinate frame given the vertical axis.

    The two horizontal axes are derived via the Gram–Schmidt process using an
    arbitrary reference vector to complete the frame.

    Args:
        v_vert: (3,) unit vertical axis.

    Returns:
        v_vert:   (3,) vertical axis (unchanged).
        v_horiz1: (3,) first horizontal axis.
        v_horiz2: (3,) second horizontal axis, orthogonal to both.
    """
    # Choose a reference vector not parallel to v_vert
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
    Full calibration pipeline from top/bottom masks and depth maps.

    Returns a dict with keys:
        'v_vert'    – (3,) unit vertical axis
        'v_horiz1'  – (3,) first horizontal axis
        'v_horiz2'  – (3,) second horizontal axis
        'c_top'     – (3,) top centroid
        'c_bot'     – (3,) bottom centroid
    """
    c_top = mask_centroid_3d(mask_top, depth_top, K_top, extrinsic_top)
    c_bot = mask_centroid_3d(mask_bot, depth_bot, K_bot, extrinsic_bot)
    v_vert = calibrate_vertical_axis(c_top, c_bot)
    v_vert, v_horiz1, v_horiz2 = build_coordinate_frame(v_vert)

    return {
        "v_vert":   v_vert,
        "v_horiz1": v_horiz1,
        "v_horiz2": v_horiz2,
        "c_top":    c_top,
        "c_bot":    c_bot,
    }


def normalize_image_color(image: np.ndarray) -> np.ndarray:
    """
    Per-channel histogram equalization to reduce cross-view appearance
    variation before passing images to GPT-4o.

    Args:
        image: (H, W, 3) BGR or RGB uint8 image.

    Returns:
        normalized: (H, W, 3) uint8 image with equalized channels.
    """
    normalized = np.empty_like(image)
    for c in range(image.shape[2]):
        normalized[..., c] = cv2.equalizeHist(image[..., c])
    return normalized
