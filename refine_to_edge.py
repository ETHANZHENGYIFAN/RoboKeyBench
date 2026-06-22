"""Functional geometric optimization utilities.

The module contains release-level implementations for:

1. 2D edge-aware refinement on rendered masks.
2. 3D mesh-distance refinement for projection onto a valid mesh surface.
3. Symmetry-aware augmentation with local normal consistency validation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import cv2
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree
except ModuleNotFoundError as exc:
    cv2 = np = trimesh = cKDTree = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def require_geometry_deps() -> None:
    if _IMPORT_ERROR is not None:
        missing = _IMPORT_ERROR.name or "geometry dependency"
        raise ModuleNotFoundError(
            f"{missing} is required for geometric refinement. Install the project "
            "geometry dependencies before running this command."
        ) from _IMPORT_ERROR


def get_edge(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No object contour found.")
    contour = max(contours, key=cv2.contourArea)
    return contour.squeeze()


def find_adjust(center: np.ndarray, current_point: np.ndarray, edge_points: np.ndarray) -> np.ndarray:
    current_vec = current_point - center
    current_norm = np.linalg.norm(current_vec)
    if current_norm < 1e-8:
        return edge_points[np.argmin(np.linalg.norm(edge_points - current_point, axis=1))]

    edge_vecs = edge_points - center
    edge_norms = np.linalg.norm(edge_vecs, axis=1)
    valid = edge_norms > 1e-8
    cosines = np.full(len(edge_points), -np.inf)
    cosines[valid] = (edge_vecs[valid] @ current_vec) / (edge_norms[valid] * current_norm)
    return edge_points[int(np.argmax(cosines))]


def refine_2d_keypoints_to_edge(
    image: np.ndarray,
    keypoints: np.ndarray,
    center: np.ndarray | None = None,
    radial_threshold: float = 0.7,
) -> np.ndarray:
    edge_points = get_edge(image)
    center = np.asarray(center if center is not None else edge_points.mean(axis=0), dtype=float)
    adjusted = []
    for row in keypoints:
        point_id, x_coord, y_coord = row
        current = np.array([x_coord, y_coord], dtype=float)
        candidate = find_adjust(center, current, edge_points)
        len_to_target = np.linalg.norm(current - center)
        len_to_edge = np.linalg.norm(candidate - center)
        ratio = len_to_target / len_to_edge if len_to_edge > 1e-8 else 0.0
        adjusted.append([point_id, candidate[0], candidate[1]] if ratio >= radial_threshold else row)
    return np.asarray(adjusted)


def _safe_vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    if normals.shape != np.asarray(mesh.vertices).shape:
        normals = np.zeros_like(np.asarray(mesh.vertices, dtype=float))
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-8)


def mesh_surface_samples(
    mesh: trimesh.Trimesh,
    include_edges: bool = True,
    return_normals: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    samples = [vertices]
    normals = [_safe_vertex_normals(mesh)]

    if len(mesh.faces) > 0:
        samples.append(np.asarray(mesh.triangles_center, dtype=float))
        normals.append(np.asarray(mesh.face_normals, dtype=float))

    if include_edges and hasattr(mesh, "edges_unique") and len(mesh.edges_unique) > 0:
        edges = vertices[mesh.edges_unique]
        samples.append(edges.mean(axis=1))
        vertex_normals = _safe_vertex_normals(mesh)
        edge_normals = vertex_normals[mesh.edges_unique].mean(axis=1)
        norms = np.linalg.norm(edge_normals, axis=1, keepdims=True)
        normals.append(edge_normals / np.maximum(norms, 1e-8))

    sample_array = np.concatenate(samples, axis=0)
    normal_array = np.concatenate(normals, axis=0)
    return (sample_array, normal_array) if return_normals else sample_array


def closest_points_by_mesh_distance(
    keypoints: np.ndarray,
    mesh: trimesh.Trimesh,
    include_edges: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.asarray(keypoints, dtype=float)
    try:
        closest, distances, _ = trimesh.proximity.closest_point(mesh, keypoints)
        return np.asarray(closest, dtype=float), np.asarray(distances, dtype=float)
    except Exception:
        samples = mesh_surface_samples(mesh, include_edges=include_edges)
        tree = cKDTree(samples)
        distances, indices = tree.query(keypoints, k=1)
        return samples[indices], distances


def refine_keypoints_to_mesh(
    keypoints: np.ndarray,
    mesh: trimesh.Trimesh,
    max_distance: float | None = None,
    include_edges: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    refined, distances = closest_points_by_mesh_distance(
        keypoints,
        mesh,
        include_edges=include_edges,
    )
    if max_distance is not None:
        keep_original = distances > max_distance
        refined[keep_original] = keypoints[keep_original]
    return refined, distances


def pca_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered / max(len(points), 1)
    _, eigenvectors = np.linalg.eigh(cov)
    return centroid, eigenvectors[:, ::-1].T


def reflect_across_plane(points: np.ndarray, centroid: np.ndarray, normal: np.ndarray) -> np.ndarray:
    normal = normal / np.linalg.norm(normal)
    offsets = points - centroid
    signed = offsets @ normal
    return points - 2.0 * signed[:, None] * normal[None, :]


def symmetry_augment_keypoints(
    keypoints: np.ndarray,
    mesh: trimesh.Trimesh,
    tolerance: float = 0.03,
    duplicate_tolerance: float = 0.01,
    normal_alignment_threshold: float = 0.25,
) -> np.ndarray:
    surface, normals = mesh_surface_samples(mesh, include_edges=True, return_normals=True)
    surface_tree = cKDTree(surface)
    existing = [np.asarray(point, dtype=float) for point in keypoints]
    source_distances, source_indices = surface_tree.query(np.asarray(keypoints, dtype=float), k=1)

    centroid, axes = pca_axes(surface)
    for axis in axes:
        reflected = reflect_across_plane(np.asarray(keypoints, dtype=float), centroid, axis)
        distances, indices = surface_tree.query(reflected, k=1)
        for source_index, distance, index in zip(source_indices, distances, indices):
            if distance > tolerance:
                continue
            candidate = surface[index]
            normal_match = abs(float(normals[source_index] @ normals[index]))
            if normal_match < normal_alignment_threshold:
                continue
            if any(np.linalg.norm(candidate - point) < duplicate_tolerance for point in existing):
                continue
            existing.append(candidate)

    return np.asarray(existing)


def load_keypoint_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0].astype(int), data[:, 1:]


def save_keypoint_csv(path: Path, ids: np.ndarray, coords: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["id", "x", "y"] if coords.shape[1] == 2 else ["id", "x", "y", "z"]
        writer.writerow(header)
        for point_id, coord in zip(ids, coords):
            writer.writerow([int(point_id), *[float(x) for x in coord]])


def adjust_edge(img_file: str | Path, key_points_file: str | Path, threshold: float = 0.7) -> np.ndarray:
    image = cv2.imread(str(img_file))
    if image is None:
        raise FileNotFoundError(img_file)
    ids, coords = load_keypoint_csv(Path(key_points_file))
    rows = np.column_stack([ids, coords])
    return refine_2d_keypoints_to_edge(image, rows, radial_threshold=threshold)


def run_image_mode(args: argparse.Namespace) -> None:
    adjusted = adjust_edge(args.image, args.keypoints, threshold=args.threshold)
    save_keypoint_csv(args.output, adjusted[:, 0], adjusted[:, 1:])


def run_mesh_mode(args: argparse.Namespace) -> None:
    mesh = trimesh.load(args.mesh, force="mesh")
    ids, coords = load_keypoint_csv(args.keypoints)
    refined, distances = refine_keypoints_to_mesh(
        coords,
        mesh,
        max_distance=args.max_distance,
        include_edges=not args.no_edge_samples,
    )
    if args.symmetry:
        augmented = symmetry_augment_keypoints(
            refined,
            mesh,
            tolerance=args.symmetry_tolerance,
            duplicate_tolerance=args.duplicate_tolerance,
            normal_alignment_threshold=args.normal_alignment_threshold,
        )
        if len(augmented) > len(refined):
            extra_count = len(augmented) - len(refined)
            extra_ids = np.arange(ids.max() + 1, ids.max() + 1 + extra_count)
            ids = np.concatenate([ids, extra_ids])
            refined = augmented

    save_keypoint_csv(args.output, ids, refined)
    if args.distance_output:
        args.distance_output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.distance_output, distances, delimiter=",")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine RoboKeyBench keypoints with geometric constraints.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    image_parser = subparsers.add_parser("image", help="2D contour edge refinement.")
    image_parser.add_argument("--image", type=Path, required=True)
    image_parser.add_argument("--keypoints", type=Path, required=True)
    image_parser.add_argument("--output", type=Path, required=True)
    image_parser.add_argument("--threshold", type=float, default=0.7)
    image_parser.set_defaults(func=run_image_mode)

    mesh_parser = subparsers.add_parser("mesh", help="3D mesh-distance and symmetry refinement.")
    mesh_parser.add_argument("--mesh", type=Path, required=True)
    mesh_parser.add_argument("--keypoints", type=Path, required=True)
    mesh_parser.add_argument("--output", type=Path, required=True)
    mesh_parser.add_argument("--max-distance", type=float)
    mesh_parser.add_argument("--distance-output", type=Path)
    mesh_parser.add_argument("--no-edge-samples", action="store_true")
    mesh_parser.add_argument("--symmetry", action="store_true")
    mesh_parser.add_argument("--symmetry-tolerance", type=float, default=0.03)
    mesh_parser.add_argument("--duplicate-tolerance", type=float, default=0.01)
    mesh_parser.add_argument("--normal-alignment-threshold", type=float, default=0.25)
    mesh_parser.set_defaults(func=run_mesh_mode)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    require_geometry_deps()
    args.func(args)


if __name__ == "__main__":
    main()
