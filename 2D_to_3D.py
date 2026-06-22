"""Convert 2D keypoint annotations to 3D mesh coordinates.

The default projection model is orthographic, matching the released
annotation-view convention. Perspective rays remain available for compatibility
with older rendered views.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    import numpy as np
    import trimesh
    from PIL import Image, ImageDraw
    from scipy.spatial.distance import cdist
    from trimesh.ray import ray_triangle
    from trimesh.scene import Camera
except ModuleNotFoundError as exc:
    np = trimesh = Image = ImageDraw = cdist = ray_triangle = Camera = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def require_projection_deps() -> None:
    if _IMPORT_ERROR is not None:
        missing = _IMPORT_ERROR.name or "projection dependency"
        raise ModuleNotFoundError(
            f"{missing} is required for 2D-to-3D projection. Install the project "
            "geometry dependencies before running this command."
        ) from _IMPORT_ERROR


VIEW_SETS = {
    "annotation8": [
        (90, 0, 0),
        (90, 0, 90),
        (90, 0, 180),
        (90, 0, 270),
        (135, 0, 60),
        (135, 0, 120),
        (135, 0, 240),
        (135, 0, 300),
    ],
    "calibration6": [
        (90, 0, 0),
        (90, 0, 90),
        (90, 0, 180),
        (90, 0, 270),
        (0, 0, 0),
        (180, 0, 0),
    ],
}


def parse_views(value: str) -> list[tuple[int, int, int]]:
    if value in VIEW_SETS:
        return VIEW_SETS[value]
    views = []
    for item in value.split(";"):
        parts = [int(float(x.strip())) for x in item.split(",") if x.strip()]
        if len(parts) != 3:
            raise ValueError("Custom views must use 'x,y,z;x,y,z' format.")
        views.append(tuple(parts))
    return views


def convert_pixels(
    pixel_records: list[dict],
    original_size: int = 640,
    target_size: int = 800,
) -> tuple[list[tuple[float, float]], list[str]]:
    scale = target_size / original_size
    coords = []
    labels = []
    for record in pixel_records:
        x_coord, y_coord = record["coordinates"]
        coords.append((x_coord * scale, y_coord * scale))
        labels.append(record.get("label", "F"))
    return coords, labels


def build_scene_from_mujoco(xml_file: Path, obj_folder: Path, texture_folder: Path) -> trimesh.Scene:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    mesh_name_to_file = {}
    for mesh in root.findall(".//mesh"):
        mesh_name = mesh.attrib.get("name")
        mesh_file = mesh.attrib.get("file")
        if mesh_name and mesh_file:
            mesh_name_to_file[mesh_name] = mesh_file

    material_to_texture = {}
    for material in root.findall(".//material"):
        material_name = material.attrib.get("name")
        texture_name = material.attrib.get("texture")
        if material_name and texture_name:
            material_to_texture[material_name] = texture_name

    texture_files = {}
    for texture in root.findall(".//texture"):
        texture_name = texture.attrib.get("name")
        texture_file = texture.attrib.get("file")
        if texture_name and texture_file:
            texture_files[texture_name] = texture_folder / texture_file

    scene = trimesh.Scene()
    for geom in root.findall(".//geom"):
        mesh_name = geom.attrib.get("mesh")
        if not mesh_name or "collision" in mesh_name:
            continue
        mesh_file = mesh_name_to_file.get(mesh_name)
        if not mesh_file:
            continue
        obj_file = obj_folder / Path(mesh_file).name
        if not obj_file.exists():
            continue
        mesh = trimesh.load(obj_file)
        material_name = geom.attrib.get("material")
        texture_name = material_to_texture.get(material_name)
        texture_file = texture_files.get(texture_name)
        if texture_file and texture_file.exists() and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=mesh.visual.uv,
                image=Image.open(texture_file),
            )
        scene.add_geometry(mesh)

    if not scene.geometry:
        raise ValueError(f"No visual geometry found for {xml_file}")
    return scene


def combined_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    meshes = [geom for geom in scene.geometry.values() if isinstance(geom, trimesh.Trimesh)]
    if not meshes:
        raise ValueError("No mesh geometry available.")
    return trimesh.util.concatenate(meshes)


def view_rotation(view: tuple[int, int, int]) -> np.ndarray:
    return trimesh.transformations.euler_matrix(
        np.radians(view[0]),
        np.radians(view[1]),
        np.radians(view[2]),
    )[:3, :3]


def orthographic_view_params(
    mesh: trimesh.Trimesh,
    view: tuple[int, int, int],
    resolution: int,
    ortho_scale: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    rotation = view_rotation(view)
    verts_view = np.asarray(mesh.vertices) @ rotation
    xy = verts_view[:, :2]
    xy_center = (xy.min(axis=0) + xy.max(axis=0)) / 2.0
    span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-8)
    pixel_scale = (resolution * ortho_scale) / span
    return rotation, xy_center, pixel_scale, float(verts_view[:, 2].min()), float(verts_view[:, 2].max())


def set_camera(scene: trimesh.Scene, view: tuple[int, int, int], resolution: int, focal: float, distance: float) -> None:
    rotation = view_rotation(view)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = rotation
    camera_pose[:3, 3] = rotation[:, 2] * distance
    scene.camera = Camera(resolution=(resolution, resolution), focal=(focal, focal))
    scene.camera_transform = camera_pose


def ray_from_pixel(
    scene: trimesh.Scene,
    mesh: trimesh.Trimesh,
    pixel: tuple[float, float],
    view: tuple[int, int, int],
    projection: str,
    resolution: int,
    focal: float,
    distance: float,
    ortho_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    u_coord, v_coord = pixel
    if projection == "orthographic":
        rotation, xy_center, pixel_scale, z_min, z_max = orthographic_view_params(
            mesh,
            view,
            resolution,
            ortho_scale,
        )
        view_x = (u_coord - resolution / 2.0) / pixel_scale + xy_center[0]
        view_y = -(v_coord - resolution / 2.0) / pixel_scale + xy_center[1]
        span_z = max(z_max - z_min, 1e-6)
        origin_view = np.array([view_x, view_y, z_max + span_z + distance], dtype=float)
        direction_view = np.array([0.0, 0.0, -1.0], dtype=float)
        inv_rotation = np.linalg.inv(rotation)
        origin = origin_view @ inv_rotation
        direction = direction_view @ inv_rotation
        return origin, direction / np.linalg.norm(direction)

    set_camera(scene, view, resolution, focal, distance)
    k_matrix = copy.deepcopy(scene.camera.K)
    camera_transform = copy.deepcopy(scene.camera_transform)
    rotation_inv = np.linalg.inv(camera_transform[:3, :3])
    pixel_ray = np.array(
        [
            (u_coord - k_matrix[0, 2]) / k_matrix[0, 0],
            (v_coord - k_matrix[1, 2]) / k_matrix[1, 1],
            1.0,
        ]
    )
    ray_direction = rotation_inv @ pixel_ray
    return camera_transform[:3, 3], ray_direction / np.linalg.norm(ray_direction)


def intersect_first(
    intersector: ray_triangle.RayMeshIntersector,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
) -> np.ndarray | None:
    locations, _, _ = intersector.intersects_location(
        ray_origins=[ray_origin],
        ray_directions=[ray_direction],
        ray_tolerance=1e-5,
        multiple_hits=True,
    )
    if len(locations) == 0:
        return None
    distances = np.linalg.norm(locations - ray_origin, axis=1)
    return locations[np.argsort(distances)[0]]


def pixel_to_3d_batch(
    scene: trimesh.Scene,
    mesh: trimesh.Trimesh,
    pixel_coords: list[tuple[float, float]],
    labels: list[str],
    view: tuple[int, int, int],
    projection: str,
    resolution: int,
    focal: float,
    distance: float,
    ortho_scale: float,
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    keypoints_3d_c: list[np.ndarray | None] = []
    keypoints_3d_f: list[np.ndarray | None] = []
    intersector = ray_triangle.RayMeshIntersector(mesh)

    for pixel_coord, label in zip(pixel_coords, labels):
        ray_origin, ray_direction = ray_from_pixel(
            scene,
            mesh,
            pixel_coord,
            view,
            projection,
            resolution,
            focal,
            distance,
            ortho_scale,
        )
        point = intersect_first(intersector, ray_origin, ray_direction)
        if point is None:
            point = intersect_first(intersector, ray_origin, -ray_direction)
        if label == "C":
            keypoints_3d_c.append(point)
        else:
            keypoints_3d_f.append(point)

    return keypoints_3d_c, keypoints_3d_f


def merge_close_points(points: list[np.ndarray], threshold: float = 0.01, mean: bool = False) -> np.ndarray:
    if not points:
        return np.empty((0, 3))
    points_array = np.asarray(points, dtype=float)
    distance_matrix = cdist(points_array, points_array)
    visited: set[int] = set()
    merged_points = []
    for idx in range(len(points_array)):
        if idx in visited:
            continue
        close_indices = np.where(distance_matrix[idx] < threshold)[0]
        merged_point = (
            np.mean(points_array[close_indices], axis=0)
            if mean
            else points_array[close_indices][0]
        )
        merged_points.append(merged_point)
        visited.update(int(i) for i in close_indices)
    return np.asarray(merged_points)


def add_axis_to_scene(scene: trimesh.Scene, main_point: np.ndarray) -> None:
    axes = [
        (np.array([0.7, 0.0, 0.0]), [255, 0, 0, 255]),
        (np.array([0.0, 0.7, 0.0]), [0, 255, 0, 255]),
        (np.array([0.0, 0.0, 0.7]), [0, 0, 255, 255]),
    ]
    for direction, color in axes:
        for sign in (1, -1):
            end_point = main_point + sign * direction
            line = trimesh.load_path(
                np.hstack([main_point, end_point]).reshape(-1, 2, 3),
                colors=[color],
            )
            scene.add_geometry(line)


def add_keypoints_to_scene(scene: trimesh.Scene, keypoints_3d: np.ndarray, color: list[int] | None = None) -> None:
    color = color or [255, 0, 0, 255]
    for idx, point in enumerate(keypoints_3d):
        sphere = trimesh.creation.icosphere(radius=0.005, subdivisions=3)
        sphere.visual.vertex_colors = color
        sphere.apply_translation(point)
        scene.add_geometry(sphere, node_name=f"Keypoint_{idx}")


def read_keypoint_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    keypoints_data = data.get("keypoints", {})
    sorted_keys = sorted(keypoints_data.keys(), key=lambda x: int(x))
    return [keypoints_data[key] for key in sorted_keys]


def render_orthographic_scene(scene: trimesh.Scene, view: tuple[int, int, int], resolution: int, ortho_scale: float) -> bytes:
    mesh = combined_mesh(scene)
    rotation, xy_center, pixel_scale, _, _ = orthographic_view_params(
        mesh,
        view,
        resolution,
        ortho_scale,
    )
    verts_view = np.asarray(mesh.vertices) @ rotation
    pixels = np.empty((len(mesh.vertices), 2), dtype=float)
    pixels[:, 0] = (verts_view[:, 0] - xy_center[0]) * pixel_scale + resolution / 2.0
    pixels[:, 1] = -(verts_view[:, 1] - xy_center[1]) * pixel_scale + resolution / 2.0

    image = Image.new("RGBA", (resolution, resolution), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    face_order = np.argsort(verts_view[mesh.faces, 2].mean(axis=1))
    for face_idx in face_order:
        color = (200, 200, 200, 255)
        face_colors = getattr(mesh.visual, "face_colors", None)
        if face_colors is not None and len(face_colors) > face_idx:
            color = tuple(int(c) for c in face_colors[face_idx])
        polygon = [tuple(pixels[v]) for v in mesh.faces[face_idx]]
        draw.polygon(polygon, fill=color)

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def process_instance(
    category: str,
    obj_name: str,
    source_root: Path,
    viewpoint_root: Path,
    output_dir: Path,
    views: list[tuple[int, int, int]],
    original_size: int,
    target_size: int,
    focal: float,
    distance: float,
    render_diagnostics: bool,
    projection: str,
    ortho_scale: float,
) -> np.ndarray:
    obj_dir = source_root / category / obj_name
    xml_file = obj_dir / "model.xml"
    scene = build_scene_from_mujoco(xml_file, obj_dir / "visual", obj_dir)
    mesh = combined_mesh(scene)

    valid_c: list[np.ndarray] = []
    valid_f: list[np.ndarray] = []
    for view in views:
        angle_x, angle_y, angle_z = view
        result_path = (
            viewpoint_root
            / category
            / obj_name
            / f"{obj_name}_x{angle_x}_y{angle_y}_z{angle_z}_results.json"
        )
        if not result_path.exists():
            continue
        pixel_records = read_keypoint_json(result_path)
        pixel_coords, labels = convert_pixels(pixel_records, original_size, target_size)
        keypoints_c, keypoints_f = pixel_to_3d_batch(
            scene,
            mesh,
            pixel_coords,
            labels,
            view,
            projection,
            target_size,
            focal,
            distance,
            ortho_scale,
        )
        valid_c.extend(point for point in keypoints_c if point is not None)
        valid_f.extend(point for point in keypoints_f if point is not None)

    filtered_c = merge_close_points(valid_c, threshold=0.1, mean=True)
    filtered_f = merge_close_points(valid_f, threshold=0.2, mean=False)
    filtered_points = np.concatenate([filtered_c, filtered_f], axis=0)

    if render_diagnostics and len(filtered_points) > 0:
        main_point = filtered_points.mean(axis=0)
        add_axis_to_scene(scene, main_point)
        add_keypoints_to_scene(scene, filtered_points)
        output_dir.mkdir(parents=True, exist_ok=True)
        for view in views:
            angle_x, angle_y, angle_z = view
            out_file = output_dir / f"{obj_name}_x{angle_x}_y{angle_y}_z{angle_z}.png"
            if projection == "orthographic":
                out_file.write_bytes(render_orthographic_scene(scene, view, target_size, ortho_scale))
            else:
                set_camera(scene, view, target_size, focal, distance)
                image_bytes = scene.save_image(resolution=(target_size, target_size), visible=True)
                if image_bytes:
                    out_file.write_bytes(image_bytes)

    return filtered_points


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RoboKeyBench 2D annotations to 3D points.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--viewpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--views", default="annotation8")
    parser.add_argument("--original-size", type=int, default=640)
    parser.add_argument("--target-size", type=int, default=800)
    parser.add_argument("--focal", type=float, default=1000.0)
    parser.add_argument("--distance", type=float, default=2.0)
    parser.add_argument("--projection", choices=("orthographic", "perspective"), default="orthographic")
    parser.add_argument("--ortho-scale", type=float, default=0.85)
    parser.add_argument("--render-diagnostics", action="store_true")
    parser.add_argument("--save-npy", action="store_true")
    args = parser.parse_args()
    require_projection_deps()

    views = parse_views(args.views)
    for category in args.categories:
        category_root = args.source_root / category
        if not category_root.exists():
            print(f"skip missing category {category_root}")
            continue
        for obj_dir in sorted(path for path in category_root.iterdir() if path.is_dir()):
            try:
                points = process_instance(
                    category,
                    obj_dir.name,
                    args.source_root,
                    args.viewpoint_root,
                    args.output_dir,
                    views,
                    args.original_size,
                    args.target_size,
                    args.focal,
                    args.distance,
                    args.render_diagnostics,
                    args.projection,
                    args.ortho_scale,
                )
                print(f"{category}/{obj_dir.name}: {len(points)} points")
                if args.save_npy:
                    out = args.output_dir / category
                    out.mkdir(parents=True, exist_ok=True)
                    np.save(out / f"{obj_dir.name}_keypoints_3d.npy", points)
            except Exception as exc:
                print(f"failed {category}/{obj_dir.name}: {exc}")


if __name__ == "__main__":
    main()
