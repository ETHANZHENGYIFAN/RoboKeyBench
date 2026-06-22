"""Render RoboKeyBench multi-view object images from MuJoCo XML assets.

The default renderer follows the manuscript's annotation convention: four
horizontal orthographic views and four oblique orthographic projections.
Perspective rendering remains available as a compatibility option.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    import numpy as np
    import trimesh
    from PIL import Image, ImageDraw
    from trimesh.scene import Camera
except ModuleNotFoundError as exc:
    np = trimesh = Image = ImageDraw = Camera = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def require_render_deps() -> None:
    if _IMPORT_ERROR is not None:
        missing = _IMPORT_ERROR.name or "rendering dependency"
        raise ModuleNotFoundError(
            f"{missing} is required for rendering. Install the project rendering "
            "dependencies before running this command."
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
    "extended12": [
        (90, 0, 0),
        (90, 0, 90),
        (90, 0, 180),
        (90, 0, 270),
        (135, 0, 0),
        (135, 0, 120),
        (135, 0, 240),
        (45, 0, 0),
        (45, 0, 120),
        (45, 0, 240),
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
            mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, image=Image.open(texture_file))
        scene.add_geometry(mesh)

    if not scene.geometry:
        raise ValueError(f"No visual geometry found for {xml_file}")
    return scene


def camera_pose_for_view(angle_x: int, angle_y: int, angle_z: int, distance: float) -> np.ndarray:
    transform = trimesh.transformations.euler_matrix(
        np.radians(angle_x),
        np.radians(angle_y),
        np.radians(angle_z),
    )
    pose = np.eye(4)
    pose[:3, :3] = transform[:3, :3]
    pose[:3, 3] = transform[:3, 2] * distance
    return pose


def _combined_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    meshes = [geom for geom in scene.geometry.values() if isinstance(geom, trimesh.Trimesh)]
    if not meshes:
        raise ValueError("No mesh geometry available for orthographic rendering.")
    return trimesh.util.concatenate(meshes)


def _view_rotation(view: tuple[int, int, int]) -> np.ndarray:
    return trimesh.transformations.euler_matrix(
        np.radians(view[0]),
        np.radians(view[1]),
        np.radians(view[2]),
    )[:3, :3]


def _face_color(mesh: trimesh.Trimesh, face_idx: int) -> tuple[int, int, int, int]:
    face_colors = getattr(mesh.visual, "face_colors", None)
    if face_colors is not None and len(face_colors) > face_idx:
        return tuple(int(c) for c in face_colors[face_idx])
    vertex_colors = getattr(mesh.visual, "vertex_colors", None)
    if vertex_colors is not None and len(vertex_colors) >= len(mesh.vertices):
        color = vertex_colors[mesh.faces[face_idx]].mean(axis=0)
        return tuple(int(c) for c in color)
    return (200, 200, 200, 255)


def render_orthographic_image(
    scene: trimesh.Scene,
    view: tuple[int, int, int],
    resolution: int,
    ortho_scale: float,
) -> bytes:
    mesh = _combined_mesh(scene)
    rotation = _view_rotation(view)
    verts_view = np.asarray(mesh.vertices) @ rotation
    xy = verts_view[:, :2]
    xy_center = (xy.min(axis=0) + xy.max(axis=0)) / 2.0
    span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-8)
    scale = (resolution * ortho_scale) / span

    pixels = np.empty_like(xy)
    pixels[:, 0] = (xy[:, 0] - xy_center[0]) * scale + resolution / 2.0
    pixels[:, 1] = -(xy[:, 1] - xy_center[1]) * scale + resolution / 2.0

    image = Image.new("RGBA", (resolution, resolution), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    face_order = np.argsort(verts_view[mesh.faces, 2].mean(axis=1))
    for face_idx in face_order:
        polygon = [tuple(pixels[v]) for v in mesh.faces[face_idx]]
        draw.polygon(polygon, fill=_face_color(mesh, int(face_idx)))

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def render_views(
    scene: trimesh.Scene,
    output_folder: Path,
    obj_prefix: str,
    views: list[tuple[int, int, int]],
    resolution: int = 800,
    focal: float = 1000.0,
    distance: float = 2.0,
    projection: str = "orthographic",
    ortho_scale: float = 0.85,
    overwrite: bool = False,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    camera = Camera(resolution=(resolution, resolution), focal=(focal, focal))

    for angle_x, angle_y, angle_z in views:
        image_path = output_folder / f"{obj_prefix}_x{angle_x}_y{angle_y}_z{angle_z}.png"
        if image_path.exists() and not overwrite:
            continue

        if projection == "orthographic":
            image_bytes = render_orthographic_image(
                scene,
                (angle_x, angle_y, angle_z),
                resolution,
                ortho_scale,
            )
        else:
            scene.camera = camera
            scene.camera_transform = camera_pose_for_view(angle_x, angle_y, angle_z, distance)
            image_bytes = scene.save_image(resolution=(resolution, resolution), visible=True)
            if image_bytes is None:
                raise RuntimeError("Renderer returned no image bytes.")
        image_path.write_bytes(image_bytes)
        print(f"saved {image_path}")


def process_model_xml(
    xml_file: Path,
    output_root: Path,
    views: list[tuple[int, int, int]],
    overwrite: bool = False,
    resolution: int = 800,
    focal: float = 1000.0,
    distance: float = 2.0,
    projection: str = "orthographic",
    ortho_scale: float = 0.85,
) -> None:
    obj_dir = xml_file.parent
    obj_prefix = obj_dir.name
    category = obj_prefix.rsplit("_", 1)[0]
    output_folder = output_root / category / obj_prefix
    if output_folder.exists() and any(output_folder.iterdir()) and not overwrite:
        return

    scene = build_scene_from_mujoco(xml_file, obj_dir / "visual", obj_dir)
    render_views(
        scene,
        output_folder,
        obj_prefix,
        views,
        resolution=resolution,
        focal=focal,
        distance=distance,
        projection=projection,
        ortho_scale=ortho_scale,
        overwrite=overwrite,
    )


def iter_model_xml(source_root: Path, categories: list[str] | None) -> list[Path]:
    roots = [source_root / category for category in categories] if categories else [source_root]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(sorted(root.rglob("model.xml")))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Render RoboKeyBench multi-view images.")
    parser.add_argument("--source-root", type=Path, required=True, help="Root containing MuJoCo object folders.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for rendered images.")
    parser.add_argument("--categories", nargs="*", help="Optional category names to process.")
    parser.add_argument("--views", default="annotation8", help="View set name or custom 'x,y,z;x,y,z' list.")
    parser.add_argument("--resolution", type=int, default=800)
    parser.add_argument("--focal", type=float, default=1000.0)
    parser.add_argument("--distance", type=float, default=2.0)
    parser.add_argument("--projection", choices=("orthographic", "perspective"), default="orthographic")
    parser.add_argument("--ortho-scale", type=float, default=0.85)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    require_render_deps()

    views = parse_views(args.views)
    xml_files = iter_model_xml(args.source_root, args.categories)
    if not xml_files:
        raise FileNotFoundError("No model.xml files found.")

    for xml_file in xml_files:
        try:
            process_model_xml(
                xml_file,
                args.output_root,
                views,
                overwrite=args.overwrite,
                resolution=args.resolution,
                focal=args.focal,
                distance=args.distance,
                projection=args.projection,
                ortho_scale=args.ortho_scale,
            )
        except Exception as exc:
            print(f"failed {xml_file}: {exc}")


if __name__ == "__main__":
    main()
