import argparse
import json
import math
import sys
from pathlib import Path

try:
    import bpy
except ImportError as exc:
    raise SystemExit(
        "Run inside Blender: blender --background --python "
        "scripts/blender/render_object_a_mesh_diagnostic_views.py -- ..."
    ) from exc

from mathutils import Vector
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_task1_fusion_scene as prep


PROJECT_ROOT = Path("/home/dechao/cv_final_pj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Object A mesh-only diagnostic orbit views.")
    parser.add_argument("--placement-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="object_A_mesh")
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=900)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_object_a_spec(placement_json: Path) -> dict:
    placement = json.loads(project_path(placement_json).read_text())
    object_a = dict(placement["placements"]["object_A"])
    object_a["location"] = [0.0, 0.0, 0.0]
    object_a["rotation_euler"] = [0.0, 0.0, 0.0]
    object_a["scale"] = [1.0, 1.0, 1.0]
    return object_a


def add_lights() -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.color = (0.86, 0.86, 0.86)

    bpy.ops.object.light_add(type="AREA", location=(-2.5, -3.0, 4.0))
    key = bpy.context.object
    key.name = "Object_A_Mesh_Diagnostic_Key_Light"
    key.data.energy = 520
    key.data.size = 4.0

    bpy.ops.object.light_add(type="AREA", location=(3.0, 2.0, 2.8))
    fill = bpy.context.object
    fill.name = "Object_A_Mesh_Diagnostic_Fill_Light"
    fill.data.energy = 130
    fill.data.size = 5.0


def scene_mesh_bbox() -> tuple[Vector, Vector]:
    bpy.context.view_layer.update()
    corners = [
        obj.matrix_world @ Vector(corner)
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
        for corner in obj.bound_box
    ]
    if not corners:
        raise RuntimeError("no mesh objects in diagnostic scene")
    bbox_min = Vector((min(p.x for p in corners), min(p.y for p in corners), min(p.z for p in corners)))
    bbox_max = Vector((max(p.x for p in corners), max(p.y for p in corners), max(p.z for p in corners)))
    return bbox_min, bbox_max


def add_camera(resolution: int, ortho_scale: float):
    data = bpy.data.cameras.new("Object_A_Mesh_Diagnostic_Camera")
    data.type = "ORTHO"
    data.ortho_scale = ortho_scale
    camera = bpy.data.objects.new("Object_A_Mesh_Diagnostic_Camera", data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.engine = (
        "BLENDER_EEVEE_NEXT"
        if "BLENDER_EEVEE_NEXT"
        in {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
        else "BLENDER_EEVEE"
    )
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    return camera


def look_at(camera, target: Vector) -> None:
    forward = (target - camera.location).normalized()
    world_up = Vector((0.0, 1.0, 0.0))
    right = forward.cross(world_up)
    if right.length < 1e-6:
        world_up = Vector((0.0, 0.0, 1.0))
        right = forward.cross(world_up)
    right.normalize()
    up = right.cross(forward).normalized()
    rotation = Matrix(
        (
            (right.x, up.x, -forward.x),
            (right.y, up.y, -forward.y),
            (right.z, up.z, -forward.z),
        )
    )
    camera.rotation_euler = rotation.to_euler("XYZ")


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prep.clear_scene()
    spec = load_object_a_spec(args.placement_json)

    page_material = prep.make_material("mat_object_A_diag_pages_fallback", (0.86, 0.78, 0.62, 1.0))
    spine_material = prep.make_material("mat_object_A_diag_spine_fallback", (0.72, 0.08, 0.12, 1.0))
    cover_material = prep.make_material("mat_object_A_diag_cover_fallback", (0.75, 0.10, 0.14, 1.0))
    prep.import_object_a(spec, page_material, spine_material, cover_material)
    add_lights()

    bbox_min, bbox_max = scene_mesh_bbox()
    center = (bbox_min + bbox_max) * 0.5
    extent = bbox_max - bbox_min
    radius = max(extent.x, extent.z, extent.y) * 3.0
    radius = max(radius, 2.0)
    ortho_scale = max(extent.y * 1.18, extent.x * 1.55, extent.z * 2.0, 1.0)
    camera = add_camera(args.resolution, ortho_scale)

    target = Vector((center.x, center.y, center.z))
    for index in range(args.views):
        theta = 2.0 * math.pi * index / args.views
        camera.location = (
            center.x + radius * math.sin(theta),
            center.y + extent.y * 0.06,
            center.z + radius * math.cos(theta),
        )
        look_at(camera, target)
        bpy.context.scene.render.filepath = str(output_dir / f"{args.prefix}_view_{index:03d}.png")
        bpy.ops.render.render(write_still=True)

    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / f"{args.prefix}_diagnostic.blend"))
    print(f"Wrote {args.views} Object A mesh diagnostic views to {output_dir}")


if __name__ == "__main__":
    main()
