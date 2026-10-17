import argparse
import math
import sys
from pathlib import Path

try:
    import bpy
except ImportError as exc:
    raise SystemExit(
        "Run inside Blender: blender --background --python scripts/blender/render_mesh_probe.py -- ..."
    ) from exc

from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render quick orbit probes for a colored mesh.")
    parser.add_argument("--input_mesh", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", default="mesh_probe")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--views", type=int, default=6)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_vertex_color_material(obj):
    mat = bpy.data.materials.new("vertex_color_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.8

    color_attrs = list(obj.data.color_attributes)
    if color_attrs:
        attr = nodes.new(type="ShaderNodeAttribute")
        attr.attribute_name = color_attrs[0].name
        mat.node_tree.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
        print(f"Using vertex color attribute: {color_attrs[0].name}")
    else:
        bsdf.inputs["Base Color"].default_value = (0.75, 0.12, 0.12, 1.0)
        print("No vertex color attribute found; using fallback red material")

    mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    obj.data.materials.append(mat)


def import_mesh(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise ValueError(f"Unsupported mesh suffix: {path.suffix}")
    obj = bpy.context.object
    obj.name = path.stem
    if suffix == ".ply" or not obj.data.materials:
        make_vertex_color_material(obj)
    return obj


def normalize_object(obj) -> float:
    bpy.context.view_layer.update()
    world_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    bbox_min = Vector((min(p.x for p in world_corners), min(p.y for p in world_corners), min(p.z for p in world_corners)))
    bbox_max = Vector((max(p.x for p in world_corners), max(p.y for p in world_corners), max(p.z for p in world_corners)))
    center = (bbox_min + bbox_max) * 0.5
    extent = bbox_max - bbox_min
    max_extent = max(extent.x, extent.y, extent.z)
    scale = 2.4 / max_extent if max_extent > 0 else 1.0
    obj.location -= center
    obj.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    return max_extent


def add_lights() -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.color = (0.78, 0.78, 0.78)

    bpy.ops.object.light_add(type="AREA", location=(-2.5, -3.0, 4.0))
    key = bpy.context.object
    key.name = "Probe_Key_Area"
    key.data.energy = 500
    key.data.size = 4.0

    bpy.ops.object.light_add(type="AREA", location=(3.0, 2.5, 2.5))
    fill = bpy.context.object
    fill.name = "Probe_Fill_Area"
    fill.data.energy = 120
    fill.data.size = 5.0


def look_at(camera, target: Vector) -> None:
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera(resolution: int):
    data = bpy.data.cameras.new("Probe_Camera")
    data.type = "ORTHO"
    data.ortho_scale = 3.0
    camera = bpy.data.objects.new("Probe_Camera", data)
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


def main() -> None:
    args = parse_args()
    input_mesh = Path(args.input_mesh)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clear_scene()
    obj = import_mesh(input_mesh)
    original_extent = normalize_object(obj)
    add_lights()
    camera = add_camera(args.resolution)

    radius = 4.5
    for index in range(args.views):
        theta = 2.0 * math.pi * index / args.views
        camera.location = (radius * math.cos(theta), radius * math.sin(theta), 1.8)
        look_at(camera, Vector((0, 0, 0)))
        bpy.context.scene.render.filepath = str(output_dir / f"{args.prefix}_view_{index:02d}.png")
        bpy.ops.render.render(write_still=True)

    print(f"Rendered {args.views} views for {input_mesh}")
    print(f"Original max extent: {original_extent}")


if __name__ == "__main__":
    main()
