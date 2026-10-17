import json
import math
import os
import sys
from pathlib import Path

try:
    import bpy
except ImportError as exc:
    raise SystemExit(
        "This script must be run inside Blender: blender --background --python "
        "scripts/blender/render_task1_foreground_alpha_from_2dgs_path.py"
    ) from exc

from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_task1_fusion_scene as prep


PROJECT_ROOT = Path("/home/dechao/cv_final_pj")
SPEC_PATH = PROJECT_ROOT / "configs/blender_fusion/scene_preparation_task1.json"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


DEFAULT_COMPOSITING_DIR = (
    PROJECT_ROOT
    / "outputs/renders/blender_fusion/task1_render_level_compositing"
)
COMPOSITING_DIR = project_path(os.environ.get("TASK1_COMPOSITING_DIR", DEFAULT_COMPOSITING_DIR))
CAMERA_PATH_JSON = project_path(
    os.environ.get(
        "TASK1_CAMERA_PATH_JSON",
        DEFAULT_COMPOSITING_DIR / "metadata/background_counter_render_path_cameras.json",
    )
)
FRAME_MODE = os.environ.get("TASK1_FRAME_MODE", "keyframes")
if FRAME_MODE not in {"keyframes", "all"}:
    raise SystemExit("TASK1_FRAME_MODE must be 'keyframes' or 'all'")

OUTPUT_DIR = COMPOSITING_DIR / "foreground_alpha" / (
    "frames" if FRAME_MODE == "all" else "keyframes"
)
BLEND_PATH = COMPOSITING_DIR / "foreground_alpha/foreground_alpha.blend"
ACTIVE_ALIGNMENT_JSON = COMPOSITING_DIR / "metadata/foreground_alignment_active.json"
SURFACE_PLACEMENT_JSON = project_path(
    os.environ.get(
        "TASK1_SURFACE_PLACEMENT_JSON",
        COMPOSITING_DIR / "metadata/foreground_surface_placement.json",
    )
)
ALLOW_OVERWRITE = os.environ.get("TASK1_ALLOW_OVERWRITE", "0") == "1"


def set_camera_from_2dgs_record(camera_obj, record) -> None:
    c2w = Matrix(record["camera_to_world"])
    opencv_to_blender = Matrix(
        (
            (1, 0, 0, 0),
            (0, -1, 0, 0),
            (0, 0, -1, 0),
            (0, 0, 0, 1),
        )
    )
    camera_obj.matrix_world = c2w @ opencv_to_blender


def add_foreground_camera(records):
    first = records[0]
    data = bpy.data.cameras.new("Camera_2DGS_render_path_foreground_alpha")
    data.angle_x = first["FoVx"]
    data.angle_y = first["FoVy"]
    camera = bpy.data.objects.new("Camera_2DGS_render_path_foreground_alpha", data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def configure_alpha_render(record):
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 240
    scene.render.resolution_x = record["image_width"]
    scene.render.resolution_y = record["image_height"]
    scene.render.fps = 60
    scene.render.film_transparent = True
    scene.render.engine = (
        "BLENDER_EEVEE_NEXT"
        if "BLENDER_EEVEE_NEXT"
        in {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
        else "BLENDER_EEVEE"
    )
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1


def surface_rotation_euler(entry):
    normal = Vector(entry["surface_normal"]).normalized()
    tangent_x = Vector(entry["surface_tangent_x"]).normalized()
    tangent_x -= normal * tangent_x.dot(normal)
    tangent_x.normalize()
    tangent_y = normal.cross(tangent_x)
    tangent_y.normalize()

    yaw = entry.get("surface_yaw", 0.0)
    if yaw:
        yaw_matrix = Matrix.Rotation(yaw, 3, normal)
        tangent_x = yaw_matrix @ tangent_x
        tangent_y = yaw_matrix @ tangent_y

    basis = Matrix(
        (
            (tangent_x.x, tangent_y.x, normal.x),
            (tangent_x.y, tangent_y.y, normal.y),
            (tangent_x.z, tangent_y.z, normal.z),
        )
    )
    return list(basis.to_euler("XYZ"))


def upright_book_rotation_euler(entry):
    normal = Vector(entry["surface_normal"]).normalized()
    to_camera = Vector(entry["camera_center"]) - Vector(entry["location"])
    cover_normal = to_camera - normal * to_camera.dot(normal)
    if cover_normal.length < 1e-6:
        cover_normal = Vector(entry["surface_tangent_y"]).normalized()
    else:
        cover_normal.normalize()

    yaw = entry.get("surface_yaw", 0.0)
    if yaw:
        yaw_matrix = Matrix.Rotation(yaw, 3, normal)
        cover_normal = yaw_matrix @ cover_normal

    local_y = normal
    local_z = cover_normal
    local_x = local_y.cross(local_z)
    local_x.normalize()
    basis = Matrix(
        (
            (local_x.x, local_y.x, local_z.x),
            (local_x.y, local_y.y, local_z.y),
            (local_x.z, local_y.z, local_z.z),
        )
    )
    return list(basis.to_euler("XYZ"))


def estimate_focus(records):
    def outer(direction):
        return Matrix(
            (
                (direction.x * direction.x, direction.x * direction.y, direction.x * direction.z),
                (direction.y * direction.x, direction.y * direction.y, direction.y * direction.z),
                (direction.z * direction.x, direction.z * direction.y, direction.z * direction.z),
            )
        )

    centers = []
    dirs = []
    for record in records:
        c2w = Matrix(record["camera_to_world"])
        centers.append(c2w.translation)
        dirs.append(c2w.to_3x3() @ Vector((0, 0, 1)))

    a = Matrix(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    b = Vector((0.0, 0.0, 0.0))
    for center, direction in zip(centers, dirs):
        direction.normalize()
        m = Matrix.Identity(3) - outer(direction)
        a += m
        b += m @ center
    return a.inverted() @ b


def apply_smoke_alignment(spec, focus):
    ground_z = focus.z - 0.78
    spec["object_A"] = dict(spec["object_A"])
    spec["object_B_anchor"] = dict(spec["object_B_anchor"])
    spec["object_C_anchor"] = dict(spec["object_C_anchor"])

    spec["object_A"]["location"] = [focus.x - 0.45, focus.y - 0.05, ground_z]
    spec["object_A"]["rotation_euler"] = [0.0, 0.0, 0.18]
    spec["object_A"]["visible_proxy_dimensions"] = [0.58, 0.98, 0.14]
    spec["object_A"]["ground_z"] = ground_z

    spec["object_B_anchor"]["location"] = [focus.x + 0.46, focus.y - 0.18, ground_z + 0.25]
    spec["object_B_anchor"]["scale"] = [0.46, 0.46, 0.66]
    spec["object_B_anchor"]["ground_z"] = ground_z

    spec["object_C_anchor"]["location"] = [focus.x + 0.12, focus.y + 0.42, ground_z + 0.24]
    spec["object_C_anchor"]["scale"] = [0.32, 0.30, 0.28]
    spec["object_C_anchor"]["ground_z"] = ground_z
    return spec


def apply_surface_placement(spec, placement):
    placements = placement["placements"]
    spec["object_A"] = dict(spec["object_A"])
    spec["object_B_anchor"] = dict(spec["object_B_anchor"])
    spec["object_C_anchor"] = dict(spec["object_C_anchor"])

    object_a = placements["object_A"]
    spec["object_A"]["location"] = object_a["location"]
    spec["object_A"]["surface_normal"] = object_a["surface_normal"]
    spec["object_A"]["orientation_mode"] = object_a.get("orientation_mode", "upright_book")
    if spec["object_A"]["orientation_mode"] in {"upright_book", "upright_open_book"}:
        spec["object_A"]["rotation_euler"] = upright_book_rotation_euler(object_a)
    else:
        spec["object_A"]["rotation_euler"] = surface_rotation_euler(object_a)
    spec["object_A"]["visible_proxy_dimensions"] = object_a["visible_proxy_dimensions"]
    for mesh_key in ["asset_path", "asset_render_mode", "mesh_fit_dimensions", "mesh_fit_contact_axis"]:
        if mesh_key in object_a:
            spec["object_A"][mesh_key] = object_a[mesh_key]
    for texture_key in [
        "front_texture_image",
        "back_texture_image",
        "pages_texture_image",
        "inner_page_texture_image",
    ]:
        if texture_key in object_a:
            spec["object_A"][texture_key] = object_a[texture_key]
    for retopo_key in [
        "retopo_spine_width",
        "retopo_cover_thickness",
        "retopo_page_margin_x",
        "retopo_page_margin_y",
        "retopo_page_line_count",
        "retopo_open_angle_degrees",
        "retopo_page_cover_gap",
        "retopo_page_block_width",
        "retopo_page_block_depth",
        "retopo_cover_gap_x",
    ]:
        if retopo_key in object_a:
            spec["object_A"][retopo_key] = object_a[retopo_key]
    if "open_angle_degrees" in object_a:
        spec["object_A"]["open_angle_degrees"] = object_a["open_angle_degrees"]

    object_b = placements["object_B_anchor"]
    spec["object_B_anchor"]["location"] = object_b["location"]
    spec["object_B_anchor"]["surface_normal"] = object_b["surface_normal"]
    spec["object_B_anchor"]["rotation_euler"] = surface_rotation_euler(object_b)
    spec["object_B_anchor"]["scale"] = object_b["scale"]
    for asset_key in ["preferred_asset", "proxy_shape", "label", "final_decision"]:
        if asset_key in object_b:
            spec["object_B_anchor"][asset_key] = object_b[asset_key]

    object_c = placements["object_C_anchor"]
    spec["object_C_anchor"]["location"] = object_c["location"]
    spec["object_C_anchor"]["surface_normal"] = object_c["surface_normal"]
    spec["object_C_anchor"]["rotation_euler"] = surface_rotation_euler(object_c)
    spec["object_C_anchor"]["scale"] = object_c["scale"]
    for asset_key in ["preferred_asset", "proxy_shape", "label", "final_decision", "current_main"]:
        if asset_key in object_c:
            spec["object_C_anchor"][asset_key] = object_c[asset_key]
    return spec


def make_contact_shadow_material(alpha: float):
    mat = bpy.data.materials.new("mat_task1_contact_shadow")
    mat.diffuse_color = (0.0, 0.0, 0.0, alpha)
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    mat.show_transparent_back = True
    nodes = mat.node_tree.nodes
    shader = nodes.get("Principled BSDF")
    if shader is not None:
        shader.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, alpha)
        shader.inputs["Alpha"].default_value = alpha
        shader.inputs["Roughness"].default_value = 1.0
    return mat


def add_contact_shadow(name: str, entry, material) -> None:
    shadow = entry.get("contact_shadow")
    if not shadow or not shadow.get("enabled", True):
        return

    normal = Vector(entry["surface_normal"]).normalized()
    tangent_x = Vector(entry["surface_tangent_x"]).normalized()
    tangent_y = Vector(entry["surface_tangent_y"]).normalized()
    center = Vector(entry.get("location_original_surface", entry["location"]))
    center += normal * shadow.get("normal_lift", 0.004)

    radius_x = shadow.get("radius_x", 0.12)
    radius_y = shadow.get("radius_y", 0.08)
    segments = int(shadow.get("segments", 48))
    vertices = [tuple(center)]
    for index in range(segments):
        theta = 2.0 * 3.141592653589793 * index / segments
        point = center + tangent_x * (radius_x * math.cos(theta)) + tangent_y * (radius_y * math.sin(theta))
        vertices.append(tuple(point))
    faces = [
        tuple([0, index, 1 + (index % segments)])
        for index in range(1, segments + 1)
    ]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)


def add_contact_shadows(placement) -> None:
    if placement is None:
        return
    material = make_contact_shadow_material(0.24)
    for object_key, entry in placement["placements"].items():
        add_contact_shadow(f"{object_key}_contact_shadow", entry, material)


def write_active_alignment_metadata(status, spec, *, focus=None, placement=None):
    ACTIVE_ALIGNMENT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "foreground_output_dir": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)),
        "foreground_blend": str(BLEND_PATH.relative_to(PROJECT_ROOT)),
        "object_A_location": spec["object_A"]["location"],
        "object_B_anchor_location": spec["object_B_anchor"]["location"],
        "object_C_anchor_location": spec["object_C_anchor"]["location"],
        "object_A_asset_path": spec["object_A"].get("asset_path"),
        "object_B_preferred_asset": spec["object_B_anchor"].get("preferred_asset"),
        "object_C_preferred_asset": spec["object_C_anchor"].get("preferred_asset"),
        "note": "Active foreground placement for render-level compositing preview only; this is not a final fusion render.",
    }
    if "ground_z" in spec["object_A"] and "surface_normal" not in spec["object_A"]:
        payload["object_A_ground_z"] = spec["object_A"]["ground_z"]
    if "ground_z" in spec["object_B_anchor"] and "surface_normal" not in spec["object_B_anchor"]:
        payload["object_B_ground_z"] = spec["object_B_anchor"]["ground_z"]
    if "ground_z" in spec["object_C_anchor"] and "surface_normal" not in spec["object_C_anchor"]:
        payload["object_C_ground_z"] = spec["object_C_anchor"]["ground_z"]
    if "surface_normal" in spec["object_A"]:
        payload["surface_normal"] = spec["object_A"]["surface_normal"]
    if focus is not None:
        payload["estimated_focus"] = [focus.x, focus.y, focus.z]
    if placement is not None:
        payload["surface_placement_json"] = str(SURFACE_PLACEMENT_JSON)
        payload["anchor_frame"] = placement["anchor_frame"]
        payload["placement_status"] = placement["status"]
        payload["object_A_orientation_mode"] = spec["object_A"].get("orientation_mode")
    ACTIVE_ALIGNMENT_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
        )
    )


def build_foreground_scene(records):
    spec = json.loads(SPEC_PATH.read_text())
    if SURFACE_PLACEMENT_JSON.exists():
        placement = json.loads(SURFACE_PLACEMENT_JSON.read_text())
        spec = apply_surface_placement(spec, placement)
        write_active_alignment_metadata(
            "surface_depth_alignment_preview_not_final",
            spec,
            placement=placement,
        )
    else:
        focus = estimate_focus(records)
        spec = apply_smoke_alignment(spec, focus)
        write_active_alignment_metadata(
            "smoke_alignment_only_not_final",
            spec,
            focus=focus,
        )

    prep.clear_scene()

    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.color = (0, 0, 0)

    mat_book_pages = prep.make_material("mat_object_A_book_pages", (0.86, 0.78, 0.62, 1.0))
    mat_book_spine = prep.make_material("mat_object_A_book_red_spine", (0.72, 0.08, 0.12, 1.0))
    mat_book_cover = prep.make_image_material(
        "mat_object_A_book_cover_texture",
        spec["object_A"]["cover_texture_image"],
    )
    mat_b = prep.make_material("mat_object_B_traffic_cone_orange", (1.0, 0.35, 0.06, 1.0))
    mat_c = prep.make_material("mat_object_C_cat_box_green", (0.55, 0.75, 0.55, 1.0))

    prep.import_object_a(spec["object_A"], mat_book_pages, mat_book_spine, mat_book_cover)
    prep.import_mesh_or_proxy(spec["object_B_anchor"], mat_b)
    prep.import_mesh_or_proxy(spec["object_C_anchor"], mat_c)
    add_contact_shadows(placement if SURFACE_PLACEMENT_JSON.exists() else None)
    prep.add_lights(spec["lights"])


def fail_if_outputs_exist(frames) -> None:
    if ALLOW_OVERWRITE:
        return

    existing = []
    for frame_index in frames:
        output = OUTPUT_DIR / f"foreground_{frame_index:05d}.png"
        if output.exists():
            existing.append(output)
    for output in [BLEND_PATH, ACTIVE_ALIGNMENT_JSON]:
        if output.exists():
            existing.append(output)

    if existing:
        paths = "\n".join(f"  - {path}" for path in existing[:20])
        raise SystemExit(
            "Refusing to overwrite existing foreground render outputs. "
            "Use a fresh TASK1_COMPOSITING_DIR for a new run.\n"
            f"{paths}"
        )


def main() -> None:
    camera_path = json.loads(CAMERA_PATH_JSON.read_text())
    records = camera_path["cameras"]
    frames = list(range(len(records))) if FRAME_MODE == "all" else [0, 40, 80, 120, 160, 200, 239]
    fail_if_outputs_exist(frames)

    build_foreground_scene(records)
    camera = add_foreground_camera(records)
    configure_alpha_render(records[0])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BLEND_PATH.parent.mkdir(parents=True, exist_ok=True)

    for frame_index in frames:
        record = records[frame_index]
        set_camera_from_2dgs_record(camera, record)
        bpy.context.scene.frame_set(frame_index + 1)
        bpy.context.scene.render.filepath = str(OUTPUT_DIR / f"foreground_{frame_index:05d}.png")
        bpy.ops.render.render(write_still=True)

    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    print(f"Wrote foreground alpha keyframes to {OUTPUT_DIR}")
    print(f"Wrote {BLEND_PATH}")


if __name__ == "__main__":
    main()
