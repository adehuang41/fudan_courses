import json
import math
import sys
from pathlib import Path

try:
    import bpy
except ImportError as exc:
    raise SystemExit(
        "This script must be run inside Blender: blender --background --python "
        "scripts/blender/prepare_task1_fusion_scene.py"
    ) from exc

from mathutils import Euler, Matrix, Vector


PROJECT_ROOT = Path("/home/dechao/cv_final_pj")
SPEC_PATH = PROJECT_ROOT / "configs/blender_fusion/scene_preparation_task1.json"


def abs_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_material(name: str, color: tuple[float, float, float, float]):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def make_image_material(name: str, image_path: str, *, emission: bool = False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    image = nodes.new(type="ShaderNodeTexImage")
    image.image = bpy.data.images.load(str(abs_path(image_path)))
    image.extension = "EXTEND"

    if emission:
        shader = nodes.new(type="ShaderNodeEmission")
        shader.inputs["Strength"].default_value = 1.0
        mat.node_tree.links.new(image.outputs["Color"], shader.inputs["Color"])
    else:
        shader = nodes.new(type="ShaderNodeBsdfPrincipled")
        shader.inputs["Roughness"].default_value = 0.72
        mat.node_tree.links.new(image.outputs["Color"], shader.inputs["Base Color"])

    mat.node_tree.links.new(shader.outputs[0], output.inputs["Surface"])
    return mat


def make_vertex_color_material(name: str, mesh):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    attr = nodes.new(type="ShaderNodeAttribute")
    shader = nodes.new(type="ShaderNodeBsdfPrincipled")
    shader.inputs["Roughness"].default_value = 0.78

    color_attrs = list(mesh.color_attributes)
    attr.attribute_name = color_attrs[0].name if color_attrs else "Col"
    mat.node_tree.links.new(attr.outputs["Color"], shader.inputs["Base Color"])
    mat.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return mat


def create_empty(name: str, location, rotation, scale):
    empty = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(empty)
    empty.empty_display_type = "ARROWS"
    empty.empty_display_size = 0.35
    empty.location = location
    empty.rotation_euler = rotation
    empty.scale = scale
    return empty


def mesh_descendants(parent):
    return [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.parent == parent
    ]


def align_parent_bottom_to_z(parent, ground_z: float) -> None:
    bpy.context.view_layer.update()
    meshes = mesh_descendants(parent)
    if not meshes:
        return

    min_z = min(
        (obj.matrix_world @ Vector(corner)).z
        for obj in meshes
        for corner in obj.bound_box
    )
    parent.location.z += ground_z - min_z
    bpy.context.view_layer.update()


def align_parent_bottom_to_surface(parent, plane_point, surface_normal) -> None:
    bpy.context.view_layer.update()
    meshes = mesh_descendants(parent)
    if not meshes:
        return

    plane_point = Vector(plane_point)
    surface_normal = Vector(surface_normal).normalized()
    min_distance = min(
        ((obj.matrix_world @ Vector(corner)) - plane_point).dot(surface_normal)
        for obj in meshes
        for corner in obj.bound_box
    )
    parent.location += surface_normal * (-min_distance)
    bpy.context.view_layer.update()


def import_mesh_or_proxy(anchor_spec, material):
    preferred = abs_path(anchor_spec["preferred_asset"])
    parent = create_empty(
        anchor_spec["label"],
        anchor_spec["location"],
        anchor_spec["rotation_euler"],
        anchor_spec["scale"],
    )

    if preferred.exists():
        suffix = preferred.suffix.lower()
        if suffix == ".obj":
            bpy.ops.wm.obj_import(filepath=str(preferred))
        elif suffix in {".glb", ".gltf"}:
            bpy.ops.import_scene.gltf(filepath=str(preferred))
        elif suffix == ".ply":
            bpy.ops.wm.ply_import(filepath=str(preferred))
        else:
            raise ValueError(f"Unsupported asset type: {preferred}")
        imported = list(bpy.context.selected_objects)
        for obj in imported:
            obj.parent = parent
    else:
        shape = anchor_spec.get("proxy_shape", "")
        if shape == "cone":
            bpy.ops.mesh.primitive_cone_add(vertices=64, radius1=0.45, radius2=0.08, depth=1.0)
            obj = bpy.context.object
            obj.name = "Object_B_proxy_traffic_cone"
            obj.data.materials.append(material)
            obj.parent = parent
            bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.52, depth=0.08, location=(0, 0, -0.54))
            base = bpy.context.object
            base.name = "Object_B_proxy_base"
            base.data.materials.append(material)
            base.parent = parent
        elif shape == "rounded_box_with_ears":
            bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24)
            body = bpy.context.object
            body.name = "Object_C_proxy_cat_box_body"
            body.scale = (0.75, 0.55, 0.55)
            body.data.materials.append(material)
            body.parent = parent
            for x in (-0.32, 0.32):
                bpy.ops.mesh.primitive_cone_add(vertices=3, radius1=0.16, depth=0.32, location=(x, 0, 0.58), rotation=(0, 0, math.radians(30)))
                ear = bpy.context.object
                ear.name = "Object_C_proxy_ear"
                ear.data.materials.append(material)
                ear.parent = parent
        else:
            bpy.ops.mesh.primitive_cube_add(size=1.0)
            obj = bpy.context.object
            obj.name = f"{anchor_spec['label']}_proxy_cube"
            obj.data.materials.append(material)
            obj.parent = parent
    if "surface_normal" in anchor_spec:
        align_parent_bottom_to_surface(
            parent,
            anchor_spec["location"],
            anchor_spec["surface_normal"],
        )
    else:
        align_parent_bottom_to_z(parent, anchor_spec.get("ground_z", 0.0))
    return parent


def create_textured_quad(name, center, axis_u, axis_v, width, height, material):
    center = Vector(center)
    axis_u = Vector(axis_u).normalized()
    axis_v = Vector(axis_v).normalized()
    half_u = axis_u * (width / 2)
    half_v = axis_v * (height / 2)
    vertices = [
        center - half_u - half_v,
        center + half_u - half_v,
        center + half_u + half_v,
        center - half_u + half_v,
    ]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(vertex) for vertex in vertices], [], [(0, 1, 2, 3)])
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for loop, uv in zip(uv_layer.data, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        loop.uv = uv
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def rotation_from_axes(axis_x, axis_y, axis_z):
    return Matrix(
        (
            (axis_x.x, axis_y.x, axis_z.x),
            (axis_x.y, axis_y.y, axis_z.y),
            (axis_x.z, axis_y.z, axis_z.z),
        )
    ).to_euler("XYZ")


def create_object_a_open_book_proxy(spec, page_material, spine_material, front_material):
    width, height, thickness = spec.get("visible_proxy_dimensions", [0.34, 0.58, 0.08])
    base = Vector(spec["location"])
    local_axes = Euler(tuple(spec["rotation_euler"]), "XYZ").to_matrix()
    local_x = (local_axes @ Vector((1.0, 0.0, 0.0))).normalized()
    local_y = (local_axes @ Vector((0.0, 1.0, 0.0))).normalized()
    local_z = (local_axes @ Vector((0.0, 0.0, 1.0))).normalized()
    hinge = base - local_x * (width / 2)
    front_center = hinge + local_x * (width / 2) + local_y * (height / 2) + local_z * 0.004

    open_angle = math.radians(spec.get("open_angle_degrees", 10.0))
    open_matrix = Matrix.Rotation(open_angle, 3, local_y)
    back_x = (open_matrix @ local_x).normalized()
    back_z = (open_matrix @ local_z).normalized()
    back_center = hinge + back_x * (width / 2) + local_y * (height / 2) - back_z * 0.004

    back_material = (
        make_image_material("mat_object_A_book_back_texture", spec["back_texture_image"])
        if spec.get("back_texture_image")
        else front_material
    )
    pages_texture_material = (
        make_image_material("mat_object_A_book_pages_texture", spec["pages_texture_image"])
        if spec.get("pages_texture_image")
        else page_material
    )

    front = create_textured_quad(
        "Object_A_visible_proxy_open_book_front_cover",
        front_center,
        local_x,
        local_y,
        width,
        height,
        front_material,
    )
    back = create_textured_quad(
        "Object_A_visible_proxy_open_book_back_cover",
        back_center,
        back_x,
        local_y,
        width,
        height,
        back_material,
    )

    pages_rotation = rotation_from_axes(back_x, local_y, back_z)
    pages_center = hinge + back_x * (width * 0.16) + local_y * (height / 2) - back_z * (thickness * 0.18)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=pages_center, rotation=pages_rotation)
    pages = bpy.context.object
    pages.name = "Object_A_visible_proxy_open_book_pages_block"
    pages.scale = (width * 0.14, height * 0.96, thickness * 0.45)
    pages.data.materials.append(pages_texture_material)

    spine_center = hinge + local_y * (height / 2) - local_x * 0.012
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=spine_center, rotation=spec["rotation_euler"])
    spine = bpy.context.object
    spine.name = "Object_A_visible_proxy_open_book_spine"
    spine.scale = (0.035, height, thickness * 0.65)
    spine.data.materials.append(spine_material)

    return front


def create_object_a_proxy(spec, page_material, spine_material, cover_material):
    if spec.get("orientation_mode") == "upright_open_book":
        return create_object_a_open_book_proxy(spec, page_material, spine_material, cover_material)

    width, height, thickness = spec.get("visible_proxy_dimensions", [0.76, 1.28, 0.18])
    ground_z = spec.get("ground_z", 0.0)
    rotation = Euler(tuple(spec["rotation_euler"]), "XYZ")
    if spec.get("orientation_mode") == "upright_book":
        base = Vector(spec["location"])
        local_axes = rotation.to_matrix()
        local_x = local_axes @ Vector((1.0, 0.0, 0.0))
        local_y = local_axes @ Vector((0.0, 1.0, 0.0))
        local_z = local_axes @ Vector((0.0, 0.0, 1.0))
        center = base + local_y * (height / 2)
        spine_center = center - local_x * (width * 0.48) + local_z * 0.003
        cover_center = center + local_z * (thickness / 2 + 0.006)
    elif "surface_normal" in spec:
        base = Vector(spec["location"])
        local_axes = rotation.to_matrix()
        local_x = local_axes @ Vector((1.0, 0.0, 0.0))
        local_z = local_axes @ Vector((0.0, 0.0, 1.0))
        center = base + local_z * (thickness / 2)
        spine_center = base - local_x * (width * 0.48) + local_z * (thickness / 2 + 0.003)
        cover_center = base + local_z * (thickness + 0.006)
    else:
        center = Vector(
            (
                spec["location"][0],
                spec["location"][1],
                ground_z + thickness / 2,
            )
        )
        spine_center = Vector(
            (
                spec["location"][0] - width * 0.48,
                spec["location"][1],
                ground_z + thickness / 2 + 0.003,
            )
        )
        cover_center = Vector(
            (
                spec["location"][0],
                spec["location"][1],
                ground_z + thickness + 0.006,
            )
        )

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center, rotation=spec["rotation_euler"])
    pages = bpy.context.object
    pages.name = "Object_A_visible_proxy_book_pages"
    pages.scale = (width, height, thickness)
    pages.data.materials.append(page_material)

    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=spine_center,
        rotation=spec["rotation_euler"],
    )
    spine = bpy.context.object
    spine.name = "Object_A_visible_proxy_book_spine"
    spine.scale = (0.055, height, thickness + 0.012)
    spine.data.materials.append(spine_material)

    bpy.ops.mesh.primitive_plane_add(
        size=1.0,
        location=cover_center,
        rotation=spec["rotation_euler"],
    )
    cover = bpy.context.object
    cover.name = "Object_A_visible_proxy_book_cover_texture"
    cover.scale = (width, height, 1.0)
    cover.data.materials.append(cover_material)
    return pages


def create_object_a_texture_baked_solid_mesh(spec, page_material, spine_material, cover_material):
    width, height, thickness = spec.get("visible_proxy_dimensions", [0.58, 0.98, 0.22])

    front_material = (
        make_image_material("mat_object_A_book_front_baked_texture", spec["front_texture_image"])
        if spec.get("front_texture_image")
        else cover_material
    )
    back_material = (
        make_image_material("mat_object_A_book_back_baked_texture", spec["back_texture_image"])
        if spec.get("back_texture_image")
        else cover_material
    )
    pages_material = (
        make_image_material("mat_object_A_book_pages_baked_texture", spec["pages_texture_image"])
        if spec.get("pages_texture_image")
        else page_material
    )

    x0, x1 = -width / 2, width / 2
    y0, y1 = 0.0, height
    z0, z1 = -thickness / 2, thickness / 2
    vertices = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    faces = [
        (4, 5, 6, 7),  # front cover
        (1, 0, 3, 2),  # back cover
        (5, 1, 2, 6),  # page edge
        (0, 4, 7, 3),  # spine
        (3, 7, 6, 2),  # top edge
        (0, 1, 5, 4),  # bottom edge
    ]
    material_indices = [0, 1, 2, 3, 3, 2]

    mesh = bpy.data.meshes.new("Object_A_texture_baked_solid_book_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    mesh.materials.append(front_material)
    mesh.materials.append(back_material)
    mesh.materials.append(pages_material)
    mesh.materials.append(spine_material)

    uv_layer = mesh.uv_layers.new(name="UVMap")
    for polygon, material_index in zip(mesh.polygons, material_indices):
        polygon.material_index = material_index
        for loop_index, uv in zip(polygon.loop_indices, [(0, 0), (1, 0), (1, 1), (0, 1)]):
            uv_layer.data[loop_index].uv = uv

    obj = bpy.data.objects.new("Object_A_texture_baked_solid_book_mesh", mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = spec["location"]
    obj.rotation_euler = spec["rotation_euler"]
    obj.scale = spec.get("scale", [1.0, 1.0, 1.0])
    return obj


def create_local_box_object(
    name,
    spec,
    bounds,
    materials,
    material_indices,
    *,
    bevel_width=0.0,
    bevel_segments=2,
    rotate_y_degrees=0.0,
    rotate_y_pivot=None,
):
    x0, x1, y0, y1, z0, z1 = bounds
    vertices = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    if rotate_y_degrees:
        pivot_x, pivot_z = rotate_y_pivot if rotate_y_pivot is not None else (0.0, 0.0)
        angle = math.radians(rotate_y_degrees)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotated_vertices = []
        for x, y, z in vertices:
            dx = x - pivot_x
            dz = z - pivot_z
            rotated_vertices.append(
                (
                    pivot_x + dx * cos_a + dz * sin_a,
                    y,
                    pivot_z - dx * sin_a + dz * cos_a,
                )
            )
        vertices = rotated_vertices
    faces = [
        (4, 5, 6, 7),  # z+
        (1, 0, 3, 2),  # z-
        (5, 1, 2, 6),  # x+
        (0, 4, 7, 3),  # x-
        (3, 7, 6, 2),  # y+
        (0, 1, 5, 4),  # y-
    ]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    for material in materials:
        mesh.materials.append(material)
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for polygon, material_index in zip(mesh.polygons, material_indices):
        polygon.material_index = material_index
        for loop_index, uv in zip(polygon.loop_indices, [(0, 0), (1, 0), (1, 1), (0, 1)]):
            uv_layer.data[loop_index].uv = uv

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = spec["location"]
    obj.rotation_euler = spec["rotation_euler"]
    obj.scale = spec.get("scale", [1.0, 1.0, 1.0])
    if bevel_width > 0:
        bevel = obj.modifiers.new(f"{name}_soft_edges", "BEVEL")
        bevel.width = bevel_width
        bevel.segments = bevel_segments
        bevel.affect = "EDGES"
        normals = obj.modifiers.new(f"{name}_weighted_normals", "WEIGHTED_NORMAL")
        normals.keep_sharp = True
    return obj


def create_object_a_texture_baked_retopo_mesh(spec, page_material, spine_material, cover_material):
    width, height, thickness = spec.get("visible_proxy_dimensions", [0.50, 0.88, 0.20])
    front_material = (
        make_image_material("mat_object_A_book_front_retopo_texture", spec["front_texture_image"])
        if spec.get("front_texture_image")
        else cover_material
    )
    back_material = (
        make_image_material("mat_object_A_book_back_retopo_texture", spec["back_texture_image"])
        if spec.get("back_texture_image")
        else cover_material
    )
    pages_material = (
        make_image_material("mat_object_A_book_pages_retopo_texture", spec["pages_texture_image"])
        if spec.get("pages_texture_image")
        else page_material
    )
    inner_page_material = (
        make_image_material("mat_object_A_book_inner_page_retopo_texture", spec["inner_page_texture_image"])
        if spec.get("inner_page_texture_image")
        else page_material
    )
    cover_edge_material = make_material("mat_object_A_book_retopo_cover_edges", (0.62, 0.07, 0.10, 1.0))
    page_shadow_material = make_material("mat_object_A_book_retopo_page_lines", (0.36, 0.31, 0.25, 1.0))

    spine_width = spec.get("retopo_spine_width", width * 0.08)
    cover_t = spec.get("retopo_cover_thickness", thickness * 0.075)
    page_margin_x = spec.get("retopo_page_margin_x", width * 0.035)
    page_margin_y = spec.get("retopo_page_margin_y", height * 0.045)
    open_angle = spec.get("retopo_open_angle_degrees", 0.0)
    page_cover_gap = spec.get("retopo_page_cover_gap", thickness * 0.05)

    x_min = -width / 2
    x_max = width / 2
    hinge_x = x_min + spine_width * 0.55
    page_x0 = x_min + spine_width * 0.95
    page_x1 = x_max - page_margin_x
    page_y0 = page_margin_y
    page_y1 = height - page_margin_y
    front_z0 = thickness / 2 - cover_t
    front_z1 = thickness / 2
    back_z0 = -thickness / 2
    back_z1 = -thickness / 2 + cover_t
    page_z0 = back_z1 + page_cover_gap
    page_z1 = front_z0 - page_cover_gap
    bevel = min(width, thickness) * 0.025

    pages = create_local_box_object(
        "Object_A_retopo_pages_block",
        spec,
        (page_x0, page_x1, page_y0, page_y1, page_z0, page_z1),
        [pages_material, inner_page_material, page_material],
        [1, 1, 0, 2, 1, 1],
        bevel_width=bevel * 0.45,
        bevel_segments=2,
        rotate_y_degrees=-open_angle * 0.18,
        rotate_y_pivot=(hinge_x, 0.0),
    )
    create_local_box_object(
        "Object_A_retopo_front_cover",
        spec,
        (x_min, x_max, 0.0, height, front_z0, front_z1),
        [front_material, inner_page_material, cover_edge_material, spine_material],
        [0, 1, 2, 3, 2, 2],
        bevel_width=bevel,
        bevel_segments=3,
        rotate_y_degrees=open_angle * 0.55,
        rotate_y_pivot=(hinge_x, 0.0),
    )
    create_local_box_object(
        "Object_A_retopo_back_cover",
        spec,
        (x_min, x_max, 0.0, height, back_z0, back_z1),
        [back_material, inner_page_material, cover_edge_material, spine_material],
        [1, 0, 2, 3, 2, 2],
        bevel_width=bevel,
        bevel_segments=3,
        rotate_y_degrees=-open_angle * 0.45,
        rotate_y_pivot=(hinge_x, 0.0),
    )
    create_local_box_object(
        "Object_A_retopo_rounded_spine",
        spec,
        (x_min, x_min + spine_width, 0.0, height, -thickness / 2, thickness / 2),
        [spine_material, cover_edge_material],
        [1, 1, 1, 0, 1, 1],
        bevel_width=bevel * 1.8,
        bevel_segments=8,
    )

    line_count = int(spec.get("retopo_page_line_count", 8))
    if line_count > 0:
        line_depth = max(thickness * 0.006, 0.001)
        line_x0 = page_x1 - width * 0.006
        line_x1 = page_x1 + width * 0.006
        for index in range(1, line_count + 1):
            z = page_z0 + (page_z1 - page_z0) * index / (line_count + 1)
            create_local_box_object(
                f"Object_A_retopo_page_line_{index:02d}",
                spec,
                (line_x0, line_x1, page_y0, page_y1, z - line_depth / 2, z + line_depth / 2),
                [page_shadow_material],
                [0, 0, 0, 0, 0, 0],
            )
    return pages


def create_object_a_textured_open_book_mesh(spec, page_material, spine_material, cover_material):
    width, height, thickness = spec.get("visible_proxy_dimensions", [0.50, 0.88, 0.12])
    front_material = (
        make_image_material("mat_object_A_book_front_open_texture", spec["front_texture_image"])
        if spec.get("front_texture_image")
        else cover_material
    )
    back_material = (
        make_image_material("mat_object_A_book_back_open_texture", spec["back_texture_image"])
        if spec.get("back_texture_image")
        else cover_material
    )
    pages_edge_material = (
        make_image_material("mat_object_A_book_pages_edge_open_texture", spec["pages_texture_image"])
        if spec.get("pages_texture_image")
        else page_material
    )
    inner_page_material = (
        make_image_material("mat_object_A_book_inner_open_texture", spec["inner_page_texture_image"])
        if spec.get("inner_page_texture_image")
        else page_material
    )
    cover_edge_material = make_material("mat_object_A_book_open_cover_edges", (0.66, 0.06, 0.10, 1.0))
    page_shadow_material = make_material("mat_object_A_book_open_page_lines", (0.34, 0.29, 0.22, 1.0))

    page_block_width = spec.get("retopo_page_block_width", width * 0.22)
    page_block_depth = spec.get("retopo_page_block_depth", thickness * 0.36)
    cover_t = spec.get("retopo_cover_thickness", max(thickness * 0.055, 0.004))
    gap_x = spec.get("retopo_cover_gap_x", width * 0.018)
    margin_y = spec.get("retopo_page_margin_y", height * 0.025)
    open_angle = spec.get("retopo_open_angle_degrees", 18.0)
    cover_w = max((width - page_block_width - gap_x * 2.0) * 0.5, width * 0.20)
    bevel = min(width, thickness) * 0.018

    page_x0 = -page_block_width / 2
    page_x1 = page_block_width / 2
    page_y0 = margin_y
    page_y1 = height - margin_y
    page_z0 = -page_block_depth / 2
    page_z1 = page_block_depth / 2

    create_local_box_object(
        "Object_A_open_book_pages_block",
        spec,
        (page_x0, page_x1, page_y0, page_y1, page_z0, page_z1),
        [pages_edge_material, inner_page_material, page_material],
        [0, 0, 1, 1, 2, 2],
        bevel_width=bevel * 0.45,
        bevel_segments=2,
    )

    right_inner_x = page_x1 + gap_x
    right_outer_x = right_inner_x + cover_w
    left_inner_x = page_x0 - gap_x
    left_outer_x = left_inner_x - cover_w

    create_local_box_object(
        "Object_A_open_book_front_cover",
        spec,
        (right_inner_x, right_outer_x, 0.0, height, -cover_t / 2, cover_t / 2),
        [front_material, inner_page_material, cover_edge_material],
        [0, 1, 2, 2, 2, 2],
        bevel_width=bevel,
        bevel_segments=3,
        rotate_y_degrees=-open_angle,
        rotate_y_pivot=(right_inner_x, 0.0),
    )
    create_local_box_object(
        "Object_A_open_book_back_cover",
        spec,
        (left_outer_x, left_inner_x, 0.0, height, -cover_t / 2, cover_t / 2),
        [inner_page_material, back_material, cover_edge_material],
        [0, 1, 2, 2, 2, 2],
        bevel_width=bevel,
        bevel_segments=3,
        rotate_y_degrees=open_angle,
        rotate_y_pivot=(left_inner_x, 0.0),
    )

    spine_w = max(spec.get("retopo_spine_width", width * 0.035), cover_t * 1.6)
    for name, x0, x1 in [
        ("Object_A_open_book_front_hinge", page_x1, page_x1 + spine_w),
        ("Object_A_open_book_back_hinge", page_x0 - spine_w, page_x0),
    ]:
        create_local_box_object(
            name,
            spec,
            (x0, x1, 0.0, height, -cover_t * 0.7, cover_t * 0.7),
            [cover_edge_material, spine_material],
            [0, 0, 1, 1, 0, 0],
            bevel_width=bevel * 0.6,
            bevel_segments=3,
        )

    line_count = int(spec.get("retopo_page_line_count", 18))
    line_w = max(page_block_width * 0.045, 0.001)
    line_depth = max(page_block_depth * 0.012, 0.001)
    if line_count > 0:
        for index in range(1, line_count + 1):
            x = page_x0 + (page_x1 - page_x0) * index / (line_count + 1)
            create_local_box_object(
                f"Object_A_open_book_page_edge_line_{index:02d}",
                spec,
                (
                    x - line_w / 2,
                    x + line_w / 2,
                    page_y0,
                    page_y1,
                    page_z1 - line_depth / 2,
                    page_z1 + line_depth / 2,
                ),
                [page_shadow_material],
                [0, 0, 0, 0, 0, 0],
            )
    return None


def fit_mesh_local_bbox_to_dimensions(obj, dimensions, contact_axis=None):
    if not obj.data.vertices:
        return

    coords = [vertex.co.copy() for vertex in obj.data.vertices]
    min_x = min(coord.x for coord in coords)
    max_x = max(coord.x for coord in coords)
    min_y = min(coord.y for coord in coords)
    max_y = max(coord.y for coord in coords)
    min_z = min(coord.z for coord in coords)
    max_z = max(coord.z for coord in coords)
    extent_x = max(max_x - min_x, 1e-6)
    extent_y = max(max_y - min_y, 1e-6)
    extent_z = max(max_z - min_z, 1e-6)
    width, height, thickness = dimensions
    scale_x = width / extent_x
    scale_y = height / extent_y
    scale_z = thickness / extent_z

    def fit_coord(value, min_value, max_value, scale, axis_name):
        if contact_axis == axis_name or (contact_axis is None and axis_name == "y"):
            return (value - min_value) * scale
        center = (min_value + max_value) * 0.5
        return (value - center) * scale

    for vertex in obj.data.vertices:
        vertex.co.x = fit_coord(vertex.co.x, min_x, max_x, scale_x, "x")
        vertex.co.y = fit_coord(vertex.co.y, min_y, max_y, scale_y, "y")
        vertex.co.z = fit_coord(vertex.co.z, min_z, max_z, scale_z, "z")
    obj.data.update()


def import_object_a(spec, page_material, spine_material, cover_material):
    if spec.get("asset_render_mode") == "textured_open_book_mesh":
        return create_object_a_textured_open_book_mesh(spec, page_material, spine_material, cover_material)
    if spec.get("asset_render_mode") == "textured_solid_book_mesh":
        return create_object_a_texture_baked_solid_mesh(spec, page_material, spine_material, cover_material)
    if spec.get("asset_render_mode") == "textured_retopo_book_mesh":
        return create_object_a_texture_baked_retopo_mesh(spec, page_material, spine_material, cover_material)

    asset = abs_path(spec["asset_path"])
    if asset.exists():
        bpy.ops.wm.ply_import(filepath=str(asset))
        obj = bpy.context.object
        obj.name = "Object_A_book_point_cloud"
        if len(obj.data.polygons) == 0:
            obj.hide_render = True
            return create_object_a_proxy(spec, page_material, spine_material, cover_material)
        if spec.get("asset_render_mode") == "vertex_color_mesh":
            obj.name = "Object_A_book_2dgs_extracted_mesh"
            obj.data.materials.clear()
            obj.data.materials.append(make_vertex_color_material("mat_object_A_2dgs_vertex_color", obj.data))
            fit_mesh_local_bbox_to_dimensions(
                obj,
                spec.get("mesh_fit_dimensions", spec.get("visible_proxy_dimensions", [0.36, 0.62, 0.08])),
                spec.get("mesh_fit_contact_axis"),
            )
        else:
            obj.data.materials.append(page_material)
        obj.location = spec["location"]
        obj.rotation_euler = spec["rotation_euler"]
        obj.scale = spec["scale"]
    else:
        obj = create_object_a_proxy(spec, page_material, spine_material, cover_material)
    return obj


def add_floor(spec, material):
    bpy.ops.mesh.primitive_plane_add(size=spec["floor_size"], location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "Counter_Textured_Floor"
    floor.data.materials.append(material)


def add_camera_backplate(camera_obj, background_spec, render_spec):
    image_path = background_spec.get("backplate_image")
    if not image_path:
        return None

    mat = make_image_material("mat_counter_camera_backplate", image_path, emission=True)
    distance = background_spec.get("backplate_distance", 7.0)
    aspect = render_spec["resolution"][0] / render_spec["resolution"][1]
    height = 2 * distance * math.tan(camera_obj.data.angle_y * 0.5) * 1.08
    width = height * aspect

    bpy.ops.mesh.primitive_plane_add(size=1.0)
    backplate = bpy.context.object
    backplate.name = "Counter_Backplate_CameraPlane"
    backplate.parent = camera_obj
    backplate.location = (0, 0, -distance)
    backplate.rotation_euler = (0, 0, 0)
    backplate.scale = (width, height, 1.0)
    backplate.data.materials.append(mat)
    if hasattr(backplate, "visible_shadow"):
        backplate.visible_shadow = False
    return backplate


def add_lights(spec):
    for light_spec in spec:
        light_data = bpy.data.lights.new(light_spec["name"], light_spec["type"])
        light_obj = bpy.data.objects.new(light_spec["name"], light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = light_spec["location"]
        light_data.energy = light_spec["power"]
        if hasattr(light_data, "size"):
            light_data.size = light_spec["size"]


def add_camera(spec):
    camera_data = bpy.data.cameras.new(spec["name"])
    camera_obj = bpy.data.objects.new(spec["name"], camera_data)
    bpy.context.collection.objects.link(camera_obj)
    bpy.context.scene.camera = camera_obj
    camera_data.lens = spec["focal_length_mm"]

    target = bpy.data.objects.new("Camera_LookAt_Target", None)
    bpy.context.collection.objects.link(target)
    target.location = spec["look_at"]

    constraint = camera_obj.constraints.new(type="TRACK_TO")
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    constraint.target = target

    start = spec["frame_start"]
    end = spec["frame_end"]
    radius = spec["orbit_radius"]
    height = spec["height"]
    for frame in range(start, end + 1):
        t = (frame - start) / max(1, end - start)
        angle = 2 * math.pi * t
        camera_obj.location = (
            radius * math.cos(angle),
            radius * math.sin(angle),
            height,
        )
        camera_obj.keyframe_insert(data_path="location", frame=frame)
    return camera_obj


def configure_render(spec):
    scene = bpy.context.scene
    scene.frame_start = spec["frame_start"]
    scene.frame_end = spec["frame_end"]
    scene.frame_set(spec["frame_start"])
    scene.render.resolution_x = spec["resolution"][0]
    scene.render.resolution_y = spec["resolution"][1]
    scene.render.fps = spec["fps"]
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items} else "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text())
    clear_scene()

    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.color = spec["background"]["color"]

    mat_book_pages = make_material("mat_object_A_book_pages", (0.86, 0.78, 0.62, 1.0))
    mat_book_spine = make_material("mat_object_A_book_red_spine", (0.72, 0.08, 0.12, 1.0))
    mat_book_cover = make_image_material(
        "mat_object_A_book_cover_texture",
        spec["object_A"]["cover_texture_image"],
    )
    mat_b = make_material("mat_object_B_traffic_cone_orange", (1.0, 0.35, 0.06, 1.0))
    mat_c = make_material("mat_object_C_cat_box_green", (0.55, 0.75, 0.55, 1.0))
    if spec["background"].get("floor_texture_image"):
        mat_floor = make_image_material(
            "mat_counter_textured_floor",
            spec["background"]["floor_texture_image"],
        )
    else:
        mat_floor = make_material("mat_counter_floor", (*spec["background"]["floor_color"], 1.0))

    add_floor(spec["background"], mat_floor)
    import_object_a(spec["object_A"], mat_book_pages, mat_book_spine, mat_book_cover)
    import_mesh_or_proxy(spec["object_B_anchor"], mat_b)
    import_mesh_or_proxy(spec["object_C_anchor"], mat_c)
    add_lights(spec["lights"])
    camera = add_camera(spec["camera"])
    add_camera_backplate(camera, spec["background"], spec["camera"])
    configure_render(spec["camera"])

    output_spec = spec["render_outputs"]
    blend_path = abs_path(output_spec["blend"])
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    preview_path = abs_path(output_spec["preview_video"])
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(preview_path)
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
