import json
import math
import os
import statistics
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path("/home/dechao/cv_final_pj")


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
DEPTH_DIR = project_path(
    os.environ.get(
        "TASK1_DEPTH_DIR",
        "outputs/reconstruction_2dgs/background_counter/2dgs_final_r4_30k/traj/ours_30000/vis",
    )
)
OUTPUT_JSON = project_path(
    os.environ.get(
        "TASK1_SURFACE_PLACEMENT_JSON",
        COMPOSITING_DIR / "metadata/foreground_surface_placement.json",
    )
)
ALLOW_OVERWRITE = os.environ.get("TASK1_ALLOW_OVERWRITE", "0") == "1"
PLACEMENT_PRESET = os.environ.get("TASK1_PLACEMENT_PRESET", "counter")

ANCHOR_FRAME = 120
DEPTH_WINDOW_RADIUS = 4

PRESETS = {
    "counter": {
        "scene": "mipnerf360_counter",
        "surface_name": "counter",
        # Pixels are manually selected on frame 00120.png. Plane pixels come from
        # visible dark-counter patches; object anchors are shifted inward from
        # the front edge so they remain visible across more of the render path.
        "plane_pixels": {
            "left_front": [335, 458],
            "right_front": [610, 455],
            "right_back": [650, 420],
        },
        "anchors": {
            "object_A": {
                "pixel": [455, 410],
                "role": "central lower counter/cutting-board area for book proxy",
                "orientation_mode": "upright_open_book",
                "surface_yaw": 0.12,
                "visible_proxy_dimensions": [0.34, 0.58, 0.08],
                "open_angle_degrees": 11.0,
                "front_texture_image": "outputs/renders/blender_fusion/task1_render_level_compositing/textures/object_A_book_front_render_00085_crop.png",
                "back_texture_image": "outputs/renders/blender_fusion/task1_render_level_compositing/textures/object_A_book_back_render_00020_crop.png",
                "pages_texture_image": "outputs/renders/blender_fusion/task1_render_level_compositing/textures/object_A_book_pages_render_00120_crop.png",
            },
            "object_B_anchor": {
                "pixel": [640, 382],
                "role": "right counter patch for traffic cone",
                "surface_yaw": 0.04,
                "scale": [0.32, 0.32, 0.48],
            },
            "object_C_anchor": {
                "pixel": [560, 430],
                "role": "right lower counter/cutting-board area for cat box",
                "surface_yaw": -0.10,
                "scale": [0.24, 0.22, 0.20],
            },
        },
    },
    "garden": {
        "scene": "mipnerf360_garden",
        "surface_name": "round wooden garden table",
        # Pixels are manually selected on official Mip-NeRF 360 garden frame
        # 00120.png. Plane points avoid the center vase and lie on the visible
        # wooden table top.
        "plane_pixels": {
            "left_front": [420, 540],
            "right_front": [890, 535],
            "right_back": [785, 365],
        },
        "anchors": {
            "object_A": {
                "pixel": [475, 500],
                "role": "left-front table area for upright open book",
                "orientation_mode": "upright_open_book",
                "surface_yaw": -0.10,
                "visible_proxy_dimensions": [0.36, 0.62, 0.08],
                "open_angle_degrees": 14.0,
                "front_texture_image": "outputs/renders/blender_fusion/task1_render_level_compositing/textures/object_A_book_front_render_00085_crop.png",
                "back_texture_image": "outputs/renders/blender_fusion/task1_render_level_compositing/textures/object_A_book_back_render_00020_crop.png",
                "pages_texture_image": "outputs/renders/blender_fusion/task1_render_level_compositing/textures/object_A_book_pages_render_00120_crop.png",
            },
            "object_B_anchor": {
                "pixel": [865, 530],
                "role": "right-front table area for upright traffic cone",
                "surface_yaw": 0.02,
                "scale": [0.58, 0.58, 0.82],
            },
            "object_C_anchor": {
                "pixel": [650, 560],
                "role": "front-center table area for upright cat box",
                "surface_yaw": -0.08,
                "scale": [0.30, 0.27, 0.25],
            },
        },
    },
}

if PLACEMENT_PRESET not in PRESETS:
    valid = ", ".join(sorted(PRESETS))
    raise SystemExit(f"Unknown TASK1_PLACEMENT_PRESET={PLACEMENT_PRESET!r}. Valid: {valid}")

PRESET = PRESETS[PLACEMENT_PRESET]
PLANE_PIXELS = PRESET["plane_pixels"]
ANCHORS = PRESET["anchors"]


def median_depth(depth_image: Image.Image, x: int, y: int, radius: int) -> float:
    width, height = depth_image.size
    values = []
    for yy in range(max(0, y - radius), min(height, y + radius + 1)):
        for xx in range(max(0, x - radius), min(width, x + radius + 1)):
            value = float(depth_image.getpixel((xx, yy)))
            if math.isfinite(value) and value > 0:
                values.append(value)
    if not values:
        raise ValueError(f"No valid depth samples around pixel {(x, y)}")
    return float(statistics.median(values))


def transform_point(matrix, point):
    return [
        sum(matrix[row][col] * point[col] for col in range(4))
        for row in range(4)
    ]


def subtract(a, b):
    return [a[index] - b[index] for index in range(3)]


def dot(a, b):
    return sum(a[index] * b[index] for index in range(3))


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def normalize(vector):
    length = math.sqrt(dot(vector, vector))
    if length == 0:
        raise ValueError("Cannot normalize zero-length vector")
    return [value / length for value in vector]


def backproject_pixel(record, x: int, y: int, depth: float, width: int, height: int):
    tan_x = math.tan(record["FoVx"] * 0.5)
    tan_y = math.tan(record["FoVy"] * 0.5)

    ndc_x = 2.0 * ((x + 0.5) / width) - 1.0
    # The foreground Blender camera is set with camera_to_world @ diag(1,-1,-1).
    # Keeping image-y positive downward in the 2DGS camera frame maps to Blender's
    # local negative-y screen direction after that conversion.
    ndc_y = 2.0 * ((y + 0.5) / height) - 1.0
    camera_point = [
        ndc_x * tan_x * depth,
        ndc_y * tan_y * depth,
        depth,
        1.0,
    ]
    world_point = transform_point(record["camera_to_world"], camera_point)
    return camera_point[:3], world_point[:3]


def backproject_anchor(record, depth_image, x: int, y: int):
    width, height = depth_image.size
    depth = median_depth(depth_image, x, y, DEPTH_WINDOW_RADIUS)
    camera_point, world_point = backproject_pixel(record, x, y, depth, width, height)
    return {
        "pixel": [x, y],
        "depth_median_window_radius": DEPTH_WINDOW_RADIUS,
        "depth": depth,
        "camera_point": camera_point,
        "world_point": world_point,
    }


def estimate_surface_basis(record, depth_image):
    plane_points = {
        name: backproject_anchor(record, depth_image, *pixel)
        for name, pixel in PLANE_PIXELS.items()
    }
    left = plane_points["left_front"]["world_point"]
    right = plane_points["right_front"]["world_point"]
    back = plane_points["right_back"]["world_point"]

    tangent_x = normalize(subtract(right, left))
    tangent_to_back = subtract(back, left)
    normal = normalize(cross(tangent_x, tangent_to_back))

    center = [
        (left[index] + right[index] + back[index]) / 3.0
        for index in range(3)
    ]
    to_camera = subtract(record["camera_center"], center)
    if dot(normal, to_camera) < 0:
        normal = [-value for value in normal]

    tangent_x = normalize(
        subtract(tangent_x, [normal[index] * dot(tangent_x, normal) for index in range(3)])
    )
    tangent_y = normalize(cross(normal, tangent_x))
    return {
        "plane_pixels": PLANE_PIXELS,
        "plane_points": plane_points,
        "surface_normal": normal,
        "surface_tangent_x": tangent_x,
        "surface_tangent_y": tangent_y,
    }


def main() -> None:
    if OUTPUT_JSON.exists() and not ALLOW_OVERWRITE:
        raise SystemExit(
            "Refusing to overwrite existing surface placement metadata. "
            "Use a fresh TASK1_COMPOSITING_DIR or TASK1_SURFACE_PLACEMENT_JSON."
        )

    camera_path = json.loads(CAMERA_PATH_JSON.read_text())
    records = camera_path["cameras"]
    record = records[ANCHOR_FRAME]
    depth_image = Image.open(DEPTH_DIR / f"depth_{ANCHOR_FRAME:05d}.tiff")
    width, height = depth_image.size
    surface_basis = estimate_surface_basis(record, depth_image)

    placements = {}
    for object_key, anchor in ANCHORS.items():
        x, y = anchor["pixel"]
        anchor_projection = backproject_anchor(record, depth_image, x, y)
        world_point = anchor_projection["world_point"]
        placements[object_key] = {
            **anchor,
            **anchor_projection,
            "surface_normal": surface_basis["surface_normal"],
            "surface_tangent_x": surface_basis["surface_tangent_x"],
            "surface_tangent_y": surface_basis["surface_tangent_y"],
            "camera_center": record["camera_center"],
            "location": world_point,
        }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "status": "surface_depth_plane_alignment_preview_not_final",
                "scheme": "render_level_compositing_scheme_A",
                "placement_preset": PLACEMENT_PRESET,
                "background_scene": PRESET["scene"],
                "surface_name": PRESET["surface_name"],
                "anchor_frame": ANCHOR_FRAME,
                "background_frame": f"renders/{ANCHOR_FRAME:05d}.png",
                "depth_frame": f"vis/depth_{ANCHOR_FRAME:05d}.tiff",
                "image_size": [width, height],
                "coordinate_note": (
                    "World positions are back-projected from the 2DGS render-path "
                    "camera and depth map. A local surface plane is estimated from "
                    "manual pixels on the selected support surface and used to "
                    "orient object local-up axes. This remains preview placement; no final occlusion or "
                    "lighting solve is implied."
                ),
                "surface_basis": surface_basis,
                "placements": placements,
            },
            indent=2,
        )
    )
    print(f"Wrote {OUTPUT_JSON}")
    for object_key, placement in placements.items():
        print(
            object_key,
            "pixel",
            placement["pixel"],
            "depth",
            round(placement["depth"], 4),
            "world",
            [round(value, 4) for value in placement["world_point"]],
        )
    print("surface_normal", [round(value, 4) for value in surface_basis["surface_normal"]])


if __name__ == "__main__":
    main()
