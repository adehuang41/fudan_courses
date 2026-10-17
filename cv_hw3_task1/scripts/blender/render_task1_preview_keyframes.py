from pathlib import Path

try:
    import bpy
except ImportError as exc:
    raise SystemExit(
        "This script must be run inside Blender: blender --background "
        "task1_locked_layout.blend --python scripts/blender/render_task1_preview_keyframes.py"
    ) from exc


PROJECT_ROOT = Path("/home/dechao/cv_final_pj")
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/renders/blender_fusion/task1_scene_preparation/preview_frames"
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    scene.render.resolution_x = 640
    scene.render.resolution_y = 360
    scene.render.image_settings.file_format = "PNG"

    for frame in [1, 25, 49, 73, 96]:
        scene.frame_set(frame)
        scene.render.filepath = str(OUTPUT_DIR / f"frame_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
