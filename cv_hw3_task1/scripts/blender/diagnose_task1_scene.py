from pathlib import Path

try:
    import bpy
except ImportError as exc:
    raise SystemExit(
        "This script must be run inside Blender: blender --background "
        "task1_locked_layout.blend --python scripts/blender/diagnose_task1_scene.py"
    ) from exc


PROJECT_ROOT = Path("/home/dechao/cv_final_pj")
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs/renders/blender_fusion/task1_scene_preparation/scene_diagnostics_20260609.txt"
)


def keyframes_for(obj):
    if not obj.animation_data or not obj.animation_data.action:
        return []
    rows = []
    for curve in obj.animation_data.action.fcurves:
        frames = [int(point.co.x) for point in curve.keyframe_points]
        rows.append(
            {
                "data_path": curve.data_path,
                "array_index": curve.array_index,
                "frame_count": len(frames),
                "first_frames": frames[:10],
                "last_frames": frames[-5:],
            }
        )
    return rows


def fmt_vec(values):
    return "(" + ", ".join(f"{value:.6g}" for value in values) + ")"


def main() -> None:
    scene = bpy.context.scene
    lines = [
        "Task 1 Blender Scene Diagnostics",
        "================================",
        f"frame_start: {scene.frame_start}",
        f"frame_end: {scene.frame_end}",
        f"current_frame: {scene.frame_current}",
        f"camera: {scene.camera.name if scene.camera else 'None'}",
        "",
        "Objects",
        "-------",
    ]

    animated_objects = []
    camera_parented_objects = []
    for obj in scene.objects:
        keys = keyframes_for(obj)
        if keys:
            animated_objects.append(obj.name)
        if obj.parent and scene.camera and obj.parent.name == scene.camera.name:
            camera_parented_objects.append(obj.name)

        lines.extend(
            [
                f"name: {obj.name}",
                f"  type: {obj.type}",
                f"  parent: {obj.parent.name if obj.parent else 'None'}",
                f"  location: {fmt_vec(obj.location)}",
                f"  rotation_euler: {fmt_vec(obj.rotation_euler)}",
                f"  scale: {fmt_vec(obj.scale)}",
                f"  hide_render: {obj.hide_render}",
                f"  keyframes: {keys if keys else 'none'}",
            ]
        )
        if obj.type == "MESH":
            lines.extend(
                [
                    f"  vertices: {len(obj.data.vertices)}",
                    f"  polygons: {len(obj.data.polygons)}",
                    f"  materials: {[mat.name for mat in obj.data.materials]}",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "Animation Summary",
            "-----------------",
            f"animated_objects: {animated_objects if animated_objects else 'none'}",
            f"camera_parented_objects: {camera_parented_objects if camera_parented_objects else 'none'}",
            "",
            "Diagnosis",
            "---------",
            "The only animated object in the current preview is the camera.",
            "Object_A_visible_proxy_book_* are manual Object A preview proxies, not the counter background.",
            "Counter_Textured_Floor is a manual textured floor proxy, not the reconstructed counter geometry.",
            "Counter_Backplate_CameraPlane is parented to the camera, so it remains static in screen space.",
            "Therefore the current video is a placeholder/smoke preview: foreground objects are viewed by a moving camera, but the counter background is a static camera-attached backplate and cannot show true parallax.",
        ]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
