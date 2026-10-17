# Object B Notes

Temporary human observations and prompt ideas for the Object B workflow go here.

## 2026-05-31 smoke test

- Object B is locked to the wooden duck text-to-3D prompt:

```text
a small matte wooden duck toy, hand-painted yellow body and orange beak, simple smooth geometry, realistic object, clean shape, non-transparent, no fur, no thin structures
```

- Reproducible config: `configs/aigc_assets/object_B_wooden_duck.yaml`.
- Smoke-test entrypoint: `scripts/aigc/run_object_B_smoke.sh`.
- Target output root: `outputs/aigc_assets/object_B_text_to_3d/`.
- Threestudio dependency recovery is complete enough for non-gradio `launch.py` use: `pip check` passes, core CUDA extension imports pass, and `threestudio` / `BaseSystem` imports pass.
- First launch failed before training because inherited `HF_ENDPOINT=https://hf-mirror.com` could not resolve the default `stabilityai/stable-diffusion-2-1-base` tokenizer.
- Smoke script now uses local `AIGC_SD15_MODEL` (`runwayml/stable-diffusion-v1-5`) for both prompt processing and SD guidance.
- Successful run: `outputs/aigc_assets/object_B_text_to_3d/smoke_test/dreamfusion-sd/a_small_matte_wooden_duck_toy,_hand-painted_yellow_body_and_orange_beak,_simple_smooth_geometry,_realistic_object,_clean_shape,_non-transparent,_no_fur,_no_thin_structures@20260531-104828/`.
- Run settings: CUDA device 6, `trainer.max_steps=25`, `trainer.val_check_interval=25`, `checkpoint.every_n_train_steps=25`.
- Output check: `2` checkpoints, `1` validation PNG, `120` test PNGs, `it25-test.mp4`, config snapshots, TensorBoard logs, and CSV metrics were written.
- Visual note: `25` steps produce only an initial colored density blob, so this is a runtime smoke pass, not final Object B geometry.

## 2026-05-31 formal logging entrypoint

- Formal script: `scripts/aigc/run_object_B_formal.sh`.
- Default output root: `outputs/aigc_assets/object_B_text_to_3d/final/`.
- Default settings after the rubber-duck retry update: `10000` steps, validation every `1000` steps, checkpoint every `1000` steps.
- The script precomputes the trial directory and writes realtime stdout/stderr to `train.log` with `tee -a`.
- `TrainMetricsPrinterCallback` prints `[TRAIN_METRICS]` lines every `AIGC_TRAIN_LOG_INTERVAL` steps; default interval is `50`.
- Expected metric names include `train/loss_sds`, `train/loss_orient`, `train/loss_sparsity`, `train/loss_opaque`, `train/grad_norm`, and `lr-Adam/*` when available.

## 2026-05-31 wooden duck formal attempt

- Historical run preserved at `outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/wooden_duck_formal_10000steps@20260531-144954/`.
- It completed normally at `10000/10000` steps on CUDA device `6`; it was not interrupted.
- Preserved outputs include `train.log`, `cmd.txt`, config snapshots, TensorBoard/CSV logs, checkpoints every `1000` steps from `1000` through `10000`, validation PNGs every `500` steps, `120` final test PNGs, and `save/it10000-test.mp4`.
- Additional videos were later generated from the preserved `1000` and `2000` checkpoints as `save/it1000-test.mp4` and `save/it2000-test.mp4`.
- Visual acceptance failed: the generated shape was not a clean recognizable duck toy, so this run should be reported as an AIGC prompt/optimization failure rather than overwritten.

## 2026-05-31 rubber duck retry setup

- New reproducible config: `configs/aigc_assets/object_B_rubber_duck.yaml`.
- Updated formal script defaults to a simpler yellow rubber-duck toy prompt:

```text
a small yellow rubber duck toy with a single short rounded orange beak, closed beak, black dot eyes, smooth simple shape, clean cartoon style, glossy plastic surface, centered, highly recognizable silhouette, no feathers, no complex texture
```

- Updated default negative prompt:

```text
two beaks, double beak, multiple beaks, open beak, long sharp beak, extra limbs, extra parts, deformed face, asymmetrical beak, realistic bird, feathers, noisy texture, dirty surface, blurry shape, floaters, artifacts
```

- `scripts/aigc/run_object_B_formal.sh` now accepts `AIGC_OBJECT_B_PROMPT` and `AIGC_OBJECT_B_NEGATIVE_PROMPT`, passes `system.prompt_processor.negative_prompt`, and defaults the trial tag to `rubber_duck_formal_<steps>steps`.
- Default Object B formal validation interval is now `1000` steps so each 1000-step checkpoint has both a validation image and a test video.
- New helper script: `scripts/aigc/render_object_B_checkpoint_videos.sh`.
- The helper monitors the active trial directory, waits for stable `epoch=0-step=<N>.ckpt` files, and runs non-training `launch.py --test` to create `save/it<N>-test.mp4` every `AIGC_OBJECT_B_VIDEO_INTERVAL` steps while main training continues.
- The helper skips the final `MAX_STEPS` checkpoint because the main threestudio training command already runs `trainer.test(...)` at the end and produces the final video.

## 2026-05-31 rubber duck formal run started

- Active run directory: `outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubber_duck_formal_10000steps@20260531-171926/`.
- Launcher PID: `377309`.
- Main training Python PID at launch check: `377315`.
- Checkpoint-video monitor PID: `377312`.
- Main training device: CUDA `6`.
- Intermediate video render device: CUDA `5`.
- Run settings: `10000` steps, validation every `1000` steps, checkpoint every `1000` steps, checkpoint-video monitor every `1000` steps, metric print interval `50`.
- Realtime training log: `outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubber_duck_formal_10000steps@20260531-171926/train.log`.
- Intermediate videos will appear as `save/it1000-test.mp4`, `save/it2000-test.mp4`, etc. The first one is expected after the `1000` checkpoint is complete.

## 2026-05-31 rubber duck visual check at 1000/2000/3000

- Compared the first three checkpoint videos: `save/it1000-test.mp4`, `save/it2000-test.mp4`, and `save/it3000-test.mp4`.
- Auxiliary comparison sheets were generated under `/tmp/` for inspection only:
  - `/tmp/objectB_rubber_duck_1000_2000_3000_rgb_sheet.jpg`
  - `/tmp/objectB_rubber_duck_beak_closeups.jpg`
  - `/tmp/objectB_f40_full_panels_steps.jpg`
- Observation: the beak/mouth does not appear to get monotonically wider in actual opacity geometry. The silhouette becomes cleaner by `3000`.
- Concern: the orange/red beak color remains too broad across the front views and by `3000` reads more like a horizontal band wrapping the face/body than a single short rounded protruding beak.
- Current acceptance status: not accepted yet. `2000` looks cleaner than `1000` in several views, but `3000` introduces darker material and a band-like mouth appearance.

## 2026-05-31 rubber duck formal run stopped after drift

- The rubber-duck retry was manually stopped after the preview videos showed clear visual drift.
- Preserved failed run directory: `outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubber_duck_formal_10000steps@20260531-171926/`.
- Saved evidence includes `train.log`, checkpoint-video monitor logs, checkpoints from `1000` through `7000`, validation PNGs from `1000` through `7000`, and test videos `save/it1000-test.mp4` through `save/it7000-test.mp4`.
- Process check after stopping found no remaining `launch.py` or checkpoint-video monitor process for this run. A zero-byte `training.done` marker exists because the older shell exit trap ran during termination; `training.stopped` was added as the authoritative marker. Treat this run as intentionally stopped/rejected, not accepted.
- Visual conclusion: `1000` is the most useful early checkpoint, `2000` is still plausible in some views, and later checkpoints increasingly turn the beak into a red/orange horizontal band. This is report-worthy evidence that longer SDS optimization can worsen prompt-aligned geometry/material details.

## 2026-05-31 next rubber duck retry policy

- Do not use `10000` steps as the Object B default for the next retry. Cap small trials at `3000` steps and choose among `1000`, `2000`, and `3000` videos instead of assuming the final checkpoint is best.
- Simplify the default prompt to:

```text
a simple yellow rubber duck bath toy, smooth rounded body, small orange beak, black dot eyes, glossy plastic
```

- Use a negative prompt that directly targets the observed failure mode:

```text
red stripe, orange band, ring around body, scarf, oversized beak, wide mouth, two beaks, extra parts, extra limbs, feathers, realistic bird, noisy texture, dirty surface, floaters, artifacts
```

- The base DreamFusion config has `seed: 0`, `guidance_scale: 100`, `min_step_percent: 0.02`, and `max_step_percent: 0.98`. The next script defaults lower guidance to `70` and narrows the diffusion timestep range to `0.05`-`0.80`.
- Suggested next search: run short `3000`-step trials with seeds such as `0`, `7`, and `42`, then accept the best checkpoint video rather than training longer by default.
- `scripts/aigc/run_object_B_formal.sh` now refuses to reuse a non-empty trial directory so failed attempts stay preserved for the report.

## 2026-05-31 strict single-GPU short seed sweep

- The Object B short candidate sweep used only CUDA `6`; CUDA `5` remained available for other users.
- The seed `0` run initially launched a checkpoint-video monitor on CUDA `5`, but that monitor was stopped before it rendered any checkpoint video. The main seed `0` training process continued on CUDA `6`.
- `scripts/aigc/run_object_B_formal.sh` now defaults `AIGC_OBJECT_B_ENABLE_VIDEO_MONITOR=0`, so future Object B formal runs train first and render checkpoint videos afterward.
- `scripts/aigc/render_object_B_checkpoint_videos.sh` now supports `AIGC_OBJECT_B_RENDER_FINAL=1` so post-training rendering can verify or generate `1000`, `2000`, and `3000` videos sequentially on CUDA `6`.
- Current sweep policy: run seed `0`, then render checkpoint videos on CUDA `6`; only after those videos are generated and inspected should seed `7` start, followed by seed `42` under the same pattern.

## 2026-05-31 rubber duck short seed sweep results

Shared settings:

```text
prompt: a simple yellow rubber duck bath toy, smooth rounded body, small orange beak, black dot eyes, glossy plastic
negative prompt: red stripe, orange band, ring around body, scarf, oversized beak, wide mouth, two beaks, extra parts, extra limbs, feathers, realistic bird, noisy texture, dirty surface, floaters, artifacts
max_steps: 3000
guidance_scale: 70
min_step_percent: 0.05
max_step_percent: 0.80
validation/checkpoint interval: 1000
GPU policy: CUDA 6 only; train first, then render checkpoint videos on CUDA 6 before starting the next seed
```

- Combined video preview contact sheet: `outputs/aigc_assets/object_B_text_to_3d/final/object_B_rubber_duck_seed_sweep_preview_contact_sheet.jpg`.
- Seed `0` run: `outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubber_duck_simple_seed0_g70_3000steps@20260531-181724/`.
  - Completed `3000/3000`.
  - Preserved checkpoints: `1000`, `2000`, `3000`.
  - Preserved previews: `it1000-0.png`, `it2000-0.png`, `it3000-0.png`, `it1000-test.mp4`, `it2000-test.mp4`, `it3000-test.mp4`.
  - Visual result: not acceptable. The object stays pink/purple rather than yellow and never becomes a clear rubber duck.
  - Best checkpoint for this seed: `2000` only as a failure comparison; no final candidate.
- Seed `7` run: `outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubber_duck_simple_seed7_g70_3000steps@20260531-184143/`.
  - Completed `3000/3000`.
  - Preserved checkpoints: `1000`, `2000`, `3000`.
  - Preserved previews: `it1000-0.png`, `it2000-0.png`, `it3000-0.png`, `it1000-test.mp4`, `it2000-test.mp4`, `it3000-test.mp4`.
  - Visual result: best of the three seeds but still weak. `1000` gives a yellow rounded silhouette; `2000` has the clearest yellow body plus some face/beak cue; `3000` becomes darker, noisier, and less toy-like.
  - Best checkpoint for this seed: `2000`.
- Seed `42` run: `outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubber_duck_simple_seed42_g70_3000steps@20260531-190238/`.
  - Stopped after the `1000` checkpoint because the validation preview already showed a split/two-object duck-like shape, which matches the early-stop rule for severe drift.
  - Preserved checkpoint: `1000`.
  - Preserved previews: `it1000-0.png`, `it1000-test.mp4`.
  - Marker: `training.stopped`.
  - Best checkpoint for this seed: none; reject.
- Overall conclusion: no seed is strong enough to accept confidently as the final Object B asset. If a weak emergency candidate is needed, use seed `7` step `2000`; otherwise do not continue blindly with the same DreamFusion setup.

## 2026-06-02 resumed Object B mainline: toy traffic cone

- New Object B concept: a small toy traffic cone. Duck variants remain preserved as failure evidence and should not be overwritten.
- Reproducible config: `configs/aigc_assets/object_B_traffic_cone.yaml`.
- Prompt:

```text
a small toy traffic cone with a simple orange cone body, one thick white reflective band, a simple dark base, matte plastic, clean silhouette, centered, single solid object, no text, no logo
```

- Negative prompt:

```text
animal, bird, duck, beak, face, eyes, feathers, fur, person, thin parts, extra parts, holes, dirty texture, noisy texture, floaters, transparent, deformed, melted, double object, sign, pole, road sign, party hat
```

- A simple dark base is allowed; road signs, poles, text/logo, party-hat shapes, animals, double objects, and detached floaters are rejection reasons.
- Validation policy update: Object B must use four validation views per checkpoint, not the old single validation view, because the rubber-duck failures showed that one good view can hide front/side failures.
- Screening policy: CUDA `6` only; seeds `0`, `7`, and `42`; start with `1500` steps per seed, validation/checkpoint every `500` steps, and render checkpoint videos after training or early stop on the same GPU.

## 2026-06-02 traffic-cone resume results

Shared settings:

```text
max_steps: 1500 for seed screening, then seed 7 resumed to 3000
guidance_scale: 70
min_step_percent: 0.05
max_step_percent: 0.80
validation/checkpoint interval: 500
validation views per checkpoint: 4
GPU policy: CUDA 6 only
```

- Seed `0` run: `outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/traffic_cone_seed0_g70_1500steps@20260602-1735/`.
  - Stopped after inspecting the `1000` validation set.
  - Preserved checkpoints: `500`, `1000`.
  - Evidence sheets: `outputs/previews/aigc_assets/object_B_text_to_3d/traffic_cone/object_B_traffic_cone_seed0_it500_views.jpg`, `outputs/previews/aigc_assets/object_B_text_to_3d/traffic_cone/object_B_traffic_cone_seed0_it1000_views.jpg`.
  - Visual result: rejected. The object had smoky density and a double-subject tendency.
- Seed `7` initial run: `outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/traffic_cone_seed7_g70_1500steps@20260602-1743/`.
  - Completed `1500/1500`.
  - Evidence sheets: `outputs/previews/aigc_assets/object_B_text_to_3d/traffic_cone/object_B_traffic_cone_seed7_it500_views.jpg`, `outputs/previews/aigc_assets/object_B_text_to_3d/traffic_cone/object_B_traffic_cone_seed7_it1000_views.jpg`, `outputs/previews/aigc_assets/object_B_text_to_3d/traffic_cone/object_B_traffic_cone_seed7_it1500_views.jpg`.
  - Visual result: best screening seed. Single traffic-cone-like form with low smoke, but the requested white reflective band was not clear.
- Seed `42` run: `outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/traffic_cone_seed42_g70_1500steps@20260602-1754/`.
  - Completed `1500/1500`.
  - Evidence sheets: `outputs/previews/aigc_assets/object_B_text_to_3d/traffic_cone/object_B_traffic_cone_seed42_it500_views.jpg`, `outputs/previews/aigc_assets/object_B_text_to_3d/traffic_cone/object_B_traffic_cone_seed42_it1000_views.jpg`, `outputs/previews/aigc_assets/object_B_text_to_3d/traffic_cone/object_B_traffic_cone_seed42_it1500_views.jpg`.
  - Visual result: rejected relative to seed `7`; single cone, but more smoke/noise and no clear white band.
- Seed `7` resume run: `outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/traffic_cone_seed7_g70_resume1500_to_3000steps@20260602-180305/`.
  - Resumed from `traffic_cone_seed7_g70_1500steps@20260602-1743/ckpts/epoch=0-step=1500.ckpt`.
  - Completed `3000/3000`; marker: `training.done`.
  - Preserved checkpoints: `2000`, `2500`, `3000`.
  - Final validation PNGs: `save/it3000-0.png` through `save/it3000-3.png`.
  - Final video: `save/it3000-test.mp4`.
  - Evidence sheets: `outputs/previews/aigc_assets/object_B_text_to_3d/traffic_cone/object_B_traffic_cone_seed7_resume_it2000_ckpt_views.jpg`, `outputs/previews/aigc_assets/object_B_text_to_3d/traffic_cone/object_B_traffic_cone_seed7_resume_it2500_ckpt_views.jpg`, `outputs/previews/aigc_assets/object_B_text_to_3d/traffic_cone/object_B_traffic_cone_seed7_resume_it3000_views.jpg`.
- Current Object B selection: seed `7` resume, step `3000`.
  - Acceptance status: usable but imperfect. The 4-view geometry is the most stable traffic-cone candidate produced so far, but material/text prompt alignment is incomplete because the white reflective band is still missing or weak.
