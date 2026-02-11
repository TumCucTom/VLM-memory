# What You Need for Step 1.2 (Export Sampled Frames) to Work

Step 1.2 expects **two** things that the default ScanNet++ download does **not** provide:

| Required | Path per scene | What the script expects | In default download? |
|----------|----------------|--------------------------|----------------------|
| **Per-frame camera** | `data/raw_data/scannetpp/data/<scene_id>/dslr/camera/` | One `.npz` per train image with keys `intrinsic` (3×3) and `extrinsic` (4×4 camera-to-world) | **No** – you have `colmap/` (text) and `nerfstudio/`, not per-frame `.npz` |
| **Rendered depth** | `data/raw_data/scannetpp/data/<scene_id>/dslr/render_depth/` | One depth image per train image (e.g. `.png`), same basenames as color | **No** – you have mesh and DSLR poses, but no pre-rendered DSLR depth |

You already have:

- `dslr/resized_undistorted_images/` (or `rgb_resized_undistorted`) – color images  
- `dslr/train_test_lists.json` – list of train image filenames  
- `dslr/colmap/` – COLMAP model (cameras.txt, images.txt, points3D.txt)  
- Mesh under `scans/` (e.g. `mesh_aligned_0.05_semantic.ply`)

So 1.2 will only work after you **generate** the `camera/` and `render_depth/` data.

---

## Option A: Generate the missing data (for DSLR / step 1.2)

### 1. Generate `camera/` (per-frame .npz from COLMAP)

A script is provided that uses the ScanNet++ toolbox to read COLMAP and write per-frame `.npz`:

- **Script:** `scripts/VLM_Dual_Mem/data-download/scannetpp/preprocessing/export_colmap_to_camera_npz.py`
- **Usage:** Set `SCANNETPP_ROOT` to the ScanNet++ toolbox repo (e.g. `~/scratch/scannetpp`), then run with `--data_root` pointing to `data/raw_data/scannetpp/data` and `--scene_list_file` to the split file (e.g. `nvs_sem_train.txt`).

The **SLURM script** `step1_2_prep_render_and_camera.slurm` clones the toolbox, runs depth rendering, and runs this camera export so you get both `render_depth/` and `camera/` in one job.

### 2. Generate `render_depth/` (depth from mesh)

The **ScanNet++ toolbox** can render depth from the mesh for DSLR viewpoints:

- Repo: **https://github.com/scannetpp/scannetpp**
- Docs: https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation  

From the docs: *“Render high-res depth maps from the mesh for DSLR and iPhone frames.”*

You need to:

1. Clone the ScanNet++ toolbox and follow its setup.
2. Use its **depth rendering** tools for DSLR views, with:
   - Input: mesh (e.g. `scans/mesh_aligned_0.05.ply` or the one they expect) and DSLR poses (e.g. from `dslr/colmap/` or `dslr/nerfstudio/`).
   - Output: one depth image per DSLR train image, saved under  
     `data/raw_data/scannetpp/data/<scene_id>/dslr/render_depth/`  
     with the **same basenames** as in `train_test_lists.json` (e.g. `DSC01752.png`).

Naming and exact script names depend on the toolbox; check its README and “render depth” / “DSLR” sections.

After both:

- `dslr/camera/<basename>.npz` for all train images  
- `dslr/render_depth/<basename>.png` for all train images  

step 1.2 can run as-is (it already uses `resized_undistorted_images` and `train_test_lists.json`).

---

## Option B: Use iPhone path (step 1.3) instead of 1.2

If you only need **some** scenes and they have iPhone data:

- Use **step 1.3** (export iPhone frames). It reads:
  - `iphone/` (video, depth, COLMAP)
  - and does **not** require `dslr/camera/` or `dslr/render_depth/`.
- Then run **step 2.2** (frame metadata) using the **output of 1.3** (e.g. point `rendered_data_dir` / frame dirs to where 1.3 wrote color/depth/pose/intrinsic), or adapt 2.2 to read from 1.3’s output layout if it currently expects the same structure as 1.2.

So: for **1.2** you must **generate** `camera/` and `render_depth/` (Option A). For **1.3** you only need the standard download plus the iPhone assets; no extra download for “1.2-style” camera/depth, but you need to point the rest of the pipeline at 1.3’s outputs.

---

## Summary

| Goal | What to do |
|------|------------|
| **Run 1.2 (DSLR sampled frames)** | 1) Run the provided script to create `dslr/camera/*.npz` from COLMAP. 2) Use the ScanNet++ toolbox to render depth into `dslr/render_depth/` with matching names. No extra “download” – only code from this repo + ScanNet++ toolbox. |
| **Avoid 1.2** | Use step 1.3 (iPhone frames) where iPhone data exists, and wire 2.2 to 1.3’s output. |
