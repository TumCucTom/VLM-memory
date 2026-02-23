# Vsibench filtered data – status for training with ArkitScenes

## Status: **Ready to train with ArkitScenes / route-plan data**

Filtered dataset JSONs are created and the config that uses them is in place. Training with **route plan** (which includes ArkitScenes) should run without collective mismatch from missing videos.

---

## What was done

1. **Filter script:** `scripts/VLM_3R/filter_vsibench_valid_paths.py`  
   - Reads scannet, scannetpp, and route_plan QA JSONs.  
   - Keeps only samples whose `video` path exists under `data/vlm_3r_data`.  
   - Writes `*_valid.json` next to the originals.

2. **Filtered config:** `scripts/VLM_3R/vsibench_data_valid.yaml`  
   - Same as `vsibench_data.yaml` but points to `*_valid.json`.  
   - Used by default in `scripts/VLM_3R/train_vsibench.sh` and `scripts/VLM_Dual_Mem/train/train_memory_phase1_1gpu.sh` (or set `DATA_YAML=scripts/VLM_3R/vsibench_data_valid.yaml` for other scripts).

3. **Regenerate filtered files when needed:**  
   ```bash
   python scripts/VLM_3R/filter_vsibench_valid_paths.py
   ```

---

## Filter report (last run)

| Dataset    | Original samples | Valid samples | Filtered out |
|-----------|-------------------|---------------|--------------|
| scannet   | 51,779            | 51,747        | 32           |
| scannetpp | 151,775           | 151,775       | 0            |
| route_plan| 4,104             | 4,103         | 1            |
| **Total** | **207,658**       | **207,625**   | **33**       |

- **Unique video paths removed:** 1  
  - `scannet/videos/scene0706_00.mp4` (missing file; all 33 removed samples reference this video).

So **33 training samples** were dropped, corresponding to **1 training video**. All other samples (including all ArkitScenes / route-plan samples that point to existing videos) are kept.

---

## How to train with ArkitScenes

- Use the **filtered** config so every sample has an existing video path:
  - `DATA_YAML=scripts/VLM_3R/vsibench_data_valid.yaml`
- Or run `scripts/VLM_3R/train_vsibench.sh` as-is (it now defaults to the filtered config).
- For working-memory-only (or other) scripts, set:
  - `DATA_YAML="scripts/VLM_3R/vsibench_data_valid.yaml"`  
  when you want **full vsibench including route plan / ArkitScenes**.

No further filtering is required for training with ArkitScenes beyond using the filtered config and regenerating it after any data changes if needed.
