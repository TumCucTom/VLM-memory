# Route-plan training readiness

## Summary

- Missing video: scannet/videos/scene0706_00.mp4
- Done: scene0706_00 .sens downloaded to data/vlm_3r_data/scannet/scans/scene0706_00/
- Export to mp4 fails for this scene (OpenCV cvtColor). Use filtered JSON or add mp4 from another source.

## Option A: Filtered route-plan JSON

Run: python3 -c \"import json,os; vf='data/vlm_3r_data'; d=json.load(open("'data/vlm_3r_data/vsibench/merged_qa_route_plan_train.json')); v=[s for s in d if not s.get('video') or os.path.exists(os.path.join(vf,s['video']))]; json.dump(v,open('data/vlm_3r_data/vsibench/merged_qa_route_plan_train_valid.json','w'),indent=2); print(len(v),'valid')\"

Then in vsibench_data.yaml use json_path: data/vlm_3r_data/vsibench/merged_qa_route_plan_train_valid.json for the route plan entry.

## Option B: Add scene0706_00.mp4

Place scene0706_00.mp4 at data/vlm_3r_data/scannet/videos/ (from another export or source).

