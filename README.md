# RoboKeyBench

Code and released annotation data for RoboKeyBench.

## Dataset release levels

This repository distinguishes three related but different data levels:

1. Representative released annotation subset: the visible `img/` directory contains a public subset of annotated objects for demonstrating the annotation format, multi-view renders, marked primitives, and file organization.
2. Dense VQA candidate pool: `generate_vqa.py` instantiates candidate questions from the annotation subset by expanding across rendered views, point/axis primitives, and question templates. These candidates are useful for verifying the automatic generation logic and are not the final balanced benchmark split.
3. Curated RoboKeyBench benchmark split: the 5,731 objects and 14,958 VQA samples reported in the manuscript refer to the curated benchmark used for experiments, constructed from the full object corpus after duplicate/ambiguous/redundant candidate filtering, question-type/category balancing, and split construction.

The candidate pool can contain many more questions per released object than the curated benchmark because it preserves dense view-primitive-template combinations before final filtering and balancing. The public benchmark package and split metadata are undergoing a release-level manual quality audit for annotation consistency, metadata integrity, filtering records, and file organization, and will be released under the same repository release page after the audit is completed.

Current VQA generation output should be written to `vqa_candidate_pool.json`:

```bash
python generate_vqa.py --output vqa_candidate_pool.json
```

Earlier draft VQA exports have been retained under `legacy/` for traceability only. They should not be used as the current benchmark split or as evidence of the current question taxonomy.

## Directory overview

- `img/`: annotated object subset organized by object category and instance. Each instance folder contains multi-view RGB renders, keypoint-overlaid images (`*_marked.png`), and annotation files (`*_arrows.json`).
- `source/`: raw 3D assets and scene descriptions when available locally.
- `results/`: evaluation outputs from multimodal LLMs.
- `legacy/`: earlier generated VQA draft files retained only for release-history traceability.

## Primitive generation pipeline

- `pipeline/data_processing/connected_component.py`: refines binary segmentation masks by filtering spurious fragments via connected component analysis.
- `pipeline/data_processing/dbscan_clustering.py`: clusters raw keypoint candidates using DBSCAN to remove noise and outliers.
- `pipeline/data_processing/strategic_sampling.py`: selects a compact representative keypoint subset via geometric anchors, saliency ranking, and farthest-point sampling.
- `pipeline/data_processing/axis_calibration.py`: calibrates the principal vertical axis from top-view and bottom-view renders for consistent cross-view coordinate frames.
- `pipeline/primitive_generation/functional_axis.py`: constructs functional point and axis primitives from keypoint correspondences and GPT-4o annotations, preserving multiple valid axis candidates when provided.
- `pipeline/primitive_generation/rjo.py`: extracts rotation axes and angular limit points for Rotational Joint Objects (RJO), using joint-region states for persistent-axis consistency and moving-part motion states for motion-plane orthogonality checks.
- `pipeline/primitive_generation/fco.py`: identifies actuation points, feedback regions, and actuation axes for Functional Control Objects (FCO).
- `pipeline/primitive_generation/primitive_builder.py`: top-level entry point that orchestrates the primitive generation pipeline.
- `pipeline/primitive_generation/closed_loop_controller.py`: explicit PASG closed-loop controller for confidence/missing-primitive checks, adaptive resampling, GPT re-query callbacks, and refinement-history logging.

`primitive_builder.py` provides the deterministic single-pass builder used after the current segmentation, sampling, and GPT annotation inputs are available. For the full PASG closed-loop execution path, use `closed_loop_controller.py` or the `build_primitives_closed_loop` wrapper in `primitive_builder.py`. The controller checks missing primitives and low-confidence GPT-aligned axes, expands the sampling budget, optionally calls a user-provided GPT/multi-view refinement callback, and rebuilds primitives until convergence or `max_iterations` is reached. RJO and FCO modules provide additional type-specific refinement branches for kinematic feasibility and actuation-feedback consistency.

## Rendering and projection helpers

- `viewpoint_material.py`: generates multi-view RGB renders from MuJoCo XML assets. The script now uses CLI arguments rather than hard-coded local paths and defaults to the released orthographic annotation-view convention.
- `2D_to_3D.py`: converts 2D image annotations to 3D world coordinates using orthographic or perspective camera rays and mesh intersection. The default orthographic projection matches `viewpoint_material.py`.

Example:

```bash
python viewpoint_material.py --source-root source --output-root output/render --categories kettle
python 2D_to_3D.py --source-root source --viewpoint-root output/viewpoint --output-dir output/3d_view --categories kettle
```

## Functional geometric optimization

- `refine_to_edge.py`: supports three release-level geometric optimization utilities:
  - 2D edge-aware refinement from rendered object contours.
  - 3D mesh-distance refinement that projects keypoints to the closest mesh surface when exact proximity is available, with a surface/edge-sample fallback.
  - symmetry-aware augmentation that proposes reflected keypoints under PCA-based geometric predicates and validates local normal consistency as an affordance-preserving check.

These utilities correspond to the edge-aware and symmetry-aware refinement described in the manuscript. They are exposed as optional post-processing tools because physical deployment may choose different mesh or rendering backends.

## VQA generation

- `generate_vqa.py`: generates a dense VQA candidate pool from annotation files, covering six question types: PSA, FTC, FTR, GTV, TGV, and PCV, with both point-primitive and axis-primitive variants. PCV generation prioritizes task-conditioned utility-axis versus placing/grasping-reference-axis pairs so angle questions reflect functional-axis constraints rather than raw keypoint-pair geometry alone.

## Result statistics

- `statistics.py`: summarizes result files under the current six-question taxonomy and capability layers.
- `statistics_open_source.py`: convenience wrapper for open-source model result files.

Example:

```bash
python statistics.py --inputs results/gpt4o/predictions_test.json results/gpt4o/predictions_val.json
python statistics_open_source.py
```
