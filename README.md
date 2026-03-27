Code and data for RoboKeyBench

--img/
-Annotated dataset organized by object category and instance.
Each instance folder contains multi-view RGB renders, keypoint-overlaid images (*_marked.png), and annotation files (*_arrows.json).

--source/
-Raw 3D assets and scene descriptions from RoboCasa

--results/
-Evaluation outputs from Multimodal LLMs (MLLMs)

--pipeline/data_processing/
-connected_component.py: Refines binary segmentation masks by filtering spurious fragments via connected component analysis
-dbscan_clustering.py: Clusters raw keypoint candidates using DBSCAN to remove noise and outliers
-strategic_sampling.py: Selects a compact representative keypoint subset via geometric anchors, saliency ranking, and farthest-point sampling
-axis_calibration.py: Calibrates the principal vertical axis from top-view and bottom-view renders for consistent cross-view coordinate frames

--pipeline/primitive_generation/
-functional_axis.py: Constructs functional manipulation axes from keypoint correspondences and GPT-4o annotations
-rjo.py: Extracts rotation axes and angular limit points for Rotational Joint Objects (RJO)
-fco.py: Identifies actuation points, feedback regions, and actuation axes for Functional Control Objects (FCO)
-primitive_builder.py: Top-level entry point that orchestrates the full primitive generation pipeline

--viewpoint_material.py
-Generates multi-perspective RGB-D renders with material variations

--2D_to_3D.py
-Converts 2D image annotations to 3D world coordinates using camera projections

--refine_to_edge.py
-Optimizes annotation boundaries through edge detection algorithms

--generate_vqa.py
-Generates the VQA dataset from annotation files, covering six question types: PSA, FTC, FTR, GTV, TGV, PCV

--statistics.py
-Analysis for MLLM evaluation metrics

--statistics_open_source.py
-Analysis for open-source MLLM evaluation metrics
