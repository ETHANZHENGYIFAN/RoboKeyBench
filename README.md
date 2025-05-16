Code for RoboKeyBench (NIPS 2025)

source/                 # Raw 3D assets and scene descriptions from RoboCasa

output/                 # Intermediate files generated during code execution

results/                # Final evaluation metrics from Multimodal LLMs (MLLMs)

viewpoint_material.py        # Generates multi-perspective RGB-D renders with material variations

2D_to_3D_copy.py           # Converts 2D image annotations to 3D world coordinates using projections

refine_to_edge.py          # Optimizes annotation boundaries through edge detection algorithms

statistics.py              # Analysis for MLLM evaluation metrics

statistics_open_source.py        # Analysis for open-source MLLM evaluation metrics
