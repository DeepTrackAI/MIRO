# Spatial Clustering of Molecular Localizations with MIRO

**MIRO** (Multifunctional Integration through Relational Optimization) is a geometric deep learning framework that enhances clustering algorithms by transforming complex point clouds into structured, compact representations. It enables more robust clustering of single-molecule localization data using recurrent graph neural networks (rGNNs).

<div align="center">
  <img src="assets/MIROw.png" width="500"/>
</div>

## How it works?
**MIRO** learns to pull together localizations belonging to the same structure, producing spatially compact, well-separated clusters. This transformation enables standard algorithms like DBSCAN to perform significantly better — especially in challenging scenarios involving varying densities, blinking artifacts, or multiple cluster types.

## Key Features  
- **Improved Clustering Performance:** MIRO increases the efficiency of existing clustering algorithms by transforming point clouds into an optimized format.  
- **Simplified Parameter Selection:** By enhancing the differentiation among clusters and their separation from the background, **MIRO** streamlines parameter selection for clustering methods like DBSCAN.
- **Single-Shot and Few-Shot Learning:** **MIRO**’s single- or few-shot learning capability allows it to generalize across scenarios with minimal training, making it highly efficient and versatile.
- **Multiscale Clustering:** **MIRO**’s recurrent structure allows for identifying patterns at different scales. 
- **Broad Applicability:** **MIRO** is effective across datasets with diverse cluster shapes and symmetries.

<!---
## Dependencies  
**MIRO** is included as part of [deeplay](https://github.com/DeepTrackAI/deeplay). 

Install deeplay and unlock the full potential of **MIRO**. 
```bash
pip install deeplay
```
-->

## Installation

To install MIRO and its dependencies:

1. Make sure you have **Python 3.9** or higher installed.

2. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/DeepTrackAI/MIRO.git

3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt

**MIRO** is included as part of [deeplay](https://github.com/DeepTrackAI/deeplay), a modular framework for deep learning.

## 📘 Tutorials

Explore MIRO's capabilities through interactive Jupyter notebooks:

- **Benchmark:** Reproduce MIRO's performance on the benchmark datasets from [Nieves et al.](https://github.com/DJ-Nieves/ARI-and-IoU-cluster-analysis-evaluation) Follow the [Benchmark Tutorial](https://github.com/DeepTrackAI/MIRO/blob/master/benchmark/tutorial.ipynb) to train your own model or load pretrained ones.

- **Single-Shot Learning:** See how MIRO achieves impressive results even when trained on a single cluster. Try it yourself in the [Single-Shot Tutorial](https://github.com/DeepTrackAI/MIRO/blob/master/single-shot/tutorial.ipynb).

- **Multiscale Clustering:** Perform simultaneous clustering of nested structures with the [Multiscale Tutorial](https://github.com/DeepTrackAI/MIRO/blob/master/multiscale/tutorial.ipynb).

- **Multishape Clustering and Classification:** Learn how MIRO can cluster and classify structures of different shapes using the [Multishape Tutorial](https://github.com/DeepTrackAI/MIRO/blob/master/multishape/tutorial.ipynb).

## Citation
If you use **MIRO** in your research, please cite:
```
@article{pineda2024spatial,
  title={Spatial Clustering of Molecular Localizations with Graph Neural Networks},
  author={Pineda, Jes{\'u}s and Mas{\'o}-Orriols, Sergi and Bertran, Joan and Goks{\"o}r, Mattias and Volpe, Giovanni and Manzo, Carlo},
  journal={arXiv preprint arXiv:2412.00173},
  year={2024}
}
```

