# Spatial Clustering of Molecular Localizations with MIRO

**MIRO** (Multimodal Integration through Relational Optimization) is a geometric deep learning framework that enhances clustering algorithms by transforming complex point clouds into an optimized structure amenable to conventional clustering methods. 

## How it works?
**MIRO** employs recurrent graph neural networks (rGNNs) to learn a transformation that squeezes localization belonging to the same cluster toward a common center, resulting in a compact representation of clusters within the point cloud.

<div align="center">
  <img src="assets/MIRO.png" width="500"/>
</div>

## Potential of MIRO  
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

## Getting Started with MIRO 

Ready to dive into **MIRO**? Getting started is easy.

1. Make sure you have **Python 3.9** or higher installed.

2. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/DeepTrackAI/MIRO.git

3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   
Then unleash **MIRO**'s full potential.

## Citation
If you use **MIRO** in your research, please cite our paper:
```
@article{pineda2024spatial,
  title={Spatial Clustering of Molecular Localizations with Graph Neural Networks},
  author={Pineda, Jes{\'u}s and Mas{\'o}-Orriols, Sergi and Bertran, Joan and Goks{\"o}r, Mattias and Volpe, Giovanni and Manzo, Carlo},
  journal={arXiv preprint arXiv:2412.00173},
  year={2024}
}
```

