"""MIRO: Multimodal Integration through Relational Optimization

This module provides the MIRO framework for multi-scale point cloud clustering,
leveraging advanced geometric deep learning techniques. MIRO transforms complex
point clouds into optimized representations, enabling more effective clustering
using traditional algorithms.

Based on the original MIRO paper by Pineda et al. [1], this implementation offers
easy-to-use methods for training the MIRO model and performing geometric-aware
clustering. It integrates recurrent graph neural networks to refine point
cloud data and enhance clustering accuracy.

[1] Pineda, Jesús, et al. "Spatial Clustering of Molecular Localizations with
    Graph Neural Networks." arXiv preprint arXiv:2412.00173 (2024).
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, List, Iterable

from .miro import MIRO
from deeplay.external import Adam, Optimizer
from deeplay.models import RecurrentMessagePassingModel


class mMIRO(MIRO):
    """Multi‐scale data point cloud clustering using MIRO (Multimodal Integration
    through Relational Optimization).

    MIRO is a geometric deep learning framework that enhances clustering
    algorithms by transforming complex point clouds into an optimized structure
    amenable to conventional clustering methods. MIRO employs recurrent graph
    neural networks (rGNNs) to learn a transformation that squeezes localization
    belonging to the same cluster toward a common center, resulting in a compact
    representation of clusters within the point cloud.

    Parameters
    ----------
    num_outputs : int
        Dimensionality of the output features, representing a displacement
        vector in Cartesian space for each node. This vector points toward
        the center of each cluster.
    num_iterations : int
        Number of recurrent steps for the model to refine the point cloud.
    cutoff_iterations : int
        Number of iterations at the small scale before switching to the
        larger spatial scale (denoted k* in the original paper).
    connectivity_radius : float
        Maximum distance between two nodes to consider them connected in the
        graph.
    model : nn.Module
        A model implementing the forward method. It should return a list of
        tensors of shape `(num_nodes, num_outputs)` representing the predicted
        displacement vectors for each node at each recurrent iteration. If not
        specified, a default model resembling the one from the original MIRO
        paper is used.
    nd_loss_weight : float
        Weight for the auxiliary loss that enforces preservation of pairwise
        distances between connected nodes. Default is 10.
    loss : torch.nn.Module
        Loss function for training. Default is `torch.nn.L1Loss`.
    optimizer : Optimizer
        Optimizer for training. Default is Adam with a learning rate of 1e-4.

    Returns
    -------
    forward : method
        Computes and returns the predicted displacement vectors for each node
        in the input graph. The output is a list of tensors representing the
        displacement vectors at each recurrent iteration.

    squeeze : method
        Applies the predicted displacement vectors from the last recurrent
        iteration (by default) to the input point cloud. This operation
        optimizes the point cloud for clustering by aligning nodes closer to
        their respective cluster centers.

    clustering : method
        Groups nodes into clusters using the DBSCAN algorithm, based on the
        predicted displacement vectors. Each node is assigned a cluster label,
        where -1 indicates background noise. Returns an array of cluster labels
        for the nodes.

    Example
    -------
    >>> # Predicts displacement vectors for each node in a point cloud at each
    >>> # recurrent iteration
    >>> displacement_vectors = model(test_graph)
    >>> print(type(displacement_vectors))
    <class 'list'>

    >>> # Applies the predicted displacement vectors to the input point cloud
    >>> squeezed = model.squeeze(test_graph)
    >>> print(squeezed.shape)
    (num_nodes, 2)

    >>> # Performs clustering using DBSCAN after MIRO squeezing
    >>> eps = 0.3  # Maximum distance for cluster connection
    >>> min_samples = 5  # Minimum points in a neighborhood for core points
    >>> clusters = model.clustering(test_graph, eps, min_samples)

    >>> # Output cluster labels
    >>> print(clusters)
    array([ 0,  0,  1,  1,  1, -1,  2,  2,  2, ...])
    # Nodes in cluster 0, 1, 2, etc.; -1 are outliers
    """

    num_outputs: int
    connectivity_radius: float
    model: nn.Module
    nd_loss_weight: float
    loss: torch.nn.Module
    metrics: list
    optimizer: Optimizer

    def __init__(
        self,
        num_outputs: int = 2,
        num_iterations: int = 20,
        cutoff_iterations: int = 10,
        connectivity_radius: float = 1.0,
        model: Optional[nn.Module] = None,
        nd_loss_weight: float = 10,
        loss: torch.nn.Module = torch.nn.L1Loss(),
        optimizer=None,
        **kwargs,
    ):

        super().__init__(
            num_outputs=num_outputs,
            num_iterations=num_iterations,
            connectivity_radius=connectivity_radius,
            model=model,
            nd_loss_weight=nd_loss_weight,
            loss=loss,
            optimizer=optimizer,
            **kwargs,
        )
        self.cutoff_iterations = cutoff_iterations

    def compute_loss(
        self,
        y_hat: List[torch.Tensor],
        y: torch.Tensor,
        edges: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the total (multi-scale) loss for the model."""
        c = self.cutoff_iterations
        ys = [*(y.y_sub,) * c, *(y.y,) * (len(y_hat) - c)]

        total = sum(
            self.loss(p, t)
            + self.nd_loss_weight * self.compute_nd_loss(p, t, edges, position)
            for p, t in zip(y_hat, ys)
        )
        return total / len(y_hat)
